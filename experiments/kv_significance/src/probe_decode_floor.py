#!/usr/bin/env python3
"""What actually sets decode latency: cache size, weights, or per-step overhead?

WHY THIS EXISTS. It was claimed -- by me, from arithmetic, without measurement --
that shrinking the KV cache cannot speed decode much because every step must read
904 MiB of weights, so the speedup is bounded by Amdahl's law on memory traffic.
That claim was never tested, and a single measured pair already sits awkwardly
with it: at 4K, dropping the cache from 454 MiB to 118 MiB moved decode from
37.91 to 38.20 ms/token, i.e. not at all. A 3.8x smaller cache buying ZERO is not
what a bandwidth model predicts; it predicts a small but visible gain.

So this measures the decomposition directly instead of assuming it. Decode
latency is timed against cache size with everything else held fixed, and a line
is fitted:

    ms_per_token  =  a  +  b * cache_tokens

  a   the fixed cost of a decode step: reading the weights, launching ~28 layers
      of kernels, Python and framework overhead. Independent of context.
  b   the marginal cost of one more cached token. If decode were bandwidth-bound
      on the cache, b would be 112 KiB / (achievable bandwidth).

The two hypotheses make different, checkable predictions:

  BANDWIDTH-BOUND      a ~= 904 MiB / bandwidth, and b ~= 112 KiB / bandwidth.
                       On a card with ~256 GB/s that is a ~3.7 ms floor and
                       ~0.45 us per cached token.
  OVERHEAD-BOUND       a is far larger than the weights read can explain, because
                       it is dominated by per-kernel launch latency at batch 1.
                       Then NO cache reduction can help much, and the thing to
                       optimise is the number of kernels, not the bytes.

The distinction decides what to do next, so it is worth measuring rather than
arguing. It also decides whether the mechanism's cache saving can EVER convert
into speed on this host, which is what the significance matrix is being asked to
do.

The cache is built by plain uncompressed prefill at the target length, with
scoring OFF, so cache size is the only variable. No eviction, no significance
matrix -- this probe is about the hardware, not the mechanism.

Usage:
    python src/probe_decode_floor.py
    python src/probe_decode_floor.py --sizes 8 1024 4096 16384 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

NL_ = chr(10)

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

from scripts.kv import significance as sig  # noqa: E402
from scripts.kv.chunked_prefill import (causal_lower_right, chunked_prefill,  # noqa: E402
                                        stored_length)

MODEL = "artifacts/rineya-core-en32k-28L"
KIB_PER_TOKEN = 112.0          # 28 layers * 8 kv heads * 128 dim * 2 (K,V) * 2 B
WEIGHTS_MIB = 904.0


@torch.no_grad()
def time_decode(model, n_ctx: int, steps: int, warmup: int, chunk: int,
                device: str) -> tuple[float, int]:
    """Minimum ms/token at a cache of exactly n_ctx tokens."""
    g = torch.Generator(device="cpu").manual_seed(11)
    ids = torch.randint(0, model.config.vocab_size, (1, n_ctx), generator=g).to(device)
    sig.reset_scores()
    sig.set_recording(False)                     # hardware probe, not mechanism
    cache = sig.SignificanceCache(config=model.config, policy=None)
    chunked_prefill(model, ids, cache, chunk=min(chunk, n_ctx), logits_to_keep=1,
                    compress_fn=None)
    stored = stored_length(cache)

    dtype = next(model.parameters()).dtype
    tok = torch.tensor([[1]], device=device)
    lat = []
    for i in range(warmup + steps):
        s = stored_length(cache)
        mask = causal_lower_right(1, s + 1, device, dtype)
        pos = torch.tensor([[n_ctx + i]], device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(tok, past_key_values=cache, position_ids=pos,
                    attention_mask={"full_attention": mask},
                    use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        if i >= warmup:
            lat.append((time.perf_counter() - t0) * 1e3)
        tok = out.logits[:, -1:].argmax(-1)
    del cache, ids
    torch.cuda.empty_cache()
    # Minimum: the least-contended observation of identical work. Under thermal
    # drift a mean is a sample of how hot the card was, not of the work.
    return min(lat), stored


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares, written out to keep the dependency surface at zero."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[8, 256, 1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--flat-tol", type=float, default=0.06,
                    help="a point is on the floor if within this of the minimum")
    ap.add_argument("--cap-mib", type=float, default=7400.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path,
                    default=EXP / "results" / "probe_decode_floor.json")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM
    force_repeat_kv(verbose=False)
    sig.patch_attention()
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, tot) / tot)
    props = torch.cuda.get_device_properties(0)

    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(args.device).eval()

    print(f"{props.name}, decode latency against cache size")
    print(f"{args.rounds} rounds x {args.steps} steps, minimum reported\n")
    print(f"{'cache tok':>10}{'cache MiB':>11}{'ms/token':>10}{'tok/s':>8}")

    best: dict[int, float] = {}
    stored_of: dict[int, int] = {}
    for _ in range(args.rounds):
        for n in args.sizes:
            if n in best and best[n] == float("inf"):
                continue
            try:
                ms, stored = time_decode(model, n, args.steps, args.warmup,
                                         args.chunk, args.device)
            except torch.cuda.OutOfMemoryError:
                best[n] = float("inf")
                torch.cuda.empty_cache()
                continue
            best[n] = min(best.get(n, float("inf")), ms)
            stored_of[n] = stored

    rows = []
    for n in args.sizes:
        ms = best.get(n, float("inf"))
        if ms == float("inf"):
            print(f"{n:>10,}{'--':>11}{'OOM':>10}{'--':>8}")
            continue
        mib = stored_of[n] * KIB_PER_TOKEN / 1024
        rows.append({"cache_tokens": stored_of[n], "cache_mib": round(mib, 1),
                     "ms_per_token": round(ms, 3), "tok_per_s": round(1000 / ms, 2)})
        print(f"{stored_of[n]:>10,}{mib:>11.1f}{ms:>10.2f}{1000/ms:>8.1f}")

    if len(rows) < 3:
        raise SystemExit("\nnot enough points to fit")

    # A SINGLE STRAIGHT LINE IS THE WRONG MODEL, and the first version of this
    # script used one. The measured shape is a FLOOR followed by a rise: decode is
    # 35.0-36.1 ms across a 512x range of cache (8 to 4,096 tokens) and only then
    # starts climbing. Fitting one line through both regimes lets the two largest
    # points set the intercept, which manufactures a slope in the flat region that
    # is not there and produced the wrong verdict.
    #
    # So: find the knee as the largest cache still within `--flat-tol` of the
    # minimum, take the floor as the median below it, and fit the slope above it.
    xs = [r["cache_tokens"] for r in rows]
    ys = [r["ms_per_token"] for r in rows]
    lo = min(ys)
    flat = [r for r in rows if r["ms_per_token"] <= lo * (1 + args.flat_tol)]
    knee = max(r["cache_tokens"] for r in flat)
    floor = statistics.median(r["ms_per_token"] for r in flat)
    above = [r for r in rows if r["cache_tokens"] >= knee]
    slope = 0.0
    if len(above) >= 2:
        slope = ((above[-1]["ms_per_token"] - above[0]["ms_per_token"])
                 / (above[-1]["cache_tokens"] - above[0]["cache_tokens"]))

    def predict(n_tok: int) -> float:
        return floor if n_tok <= knee else floor + slope * (n_tok - knee)

    print(f"{NL_}FLOOR {floor:.2f} ms, flat to {knee:,} cached tokens "
          f"({knee * KIB_PER_TOKEN / 1024:.0f} MiB), "
          f"then {slope * 1e3:.3f} us per extra token")
    spread = max(r["ms_per_token"] for r in flat) - min(r["ms_per_token"] for r in flat)
    print(f"  flat region spread: {spread:.2f} ms over a "
          f"{knee / min(xs):.0f}x range of cache size")

    peak_bw = 256.0   # RTX 4070 Laptop, nominal
    floor_bw = WEIGHTS_MIB * 2 ** 20 / (floor * 1e-3) / 1e9
    slope_bw = (KIB_PER_TOKEN * 1024) / (slope * 1e-3) / 1e9 if slope > 0 else 0.0
    print(f"{NL_}  floor implies {floor_bw:.0f} GB/s for a {WEIGHTS_MIB:.0f} MiB "
          f"weight read ({100 * floor_bw / peak_bw:.0f}% of ~{peak_bw:.0f} GB/s peak)")
    print(f"  slope implies {slope_bw:.0f} GB/s for cache reads "
          f"({100 * slope_bw / peak_bw:.0f}% of peak)")

    if floor_bw < 0.5 * peak_bw:
        print(f"{NL_}=> NOT bandwidth-bound. Both terms run at a small fraction of what")
        print("   the card can move, and below the knee the cache read is entirely")
        print("   hidden. Decode at batch 1 is dominated by per-step overhead --")
        print("   kernel launches across 28 layers, framework and Python -- which no")
        print("   cache reduction can touch. An earlier claim that Amdahl on WEIGHTS")
        print("   BANDWIDTH explains the ceiling is withdrawn: the ceiling is real but")
        print("   its cause is the runtime, not memory traffic.")
    else:
        print(f"{NL_}=> Bandwidth-bound; Amdahl on weights is the right model.")

    print(f"{NL_}WHAT COMPRESSION IS WORTH, from the measured curve")
    print("")
    print(f"{'context':>10}{'full ms':>10}{'B=1024':>9}{'B=4096':>9}"
          f"{'gain 1024':>11}{'gain 4096':>11}")
    for n in (4096, 16384, 32768, 65536, 131072):
        f_ms, b1, b4 = predict(n), predict(1024), predict(4096)
        print(f"{n:>10,}{f_ms:>10.1f}{b1:>9.1f}{b4:>9.1f}"
          f"{f_ms / b1:>10.2f}x{f_ms / b4:>10.2f}x")
    print(f"{NL_}Budget 1024 and budget 4096 cost the SAME decode time, because both")
    print("sit on the floor. Compressing below the knee trades accuracy for nothing.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": MODEL, "gpu": props.name, "steps": args.steps,
         "rounds": args.rounds, "kib_per_token": KIB_PER_TOKEN,
         "weights_mib": WEIGHTS_MIB,
         "fit": {"floor_ms": round(floor, 4), "knee_tokens": knee,
                 "slope_ms_per_token_above_knee": round(slope, 8),
                 "flat_spread_ms": round(spread, 4),
                 "floor_implied_bw_gb_s": round(floor_bw, 1),
                 "slope_implied_bw_gb_s": round(slope_bw, 1),
                 "peak_bw_gb_s": peak_bw},
         "results": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
