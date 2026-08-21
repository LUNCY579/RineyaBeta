#!/usr/bin/env python3
"""Does prompt length drop out of peak memory under chunked prefill?

The claim is an identity, not an estimate: processing in chunks of C and
compressing to a budget B between chunks bounds the resident cache at `B + C`
tokens regardless of n. This measures it against `torch.cuda.max_memory_allocated`
across a length ladder, which is the only way to know whether the identity
survives contact with activations, the allocator and the mask.

Two things are measured that the identity does not predict, and both matter:

  peak      the whole point. Should be flat in n.
  prefill   attention work per chunk is O(C * (B + C)) and there are n/C chunks,
            so total attention work is O(n * (B + C)) -- LINEAR in n, against
            O(n^2 / 2) un-chunked. At 128K with B=2048, C=4096 that is roughly a
            10x reduction in attention work, so prefill should scale sub-
            quadratically as well.

Correctness is NOT re-established here; `scripts/kv/chunked_prefill.py --gate`
does that, and it must pass before any number below means anything.

Usage:
    python src/measure_chunked.py --lengths 4096 16384 32768 65536
    python src/measure_chunked.py --lengths 131072 --budget 2048 --chunk 4096
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

from scripts.kv import significance as sig  # noqa: E402
from scripts.kv.chunked_prefill import chunked_prefill, stored_length  # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"


def resident_cache_bytes(cache) -> int:
    """De-duplicated by storage pointer -- a view must not be counted as a saving."""
    seen: dict[int, int] = {}
    for layer in getattr(cache, "layers", []) or []:
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if t is not None:
                st = t.untyped_storage()
                seen[st.data_ptr()] = st.nbytes()
    return sum(seen.values())


@torch.no_grad()
def run(model, n: int, budget: int, chunk: int, window: int, recent: int,
        device: str, chunked: bool) -> dict:
    g = torch.Generator(device="cpu").manual_seed(11)
    ids = torch.randint(0, model.config.vocab_size, (1, n), generator=g).to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sig.reset_scores()
    sig.reset_heavy_zero_count()

    policy = sig.Policy(budget=budget, sinks=4, recent=recent) if chunked else None
    cache = sig.SignificanceCache(config=model.config, policy=None)
    sig.set_recording(chunked, window=window)

    def compress_fn(c, ci):
        sig.compress(c, policy)

    t0 = time.perf_counter()
    if chunked:
        chunked_prefill(model, ids, cache, chunk=chunk, logits_to_keep=1,
                        compress_fn=compress_fn)
    else:
        model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1e3

    sig.set_recording(False)
    peak = torch.cuda.max_memory_allocated()
    stored = stored_length(cache)
    resident = resident_cache_bytes(cache)
    score_b = sig.score_bytes()
    hz = sig.heavy_zero_count()
    del cache, ids
    torch.cuda.empty_cache()

    return {
        "length": n, "chunked": chunked, "budget": budget if chunked else 0,
        "chunk": chunk if chunked else 0,
        "stored_kv": stored,
        "cache_resident_mib": round(resident / 2 ** 20, 2),
        "score_mib": round(score_b / 2 ** 20, 3),
        "peak_mib": round(peak / 2 ** 20, 1),
        "prefill_ms": round(elapsed, 1),
        "heavy_zero": hz,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[4096, 16384, 32768, 65536])
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--recent", type=int, default=512)
    ap.add_argument("--baseline", action="store_true",
                    help="also run un-chunked, which will OOM past ~16K")
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "measure_chunked.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs cuda")
    from compat import force_repeat_kv
    force_repeat_kv(verbose=False)
    sig.patch_attention()
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, total) / total)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(args.device).eval()

    print(f"chunked prefill: B={args.budget} C={args.chunk} recent={args.recent} "
          f"window={args.window}, cap {args.cap_mib:.0f} MiB")
    print(f"identity predicts resident cache <= (B + C) = {args.budget + args.chunk} "
          f"tokens = {(args.budget + args.chunk) * 112 / 1024:.0f} MiB, flat in n\n")
    print(f"{'n':>9}{'mode':>10}{'stored':>8}{'cache MiB':>11}{'sig MiB':>9}"
          f"{'peak MiB':>10}{'prefill ms':>12}")

    rows = []
    for n in args.lengths:
        modes = [False, True] if args.baseline else [True]
        for chunked in modes:
            try:
                r = run(model, n, args.budget, args.chunk, args.window,
                        args.recent, args.device, chunked)
            except torch.cuda.OutOfMemoryError:
                print(f"{n:>9,}{'chunked' if chunked else 'single':>10}   OOM")
                torch.cuda.empty_cache()
                continue
            rows.append(r)
            print(f"{r['length']:>9,}{'chunked' if chunked else 'single':>10}"
                  f"{r['stored_kv']:>8}{r['cache_resident_mib']:>11.1f}"
                  f"{r['score_mib']:>9.2f}{r['peak_mib']:>10.1f}{r['prefill_ms']:>12.1f}")

    ch = [r for r in rows if r["chunked"]]
    if len(ch) > 1:
        peaks = [r["peak_mib"] for r in ch]
        print(f"\npeak across n = {min(ch, key=lambda r: r['length'])['length']:,}"
              f" .. {max(ch, key=lambda r: r['length'])['length']:,}: "
              f"{min(peaks):.1f} .. {max(peaks):.1f} MiB "
              f"(spread {100 * (max(peaks) - min(peaks)) / min(peaks):.1f}%)")
        print(f"resident cache: {set(r['cache_resident_mib'] for r in ch)}")
        base = min(ch, key=lambda r: r["length"])
        for r in ch:
            if r["length"] == base["length"]:
                continue
            ratio_n = r["length"] / base["length"]
            ratio_t = r["prefill_ms"] / max(base["prefill_ms"], 1e-9)
            print(f"  n x{ratio_n:>5.0f} -> prefill x{ratio_t:>6.1f} "
                  f"(quadratic would be x{ratio_n ** 2:.0f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"gpu": torch.cuda.get_device_name(0), "budget": args.budget,
         "chunk": args.chunk, "recent": args.recent, "window": args.window,
         "results": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
