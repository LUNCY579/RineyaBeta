#!/usr/bin/env python3
"""Measure what the analytic model only predicts: cache bytes, prefill latency,
decode throughput.

The repository has never measured any of these -- a grep for latency, throughput,
kv_cache or prefill across `scripts/` finds only prose. This closes that gap for
the attention-shape question.

Three quantities per (variant, length):

  cache bytes    summed exactly from the live Cache tensors, then cross-checked
                 against `shapes.py`. A mismatch means the analytic model is
                 wrong and every projection built on it is void, so this is
                 reported as a relative error rather than assumed.
  prefill        wall time to process the full prompt, CUDA-event timed, warmups
                 discarded, reported as median over repeats.
  decode         tokens/s generating from a filled cache -- the number that
                 actually governs interactive use, and the one where sliding
                 window should win beyond its memory saving, because attention
                 reads fewer keys per step.

Deliberately NOT importing `scripts/train/determinism.py`: it disables the
flash and mem-efficient SDPA backends, which makes long context impossible
(the eager score matrix is 8.6 GB per layer at 16K). This runs under the
evaluation-path determinism contract instead -- greedy, fixed seed, SDPA
permitted, reproducibility verified at the score level rather than bitwise.

Usage:
    python src/measure.py --lengths 1024 4096 16384 --variants base-8KV swa-w1024-g4
    python src/measure.py --all --lengths 1024 4096 16384 32768 --out results/measure.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compat import force_repeat_kv  # noqa: E402
from shapes import Shape  # noqa: E402
from variants import REPO, Variant, catalogue, load_variant  # noqa: E402

SURGERY_ROOT = "artifacts/attn_shape"


def surgery_dir_for(v: Variant) -> str | None:
    return None if not v.needs_surgery else f"{SURGERY_ROOT}/r2-gqa{v.surgery_kv_heads}kv"


def exact_cache_bytes(past) -> int:
    total = 0
    for layer in getattr(past, "layers", []) or []:
        for t in (layer.keys, layer.values):
            if t is not None:
                total += t.numel() * t.element_size()
    return total


def timed(fn, repeats: int, warmup: int, device: str) -> list[float]:
    """Median-friendly CUDA-event timings in milliseconds."""
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            out.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t0) * 1e3)
    return out


@torch.no_grad()
def measure_one(model, shape: Shape, length: int, device: str, decode_tokens: int,
                repeats: int, warmup: int) -> dict:
    vocab = model.config.vocab_size
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.randint(0, vocab, (1, length), generator=g).to(device)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --- prefill -----------------------------------------------------------
    # logits_to_keep=1 is not an optimisation, it is the correct measurement:
    # real prefill only needs the last position's logits, and materialising all
    # of them costs 16384 x 32768 x 4 B = 2 GiB in fp32 plus 1 GiB in bf16 at
    # 16K. That alone OOM'd the first capped sweep and had nothing to do with
    # the KV cache under study.
    out = model(ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    cache_b = exact_cache_bytes(past)
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0

    predicted = shape.cache_bytes(length)
    rel_err = abs(cache_b - predicted) / max(predicted, 1)

    prefill_ms = timed(lambda: model(ids, use_cache=True, logits_to_keep=1),
                       repeats, warmup, device)

    # --- decode from a filled cache ----------------------------------------
    # Time ONLY the decode loop. An earlier version timed prefill+decode and
    # subtracted a median prefill, which at 32K is a small difference of two
    # large noisy numbers and produced decode rates that moved non-monotonically
    # with cache size. Here the cache is filled once, untimed, and each timed
    # block continues from it; the cache grows by repeats*decode_tokens, which
    # is <0.2% of the context at every length measured and is reported so the
    # bias is visible rather than hidden.
    state = {"pkv": past, "tok": out.logits[:, -1:].argmax(-1)}

    def decode_block():
        pkv, tok = state["pkv"], state["tok"]
        for _ in range(decode_tokens):
            o = model(tok, past_key_values=pkv, use_cache=True, logits_to_keep=1)
            pkv = o.past_key_values
            tok = o.logits[:, -1:].argmax(-1)
        state["pkv"], state["tok"] = pkv, tok

    decode_ms = timed(decode_block, repeats, 1, device)
    tps = [decode_tokens / (d / 1e3) for d in decode_ms]
    cache_drift = (repeats + 1) * decode_tokens

    del past, out, state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "length": length,
        "cache_bytes_measured": cache_b,
        "cache_bytes_analytic": predicted,
        "cache_rel_err": round(rel_err, 6),
        "cache_mib": round(cache_b / 2 ** 20, 2),
        "peak_alloc_mib": round(peak / 2 ** 20, 2),
        "prefill_ms_median": round(statistics.median(prefill_ms), 2),
        "prefill_ms_min": round(min(prefill_ms), 2),
        "decode_tok_per_s": round(statistics.median(tps), 1),
        "decode_tok_per_s_iqr": round(
            (max(tps) - min(tps)) if len(tps) < 4
            else statistics.quantiles(tps, n=4)[2] - statistics.quantiles(tps, n=4)[0], 1),
        "decode_cache_drift_tokens": cache_drift,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-compat", action="store_true",
                    help="leave enable_gqa alone (reproduces the math-kernel OOM)")
    ap.add_argument("--cap-mib", type=float, default=6144.0,
                    help="hard allocator ceiling; 0 disables")
    args = ap.parse_args()

    if not args.no_compat:
        force_repeat_kv()

    # Windows/WDDM does not OOM when VRAM runs out -- it spills into system RAM
    # and the run goes quiet at 0% GPU utilisation for minutes before dying with
    # no output. A per-process cap turns that silent stall into an immediate,
    # catchable OutOfMemoryError, which is the same trade `scripts/train/memory.py`
    # makes for the trainer. An uncapped 32K sweep was killed this way, losing
    # its buffered stdout entirely.
    if args.cap_mib and args.device.startswith("cuda"):
        total_mib = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        frac = min(args.cap_mib, total_mib) / total_mib
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"allocator cap: {args.cap_mib:.0f} MiB of {total_mib:.0f} MiB "
              f"({frac:.2f})")

    cat = {v.name: v for v in catalogue()}
    if args.all:
        names = list(cat)
    elif args.variants:
        names = args.variants
    else:
        names = ["base-8KV", "swa-w1024-g4", "swa-w256-g4", "gqa-2KV"]

    unknown = [n for n in names if n not in cat]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}\navailable: {list(cat)}")

    print(f"device {args.device}   lengths {args.lengths}   "
          f"decode {args.decode_tokens} tok   repeats {args.repeats}")
    if args.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        print(f"gpu {torch.cuda.get_device_name(0)}  "
              f"free {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    print()

    results = []
    for name in names:
        v = cat[name]
        sd = surgery_dir_for(v)
        if v.needs_surgery and not (REPO / sd).exists():
            print(f"{name:20} SKIP - run gqa_surgery.py first ({sd} missing)")
            continue
        try:
            model = load_variant(v, device=args.device, surgery_dir=sd)
        except Exception as e:  # noqa: BLE001
            print(f"{name:20} LOAD FAILED: {type(e).__name__}: {e}")
            continue

        # bf16 on GPU, fp32 on CPU -- tell the analytic model which.
        elem = 2 if args.device.startswith("cuda") else 4
        shape = Shape(name=v.shape.name, n_layers=v.shape.n_layers,
                      kv_heads=v.shape.kv_heads, head_dim=v.shape.head_dim,
                      bytes_per_elem=elem, windows=v.shape.windows,
                      share=v.shape.share, d_latent=v.shape.d_latent,
                      d_rope=v.shape.d_rope)

        for L in args.lengths:
            try:
                r = measure_one(model, shape, L, args.device, args.decode_tokens,
                                args.repeats, args.warmup)
            except torch.cuda.OutOfMemoryError:
                print(f"{name:20} {L:>7,}  OOM")
                torch.cuda.empty_cache()
                continue
            r["variant"] = name
            r["unhealed_surgery"] = v.needs_surgery
            results.append(r)
            flag = "" if r["cache_rel_err"] < 0.01 else "  <-- ANALYTIC MISMATCH"
            print(f"{name:20} {L:>7,}  cache {r['cache_mib']:>8.1f} MiB "
                  f"(err {r['cache_rel_err']:.4f})  peak {r['peak_alloc_mib']:>8.1f}  "
                  f"prefill {r['prefill_ms_median']:>8.1f} ms  "
                  f"decode {r['decode_tok_per_s']:>7.1f} tok/s{flag}")

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "torch": torch.__version__,
            "decode_tokens": args.decode_tokens,
            "results": results,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    bad = [r for r in results if r["cache_rel_err"] >= 0.01]
    print(f"\n{len(results)} cells measured; "
          f"{len(results) - len(bad)} agree with the analytic model within 1%")
    if bad:
        print("MISMATCHES -- shapes.py is wrong for: "
              f"{sorted({r['variant'] for r in bad})}")


if __name__ == "__main__":
    main()
