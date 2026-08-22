#!/usr/bin/env python3
"""P0.2 -- even with a perfect selector, is a sparse FFN faster on this GPU?

WHY THIS RUNS BEFORE ANY SELECTOR IS BUILT. P0.1 established that 25% of the
neurons carry 96% of the FFN output, so a perfect oracle exists. This asks the
separate question of whether exploiting it is *worth anything on this machine*,
and it needs no checkpoint and no training to answer.

The reason to doubt it is measured, not theoretical. `probe_decode_floor.py`
found decode latency flat across a 512x range of KV cache size, with a floor
implying 27 GB/s for a 904 MiB weight read on a card rated near 256 -- about 11%
of peak. A machine running at 11% of its bandwidth is not waiting on bytes, and
reading fewer bytes cannot speed it up. Sparse gathers add irregular access and
extra kernel launches, which is what an overhead-bound regime punishes. This
program has already been caught once: a 3.8x smaller KV cache changed decode by
0.29 ms.

WHAT IS TIMED. One SwiGLU block at this checkpoint's real shape (hidden 1024,
intermediate 3072), three variants:

  dense       x @ Wg, x @ Wu, silu*, @ Wd            -- the baseline
  gather      index_select the k live columns/rows every call, then matmul
              -- honest cost of a per-token selector
  pregathered the same matmul on already-narrow weights, gather excluded
              -- the unreachable upper bound, which localises the cost: if
                 pregathered wins and gather does not, the gather is the problem
                 and a different data layout might fix it; if neither wins, the
                 idea is dead here regardless of layout

Batch 1 (decode, the regime that matters for latency) and batch 8 (a serving
regime), because a negative at 1 and a positive at 8 localises the finding to
where it applies rather than condemning the idea outright.

PRE-DECLARED THRESHOLD (docs/EXPERIMENT_DESIGN_STABILIZER.md section 5): if
`gather` at 25% density is not at least 1.3x faster than dense at batch 1, the
speed claim is dead on this host whatever the selector does, and the work
continues -- if at all -- as a memory result only.

Usage:
    python src/probe_sparse_matmul.py
    python src/probe_sparse_matmul.py --iters 400 --batches 1 8 32
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
H, I_ = 1024, 3072          # this checkpoint's MLP shape
LAYERS = 28
DENSITIES = (1.0, 0.75, 0.50, 0.375, 0.25, 0.125)


def timeit(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    # Minimum, not mean: the least-contended observation of identical work. This
    # host's thermal band is 2.17x, so a mean samples temperature, not work.
    return min(ts)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "sparse_matmul.json")
    args = ap.parse_args()

    dev = args.device
    dt = torch.bfloat16
    g = torch.Generator(device="cpu").manual_seed(3)
    Wg = torch.randn(H, I_, generator=g).to(dev, dt)      # [1024, 3072]
    Wu = torch.randn(H, I_, generator=g).to(dev, dt)
    Wd = torch.randn(I_, H, generator=g).to(dev, dt)      # [3072, 1024]

    props = torch.cuda.get_device_properties(0)
    print(f"P0.2 -- sparse vs dense SwiGLU, {props.name}")
    print(f"hidden {H}, intermediate {I_}, bf16, minimum of {args.iters} iters\n")

    rows = []
    for B in args.batches:
        x = torch.randn(B, H, generator=g).to(dev, dt)

        def dense():
            a = F.silu(x @ Wg) * (x @ Wu)
            return a @ Wd

        base = timeit(dense, args.iters, args.warmup)
        print(f"batch {B}: dense {base:.4f} ms/layer  "
              f"({base*LAYERS:.2f} ms for {LAYERS} layers)")
        print(f"{'density':>9}{'gather ms':>12}{'speedup':>10}"
              f"{'pregath ms':>13}{'speedup':>10}")

        for d in DENSITIES:
            k = max(1, int(round(d * I_)))
            idx = torch.randperm(I_, generator=g)[:k].to(dev)
            Wg_k = Wg.index_select(1, idx).contiguous()
            Wu_k = Wu.index_select(1, idx).contiguous()
            Wd_k = Wd.index_select(0, idx).contiguous()

            def gather():
                gk = Wg.index_select(1, idx)
                uk = Wu.index_select(1, idx)
                dk = Wd.index_select(0, idx)
                a = F.silu(x @ gk) * (x @ uk)
                return a @ dk

            def pregathered():
                a = F.silu(x @ Wg_k) * (x @ Wu_k)
                return a @ Wd_k

            tg = timeit(gather, args.iters, args.warmup)
            tp = timeit(pregathered, args.iters, args.warmup)
            rows.append({"batch": B, "density": d, "k": k,
                         "dense_ms": base, "gather_ms": tg, "pregathered_ms": tp,
                         "gather_speedup": base / tg, "pregathered_speedup": base / tp})
            print(f"{d*100:>8.1f}%{tg:>12.4f}{base/tg:>9.2f}x"
                  f"{tp:>13.4f}{base/tp:>9.2f}x")
            del Wg_k, Wu_k, Wd_k
            torch.cuda.empty_cache()
        print()

    r25 = next((r for r in rows if r["batch"] == 1 and abs(r["density"] - 0.25) < 1e-9), None)
    verdict = "no 25% cell measured"
    if r25:
        s = r25["gather_speedup"]
        verdict = ("PASS -- sparse is worth it at batch 1" if s >= 1.3 else
                   "FAIL -- the speed claim is dead on this host; continue as a "
                   "MEMORY result only")
        print(f"batch 1, 25% density: gather {s:.2f}x vs the 1.30x threshold")
        print(f"  -> {verdict}")
        p = r25["pregathered_speedup"]
        if s < 1.3 <= p:
            print("\n  The pregathered variant DOES clear the bar, so the cost is the\n"
                  "  gather, not the narrower matmul. A layout that keeps live neurons\n"
                  "  contiguous -- or a selector fixed across many tokens -- could\n"
                  "  recover it. That is a different experiment, not this one.")
        elif p < 1.3:
            print("\n  Even the pregathered upper bound misses the bar, so no data\n"
                  "  layout rescues this: at batch 1 the FFN matmul is not what the\n"
                  "  time is going to.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"gpu": props.name, "hidden": H, "intermediate": I_, "layers": LAYERS,
         "iters": args.iters, "threshold": 1.3, "verdict": verdict,
         "results": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
