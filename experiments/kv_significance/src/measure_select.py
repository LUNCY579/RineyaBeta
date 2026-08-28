#!/usr/bin/env python3
"""O12 -- per-compression cost of the POOLING VARIANTS, with no model at all.

WHY THIS IS THE RIGHT INSTRUMENT for the comparative question. O12 asks for
"wall-clock per compression round for centred pooling, fixed backward reach, and
derived reach at matched budget". A compression round is two things: choosing
which positions to keep (`select_keep`, which is where the variants differ) and
gathering them out of the cache (identical across variants). Timing it through a
model measures both plus decode noise plus thermal drift; timing `select_keep`
directly measures exactly the part that differs, on tensors of the real shape.

It also runs in a few hundred MB, which matters: the machine this was written on
had 1.04 GB of RAM free and could not load a model at all.

WHAT IT DOES NOT MEASURE, stated so the number is not over-read: the gather, the
attention-time score accumulation (that is the ~22% already measured by
measure_decode's evict/evict_noscore arms), and the query-aware prefill pass.
This is the selection step alone.

Thermal control is not needed on CPU in the way it is on the 4070 -- but arms
are still INTERLEAVED round-robin and the statistic is the MINIMUM across
rounds, for the same reason measure_decode uses it: the minimum is the
least-contended observation of the same work.

    python experiments/kv_significance/src/measure_select.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.kv import significance as sig  # noqa: E402

# (label, Policy kwargs) -- everything matched but the pooling variant.
ARMS = [
    ("no pooling (pool=1)",      dict(pool=1)),
    ("centred (3,3)",            dict(pool=7)),
    ("causal (6,0)",             dict(pool=7, pool_back=6, pool_fwd=0)),
    ("derived reach grow=2.0",   dict(pool=7, grow=2.0, grow_max=64)),
    ("derived reach grow=5.0",   dict(pool=7, grow=5.0, grow_max=64)),
]
# (n_kv_heads, cache length) -- the real shapes this mechanism runs at.
SHAPES = [(8, 2048), (8, 8192), (8, 32768), (8, 131072)]
BUDGET, SINKS, RECENT = 64, 4, 16


def main() -> None:
    rounds, reps = 5, 20
    torch.manual_seed(0)
    out = []
    for n_kv, n in SHAPES:
        # one score tensor per shape, reused by every arm: the arms must see
        # IDENTICAL input or the comparison is of inputs, not of policies.
        s = torch.rand(1, n_kv, n) * 0.05
        s[0, :, n // 2] = 1.0
        samples = {lbl: [] for lbl, _ in ARMS}
        for _ in range(rounds):                      # interleaved, not blocked
            for lbl, kw in ARMS:
                pol = sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT, **kw)
                sig.select_keep(s, n, pol, torch.device("cpu"))   # warm
                t0 = time.perf_counter()
                for _ in range(reps):
                    sig.select_keep(s, n, pol, torch.device("cpu"))
                samples[lbl].append((time.perf_counter() - t0) / reps * 1000.0)
        base = min(samples["causal (6,0)"])
        print(f"\n  cache length {n:,} ({n_kv} KV heads), budget {BUDGET}")
        print(f"    {'arm':<26}{'ms/compression':>16}{'vs causal':>12}")
        for lbl, _ in ARMS:
            best = min(samples[lbl])
            print(f"    {lbl:<26}{best:>16.4f}{best / base:>11.2f}x")
            out.append({"n": n, "n_kv": n_kv, "arm": lbl,
                        "ms_min": round(best, 5),
                        "ms_median": round(statistics.median(samples[lbl]), 5),
                        "vs_causal": round(best / base, 4)})
    p = ROOT / "experiments/kv_significance/results/measure_select.json"
    p.write_text(json.dumps({"budget": BUDGET, "sinks": SINKS, "recent": RECENT,
                             "rounds": rounds, "reps": reps, "device": "cpu",
                             "statistic": "minimum across interleaved rounds",
                             "rows": out}, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
