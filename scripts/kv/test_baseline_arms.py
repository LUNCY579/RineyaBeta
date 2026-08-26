#!/usr/bin/env python3
"""CPU tests that the three selection arms are actually three mechanisms.

WHY. A baseline that silently runs the mechanism it is supposed to control for
is worse than no baseline: it manufactures agreement. When `--select` was added
to the language harness, `streaming` produced SIGNIFICANCE selection while
writing "streaming" into the manifest -- select_keep only special-cases
`random`, and `streaming` is realised by the CALLER spending the whole budget
on sinks plus recent so that heavy() becomes zero. A synthetic probe caught it
in one run: the streaming arm still kept the single high-scoring position that
streaming must miss.

That is the same defect shape as the pool_back/pool_fwd footgun -- a
configuration that silently yields a different mechanism than the caller asked
for, with a plausible-looking manifest. These tests pin the distinction.

The probe is deliberately synthetic and extreme: one position carries a score
1000x the floor. Significance MUST keep it, and any arm that does not rank
cannot.

    python scripts/kv/test_baseline_arms.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kv import significance as sig  # noqa: E402

FAILED = 0
N, BUDGET, SINKS, RECENT, PEAK = 64, 16, 4, 4, 40


def check(label: str, got, want) -> None:
    global FAILED
    ok = got == want
    FAILED += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + ("" if ok else f"   got {got!r}, want {want!r}"))


def keep(select: str, recent: int, seed: int = 0) -> list[int]:
    s = torch.full((1, 1, N), 0.01)
    s[0, 0, PEAK] = 10.0
    p = sig.Policy(budget=BUDGET, sinks=SINKS, recent=recent, pool=7,
                   pool_back=6, pool_fwd=0, select=select)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sorted(sig.select_keep(s, N, p, torch.device("cpu"))[0, 0].tolist())


def main() -> None:
    print("the three selection arms must be three different mechanisms\n")

    # `recent` for streaming is what the HARNESS computes, not a Policy default:
    # both passkey.py and run_eval_sig.py spend the budget down to heavy() == 0.
    stream_recent = BUDGET - SINKS

    print("significance ranks, so it must keep the one high-scoring position:")
    check("keeps the peak", PEAK in keep("significance", RECENT), True)
    check("spends 8 heavy slots",
          sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT).heavy(), 8)

    print("\nrandom keeps the SAME NUMBER of slots but ignores the score:")
    check("same heavy count as significance",
          sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT, select="random").heavy(), 8)
    # A PROBABILISTIC assertion needs a sample, not twelve draws. The first
    # version of this check used seeds 0..11 and failed -- not because the
    # baseline was broken (measured over 400 draws it keeps the peak 15.8% of
    # the time against a uniform expectation of 8/56 = 14.3%) but because twelve
    # Bernoulli draws cannot resolve that rate. This programme has a documented
    # history of exactly this error: "a 6-trial depth curve read as monotone was
    # 0.42/0.33/0.83 at 12". A flaky test is worse than no test, because it
    # trains you to ignore a red result.
    DRAWS = 400
    hits = sum(1 for sd in range(DRAWS) if PEAK in keep("random", RECENT, seed=sd))
    rate, expected = hits / DRAWS, 8 / (N - SINKS - RECENT)
    # +/- 4 binomial SD: wide enough never to flake, tight enough to catch a
    # baseline that is secretly ranking (which would push the rate towards 1.0).
    sd_band = 4 * (expected * (1 - expected) / DRAWS) ** 0.5
    check(f"selects uniformly: {rate:.1%} vs {expected:.1%} expected "
          f"(+/-{sd_band:.1%})", abs(rate - expected) < sd_band, True)
    check("and is nowhere near always keeping the peak (would mean it ranks)",
          rate < 0.5, True)

    print("\nstreaming ranks NOTHING -- heavy must be zero and the peak must be missed:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check("heavy is zero",
              sig.Policy(budget=BUDGET, sinks=SINKS, recent=stream_recent,
                         select="streaming").heavy(), 0)
    check("misses the peak", PEAK in keep("streaming", stream_recent), False)

    print("\nthe regression that motivated this file:")
    # With the harness NOT spending the budget down, `streaming` silently ranks.
    check("streaming at the significance recent DOES rank (why the harness must "
          "set recent itself)", PEAK in keep("streaming", RECENT), True)

    print("\nthe B1 floor must not resurrect a streaming arm:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = sig.Policy(budget=BUDGET, sinks=SINKS, recent=stream_recent,
                       select="streaming", min_heavy_frac=0.5)
        check("streaming keeps heavy == 0 under the floor", p.heavy(), 0)
        check("streaming keeps its full recent window", p.n_recent(), stream_recent)

    print(f"\n{'all checks pass' if not FAILED else f'{FAILED} FAILED'}")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
