#!/usr/bin/env python3
"""CPU tests for B1, the self-sizing budget. No GPU, no model, no weights.

WHAT THIS GUARDS. `recent` was a fixed constant. On a 232-token rag_dev prompt
at keep 0.25 the entire budget is ~58, so sinks(4) + recent(32) consumed 61% of
it and Policy.heavy() hit ZERO on 17,360 compressions -- the policy silently
degenerated to StreamingLLM, and the resulting scores were read as the
mechanism underperforming at aggressive compression. It was not
underperforming. It was not running.

The counter that exposed this already existed and was recorded in every
manifest. Nobody read it. These tests make the failure impossible rather than
merely observable, which is the only version of the fix that survives someone
not reading a counter again.

    python scripts/kv/test_self_sizing_budget.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kv.significance import Policy  # noqa: E402

FAILED = 0


def check(label: str, got, want) -> None:
    global FAILED
    ok = got == want
    FAILED += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + ("" if ok else f"   got {got!r}, want {want!r}"))


def main() -> None:
    print("B1 -- the self-sizing budget\n")

    print("the defect, reproduced exactly:")
    d = Policy(budget=58, sinks=4, recent=32)
    check("budget 58, fixed recent=32 -> score gets 22 of 58", d.heavy(), 22)
    check("  ...which is under 40% of the budget", d.heavy() / d.budget < 0.40, True)
    bad = Policy(budget=11, sinks=4, recent=32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check("budget 11 -> heavy() is ZERO: silently StreamingLLM", bad.heavy(), 0)

    print("\nrecent_frac -- the window scales with the budget:")
    for b, want in ((58, 14), (100, 25), (16, 4)):
        check(f"budget {b}, recent_frac 0.25 -> recent {want}",
              Policy(budget=b, sinks=4, recent=32, recent_frac=0.25).n_recent(), want)

    print("\nmin_heavy_frac -- the score's share cannot be starved:")
    # THE CONSTRAINT HIERARCHY, learned by breaking it. Three things compete
    # for a tiny budget and they are NOT equal:
    #   1. sinks  -- load-bearing and measured: sinks=0 scores 0.000 even with
    #                MORE score-chosen slots. Never yields.
    #   2. recent -- must be >= 1 because it absorbs the SCORE LAG; _evict
    #                raises otherwise. Structural, not heuristic. Yields to 1.
    #   3. heavy  -- the mechanism. Gets what is left.
    # With sinks=4 the smallest budget that can run the mechanism at all is
    # 4 + 1 + 1 = 6. Below that heavy() is legitimately 0. The first version of
    # this test asserted heavy > 0 at budget 5, which is arithmetically
    # impossible once recent >= 1 holds -- written before I knew the recent
    # window had a structural role, and three B1 arms died proving it.
    for b in (8, 11, 16, 32, 58, 128, 1024):
        p = Policy(budget=b, sinks=4, recent=32, min_heavy_frac=0.5)
        check(f"budget {b:>4} -> heavy {p.heavy()} > 0", p.heavy() > 0, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tiny = Policy(budget=5, sinks=4, recent=32, min_heavy_frac=0.5)
        check("budget 5 is BELOW the viable floor (4 sinks + 1 recent leaves none)",
              (tiny.n_sinks(), tiny.n_recent(), tiny.heavy()), (4, 1, 0))
    for b in (5, 8, 10, 12, 16, 20, 30, 58, 116, 1024):
        p = Policy(budget=b, sinks=4, recent=32, min_heavy_frac=0.78)
        check(f"budget {b:>4} -> recent {p.n_recent()} >= 1 (absorbs the score lag)",
              p.n_recent() >= 1, True)

    print("\n  the floor yields the RECENT window, never the sinks:")
    p = Policy(budget=11, sinks=4, recent=32, min_heavy_frac=0.5)
    check("budget 11: sinks still 4", p.n_sinks(), 4)
    check("budget 11: recent squeezed to 1", p.n_recent(), 1)
    check("budget 11: score gets 6", p.heavy(), 6)

    print("\n  and it never overspends the budget:")
    for b in (5, 11, 58, 128):
        p = Policy(budget=b, sinks=4, recent=32, min_heavy_frac=0.5)
        check(f"budget {b:>3}: sinks+recent+heavy == budget",
              p.n_sinks() + p.n_recent() + p.heavy(), b)

    print("\nthe streaming BASELINE must stay exempt -- heavy=0 is its purpose:")
    st = Policy(budget=58, sinks=4, recent=54, min_heavy_frac=0.5, select="streaming")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check("streaming keeps heavy == 0 despite the floor", st.heavy(), 0)
    check("streaming keeps its full recent window", st.n_recent(), 54)

    print("\nbackward compatibility -- every committed result must reproduce:")
    check("default policy is unchanged (floor off)",
          Policy(budget=128, sinks=4, recent=32).heavy(), 92)
    check("passkey 128K B=16 recent=4 is unchanged",
          Policy(budget=16, sinks=4, recent=4).heavy(), 8)
    check("  ...and its recent window is unchanged",
          Policy(budget=16, sinks=4, recent=4).n_recent(), 4)
    check("min_heavy_frac defaults to 0.0 (opt-in only)",
          Policy(budget=58, sinks=4, recent=32).min_heavy_frac, 0.0)

    print(f"\n{'all checks pass' if not FAILED else f'{FAILED} FAILED'}")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
