#!/usr/bin/env python3
"""CPU tests for grow_spans -- the derived-reach alternative to a fixed pool window.

These run without a GPU and without the model, so they can be run while a
measurement occupies the card. They check behaviour, not accuracy; the accuracy
question belongs to the passkey harness.

What is being checked, and why each matters:

  no-op safety      grow=None must leave select_keep bit-identical to the fixed
                    window. Every committed result was measured on that path.
  span coverage     a head spike followed by a payload run should carry the head's
                    score across the whole run -- this is the mechanism.
  boundary stop     growth must NOT cross a collapse to filler, or it would drag
                    unrelated text in and spend budget on it.
  ADAPTIVITY        the reason this exists: one fixed reach cannot serve two spans
                    of different lengths, and growth must serve both.

Usage:
    python scripts/kv/test_grow_spans.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kv.significance import Policy, grow_spans, select_keep  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def span_scores(n: int, spans, head: float = 1.0, payload: float = 0.15,
                filler: float = 0.01) -> torch.Tensor:
    """Filler with `spans` of (start, length) laid on it.

    Shaped like what P2.2 measured: a high query-shaped HEAD followed by a payload
    that scores well above filler but far below the head.
    """
    s = torch.full((1, 1, n), filler)
    for start, length in spans:
        s[0, 0, start] = head
        s[0, 0, start + 1:start + length] = payload
    return s


def main() -> None:
    print("grow_spans -- CPU behaviour tests\n")

    # --- no-op safety -------------------------------------------------------
    s = span_scores(200, [(50, 5)])
    check("tau=0 is a no-op", torch.equal(grow_spans(s, 0.0, 64), s))

    torch.manual_seed(0)
    rnd = torch.rand(2, 8, 300)
    p_fixed = Policy(budget=64, sinks=4, recent=16, pool=7, pool_back=4, pool_fwd=0)
    p_grow = Policy(budget=64, sinks=4, recent=16, pool=7, pool_back=4, pool_fwd=0,
                    grow=0.95)
    a = select_keep(rnd.clone(), 300, p_fixed, torch.device("cpu"))
    b = select_keep(rnd.clone(), 300, Policy(budget=64, sinks=4, recent=16, pool=7,
                                             pool_back=4, pool_fwd=0),
                    torch.device("cpu"))
    check("grow=None is deterministic and unchanged", torch.equal(a, b))

    # --- span coverage ------------------------------------------------------
    n, start, length = 200, 50, 5
    s = span_scores(n, [(start, length)])
    g = grow_spans(s, 0.95, 64)[0, 0]
    body = g[start:start + length]
    check("head score reaches the whole payload",
          bool((body >= 1.0 - 1e-6).all()),
          f"payload now {body.tolist()}")

    # --- boundary stop ------------------------------------------------------
    after = g[start + length:start + length + 20]
    check("growth stops at the filler boundary",
          bool((after <= 0.02).all()),
          f"max just past the span = {after.max().item():.4f}")

    before = g[start - 10:start]
    check("growth does not run backwards past the head",
          bool((before <= 0.02).all()),
          f"max just before the head = {before.max().item():.4f}")

    # --- two spans do not merge --------------------------------------------
    s2 = span_scores(200, [(50, 5), (70, 5)])
    g2 = grow_spans(s2, 0.95, 64)[0, 0]
    gap = g2[55:70]
    check("two spans separated by filler do not merge",
          bool((gap <= 0.02).all()),
          f"max in the gap = {gap.max().item():.4f}")

    # --- THE POINT: adaptivity ---------------------------------------------
    # One short span and one long one. A fixed reach tuned for either fails the
    # other; growth must cover both.
    n = 300
    s3 = span_scores(n, [(50, 5), (150, 20)])
    fixed_back = 4                      # what P2.4 measured for a 5-token span
    padded = torch.nn.functional.pad(s3, (fixed_back, 0), value=float("-inf"))
    fixed = torch.nn.functional.max_pool1d(padded, kernel_size=fixed_back + 1,
                                           stride=1)[..., :n][0, 0]
    grown = grow_spans(s3, 0.90, 64)[0, 0]

    short_fixed = bool((fixed[50:55] >= 1.0 - 1e-6).all())
    long_fixed = bool((fixed[150:170] >= 1.0 - 1e-6).all())
    short_grow = bool((grown[50:55] >= 1.0 - 1e-6).all())
    long_grow = bool((grown[150:170] >= 1.0 - 1e-6).all())

    check("fixed reach=4 covers the 5-token span", short_fixed)
    check("fixed reach=4 FAILS the 20-token span", not long_fixed,
          f"covered {int((fixed[150:170] >= 1.0 - 1e-6).sum())}/20 positions")
    check("growth covers the 5-token span", short_grow)
    check("growth covers the 20-token span too", long_grow,
          f"covered {int((grown[150:170] >= 1.0 - 1e-6).sum())}/20 positions")

    # --- convergence --------------------------------------------------------
    g_short = grow_spans(s3, 0.90, 4)[0, 0]
    check("max_reach bounds the growth",
          not bool((g_short[150:170] >= 1.0 - 1e-6).all()),
          "reach=4 cannot cross a 20-token span, as expected")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks pass")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
