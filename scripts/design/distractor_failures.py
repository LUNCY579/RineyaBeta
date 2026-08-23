#!/usr/bin/env python3
"""Stage 1 diagnosis: when a distractor cell fails, WHAT does the model answer?

Pre-registered prediction P1 (docs/PLAN_LONG_REASONING.md) says query-shaped decoys
should be preferentially retained and CROWD OUT the true answer, because P2.2
measured the score keeping text that resembles the query -- framing surviving in 54%
of streams against 35% for the payload. That prediction is about *which span wins*.

But there is a second way a distractor cell can fail, and it says something entirely
different: the budget is split across more spans than it can hold intact, and the
true span comes back FRAGMENTED. P2 established the signature -- a correct prefix
that stops early (4 of 5 digits, never zero).

The two are distinguishable from the emitted text, and they call for opposite fixes:

  CROWDING OUT     the answer is a DECOY's key, whole and correct.
                   The span survived; the score chose the wrong span. The fix is in
                   scoring -- the query-similarity bias is real and must be undone.

  FRAGMENTATION    the answer is a corrupted or truncated version of the TRUE key.
                   The score chose right; the budget could not hold the span. The fix
                   is in budget or span protection, not in scoring.

  NEITHER          something else. Report it rather than forcing a category.

Decoy keys are regenerated here rather than parsed from prompts: build_prompt seeds
them `random.Random(f"{key}:decoys")`, so they are a deterministic function of the
real key and reproduce exactly without storing the 128K prompt.

Usage:
    python scripts/design/distractor_failures.py
"""

from __future__ import annotations

import glob
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/kv_significance/results"


def decoys_for(key: int, n: int) -> list[int]:
    """Exactly what build_prompt draws, without needing the prompt."""
    drng = random.Random(f"{key}:decoys")
    nd = len(str(key))
    return [drng.randint(10 ** (nd - 1), 10 ** nd - 1) for _ in range(n)]


def classify(key: int, got: str, decoys: list[int]) -> str:
    digits = re.findall(r"\d+", got or "")
    if not digits:
        return "no-digits"
    first = digits[0]
    if first == str(key):
        return "correct"
    if any(first == str(d) for d in decoys):
        return "crowded-out"
    if any(str(d) in (got or "") for d in decoys):
        return "decoy-mentioned"
    if str(key).startswith(first) and len(first) >= 2:
        return "truncated"
    # same length, some digits right: a corrupted span rather than a wrong span
    if len(first) == len(str(key)) and sum(
            a == b for a, b in zip(first, str(key))) >= 2:
        return "corrupted"
    return "other"


def main() -> None:
    rows = []
    for p in sorted(glob.glob(str(RESULTS / "pk_128k_B64_dist*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        k, style = d.get("distractors"), d.get("distractor_style")
        counts: dict[str, int] = {}
        examples: dict[str, str] = {}
        n = 0
        for c in d["results"]:
            for t in c["trials"]:
                n += 1
                cat = classify(t["key"], t.get("got", ""), decoys_for(t["key"], k))
                counts[cat] = counts.get(cat, 0) + 1
                examples.setdefault(cat, f"{t['key']}->{(t.get('got') or '')[:22]!r}")
        rows.append((k, style, n, counts, examples))

    if not rows:
        print("no distractor cells found")
        return

    print("STAGE 1 -- how do distractor cells fail?\n")
    cats = ["correct", "crowded-out", "decoy-mentioned", "truncated",
            "corrupted", "no-digits", "other"]
    print(f"{'k':>3} {'style':>7} {'n':>4} " + " ".join(f"{c[:11]:>12}" for c in cats))
    for k, style, n, counts, _ in rows:
        print(f"{k:>3} {style:>7} {n:>4} " +
              " ".join(f"{counts.get(c, 0):>12}" for c in cats))

    tot_crowd = sum(c.get("crowded-out", 0) for _, _, _, c, _ in rows)
    tot_frag = sum(c.get("truncated", 0) + c.get("corrupted", 0)
                   for _, _, _, c, _ in rows)
    tot_wrong = sum(n - c.get("correct", 0) for _, _, n, c, _ in rows)

    print(f"\nof {tot_wrong} failures across all cells:")
    print(f"   crowded out by a decoy  {tot_crowd:>4}"
          f"   ({tot_crowd / tot_wrong:.0%})" if tot_wrong else "")
    print(f"   fragmented true span    {tot_frag:>4}"
          f"   ({tot_frag / tot_wrong:.0%})" if tot_wrong else "")

    print("\nVERDICT ON P1. The prediction is about crowding out -- a decoy's key")
    print("returned whole. Fragmentation of the TRUE key is a different failure and")
    print("does not confirm P1, however large the accuracy drop.")

    print("\nexamples by category:")
    seen = set()
    for _, _, _, _, ex in rows:
        for cat, e in ex.items():
            if cat != "correct" and cat not in seen:
                seen.add(cat)
                print(f"   {cat:<16} {e}")


if __name__ == "__main__":
    main()
