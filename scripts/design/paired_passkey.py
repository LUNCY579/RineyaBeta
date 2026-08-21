#!/usr/bin/env python3
"""Paired analysis of passkey cells, because the unpaired version was wrong.

WHY THIS EXISTS. A 128K depth curve was published from five arms pooled as
"n = 30 per depth", with Clopper-Pearson intervals and Fisher contrasts between
depths. Both were invalid, for the same reason: the five arms share one key
sequence, so the 30 observations at a depth were five repetitions of six needles,
and -- worse -- each depth drew a DIFFERENT six needles, putting depth and needle
identity in the same column. An identical configuration scored 0.17 at depth 0.5
with one block of keys and 0.50 with another.

The harness now pairs every cell on the same needles and records per-trial
outcomes. That makes the correct tests available, and this runs them, so the
analysis is not hand-rolled a second time.

WHAT IT DOES

  * Loads any number of passkey result files and indexes every cell by
    (length, budget, depth, rank_exponent), keyed on the needle.
  * Refuses to compare two cells that do not share a needle set. That refusal is
    the point: an unpairable comparison is reported as unpairable rather than
    silently run as if it were independent.
  * Two conditions -> exact McNemar on the discordant pairs. Concordant pairs
    carry no information about a difference and are excluded, which is exactly
    what pooling gets wrong.
  * More than two ordered conditions -> a within-subject permutation trend test.
    Each needle's outcomes are permuted across conditions, which is the null
    "condition does not matter for this needle" and preserves needle difficulty.

Usage:
    python scripts/design/paired_passkey.py experiments/kv_significance/results/pk_128k_grid.json
    python scripts/design/paired_passkey.py results/pk_128k_e*.json --by rank_exponent
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The four things that define a cell. Anything else about a run is either fixed
# by the file's manifest or irrelevant to which condition a needle was seen under.
AXES = ("length", "budget", "depth", "rank_exponent")


def load(paths: list[str]) -> list[dict]:
    """Every cell that carries per-trial records, flattened."""
    cells = []
    for pat in paths:
        for f in sorted(glob.glob(pat)) or [pat]:
            p = Path(f)
            if not p.is_absolute():
                p = ROOT / f if (ROOT / f).exists() else p
            if not p.exists():
                print(f"  skip (missing): {f}")
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            for r in d.get("results", []):
                if not r.get("trials"):
                    continue          # pre-fix file: cell accuracy only
                cells.append({
                    "file": p.name,
                    "seed": d.get("seed"),
                    **{a: r.get(a) for a in AXES},
                    "acc": r.get("acc"),
                    "hits": {t["key"]: bool(t["hit"]) for t in r["trials"]},
                })
    return cells


def mcnemar_exact(a: dict[int, bool], b: dict[int, bool]):
    """Two-sided exact McNemar on the needles both cells saw.

    Returns (n_shared, b_only, a_only, p). `b_only` is the count of needles the
    SECOND condition found and the first did not.
    """
    shared = sorted(set(a) & set(b))
    x = sum(1 for k in shared if a[k] and not b[k])      # first only
    y = sum(1 for k in shared if b[k] and not a[k])      # second only
    n = x + y
    if n == 0:
        return len(shared), y, x, 1.0
    lo = min(x, y)
    p = 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n
    return len(shared), y, x, min(1.0, p)


def trend_perm(cells: list[dict], scores: list[float], n_perm: int, seed: int):
    """Within-needle permutation test for a monotone trend across conditions.

    The statistic is sum_j score_j * (needles found under condition j). The null
    permutes each needle's own outcome vector across conditions, so needle
    difficulty is held fixed and only the condition labels move. That is the
    assumption pooling violates.
    """
    keys = sorted(set.intersection(*(set(c["hits"]) for c in cells)))
    if not keys:
        return None
    rows = [[c["hits"][k] for c in cells] for k in keys]      # needle x condition

    def stat(mat):
        return sum(s * sum(r[j] for r in mat) for j, s in enumerate(scores))

    obs = stat(rows)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        perm = []
        for r in rows:
            q = list(r)
            rng.shuffle(q)
            perm.append(q)
        if stat(perm) >= obs:
            ge += 1
    # one-sided upper, with the +1 correction that keeps the p-value valid
    return keys, obs, (ge + 1) / (n_perm + 1)


def label(c: dict) -> str:
    e = c["rank_exponent"]
    return (f"n={c['length']:,} B={c['budget']} d={c['depth']:.1f}"
            + (f" e={e:g}" if e is not None else " full"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--by", default="budget", choices=AXES,
                    help="the axis to compare along; every other axis is held fixed")
    ap.add_argument("--perm", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    cells = load(args.paths)
    if not cells:
        raise SystemExit(
            "no cells with per-trial records.\n"
            "Files written before the pairing fix store cell accuracy only, which\n"
            "cannot support a paired test. Re-run them with the current passkey.py.")

    seeds = {c["seed"] for c in cells}
    if len(seeds) > 1:
        print(f"WARNING: mixed seeds {seeds}. Cells from different seeds see")
        print("different needles and are not pairable along any axis.\n")

    print(f"{len(cells)} cells with per-trial records\n")
    print(f"{'cell':<40}{'acc':>7}{'n':>5}")
    for c in sorted(cells, key=lambda c: [str(c[a]) for a in AXES]):
        print(f"{label(c):<40}{c['acc']:>7.2f}{len(c['hits']):>5}")

    # Group by everything except the comparison axis.
    other = [a for a in AXES if a != args.by]
    groups: dict[tuple, list[dict]] = {}
    for c in cells:
        groups.setdefault(tuple(c[a] for a in other), []).append(c)

    print(f"\n=== paired contrasts along `{args.by}` ===")
    any_run = False
    for gk, gc in sorted(groups.items(), key=lambda kv: str(kv[0])):
        gc = sorted(gc, key=lambda c: (c[args.by] is None, c[args.by]))
        if len(gc) < 2:
            continue
        any_run = True
        fixed = ", ".join(f"{a}={v}" for a, v in zip(other, gk))
        print(f"\n{fixed}")
        for i in range(len(gc)):
            for j in range(i + 1, len(gc)):
                a, b = gc[i], gc[j]
                n, gain, loss, p = mcnemar_exact(a["hits"], b["hits"])
                if n == 0:
                    print(f"  {a[args.by]} vs {b[args.by]}: NOT PAIRABLE "
                          f"(no shared needles) -- do not compare these")
                    continue
                print(f"  {args.by} {a[args.by]} -> {b[args.by]}: "
                      f"{a['acc']:.2f} -> {b['acc']:.2f} on {n} shared needles, "
                      f"+{gain}/-{loss} discordant, exact McNemar p = {p:.4f}")
        vals = [c[args.by] for c in gc]
        if len(gc) > 2 and all(isinstance(v, (int, float)) for v in vals):
            res = trend_perm(gc, [float(v) for v in vals], args.perm, args.seed)
            if res:
                keys, obs, p = res
                print(f"  trend across {vals} on {len(keys)} shared needles: "
                      f"T = {obs:g}, permutation p = {p:.4f} ({args.perm:,} draws)")

    if not any_run:
        print(f"  nothing to compare: every group has one level of `{args.by}`.")

    print("\nConcordant needles are excluded from McNemar by construction -- a needle")
    print("that both conditions found, or neither did, says nothing about which is")
    print("better. Pooling counts them, which is how six needles became 'n = 30'.")


if __name__ == "__main__":
    main()
