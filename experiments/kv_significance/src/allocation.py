#!/usr/bin/env python3
"""Optimal per-layer KV budget allocation by water-filling on measured rho_l(k).

The current policy gives every layer the same budget. That is provably not the
best use of a fixed total, because the layers are not alike: R2 sends 80.5% of
layer 27's attention to the first 4 tokens but under 3% of layer 0's, so layer 27
needs almost nothing beyond its sinks while layer 0 needs a lot.

THE PROBLEM. Given a total budget B and per-layer retained-mass curves
rho_l(b) measured on the real model, choose b_1..b_L to

    minimise   sum_l ( 1 - rho_l(b_l) )        subject to  sum_l b_l = B

WHY GREEDY IS OPTIMAL. The objective is separable across layers, and each term is
decreasing in b_l. If in addition every rho_l has non-increasing marginal gains
(discrete concavity), then allocating one unit at a time to whichever layer
currently offers the largest marginal gain is exactly optimal -- this is the
standard water-filling/greedy argument for maximising a separable concave
objective under a cardinality constraint, and it is the same exchange argument
that makes the matroid-greedy optimal.

That assumption is NOT taken on faith here: `check_concavity()` measures it on
the actual curves and reports every violation, because a curve with an increasing
marginal somewhere makes greedy merely a heuristic rather than optimal, and the
honest thing is to know which one we have.

WHAT THIS OBJECTIVE IS NOT. sum_l (1 - rho_l) is a proxy, not end-to-end quality.
This repository has already measured rho > 0.996 costing 0.199 in generation
quality, so retained mass is necessary and not sufficient, and errors compose
across 28 layers rather than adding. Treat the allocation as a better-motivated
prior than uniform, and validate it against a task before believing the number.

Usage:
    python src/allocation.py
    python src/allocation.py --total 1792 --grid 8 --out results/allocation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def load_curves(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["ks"], d["rho_per_layer"], d


def rho_at(ks: list[int], row: list[float], b: float) -> float:
    """Piecewise-linear interpolation of a measured rho curve, clamped at both ends."""
    if b <= ks[0]:
        return row[0] * (b / ks[0]) if ks[0] else row[0]
    if b >= ks[-1]:
        return row[-1]
    for i in range(len(ks) - 1):
        if ks[i] <= b <= ks[i + 1]:
            t = (b - ks[i]) / (ks[i + 1] - ks[i])
            return row[i] + t * (row[i + 1] - row[i])
    return row[-1]


def check_concavity(ks: list[int], rho: list[list[float]]) -> dict:
    """Are marginal gains non-increasing? Greedy is optimal only if they are."""
    violations = []
    for li, row in enumerate(rho):
        marg = [(row[i + 1] - row[i]) / (ks[i + 1] - ks[i]) for i in range(len(ks) - 1)]
        for i in range(len(marg) - 1):
            if marg[i + 1] > marg[i] + 1e-9:
                violations.append({"layer": li, "between_k": [ks[i], ks[i + 1], ks[i + 2]],
                                   "marginal": [round(marg[i], 8), round(marg[i + 1], 8)]})
    return {"n_layers": len(rho), "violations": violations,
            "concave": not violations}


def allocate(ks, rho, total: int, grid: int, floor: int) -> list[int]:
    """Greedy marginal water-filling on a `grid`-token lattice."""
    L = len(rho)
    alloc = [floor] * L
    remaining = total - floor * L
    if remaining < 0:
        raise SystemExit(f"total {total} below the floor {floor} x {L} layers")
    while remaining >= grid:
        best_gain, best_l = -1.0, -1
        for l in range(L):
            gain = rho_at(ks, rho[l], alloc[l] + grid) - rho_at(ks, rho[l], alloc[l])
            if gain > best_gain:
                best_gain, best_l = gain, l
        if best_gain <= 0:
            break                      # no layer can still use another unit
        alloc[best_l] += grid
        remaining -= grid
    return alloc


def objective(ks, rho, alloc) -> float:
    return sum(1.0 - rho_at(ks, rho[l], alloc[l]) for l in range(len(alloc)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", type=Path,
                    default=EXP / "results" / "concentration.json")
    ap.add_argument("--total", type=int, default=None,
                    help="total tokens across all layers (default: 64 x n_layers)")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--floor", type=int, default=8,
                    help="minimum per layer; below the sink count is meaningless")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "allocation.json")
    args = ap.parse_args()

    ks, rho, meta = load_curves(args.curves)
    L = len(rho)
    total = args.total if args.total is not None else 64 * L
    uniform = total // L

    conc = check_concavity(ks, rho)
    print(f"curves: {args.curves.name}, {L} layers, k grid {ks}")
    print(f"concavity of rho_l(k): {'HOLDS' if conc['concave'] else 'VIOLATED'} "
          f"({len(conc['violations'])} violations)")
    if not conc["concave"]:
        print("  -> greedy is a heuristic here, not provably optimal. Violations:")
        for v in conc["violations"][:5]:
            print(f"     layer {v['layer']:>2} between k={v['between_k']} "
                  f"marginals {v['marginal']}")
        if len(conc["violations"]) > 5:
            print(f"     ... and {len(conc['violations']) - 5} more")
    print()

    alloc = allocate(ks, rho, total, args.grid, args.floor)
    uni = [uniform] * L
    j_opt, j_uni = objective(ks, rho, alloc), objective(ks, rho, uni)

    print(f"total budget {total} tokens across {L} layers (uniform = {uniform}/layer)")
    print(f"  uniform   sum(1-rho) = {j_uni:.5f}")
    print(f"  allocated sum(1-rho) = {j_opt:.5f}   "
          f"({100 * (j_uni - j_opt) / j_uni:+.1f}% error reduction)")
    print(f"  allocation spans {min(alloc)} .. {max(alloc)} tokens/layer\n")

    order = sorted(range(L), key=lambda l: -alloc[l])
    print(f"{'layer':>6}{'budget':>8}{'vs uniform':>12}{'rho at budget':>15}")
    for l in order:
        print(f"{l:>6}{alloc[l]:>8}{alloc[l] - uniform:>+12}"
              f"{rho_at(ks, rho[l], alloc[l]):>15.4f}")

    print("\nCAVEAT: the objective is sum(1-rho), which this repository has already")
    print("measured to be necessary but NOT sufficient -- rho > 0.996 still cost")
    print("0.199 end-to-end. This allocation is a better-motivated prior than")
    print("uniform; it is not validated until it is run against a task.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "curves": str(args.curves.name), "layers": L, "total": total,
        "grid": args.grid, "floor": args.floor,
        "concavity": conc,
        "allocation": alloc, "uniform": uniform,
        "objective_allocated": j_opt, "objective_uniform": j_uni,
        "error_reduction_pct": 100 * (j_uni - j_opt) / j_uni,
        "source_cases": meta.get("cases"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
