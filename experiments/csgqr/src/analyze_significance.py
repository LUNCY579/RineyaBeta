#!/usr/bin/env python3
"""H3 and the group-level analyses: is significance real, and is it additive?

Three questions, each with the control that makes the answer mean something.

**Is the ranking informative?** A ranking is only useful if selecting its top-k
captures more utility than selecting k groups at random. So the top-k utility
capture is compared against random selection (many draws, so the comparison has
an interval), against magnitude ranking, against the shuffled ranking, and
against the reversed ranking. Beating random is the minimum bar; beating
magnitude is the bar that shows significance carries information a cheap
heuristic does not.

**Is per-state significance better than one global ranking?** The oracle
per-token top-k is an upper bound on any router that predicts utility from the
hidden state. If the global static ranking is already close to that bound, a
learned per-state router has almost nothing to win and its complexity is not
purchasable. This is reported as the *headroom*, because it is the number that
decides whether the dynamic branch of the method is worth building.

**Are group utilities additive?** The *outputs* are exactly additive
(``F = sum_g F_g``), but utility is a difference of KL divergences and KL is not
linear, so ``U(A union B) = U(A) + U(B)`` is a genuine empirical question. The
interaction ``I(A,B) = U(A union B) - U(A) - U(B)`` is measured on sampled pairs. Large
interactions would mean independent top-k ranking is the wrong selection rule,
and that finding would matter more than any threshold sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from significance import GroupSignificance, rank_correlation      # noqa: E402
from stats import cluster_bootstrap, spearman                     # noqa: E402

SEED = 20260823


def load(trace_dir: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(l) for l in (trace_dir / "traces.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = json.loads((trace_dir / "trace_manifest.json")
                          .read_text(encoding="utf-8"))
    return rows, manifest


def capture_curve(util: torch.Tensor, order: list[int]) -> list[float]:
    """Cumulative mean utility captured by the first k groups of ``order``.

    Note this is the *sum of individual* group utilities, which is the quantity a
    top-k rule optimises under an additivity assumption. Whether that assumption
    holds is tested separately below; if it does not, this curve is a ranking
    diagnostic rather than a prediction of the routed arm's utility.
    """
    idx = torch.tensor(order, dtype=torch.long)
    per_group_mean = util.mean(dim=0)[idx]
    return torch.cumsum(per_group_mean, dim=0).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--random-draws", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trace_dir = REPO / args.trace if not Path(args.trace).is_absolute() \
        else Path(args.trace)
    rows, manifest = load(trace_dir)
    n_groups = manifest["n_groups"]
    clusters = [r["prompt_index"] for r in rows]

    util = torch.tensor([r["group_utility"] for r in rows], dtype=torch.float64)
    whole = torch.tensor([r["utility"] for r in rows], dtype=torch.float64)
    n = util.shape[0]
    print("trace  layer " + str(manifest["layer"]) + "  " + manifest["arm"]
          + "  " + str(n_groups) + " groups  " + str(n) + " positions")

    # -- global significance -----------------------------------------------
    sig = GroupSignificance(n_groups)
    sig.update(util)
    ranking = sig.ranking()
    mean_u = sig.mean_utility()
    print("\nper-group mean utility:")
    print("  best 6  " + str([(g, round(mean_u[g].item(), 5)) for g in ranking[:6]]))
    print("  worst 6 " + str([(g, round(mean_u[g].item(), 5)) for g in ranking[-6:]]))
    print("  sum of individual group utilities " + format(mean_u.sum().item(), ".5f")
          + "   whole-remnant utility " + format(whole.mean().item(), ".5f"))
    additivity_gap = mean_u.sum().item() - whole.mean().item()
    print("  additivity gap (sum - whole)      " + format(additivity_gap, "+.5f"))

    # -- ranking quality vs controls ---------------------------------------
    magnitude_order = None            # supplied only if a ranking file exists
    curves = {"significance": capture_curve(util, ranking),
              "reversed": capture_curve(util, list(reversed(ranking)))}

    g = torch.Generator().manual_seed(SEED)
    random_curves = []
    for _ in range(args.random_draws):
        perm = torch.randperm(n_groups, generator=g).tolist()
        random_curves.append(capture_curve(util, perm))
    rc = torch.tensor(random_curves, dtype=torch.float64)
    curves["random_mean"] = rc.mean(dim=0).tolist()
    curves["random_p05"] = torch.quantile(rc, 0.05, dim=0).tolist()
    curves["random_p95"] = torch.quantile(rc, 0.95, dim=0).tolist()

    total = mean_u.sum().item()
    print("\nutility captured by the top-k groups (k / fraction of the "
          "sum of individual utilities):")
    print("  {:>4s} {:>6s} {:>12s} {:>12s} {:>14s} {:>10s}".format(
        "k", "rho", "significance", "random", "random p95", "reversed"))
    ks = sorted({max(1, int(round(f * n_groups)))
                 for f in (0.125, 0.25, 0.5, 0.75, 1.0)})
    table = []
    for k in ks:
        s = curves["significance"][k - 1]
        r = curves["random_mean"][k - 1]
        r95 = curves["random_p95"][k - 1]
        rev = curves["reversed"][k - 1]
        beats = s > r95
        table.append({"k": k, "rho": k / n_groups, "significance": s,
                      "random_mean": r, "random_p95": r95, "reversed": rev,
                      "significance_frac_of_total": s / total if total else float("nan"),
                      "random_frac_of_total": r / total if total else float("nan"),
                      "beats_random_p95": bool(beats)})
        print("  {:>4d} {:>6.3f} {:>12.5f} {:>12.5f} {:>14.5f} {:>10.5f}   {}".format(
            k, k / n_groups, s, r, r95, rev, "significance > random p95" if beats else ""))

    # -- static global vs oracle per-state ---------------------------------
    # The oracle picks the best k groups for each token independently. No router
    # that reads only the hidden state can beat it, so it bounds the headroom.
    print("\nstatic global ranking vs per-token oracle:")
    headroom = []
    for k in ks:
        static_idx = torch.tensor(ranking[:k], dtype=torch.long)
        static_cap = util[:, static_idx].sum(dim=1)
        oracle_cap = torch.topk(util, k=k, dim=1).values.sum(dim=1)
        st = cluster_bootstrap(static_cap, clusters, lambda v: v.mean().item(),
                               n_boot=args.n_boot)
        orc = cluster_bootstrap(oracle_cap, clusters, lambda v: v.mean().item(),
                                n_boot=args.n_boot)
        gap = orc["point"] - st["point"]
        rel = gap / orc["point"] if orc["point"] else float("nan")
        headroom.append({"k": k, "rho": k / n_groups,
                         "static": st, "oracle": orc,
                         "absolute_headroom": gap, "relative_headroom": rel})
        print("  k={:<3d} static {:>9.5f}  oracle {:>9.5f}  headroom {:>9.5f}"
              "  ({:>6.2f}% of oracle)".format(
                  k, st["point"], orc["point"], gap, 100 * rel))

    # -- per-token ranking stability ---------------------------------------
    # If each token's best groups are wildly different from the global ranking,
    # a static ranking cannot work; if they are nearly the same, a dynamic router
    # has nothing to add. Either way this number decides it.
    sample = util[torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
                  [:min(n, 500)]]
    per_token_rho = [spearman(row, mean_u) for row in sample]
    rho_t = torch.tensor(per_token_rho, dtype=torch.float64)
    print("\nper-token agreement with the global ranking (Spearman over groups):")
    print("  mean " + format(rho_t.mean().item(), ".4f")
          + "   median " + format(rho_t.median().item(), ".4f")
          + "   p05 " + format(torch.quantile(rho_t, 0.05).item(), ".4f")
          + "   p95 " + format(torch.quantile(rho_t, 0.95).item(), ".4f"))

    # -- additivity of utilities -------------------------------------------
    interactions = None
    pairs = manifest.get("pairs") or []
    if pairs and "pair_utility" in rows[0]:
        recs = []
        for a, b in pairs:
            key = str(a) + "," + str(b)
            u_ab = torch.tensor([r["pair_utility"][key] for r in rows],
                                dtype=torch.float64)
            u_a = util[:, a]
            u_b = util[:, b]
            inter = u_ab - u_a - u_b
            ci = cluster_bootstrap(inter, clusters, lambda v: v.mean().item(),
                                   n_boot=args.n_boot)
            denom = u_a.abs().mean().item() + u_b.abs().mean().item()
            recs.append({
                "pair": [a, b],
                "mean_u_a": u_a.mean().item(), "mean_u_b": u_b.mean().item(),
                "mean_u_ab": u_ab.mean().item(),
                "mean_interaction": ci["point"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                "excludes_zero": bool(ci["lo"] > 0 or ci["hi"] < 0),
                "relative_interaction": (ci["point"] / denom if denom else float("nan")),
            })
        rel = torch.tensor([abs(r["relative_interaction"]) for r in recs],
                           dtype=torch.float64)
        n_sig = sum(1 for r in recs if r["excludes_zero"])
        interactions = {
            "n_pairs": len(recs),
            "mean_abs_relative_interaction": rel.mean().item(),
            "median_abs_relative_interaction": rel.median().item(),
            "max_abs_relative_interaction": rel.max().item(),
            "pairs_with_interaction_excluding_zero": n_sig,
            "pairs": recs,
        }
        print("\ngroup-utility interaction I(A,B) = U(A+B) - U(A) - U(B) over "
              + str(len(recs)) + " sampled pairs:")
        print("  |I| relative to |U(A)|+|U(B)|:  mean "
              + format(rel.mean().item(), ".4f") + "   median "
              + format(rel.median().item(), ".4f") + "   max "
              + format(rel.max().item(), ".4f"))
        print("  pairs whose interaction CI excludes zero: " + str(n_sig)
              + " / " + str(len(recs)))
    else:
        print("\n(no sampled pairs in this trace; rerun trace_utility.py --pairs N "
              "for the interaction test)")

    out = {
        "trace": str(trace_dir.relative_to(REPO)).replace("\\", "/"),
        "layer": manifest["layer"], "arm": manifest["arm"],
        "n_groups": n_groups, "n_positions": n,
        "n_prompts": len(set(clusters)),
        "ranking": ranking,
        "mean_group_utility": mean_u.tolist(),
        "positive_rate": sig.positive_rate().tolist(),
        "sum_of_group_utilities": mean_u.sum().item(),
        "whole_remnant_utility": whole.mean().item(),
        "additivity_gap_sum_minus_whole": additivity_gap,
        "capture_curves": curves,
        "topk_table": table,
        "static_vs_oracle": headroom,
        "per_token_ranking_agreement": {
            "n_sampled": int(sample.shape[0]),
            "mean_spearman": rho_t.mean().item(),
            "median_spearman": rho_t.median().item(),
            "p05": torch.quantile(rho_t, 0.05).item(),
            "p95": torch.quantile(rho_t, 0.95).item(),
        },
        "interactions": interactions,
        "random_draws": args.random_draws,
    }
    dest = REPO / (args.out or (str(trace_dir.relative_to(REPO)).replace("\\", "/")
                                + "/significance_analysis.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
