#!/usr/bin/env python3
"""H1 and the confidence diagnostics: does confidence predict remnant need?

The hypothesis under test is

    E[ U_t | C_t low ]  >  E[ U_t | C_t high ]

and the point of this script is to give it every fair chance to fail. So:

* all four confidence features are tested, not just the one that looks best,
  and the p-values are Holm-corrected across them;
* intervals come from a **cluster bootstrap over prompts**, because tokens from
  one prompt are not independent draws and a token-level interval would be
  narrow enough to make a null look significant;
* AUROC is reported against its own base rate, so a high AUROC on a degenerate
  label distribution cannot be mistaken for a useful detector;
* the false-skip rate -- skipping the remnant when it was needed -- is reported
  at every candidate deployment threshold, because that is the error a
  quality-first system actually cares about, and a policy can look excellent on
  mean utility while making exactly that mistake on the tokens that matter.

"Confidently wrong" is the specific catastrophe: the lens is confident, the
truncated model disagrees with the dense teacher, and the remnant would have
fixed it. Those are counted and reported separately rather than averaged away.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from confidence import expected_calibration_error                 # noqa: E402
from stats import (auroc, auprc, base_rate, bin_by,                # noqa: E402
                   cluster_bootstrap, cluster_bootstrap_pair,
                   holm_bonferroni, pearson, spearman,
                   wilcoxon_signed_rank)

FEATURES = ("max_prob", "neg_entropy", "prob_margin", "logit_margin")


def load_trace(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if not rows:
        raise SystemExit("empty trace at " + str(path))
    return rows


def col(rows: list[dict], key: str) -> torch.Tensor:
    return torch.tensor([r[key] for r in rows], dtype=torch.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True,
                    help="directory containing traces.jsonl")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="utility threshold defining 'remnant needed'")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trace_dir = REPO / args.trace if not Path(args.trace).is_absolute() else Path(args.trace)
    rows = load_trace(trace_dir / "traces.jsonl")
    manifest = json.loads((trace_dir / "trace_manifest.json").read_text(encoding="utf-8"))
    clusters = [r["prompt_index"] for r in rows]
    n_prompts = len(set(clusters))

    utility = col(rows, "utility")
    kl_del = col(rows, "kl_delete")
    kl_rem = col(rows, "kl_remnant")
    needed = (utility > args.delta).double()

    print("trace   layer " + str(manifest["layer"]) + "  arm " + manifest["arm"]
          + "  " + str(manifest["n_groups"]) + " groups  " + manifest["dataset"])
    print("        " + str(len(rows)) + " positions from " + str(n_prompts)
          + " prompts")
    print("delta   " + str(args.delta) + "  ->  base rate P(U > delta) = "
          + format(base_rate(needed), ".4f"))
    print()

    # -- damage and recovery ------------------------------------------------
    dmg = cluster_bootstrap(kl_del, clusters, lambda v: v.mean().item(),
                            n_boot=args.n_boot)
    rem = cluster_bootstrap(kl_rem, clusters, lambda v: v.mean().item(),
                            n_boot=args.n_boot)
    util_ci = cluster_bootstrap(utility, clusters, lambda v: v.mean().item(),
                               n_boot=args.n_boot)
    print("KL(dense||delete)   " + format(dmg["point"], ".5f")
          + "  [" + format(dmg["lo"], ".5f") + ", " + format(dmg["hi"], ".5f") + "]")
    print("KL(dense||remnant)  " + format(rem["point"], ".5f")
          + "  [" + format(rem["lo"], ".5f") + ", " + format(rem["hi"], ".5f") + "]")
    print("mean utility        " + format(util_ci["point"], ".5f")
          + "  [" + format(util_ci["lo"], ".5f") + ", " + format(util_ci["hi"], ".5f")
          + "]   P(U>0) = " + format((utility > 0).double().mean().item(), ".4f"))
    kl_recovery = ((dmg["point"] - rem["point"]) / dmg["point"]
                   if dmg["point"] > 0 else float("nan"))
    print("KL recovery ratio   " + format(kl_recovery, ".4f"))
    print()

    # -- H1 per feature -----------------------------------------------------
    per_feature: dict[str, dict] = {}
    pvals: dict[str, float] = {}
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]

    for f in FEATURES:
        c = col(rows, f)
        # For the split test, "low" and "high" are the tails, so the contrast is
        # the one H1 actually states rather than a correlation over the bulk.
        q1, q3 = torch.quantile(c, 0.25).item(), torch.quantile(c, 0.75).item()
        low = utility[c <= q1]
        high = utility[c >= q3]
        low_cl = [cl for cl, cv in zip(clusters, c.tolist()) if cv <= q1]
        high_cl = [cl for cl, cv in zip(clusters, c.tolist()) if cv >= q3]
        lo_ci = cluster_bootstrap(low, low_cl, lambda v: v.mean().item(),
                                  n_boot=args.n_boot)
        hi_ci = cluster_bootstrap(high, high_cl, lambda v: v.mean().item(),
                                  n_boot=args.n_boot)
        w = wilcoxon_signed_rank(torch.cat([low, -high])) if low.numel() and high.numel() \
            else {"p_value": float("nan")}

        # Correlation of (1 - C) with utility: H1 predicts positive.
        inv = (1.0 - c) if f != "logit_margin" else (-c)
        pr = pearson(inv, utility)
        sp = spearman(inv, utility)

        feat_edges = edges if f != "logit_margin" else None
        bins = bin_by(c, utility, edges) if feat_edges else \
            bin_by(c, utility, torch.quantile(
                c, torch.linspace(0, 1, 9, dtype=torch.float64)).tolist())

        a_roc = auroc(inv, needed)
        a_prc = auprc(inv, needed)

        per_feature[f] = {
            "q25": q1, "q75": q3,
            "mean_utility_low_confidence": lo_ci,
            "mean_utility_high_confidence": hi_ci,
            "gap_low_minus_high": lo_ci["point"] - hi_ci["point"],
            "pearson_inv_conf_vs_utility": pr,
            "spearman_inv_conf_vs_utility": sp,
            "auroc_needed": a_roc,
            "auprc_needed": a_prc,
            "base_rate": base_rate(needed),
            "bins": bins,
            "wilcoxon": w,
        }
        pvals[f] = w.get("p_value", float("nan"))

        print(f + ":")
        print("   mean U | C low (<=q25)   " + format(lo_ci["point"], ".5f")
              + "  [" + format(lo_ci["lo"], ".5f") + ", "
              + format(lo_ci["hi"], ".5f") + "]  n=" + str(lo_ci["n"]))
        print("   mean U | C high (>=q75)  " + format(hi_ci["point"], ".5f")
              + "  [" + format(hi_ci["lo"], ".5f") + ", "
              + format(hi_ci["hi"], ".5f") + "]  n=" + str(hi_ci["n"]))
        print("   gap (H1 predicts > 0)    "
              + format(lo_ci["point"] - hi_ci["point"], "+.5f"))
        print("   corr(1-C, U)  pearson " + format(pr, "+.4f")
              + "   spearman " + format(sp, "+.4f"))
        print("   AUROC(U>delta) " + format(a_roc, ".4f")
              + "   AUPRC " + format(a_prc, ".4f")
              + "   (base rate " + format(base_rate(needed), ".4f") + ")")
        print()

    corrected = holm_bonferroni(pvals)

    # -- false skips and confidently-wrong ---------------------------------
    dense_top1 = torch.tensor([r["dense_top1"] for r in rows])
    del_top1 = torch.tensor([r["deleted_top1"] for r in rows])
    rem_top1 = torch.tensor([r["remnant_top1"] for r in rows])
    lens_top1 = torch.tensor([r["lens_top1"] for r in rows])

    deleted_wrong = (del_top1 != dense_top1)
    remnant_fixes = deleted_wrong & (rem_top1 == dense_top1)
    print("top-token behaviour over " + str(len(rows)) + " positions:")
    print("   deleted disagrees with dense    "
          + format(deleted_wrong.double().mean().item(), ".4f")
          + "  (" + str(int(deleted_wrong.sum())) + ")")
    print("   remnant repairs that disagreement "
          + format((remnant_fixes.sum() / max(int(deleted_wrong.sum()), 1)).item(), ".4f")
          + "  (" + str(int(remnant_fixes.sum())) + " of "
          + str(int(deleted_wrong.sum())) + ")")
    print("   lens top1 == dense top1        "
          + format((lens_top1 == dense_top1).double().mean().item(), ".4f"))
    print()

    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    skip_table: dict[str, list[dict]] = {}
    for f in FEATURES:
        c = col(rows, f)
        if f == "logit_margin":
            grid = torch.quantile(c, torch.tensor(thresholds,
                                                  dtype=torch.float64)).tolist()
        else:
            grid = thresholds
        table = []
        for tau in grid:
            skip = c >= tau
            n_skip = int(skip.sum())
            fs = int((skip & (needed > 0)).sum())
            cw = int((skip & remnant_fixes).sum())
            table.append({
                "tau": float(tau),
                "skip_rate": n_skip / len(rows),
                "false_skip_count": fs,
                "false_skip_rate_of_all": fs / len(rows),
                "false_skip_rate_given_needed": (fs / max(int((needed > 0).sum()), 1)),
                "confidently_wrong_count": cw,
                "confidently_wrong_rate": cw / len(rows),
                "utility_forgone": float(utility[skip].clamp(min=0).sum())
                if n_skip else 0.0,
            })
        skip_table[f] = table

    print("false skips at candidate skip thresholds (max_prob):")
    print("   {:>6s} {:>10s} {:>12s} {:>16s} {:>18s}".format(
        "tau", "skip rate", "false skips", "P(skip|needed)", "confidently wrong"))
    for e in skip_table["max_prob"]:
        print("   {:>6.2f} {:>10.4f} {:>12d} {:>16.4f} {:>18d}".format(
            e["tau"], e["skip_rate"], e["false_skip_count"],
            e["false_skip_rate_given_needed"], e["confidently_wrong_count"]))
    print()

    # -- calibration --------------------------------------------------------
    calib = {}
    for f in ("max_prob", "neg_entropy", "prob_margin"):
        c = col(rows, f)
        agree = (lens_top1 == dense_top1).double()
        calib[f] = expected_calibration_error(c, agree)
        calib[f]["target"] = "lens_top1 == dense_top1"
        print("ECE(" + f + " vs lens/dense top-1 agreement) = "
              + format(calib[f]["ece"], ".4f")
              + "   Brier " + format(calib[f]["brier"], ".4f"))

    verdict_gap = all(per_feature[f]["gap_low_minus_high"] > 0 for f in FEATURES)
    best = max(FEATURES, key=lambda f: (per_feature[f]["auroc_needed"]
                                        if not math.isnan(per_feature[f]["auroc_needed"])
                                        else -1))

    out = {
        "trace": str(trace_dir.relative_to(REPO)).replace("\\", "/"),
        "manifest": {k: manifest[k] for k in
                     ("layer", "arm", "spec", "n_groups", "dataset", "prompts",
                      "positions", "lens_rank", "seed")},
        "delta": args.delta,
        "n_positions": len(rows),
        "n_prompts": n_prompts,
        "kl_delete": dmg, "kl_remnant": rem, "mean_utility": util_ci,
        "kl_recovery_ratio": kl_recovery,
        "base_rate_needed": base_rate(needed),
        "per_feature": per_feature,
        "holm_bonferroni": corrected,
        "top_token": {
            "deleted_disagrees_with_dense": deleted_wrong.double().mean().item(),
            "deleted_disagree_count": int(deleted_wrong.sum()),
            "remnant_repairs_count": int(remnant_fixes.sum()),
            "remnant_repair_rate_of_disagreements":
                (remnant_fixes.sum() / max(int(deleted_wrong.sum()), 1)).item(),
            "lens_agrees_with_dense": (lens_top1 == dense_top1).double().mean().item(),
        },
        "skip_thresholds": skip_table,
        "calibration": calib,
        "h1": {
            "statement": "E[U | C low] > E[U | C high]",
            "all_features_positive_gap": verdict_gap,
            "best_feature_by_auroc": best,
            "best_auroc": per_feature[best]["auroc_needed"],
        },
    }
    dest = REPO / (args.out or (str(trace_dir.relative_to(REPO)).replace("\\", "/")
                                + "/confidence_analysis.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nH1: all four features show a positive low-minus-high gap: "
          + str(verdict_gap))
    print("best feature by AUROC: " + best + " = "
          + format(per_feature[best]["auroc_needed"], ".4f"))
    print("wrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
