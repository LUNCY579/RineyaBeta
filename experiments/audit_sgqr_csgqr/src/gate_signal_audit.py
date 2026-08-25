#!/usr/bin/env python3
"""Is generic confidence an appropriate proxy for remnant need? (brief section 6)

Downstream accuracy cannot answer this. An arm can win on KL because it spends
compute where the *residual* is large without the confidence signal carrying any
information about need -- and it can lose despite a good signal because the
allocation rule wastes it. So this script measures the signal directly, on the
held-out split, against the two quantities it is supposed to track:

    deletion damage      D_t = KL(dense || all-four-deleted)_t
    recoverable benefit  U_t = D_t - KL(dense || full remnant)_t

and separately audits the two failure quadrants the brief names:

    low confidence, little benefit   -> compute spent for nothing
    high confidence, large benefit   -> recovery skipped where it mattered

Everything here is reported on the TEST trace unless stated. The probe was
fitted on calib and its ridge strength selected on val, so its test numbers are
not selected on. Lens features are untrained, so their test numbers are clean by
construction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import REPO                                                # noqa: E402
from stats import (auroc, auprc, base_rate, cluster_bootstrap_mean,   # noqa: E402
                   cluster_bootstrap_mean_pair, pearson, spearman)


def two_group_gap(values, clusters, mask_hi, mask_lo, *, n_boot=2000,
                  seed=20260824):
    """CI for mean(values[hi]) - mean(values[lo]), resampling whole prompts.

    The two groups are disjoint sets of positions, so this is NOT a paired test.
    Prompts are resampled together so that within-prompt dependence is carried
    into the interval; a draw in which either group is empty is discarded.
    """
    import torch as _t
    ids = sorted(set(clusters))
    idx_by = {c: [] for c in ids}
    for i, c in enumerate(clusters):
        idx_by[c].append(i)
    g = _t.Generator().manual_seed(seed)
    point = float(values[mask_hi].mean() - values[mask_lo].mean())
    draws = []
    for _ in range(n_boot):
        pick = _t.randint(0, len(ids), (len(ids),), generator=g).tolist()
        rows = [i for j in pick for i in idx_by[ids[j]]]
        sel = _t.tensor(rows, dtype=_t.long)
        hi = values[sel][mask_hi[sel]]
        lo = values[sel][mask_lo[sel]]
        if hi.numel() and lo.numel():
            draws.append(float(hi.mean() - lo.mean()))
    if not draws:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    d = _t.tensor(draws, dtype=_t.float64)
    return {"point": point, "lo": float(_t.quantile(d, 0.025)),
            "hi": float(_t.quantile(d, 0.975)), "n_boot": len(draws),
            "excludes_zero": bool(float(_t.quantile(d, 0.025)) > 0
                                  or float(_t.quantile(d, 0.975)) < 0)}

RES = REPO / "experiments/audit_sgqr_csgqr/results"
LENS = ("max_prob", "neg_entropy", "prob_margin", "logit_margin")


def read(split: str, bits: str = "int4"):
    rows = [json.loads(l) for l in
            (RES / ("trace_%s_%s.jsonl" % (split, bits))).read_text(
                encoding="utf-8").splitlines() if l.strip()]
    hid = torch.load(RES / ("trace_%s_%s.hidden.pt" % (split, bits)),
                     weights_only=False)["hidden"].to(torch.float64)
    return rows, hid


def col(rows, k) -> torch.Tensor:
    return torch.tensor([r[k] for r in rows], dtype=torch.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--bits", default="int4")
    ap.add_argument("--deltas", default="0.0,0.01,0.05,0.1,0.25,0.5",
                    help="need thresholds; the historical campaign used 0.1 only "
                         "and did not sweep it (STATISTICAL_ANALYSIS 7.3)")
    ap.add_argument("--probes", default="experiments/audit_sgqr_csgqr/results/probes.pt")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    rows, hid = read(args.split, args.bits)
    clusters = [r["prompt"] for r in rows]
    D = col(rows, "kl_deleted")
    R = col(rows, "kl_remnant")
    U = D - R
    probes = torch.load(REPO / args.probes, weights_only=False)
    X = (hid - probes["mu"]) / probes["sd"]
    ones = torch.ones(X.shape[0], 1, dtype=torch.float64)
    Xb = torch.cat([X, ones], dim=1)
    signals = {("lens_" + f): -col(rows, f) for f in LENS}   # higher = more need
    signals["probe_need"] = Xb @ probes["probes"]["need"]
    signals["probe_confidence"] = -(Xb @ probes["probes"]["confidence"])
    signals["oracle_utility"] = U.clone()

    out: dict = {
        "split": args.split, "bits": args.bits, "n_positions": len(rows),
        "n_prompts": len(set(clusters)),
        "orientation": "every signal is oriented so HIGHER = more remnant needed",
        "deletion_damage": cluster_bootstrap_mean(D, clusters, n_boot=args.n_boot),
        "remnant_residual": cluster_bootstrap_mean(R, clusters, n_boot=args.n_boot),
        "utility": cluster_bootstrap_mean(U, clusters, n_boot=args.n_boot),
        "kl_recovery_ratio": float(1.0 - R.mean() / D.mean()),
        "utility_distribution": {
            "frac_positive": float((U > 0).double().mean()),
            "frac_gt_0.1": float((U > 0.1).double().mean()),
            "q10": float(torch.quantile(U, 0.10)), "q50": float(torch.quantile(U, 0.50)),
            "q90": float(torch.quantile(U, 0.90)), "max": float(U.max()),
            "min": float(U.min()),
            "gini_top10pct_share": float(U.clamp(min=0).topk(max(1, len(U) // 10)).values.sum()
                                         / U.clamp(min=0).sum()),
        },
        "signals": {},
    }

    print("%s split: %d positions, %d prompts" % (args.split, len(rows), out["n_prompts"]))
    print("mean deletion damage %.5f, mean residual %.5f, recovery %.4f"
          % (out["deletion_damage"]["point"], out["remnant_residual"]["point"],
             out["kl_recovery_ratio"]))
    print("utility: %.1f%% positive, %.1f%% above 0.1, top decile holds %.1f%% of it"
          % (100 * out["utility_distribution"]["frac_positive"],
             100 * out["utility_distribution"]["frac_gt_0.1"],
             100 * out["utility_distribution"]["gini_top10pct_share"]))
    print()
    print("%-20s %8s %8s %8s %8s %8s" % ("signal", "rho(D)", "rho(U)", "r(U)",
                                         "AUROC.1", "top20%U"))
    deltas = [float(x) for x in args.deltas.split(",")]
    for name, s in signals.items():
        rec = {
            "spearman_vs_deletion_damage": spearman(s, D),
            "pearson_vs_deletion_damage": pearson(s, D),
            "spearman_vs_utility": spearman(s, U),
            "pearson_vs_utility": pearson(s, U),
            "auroc_by_delta": {}, "auprc_by_delta": {}, "base_rate_by_delta": {},
        }
        for d in deltas:
            lab = (U > d).double()
            if 0 < float(lab.mean()) < 1:
                rec["auroc_by_delta"][str(d)] = auroc(s, lab)
                rec["auprc_by_delta"][str(d)] = auprc(s, lab)
            rec["base_rate_by_delta"][str(d)] = base_rate(lab)
        # what fraction of total utility sits in the positions this signal ranks
        # highest -- the quantity a token gate actually exploits
        n = s.numel()
        for frac in (0.1, 0.2, 0.3, 0.5):
            k = max(1, int(round(frac * n)))
            sel = torch.argsort(s, descending=True)[:k]
            rec.setdefault("utility_captured", {})[str(frac)] = float(
                U[sel].sum() / U.sum())
        rec["captured_by_oracle"] = {
            str(f): float(U[torch.argsort(U, descending=True)[
                :max(1, int(round(f * n)))]].sum() / U.sum())
            for f in (0.1, 0.2, 0.3, 0.5)}
        out["signals"][name] = rec
        print("%-20s %+8.4f %+8.4f %+8.4f %8.4f %8.4f"
              % (name, rec["spearman_vs_deletion_damage"], rec["spearman_vs_utility"],
                 rec["pearson_vs_utility"], rec["auroc_by_delta"].get("0.1", float("nan")),
                 rec["utility_captured"]["0.2"]))

    # ------------------------------------------------------- failure quadrants
    print("\nfailure quadrants (quartile split on each signal, delta = 0.1)")
    quad = {}
    for name, s in signals.items():
        if name == "oracle_utility":
            continue
        q25, q75 = torch.quantile(s, 0.25), torch.quantile(s, 0.75)
        low_need = s <= q25            # signal says: skip
        high_need = s >= q75           # signal says: run
        big_benefit = U > 0.1
        wasted = high_need & ~big_benefit        # ran for nothing
        missed = low_need & big_benefit          # skipped where it mattered
        quad[name] = {
            "n_signal_low": int(low_need.sum()), "n_signal_high": int(high_need.sum()),
            "frac_high_signal_with_little_benefit": float(
                (wasted.sum() / max(1, int(high_need.sum())))),
            "frac_low_signal_with_large_benefit": float(
                (missed.sum() / max(1, int(low_need.sum())))),
            "mean_U_signal_low": float(U[low_need].mean()),
            "mean_U_signal_high": float(U[high_need].mean()),
            "gap_high_minus_low": float(U[high_need].mean() - U[low_need].mean()),
            "utility_lost_in_low_quartile": float(U[low_need].clamp(min=0).sum()
                                                  / U.clamp(min=0).sum()),
            "gap_ci": two_group_gap(U, clusters, high_need, low_need),
        }
        print("  %-20s  E[U|high]-E[U|low] = %+.5f [%+.5f,%+.5f]   "
              "ran-for-nothing %.3f   skipped-but-needed %.3f   "
              "utility in skip quartile %.3f"
              % (name, quad[name]["gap_high_minus_low"],
                 quad[name]["gap_ci"]["lo"], quad[name]["gap_ci"]["hi"],
                 quad[name]["frac_high_signal_with_little_benefit"],
                 quad[name]["frac_low_signal_with_large_benefit"],
                 quad[name]["utility_lost_in_low_quartile"]))
    out["failure_quadrants"] = quad

    # --------------------------------------------- does the remnant change the answer
    dense = col(rows, "dense_top1")
    dele = col(rows, "deleted_top1")
    rem = col(rows, "remnant_top1")
    gold = col(rows, "gold")
    changed = (rem != dele).double()
    changed_to_correct = ((rem != dele) & (rem == gold)).double()
    changed_to_wrong = ((rem != dele) & (dele == gold) & (rem != gold)).double()
    out["answer_change"] = {
        "frac_remnant_changes_deleted_answer": float(changed.mean()),
        "frac_change_lands_on_gold": float(changed_to_correct.sum() / changed.sum())
        if float(changed.sum()) else None,
        "frac_change_breaks_a_correct_answer": float(changed_to_wrong.sum() / changed.sum())
        if float(changed.sum()) else None,
        "net_gold_top1_gain_from_remnant": float(
            (rem == gold).double().mean() - (dele == gold).double().mean()),
        "gold_top1_dense": float((dense == gold).double().mean()),
        "gold_top1_deleted": float((dele == gold).double().mean()),
        "gold_top1_remnant": float((rem == gold).double().mean()),
        "paired_gain": cluster_bootstrap_mean_pair(
            (rem == gold).double(), (dele == gold).double(), clusters,
            n_boot=args.n_boot),
    }
    ac = out["answer_change"]
    print("\nanswer changes caused by running the whole remnant:")
    print("  changes the deleted model's argmax on %.1f%% of positions"
          % (100 * ac["frac_remnant_changes_deleted_answer"]))
    print("  of those, %.1f%% land on the gold token and %.1f%% break a correct one"
          % (100 * (ac["frac_change_lands_on_gold"] or 0),
             100 * (ac["frac_change_breaks_a_correct_answer"] or 0)))
    print("  net gold-top-1: deleted %.4f -> remnant %.4f (dense %.4f), paired "
          "delta %+.5f CI [%+.5f, %+.5f]"
          % (ac["gold_top1_deleted"], ac["gold_top1_remnant"], ac["gold_top1_dense"],
             ac["paired_gain"]["point"], ac["paired_gain"]["lo"], ac["paired_gain"]["hi"]))

    # ---- is the signal predicting need, or predicting position in the reply?
    pos = torch.tensor([r["pos"] for r in rows], dtype=torch.float64)
    out["position_confound"] = {
        "spearman_position_vs_utility": spearman(pos, U),
        "spearman_position_vs_deletion_damage": spearman(pos, D),
        **{("spearman_position_vs_" + k): spearman(pos, v) for k, v in signals.items()},
    }
    print("\nposition-in-reply confound: rho(pos,U) = %+.4f, rho(pos,probe) = %+.4f"
          % (out["position_confound"]["spearman_position_vs_utility"],
             out["position_confound"]["spearman_position_vs_probe_need"]))

    dest = RES / ("gate_signal_audit_%s.json" % args.split)
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("\nwrote %s" % dest.name)


if __name__ == "__main__":
    main()
