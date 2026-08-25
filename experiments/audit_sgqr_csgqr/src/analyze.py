#!/usr/bin/env python3
"""Turn the frontier run into the decision tables the audit has to answer with.

Four blocks, matching the brief:

* **Factorial (section 11).** The 2x2 of {fixed, gated} x {random order,
  significance order} at each matched budget, with main effects and the
  interaction. Everything is a paired per-position difference, so the
  decomposition is on the same tokens throughout.
* **Confidence gate (sections 5 and 4).** CSGQR against SGQR, against its own
  permuted-budget control, and against the oracle ceiling, at every budget.
* **Significance (section 10).** Interventional against random (several seeds)
  and against magnitude, with the random arms pooled so the comparison carries
  seed variance rather than one draw.
* **Multiplicity.** Holm-Bonferroni within each pre-declared family. Families are
  declared here, before the numbers are read, and the raw intervals are printed
  next to the adjusted decision so a reader can see both.

Bootstrap p-values are two-sided and derived from the paired clustered bootstrap
distribution: p = 2 * min(P(d <= 0), P(d >= 0)), floored at 1/n_boot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import REPO                                              # noqa: E402
from stats import cluster_bootstrap_mean_pair, holm_bonferroni     # noqa: E402

RES = REPO / "experiments/audit_sgqr_csgqr/results"


def load_per_example(tags: list[str]) -> dict[str, dict]:
    """arm -> {kl, nll, gold, clusters} as tensors, from the per-example JSONL.

    Several tags may be merged: the frontier is run one budget point per
    invocation so that a crash costs one budget rather than the sweep. Arms that
    appear in more than one file (the shared endpoints) keep the first copy, and
    the copies are checked to agree -- if a repeated endpoint disagreed between
    runs the whole sweep would be non-deterministic and the merge must fail loudly.
    """
    by: dict[str, dict[str, list]] = {}
    for tag in tags:
        path = RES / ("per_example_results%s.jsonl" % (("_" + tag) if tag else ""))
        seen_here: dict[str, dict[str, list]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            d = seen_here.setdefault(r["arm"], {"kl": [], "nll": [], "gold": [], "cl": []})
            d["kl"].append(r["kl"])
            d["nll"].append(r["nll"])
            d["gold"].append(r["gold_top1"])
            d["cl"].append(r["prompt"])
        for k, d in seen_here.items():
            if k in by:
                if by[k]["kl"] != d["kl"]:
                    raise SystemExit(
                        "arm %s differs between runs %s -- the sweep is not "
                        "deterministic and the merge cannot be trusted" % (k, tag))
                continue
            by[k] = d
    out = {}
    for k, d in by.items():
        out[k] = {"kl": torch.tensor(d["kl"], dtype=torch.float64),
                  "nll": torch.tensor(d["nll"], dtype=torch.float64),
                  "gold": torch.tensor(d["gold"], dtype=torch.float64),
                  "clusters": d["cl"]}
    return out


def boot_p(a: torch.Tensor, b: torch.Tensor, clusters, n_boot: int) -> float:
    """Two-sided bootstrap p for mean(a) - mean(b), clustered by prompt."""
    d = (a - b).to(torch.float64)
    ids = sorted(set(clusters))
    idx = {c: [] for c in ids}
    for i, c in enumerate(clusters):
        idx[c].append(i)
    sums = torch.tensor([float(d[idx[c]].sum()) for c in ids], dtype=torch.float64)
    cnts = torch.tensor([len(idx[c]) for c in ids], dtype=torch.float64)
    g = torch.Generator().manual_seed(20260824)
    draw = torch.randint(0, len(ids), (n_boot, len(ids)), generator=g)
    means = sums[draw].sum(dim=1) / cnts[draw].sum(dim=1)
    p_le = float((means <= 0).double().mean())
    p_ge = float((means >= 0).double().mean())
    return max(1.0 / n_boot, min(1.0, 2.0 * min(p_le, p_ge)))


def pooled(per: dict, keys: list[str], metric: str) -> torch.Tensor | None:
    ks = [k for k in keys if k in per]
    if not ks:
        return None
    return torch.stack([per[k][metric] for k in ks]).mean(dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tags", default="",
                    help="comma-separated frontier tags to merge (one per budget)")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    tags = [t for t in args.tags.split(",") if t] or [""]
    fronts = [json.loads((RES / ("frontier_%s%s.json"
                                 % (args.split, ("_" + t) if t else "")))
                         .read_text(encoding="utf-8")) for t in tags]
    front = dict(fronts[0])
    arms = {}
    budgets: list[float] = []
    for f in fronts:
        arms.update(f["arms"])
        for b in f["budgets"]:
            if b not in budgets:
                budgets.append(b)
    budgets.sort()
    front["arms"] = arms
    front["budgets"] = budgets
    front["merged_tags"] = tags
    per = load_per_example(tags)
    clusters = per[next(iter(per))]["clusters"]
    seeds = fronts[0]["seeds"]
    tag = ("_" + args.out_tag) if args.out_tag else ""
    NB = args.n_boot

    def has(k):
        return k in per and k in arms

    def contrast(a, b, metric="kl"):
        if not (has(a) and has(b)):
            return None
        r = cluster_bootstrap_mean_pair(per[a][metric], per[b][metric], clusters,
                                        n_boot=NB)
        r["p"] = boot_p(per[a][metric], per[b][metric], clusters, NB)
        r["a"], r["b"], r["metric"] = a, b, metric
        r["a_point"] = float(per[a][metric].mean())
        r["b_point"] = float(per[b][metric].mean())
        r["compute_gap"] = (arms[a]["executed_group_tokens"]
                            - arms[b]["executed_group_tokens"])
        return r

    out: dict = {"split": args.split, "budgets": budgets, "seeds": seeds,
                 "n_boot": NB,
                 "n_positions": len(clusters), "n_prompts": len(set(clusters)),
                 "families": {}, "factorial": {}, "tables": {}}

    # ============================================================ factorial
    print("=" * 78)
    print("FACTORIAL DECOMPOSITION  (KL to dense; lower is better)")
    print("=" * 78)
    print("%6s %11s %11s %11s %11s | %11s %11s %11s"
          % ("budget", "flat+rand", "flat+sig", "gate+rand", "gate+sig",
             "sig effect", "gate effect", "interaction"))
    fac_rows = []
    for b in budgets:
        t = format(b, ".2f")
        cells = {"flat_rand": "sgqr-rand-s%d-%s" % (seeds[0], t),
                 "flat_sig": "sgqr-sig-" + t,
                 "gate_rand": "csgqr-probe-rand-" + t,
                 "gate_sig": "csgqr-probe-sig-" + t}
        if not all(has(v) for v in cells.values()):
            continue
        m = {k: per[v]["kl"] for k, v in cells.items()}
        sig_eff_flat = cluster_bootstrap_mean_pair(m["flat_sig"], m["flat_rand"],
                                                   clusters, n_boot=NB)
        sig_eff_gate = cluster_bootstrap_mean_pair(m["gate_sig"], m["gate_rand"],
                                                   clusters, n_boot=NB)
        gate_eff_rand = cluster_bootstrap_mean_pair(m["gate_rand"], m["flat_rand"],
                                                    clusters, n_boot=NB)
        gate_eff_sig = cluster_bootstrap_mean_pair(m["gate_sig"], m["flat_sig"],
                                                   clusters, n_boot=NB)
        inter = cluster_bootstrap_mean_pair(
            m["gate_sig"] - m["gate_rand"], m["flat_sig"] - m["flat_rand"],
            clusters, n_boot=NB)
        rand_keys = ["sgqr-rand-s%d-%s" % (sd, t) for sd in seeds]
        pooled_rand = pooled(per, rand_keys, "kl")
        sig_eff_pooled = (cluster_bootstrap_mean_pair(m["flat_sig"], pooled_rand,
                                                      clusters, n_boot=NB)
                          if pooled_rand is not None else None)
        row = {
            "budget": b,
            "cells": {k: float(v.mean()) for k, v in m.items()},
            "significance_effect_vs_pooled_random": sig_eff_pooled,
            "random_seed_kls": [float(per[k]["kl"].mean()) for k in rand_keys
                                if k in per],
            "note": ("the factorial's two random cells use ONE random ordering "
                     "(seed %d) so that the 2x2 is paired; the pooled column "
                     "carries the seed variance, which is large" % seeds[0]),
            "arms": cells,
            "significance_effect_at_flat": sig_eff_flat,
            "significance_effect_at_gate": sig_eff_gate,
            "gate_effect_at_random_order": gate_eff_rand,
            "gate_effect_at_significance_order": gate_eff_sig,
            "interaction": inter,
            "compute_matched": len(set(arms[v]["executed_group_tokens"]
                                       for v in cells.values())) == 1,
        }
        fac_rows.append(row)
        print("%6.2f %11.5f %11.5f %11.5f %11.5f | %+11.5f %+11.5f %+11.5f%s"
              % (b, row["cells"]["flat_rand"], row["cells"]["flat_sig"],
                 row["cells"]["gate_rand"], row["cells"]["gate_sig"],
                 sig_eff_flat["point"], gate_eff_rand["point"], inter["point"],
                 "" if row["compute_matched"] else "  [COMPUTE NOT MATCHED]"))
        if sig_eff_pooled is not None:
            print("       significance effect against the POOLED random seeds: "
                  "%+.5f [%+.5f, %+.5f]   seed KLs %s"
                  % (sig_eff_pooled["point"], sig_eff_pooled["lo"],
                     sig_eff_pooled["hi"],
                     ", ".join(format(v, ".5f") for v in row["random_seed_kls"])))
    out["factorial"] = fac_rows
    print("\n  sig effect  = (flat+sig) - (flat+rand)     negative = significance helps")
    print("  gate effect = (gate+rand) - (flat+rand)    negative = the gate helps")
    print("  interaction = [gate: sig-rand] - [flat: sig-rand]")

    # ================================================== confidence gate family
    print("\n" + "=" * 78)
    print("CONFIDENCE GATE AT MATCHED COMPUTE")
    print("=" * 78)
    fam_gate: dict[str, float] = {}
    gate_rows = []
    for b in budgets:
        t = format(b, ".2f")
        for a, bb, why in [
            ("csgqr-probe-sig-" + t, "sgqr-sig-" + t, "CSGQR vs SGQR"),
            ("csgqr-conf-sig-" + t, "sgqr-sig-" + t, "CSGQR(conf probe) vs SGQR"),
            ("csgqr-lens-sig-" + t, "sgqr-sig-" + t, "CSGQR(exact lens) vs SGQR"),
            ("csgqr-binary-probe-" + t, "sgqr-sig-" + t, "binary gate vs SGQR"),
            ("oracle-sig-" + t, "sgqr-sig-" + t, "ORACLE ceiling vs SGQR"),
        ]:
            r = contrast(a, bb)
            if r:
                r["question"] = why
                r["budget"] = b
                gate_rows.append(r)
                fam_gate[a + " vs " + bb] = r["p"]
        for s in seeds:
            r = contrast("csgqr-probe-sig-" + t, "randtok-graded-s%d-%s" % (s, t))
            if r:
                r["question"] = "CSGQR vs its own budgets permuted across tokens"
                r["budget"] = b
                gate_rows.append(r)
                fam_gate[r["a"] + " vs " + r["b"]] = r["p"]
    holm_gate_full = holm_bonferroni(fam_gate)
    holm_gate = holm_gate_full["results"]
    for r in gate_rows:
        key = r["a"] + " vs " + r["b"]
        hh = holm_gate.get(key, {})
        r["holm"] = hh
        print("  b=%.2f %-30s vs %-26s dKL %+9.5f [%+9.5f,%+9.5f] p=%.4f "
              "adj=%.4f %s  dcompute=%d"
              % (r["budget"], r["a"], r["b"], r["point"], r["lo"], r["hi"], r["p"],
                 hh.get("adjusted_p", float("nan")),
                 "SIG " if hh.get("reject") else "ns  ", r["compute_gap"]))
    out["tables"]["confidence_gate"] = gate_rows
    out["families"]["confidence_gate"] = holm_gate_full

    # ==================================================== significance family
    print("\n" + "=" * 78)
    print("SIGNIFICANCE SELECTION AT MATCHED COMPUTE")
    print("=" * 78)
    fam_sig: dict[str, float] = {}
    sig_rows = []
    for b in budgets:
        t = format(b, ".2f")
        rand_keys = ["sgqr-rand-s%d-%s" % (s, t) for s in seeds]
        pooled_rand = pooled(per, rand_keys, "kl")
        if pooled_rand is not None and has("sgqr-sig-" + t):
            r = cluster_bootstrap_mean_pair(per["sgqr-sig-" + t]["kl"], pooled_rand,
                                            clusters, n_boot=NB)
            r["p"] = boot_p(per["sgqr-sig-" + t]["kl"], pooled_rand, clusters, NB)
            r.update({"a": "sgqr-sig-" + t, "b": "mean of %d random seeds" % len(rand_keys),
                      "question": "significance vs random (seeds pooled)",
                      "budget": b, "metric": "kl",
                      "a_point": float(per["sgqr-sig-" + t]["kl"].mean()),
                      "b_point": float(pooled_rand.mean()),
                      "random_seed_spread": [float(per[k]["kl"].mean()) for k in rand_keys
                                             if k in per],
                      "compute_gap": 0})
            sig_rows.append(r)
            fam_sig[r["a"] + " vs pooled-random"] = r["p"]
        for a, bb, why in [("sgqr-sig-" + t, "sgqr-mag-" + t,
                            "significance vs magnitude"),
                           ("sgqr-sig-" + t, "sgqr-rev-" + t,
                            "significance vs reversed"),
                           ("sgqr-mag-" + t, "sgqr-rand-s%d-%s" % (seeds[0], t),
                            "magnitude vs random")]:
            r = contrast(a, bb)
            if r:
                r["question"] = why
                r["budget"] = b
                sig_rows.append(r)
                fam_sig[a + " vs " + bb] = r["p"]
    holm_sig_full = holm_bonferroni(fam_sig)
    holm_sig = holm_sig_full["results"]
    for r in sig_rows:
        key = ((r["a"] + " vs pooled-random") if r["b"].startswith("mean of")
               else (r["a"] + " vs " + r["b"]))
        hh = holm_sig.get(key, {})
        r["holm"] = hh
        extra = ""
        if "random_seed_spread" in r:
            v = r["random_seed_spread"]
            extra = "  seed spread [%.5f, %.5f]" % (min(v), max(v))
        print("  b=%.2f %-24s vs %-26s dKL %+9.5f [%+9.5f,%+9.5f] p=%.4f "
              "adj=%.4f %s%s"
              % (r["budget"], r["a"], r["b"], r["point"], r["lo"], r["hi"], r["p"],
                 hh.get("adjusted_p", float("nan")),
                 "SIG " if hh.get("reject") else "ns  ", extra))
    out["tables"]["significance"] = sig_rows
    out["families"]["significance"] = holm_sig_full

    # ================================================ capability contrasts
    print()
    print("=" * 78)
    print("THE SAME CONTRASTS ON THE CAPABILITY METRIC (gold top-1; higher is better)")
    print("=" * 78)
    fam_cap: dict[str, float] = {}
    cap_contrasts = []
    for b in budgets:
        t = format(b, ".2f")
        pairs = [("sgqr-sig-" + t, "off-delete4", "does SGQR recover capability?"),
                 ("csgqr-probe-sig-" + t, "sgqr-sig-" + t, "does the gate add capability?"),
                 ("sgqr-sig-" + t, "sgqr-rand-s%d-%s" % (seeds[0], t),
                  "does significance add capability?"),
                 ("oracle-sig-" + t, "sgqr-sig-" + t, "ceiling on the gate"),
                 ("sgqr-sig-" + t, "dense", "residual capability gap vs dense")]
        for a, bb, why in pairs:
            r = contrast(a, bb, metric="gold")
            if r:
                r["question"] = why
                r["budget"] = b
                cap_contrasts.append(r)
                fam_cap[a + " vs " + bb + " [gold]"] = r["p"]
    for a, bb in (("on-int4-full", "off-delete4"), ("on-int4-full", "dense")):
        r = contrast(a, bb, metric="gold")
        if r:
            r["question"] = "always-on remnant vs " + bb
            r["budget"] = 1.0
            cap_contrasts.append(r)
            fam_cap[a + " vs " + bb + " [gold]"] = r["p"]
    holm_cap_full = holm_bonferroni(fam_cap)
    holm_cap = holm_cap_full["results"]
    for r in cap_contrasts:
        hh = holm_cap.get(r["a"] + " vs " + r["b"] + " [gold]", {})
        r["holm"] = hh
        print("  b=%.2f %-26s vs %-22s dGold %+8.5f [%+8.5f,%+8.5f] p=%.4f "
              "adj=%.4f %s"
              % (r["budget"], r["a"], r["b"], r["point"], r["lo"], r["hi"], r["p"],
                 hh.get("adjusted_p", float("nan")),
                 "SIG " if hh.get("reject") else "ns  "))
    out["tables"]["capability_contrasts"] = cap_contrasts
    out["families"]["capability"] = holm_cap_full

    # ============================================ mechanistic vs capability
    print("\n" + "=" * 78)
    print("MECHANISTIC RECOVERY vs CAPABILITY RECOVERY")
    print("=" * 78)
    ref_dense = arms.get("dense")
    ref_del = arms.get("off-delete4")
    cap_rows = []
    print("%-30s %9s %9s %9s %9s" % ("arm", "KL", "gold@1", "ppl", "act"))
    for k in sorted(arms, key=lambda z: arms[z]["kl"]["point"]):
        a = arms[k]
        cap_rows.append({"arm": k, "kl": a["kl"]["point"],
                         "gold_top1": a["gold_top1"]["point"],
                         "perplexity": a["perplexity_gold"],
                         "active_fraction": a["active_fraction"],
                         "kl_recovery": (1 - a["kl"]["point"] / ref_del["kl"]["point"])
                         if ref_del else None,
                         "gold_recovery": ((a["gold_top1"]["point"] - ref_del["gold_top1"]["point"])
                                           / (ref_dense["gold_top1"]["point"]
                                              - ref_del["gold_top1"]["point"]))
                         if ref_del and ref_dense and
                         abs(ref_dense["gold_top1"]["point"] - ref_del["gold_top1"]["point"]) > 1e-9
                         else None})
        print("%-30s %9.5f %9.4f %9.2f %9.4f"
              % (k, a["kl"]["point"], a["gold_top1"]["point"],
                 a["perplexity_gold"], a["active_fraction"]))
    out["tables"]["capability"] = cap_rows

    # correlation between mechanistic and capability recovery across arms
    kl_v = torch.tensor([r["kl"] for r in cap_rows], dtype=torch.float64)
    gd_v = torch.tensor([r["gold_top1"] for r in cap_rows], dtype=torch.float64)
    pp_v = torch.tensor([r["perplexity"] for r in cap_rows], dtype=torch.float64)
    from stats import pearson, spearman
    out["mechanistic_vs_capability"] = {
        "n_arms": len(cap_rows),
        "spearman_kl_vs_gold_top1": spearman(kl_v, gd_v),
        "pearson_kl_vs_gold_top1": pearson(kl_v, gd_v),
        "spearman_kl_vs_perplexity": spearman(kl_v, pp_v),
        "pearson_kl_vs_perplexity": pearson(kl_v, pp_v),
    }
    print("\nacross %d arms: rho(KL, gold@1) = %+.4f ; rho(KL, perplexity) = %+.4f"
          % (out["mechanistic_vs_capability"]["n_arms"],
             out["mechanistic_vs_capability"]["spearman_kl_vs_gold_top1"],
             out["mechanistic_vs_capability"]["spearman_kl_vs_perplexity"]))

    (RES / ("analysis_%s%s.json" % (args.split, tag))).write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("\nwrote analysis_%s%s.json" % (args.split, tag))


if __name__ == "__main__":
    main()
