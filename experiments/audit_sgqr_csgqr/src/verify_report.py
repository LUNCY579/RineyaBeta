#!/usr/bin/env python3
"""Re-derive every number quoted in SGQR_CSGQR_DUAL_AUDIT.md from the artifacts.

The report and the artifacts must not be able to drift apart. Each row below
names a quantity as the report states it and recomputes it from the committed
JSON. A mismatch is a failure, not a rounding note: the report is wrong or the
artifact is, and either way the discrepancy has to be visible.

This is the same discipline the campaign applied to itself with
``verify_claims.py``; it is applied here to the audit's own claims so that the
audit is held to the standard it audits against.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUD = ROOT / "experiments/audit_sgqr_csgqr"
RES = AUD / "results"

_c: dict[str, dict] = {}


def J(name: str) -> dict:
    if name not in _c:
        _c[name] = json.loads((RES / name).read_text(encoding="utf-8"))
    return _c[name]


ROWS: list[dict] = []


def pin(what: str, source: str, stated, actual, tol: float = 5e-5) -> None:
    if isinstance(stated, bool) or isinstance(actual, bool):
        ok = bool(stated) == bool(actual)
    elif isinstance(stated, str) or isinstance(actual, str):
        ok = str(stated) == str(actual)
    elif isinstance(stated, (list, tuple)):
        ok = (len(stated) == len(actual)
              and all(abs(float(a) - float(b)) <= tol for a, b in zip(stated, actual)))
    else:
        ok = actual is not None and abs(float(actual) - float(stated)) <= tol
    ROWS.append({"quantity": what, "source": source, "report": stated,
                 "recomputed": actual, "tol": tol, "ok": bool(ok)})


def arm(tag: str, key: str, field: str = "kl") -> float:
    a = J("frontier_test_%s.json" % tag)["arms"][key]
    v = a[field]
    return v["point"] if isinstance(v, dict) else float(v)


def main() -> int:
    # ---------------------------------------------------------------- A
    ev = json.loads((AUD / "evidence/evidence_manifest.json").read_text(encoding="utf-8"))
    pin("artifacts hashed", "evidence_manifest.json", 132, ev["n_artifacts"], 0)
    pin("untracked artifacts", "evidence_manifest.json", 22, len(ev["untracked"]), 0)
    pin("results predating a producer change", "evidence_manifest.json", 4,
        len(ev["produced_by_since_changed_code"]), 0)

    ce = J("claim_evidence_table.json")
    pin("claims checked", "claim_evidence_table.json", 194, ce["n_claims"], 0)
    for k, v in (("VERIFIED", 188), ("INVALID", 3), ("AMBIGUOUS", 2),
                 ("MISSING_PROVENANCE", 1)):
        pin("claims %s" % k, "claim_evidence_table.json", v,
            ce["status_counts"].get(k, 0), 0)

    vm = J("verify_mechanism.json")
    pin("mechanism checks passed", "verify_mechanism.json", 33, vm["passed"], 0)
    pin("mechanism checks failed", "verify_mechanism.json", 0, vm["failed"], 0)

    ra = J("repro_A_baselines.json")["arms"]
    hist = json.loads((ROOT / "experiments/csgqr/results/"
                       "forced_heldout_A_baselines.json").read_text(encoding="utf-8"))["arms"]
    for k in ("b1-delete", "b4-int4-all", "b13-wrong-layer"):
        pin("reproduction of %s is bit-identical" % k, "repro_A_baselines.json",
            hist[k]["kl"]["point"], ra[k]["kl"]["point"], 0.0)

    # ---------------------------------------------------------------- C
    for tag, b, tot in (("b030", "0.30", 1509896), ("b050", "0.50", 2516496)):
        f = J("frontier_test_%s.json" % tag)
        totals = sorted(set(v["executed_group_tokens"] for v in f["arms"].values()
                            if v["mode"] not in ("dense", "none")))
        pin("every routed arm at rho=%s executes %d group-tokens" % (b, tot),
            "frontier_test_%s.json" % tag, str(sorted({tot, 5032992})), str(totals), 0)

    for tag, key, val in (
            ("b030", "dense", 0.0), ("b030", "off-delete4", 0.84122),
            ("b030", "on-int4-full", 0.02415), ("b030", "oracle-sig-0.30", 0.24913),
            ("b030", "sgqr-sig-0.30", 0.39964),
            ("b030", "csgqr-probe-sig-0.30", 0.39662),
            ("b030", "csgqr-conf-sig-0.30", 0.38694),
            ("b030", "csgqr-lens-sig-0.30", 0.40217),
            ("b030", "csgqr-binary-probe-0.30", 0.42495),
            ("b030", "sgqr-mag-0.30", 0.60230), ("b030", "sgqr-rev-0.30", 0.77932),
            ("b050", "oracle-sig-0.50", 0.14617), ("b050", "sgqr-sig-0.50", 0.30779),
            ("b050", "csgqr-probe-sig-0.50", 0.30647),
            ("b050", "csgqr-conf-sig-0.50", 0.27715),
            ("b050", "csgqr-lens-sig-0.50", 0.28175),
            ("b050", "csgqr-binary-probe-0.50", 0.29030),
            ("b050", "sgqr-mag-0.50", 0.45166), ("b050", "sgqr-rev-0.50", 0.68892)):
        pin("%s KL" % key, "frontier_test_%s.json" % tag, val, arm(tag, key))

    an = J("analysis_test_main.json")
    fac = {round(r["budget"], 2): r for r in an["factorial"]}
    pin("significance effect at rho=0.30", "analysis_test_main.json", -0.18833,
        fac[0.3]["significance_effect_at_flat"]["point"])
    pin("confidence effect at rho=0.30", "analysis_test_main.json", -0.00284,
        fac[0.3]["gate_effect_at_random_order"]["point"])
    pin("interaction at rho=0.30", "analysis_test_main.json", -0.00019,
        fac[0.3]["interaction"]["point"])
    pin("significance effect vs pooled random at rho=0.30",
        "analysis_test_main.json", -0.17267,
        fac[0.3]["significance_effect_vs_pooled_random"]["point"])
    pin("significance effect vs pooled random at rho=0.50",
        "analysis_test_main.json", -0.06554,
        fac[0.5]["significance_effect_vs_pooled_random"]["point"])
    pin("random seed KLs at rho=0.30 span 0.47218-0.65679",
        "analysis_test_main.json", [0.47218, 0.65679],
        [min(fac[0.3]["random_seed_kls"]), max(fac[0.3]["random_seed_kls"])])
    pin("random seed KLs at rho=0.50 span 0.31194-0.46519",
        "analysis_test_main.json", [0.31194, 0.46519],
        [min(fac[0.5]["random_seed_kls"]), max(fac[0.5]["random_seed_kls"])])

    def gate(bud, a, b):
        for r in an["tables"]["confidence_gate"]:
            if abs(r["budget"] - bud) < 1e-9 and r["a"] == a and r["b"] == b:
                return r
        return None

    for bud, a, b, d in ((0.30, "csgqr-probe-sig-0.30", "sgqr-sig-0.30", -0.00302),
                         (0.30, "oracle-sig-0.30", "sgqr-sig-0.30", -0.15051),
                         (0.50, "csgqr-probe-sig-0.50", "sgqr-sig-0.50", -0.00133),
                         (0.50, "oracle-sig-0.50", "sgqr-sig-0.50", -0.16162),
                         (0.50, "csgqr-conf-sig-0.50", "sgqr-sig-0.50", -0.03064),
                         (0.30, "csgqr-conf-sig-0.30", "sgqr-sig-0.30", -0.01270)):
        r = gate(bud, a, b)
        pin("%s vs %s dKL" % (a, b), "analysis_test_main.json", d,
            r["point"] if r else None)

    def cap(a, b):
        for r in an["tables"]["capability_contrasts"]:
            if r["a"] == a and r["b"] == b:
                return r
        return None

    for a, b, d in (("sgqr-sig-0.30", "off-delete4", 0.02828),
                    ("csgqr-probe-sig-0.30", "sgqr-sig-0.30", -0.00149),
                    ("sgqr-sig-0.30", "sgqr-rand-s1-0.30", 0.03015),
                    ("on-int4-full", "off-delete4", 0.03796),
                    ("on-int4-full", "dense", -0.00112)):
        r = cap(a, b)
        pin("%s vs %s dGold" % (a, b), "analysis_test_main.json", d,
            r["point"] if r else None)

    pin("Spearman(KL, gold top-1) across arms", "analysis_test_main.json",
        -0.885, an["mechanistic_vs_capability"]["spearman_kl_vs_gold_top1"], 1e-3)
    pin("Spearman(KL, perplexity) across arms", "analysis_test_main.json",
        0.803, an["mechanistic_vs_capability"]["spearman_kl_vs_perplexity"], 1e-3)
    pin("arms in the merged analysis", "analysis_test_main.json", 39,
        an["mechanistic_vs_capability"]["n_arms"], 0)

    # ---------------------------------------------------------------- D
    g = J("gate_signal_audit_test.json")
    pin("mean deletion damage on test", "gate_signal_audit_test.json", 0.84122,
        g["deletion_damage"]["point"])
    pin("KL recovery of the always-on remnant", "gate_signal_audit_test.json",
        0.9713, g["kl_recovery_ratio"], 5e-4)
    pin("fraction of positions with positive utility",
        "gate_signal_audit_test.json", 0.9978,
        g["utility_distribution"]["frac_positive"], 5e-4)
    pin("fraction of positions with utility > 0.1",
        "gate_signal_audit_test.json", 0.8508,
        g["utility_distribution"]["frac_gt_0.1"], 5e-4)
    pin("top decile share of utility", "gate_signal_audit_test.json", 0.392,
        g["utility_distribution"]["gini_top10pct_share"], 5e-4)
    for k, rho, au, cap20 in (("lens_max_prob", 0.0438, 0.5548, 0.1861),
                              ("lens_neg_entropy", 0.0892, 0.5733, 0.1975),
                              ("probe_need", 0.3051, 0.6149, 0.3188),
                              ("probe_confidence", 0.0755, 0.4169, 0.2270),
                              ("oracle_utility", 1.0, 1.0, 0.5786)):
        s = g["signals"][k]
        pin("%s spearman vs utility" % k, "gate_signal_audit_test.json", rho,
            s["spearman_vs_utility"], 5e-4)
        pin("%s AUROC at delta=0.1" % k, "gate_signal_audit_test.json", au,
            s["auroc_by_delta"]["0.1"], 5e-4)
        pin("%s utility captured in top 20%%" % k, "gate_signal_audit_test.json",
            cap20, s["utility_captured"]["0.2"], 5e-4)
    pin("remnant changes the deleted argmax on 40.5% of positions",
        "gate_signal_audit_test.json", 0.405,
        g["answer_change"]["frac_remnant_changes_deleted_answer"], 5e-4)
    pin("net gold top-1 gain from the remnant", "gate_signal_audit_test.json",
        0.03796, g["answer_change"]["paired_gain"]["point"])

    w = J("within_prompt_signal_test.json")
    pin("between-prompt share of utility variance",
        "within_prompt_signal_test.json", 0.1204,
        w["between_prompt_variance_share_utility"], 5e-4)
    for k, v in (("lens_max_prob", -0.0016), ("lens_neg_entropy", 0.0364),
                 ("lens_prob_margin", -0.0062), ("lens_logit_margin", -0.0164),
                 ("probe_need", 0.3316), ("position_in_reply", 0.1752)):
        pin("%s within-prompt spearman" % k, "within_prompt_signal_test.json", v,
            w["signals"][k]["spearman_within_prompt"], 5e-4)

    pc = J("probe_ceiling.json")
    pin("linear probe test spearman", "probe_ceiling.json", 0.3051,
        pc["models"]["linear, val-selected ridge"]["test_spearman"], 5e-4)
    pin("MLP probe test spearman", "probe_ceiling.json", 0.1467,
        pc["models"]["MLP, val-selected"]["test_spearman"], 5e-4)
    pin("unregularized linear test spearman", "probe_ceiling.json", -0.0831,
        pc["models"]["linear, IN-SAMPLE ceiling"]["test_spearman"], 5e-4)

    # ---------------------------------------------------------------- E
    sg = J("significance_96.json")
    pin("mean four-layer deletion KL on calibration", "significance_96.json",
        0.80966, sg["mean_kl_delete"], 5e-5)
    pin("groups with negative marginal utility", "significance_96.json", 38,
        sg["n_negative_groups"], 0)
    pin("rank-stability Spearman mean", "significance_96.json", 0.9414,
        sg["rank_stability"]["spearman_mean"], 5e-4)
    pin("rank-stability Spearman p05", "significance_96.json", 0.8969,
        sg["rank_stability"]["spearman_p05"], 5e-4)
    pin("top-24 overlap mean", "significance_96.json", 0.8514,
        sg["rank_stability"]["topk_overlap_mean"]["24"], 5e-4)

    def sigrow(bud, a, b):
        for r in an["tables"]["significance"]:
            if abs(r["budget"] - bud) < 1e-9 and r["a"] == a and r["b"] == b:
                return r
        return None
    for bud, a, b, d in ((0.30, "sgqr-sig-0.30", "sgqr-mag-0.30", -0.20265),
                         (0.30, "sgqr-sig-0.30", "sgqr-rev-0.30", -0.37967),
                         (0.50, "sgqr-sig-0.50", "sgqr-mag-0.50", -0.14387),
                         (0.50, "sgqr-mag-0.50", "sgqr-rand-s1-0.50", 0.13973)):
        r = sigrow(bud, a, b)
        pin("%s vs %s dKL" % (a, b), "analysis_test_main.json", d,
            r["point"] if r else None)

    # -------------------------------------------------------- optional blocks
    for name, pins in (("frontier_test_perm.json", []),
                       ("reference_arms_test.json", []),
                       ("systems_audit.json", []),
                       ("capacity_controls.json", [])):
        if not (RES / name).exists():
            ROWS.append({"quantity": "block present: " + name, "source": name,
                         "report": "expected", "recomputed": "MISSING",
                         "tol": 0, "ok": False})

    n_ok = sum(1 for r in ROWS if r["ok"])
    payload = {"audit": "sgqr_csgqr_dual", "total": len(ROWS), "ok": n_ok,
               "failed": len(ROWS) - n_ok, "rows": ROWS}
    (RES / "verify_report.json").write_text(json.dumps(payload, indent=1) + "\n",
                                            encoding="utf-8")
    print("report pins: %d/%d re-derive" % (n_ok, len(ROWS)))
    for r in ROWS:
        if not r["ok"]:
            print("  MISMATCH  %-58s report=%s  recomputed=%s"
                  % (r["quantity"], r["report"], r["recomputed"]))
    return 0 if n_ok == len(ROWS) else 1


if __name__ == "__main__":
    sys.exit(main())
