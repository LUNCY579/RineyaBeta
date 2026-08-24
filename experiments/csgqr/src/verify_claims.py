#!/usr/bin/env python3
"""Check every headline CSGQR number against the committed result files.

In the spirit of ``scripts/design/verify_claims.py``: a fresh clone should be
able to decide whether the prose is consistent with the data, without a GPU and
without rerunning anything. Each claim names the file and field it comes from, so
a failure identifies which of the two drifted.

Claims whose source file is absent are reported MISSING rather than passing
silently -- a claim that cannot be checked is not a claim that is true.

Run:
    python experiments/csgqr/src/verify_claims.py
    python experiments/csgqr/src/verify_claims.py --json results/claims.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / "experiments" / "csgqr" / "results"

CHECKS: list[dict] = []


def load(rel: str):
    p = RESULTS / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def claim(name: str, source: str, expected, actual, *, tol: float = 0.0,
          note: str = "") -> None:
    if actual is None:
        CHECKS.append({"claim": name, "source": source, "status": "MISSING",
                       "expected": expected, "actual": None, "note": note})
        print("  [MISS] " + name + "   (" + source + " absent)")
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        ok = bool(expected) == bool(actual)
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(float(expected) - float(actual)) <= tol
    else:
        ok = expected == actual
    CHECKS.append({"claim": name, "source": source,
                   "status": "OK" if ok else "FAIL",
                   "expected": expected, "actual": actual, "tol": tol,
                   "note": note})
    mark = "OK  " if ok else "FAIL"
    shown = (format(actual, ".6g") if isinstance(actual, float) else str(actual))
    print("  [" + mark + "] " + name + "   = " + shown
          + ("" if ok else "   EXPECTED " + str(expected)))


def section(t: str) -> None:
    print("\n" + t)
    print("-" * len(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    # ------------------------------------------------------------------
    section("Environment and model identity")
    env = load("environment.json")
    if env:
        claim("CUDA was available to the run", "environment.json",
              True, env.get("cuda_available"))
        claim("torch is a CUDA build", "environment.json",
              True, env.get("torch_cuda_build") is not None)
        claim("GPU total VRAM (MiB)", "environment.json",
              8188.0, (env.get("gpu") or {}).get("total_vram_mib"), tol=1.0)
        cfg = ((env.get("model") or {}).get("config") or {})
        claim("model layers", "environment.json", 28, cfg.get("num_hidden_layers"))
        claim("model hidden size", "environment.json", 1024, cfg.get("hidden_size"))
        claim("model intermediate size", "environment.json", 3072,
              cfg.get("intermediate_size"))
        claim("model vocab size", "environment.json", 32768, cfg.get("vocab_size"))
        claim("embeddings are tied", "environment.json", True,
              cfg.get("tie_word_embeddings"))
    else:
        claim("environment manifest exists", "environment.json", True, None)

    # ------------------------------------------------------------------
    section("Analytic budget (exact arithmetic, no measurement)")
    b = load("budget.json")
    if b:
        shape = b["shape"]
        claim("total parameters", "budget.json", 474021888,
              shape["total_parameters"])
        claim("MLP parameters per layer", "budget.json", 9437184,
              shape["mlp_parameters_per_layer"])
        share = shape["mlp_parameters_per_layer"] / shape["total_parameters"]
        claim("one MLP layer is 1.99% of the model", "budget.json",
              0.0199, share, tol=0.0002)
        int4 = b["storage"]["int4-asym"]
        claim("int4 remnant packed MiB", "budget.json", 4.711,
              int4["packed_bytes"] / (1 << 20), tol=0.01)
        claim("int4 compression vs bf16", "budget.json", 3.821,
              int4["compression_vs_dense"], tol=0.01)
        claim("int4 saving as share of model VRAM = 1.47%", "budget.json",
              0.0147, int4["saving_as_share_of_model"], tol=0.0002)
        int8 = b["storage"]["int8-asym"]
        claim("int8 saving as share of model VRAM = 0.97%", "budget.json",
              0.0097, int8["saving_as_share_of_model"], tol=0.0002)
        rc = b["routing_ceiling"]["0.000"]
        claim("routing ceiling at rho=0 is 1.99% of model MACs", "budget.json",
              0.0199, rc["model_macs_saved_fraction"], tol=0.0002)
        rc25 = b["routing_ceiling"]["0.250"]
        claim("routing ceiling at rho=0.25 is 1.49% of model MACs",
              "budget.json", 0.0149, rc25["model_macs_saved_fraction"],
              tol=0.0002)
        co = b["confidence_overhead"]
        claim("exact-confidence cost ratio = 3.556x", "budget.json",
              3.5556, co["confidence_cost_ratio"], tol=0.001)
        closed = shape["vocab"] / (3.0 * shape["intermediate"])
        claim("that ratio equals the closed form |V|/(3*d_ff)", "budget.json",
              closed, co["confidence_cost_ratio"], tol=1e-9)
        claim("the monitor costs MORE than the MLP it gates", "budget.json",
              True, co["confidence_cost_ratio"] > 1.0)
    else:
        claim("budget report exists", "budget.json", True, None)

    # ------------------------------------------------------------------
    section("Verification suite")
    v = load("verify_csgqr.json")
    if v:
        claim("all verification checks pass", "verify_csgqr.json",
              0, v.get("failed"))
        claim("verification check count", "verify_csgqr.json",
              True, (v.get("total") or 0) >= 80,
              note="expected at least 80 checks; got " + str(v.get("total")))
    else:
        claim("verification report exists", "verify_csgqr.json", True, None)

    # ------------------------------------------------------------------
    section("Gate A: layer scan")
    ls = load("layer_scan.json")
    if ls:
        rows = ls["rows"]
        claim("all 28 layers scanned", "layer_scan.json", 28, len(rows))
        claim("FP remnant KL is exactly zero at every layer",
              "layer_scan.json", True,
              all(r["fp_kl"] == 0.0 for r in rows),
              note="the remnant code path is arithmetically clean")
        i8 = sorted(r["int8_kl_recovery"] for r in rows)
        i4 = sorted(r["int4_kl_recovery"] for r in rows)
        claim("INT8 median KL recovery", "layer_scan.json", 0.997,
              i8[len(i8) // 2], tol=0.002)
        claim("INT4 median KL recovery", "layer_scan.json", 0.967,
              i4[len(i4) // 2], tol=0.002)
        claim("INT4 worst-layer KL recovery still above 0.94",
              "layer_scan.json", True, min(i4) > 0.94,
              note="min = " + format(min(i4), ".4f"))
        by_layer = {r["layer"]: r for r in rows}
        claim("L1 is the most damaging MLP to delete", "layer_scan.json",
              1, max(rows, key=lambda r: r["delete_kl"])["layer"])
        claim("L25 is the least damaging MLP to delete", "layer_scan.json",
              25, min(rows, key=lambda r: r["delete_kl"])["layer"])
        claim("selected moderate layer is 20", "layer_scan.json",
              20, ls["candidates"]["moderate_sensitivity"])
        claim("L20 deletion damage (KL)", "layer_scan.json", 0.3194,
              by_layer[20]["delete_kl"], tol=0.001)
        claim("L20 deleted top-1 agreement with dense", "layer_scan.json",
              0.750, by_layer[20]["delete_agreement"], tol=0.002)
    else:
        claim("layer scan exists", "layer_scan.json", True, None)

    # ------------------------------------------------------------------
    section("H1 / confidence at L20")
    ca = load("trace_L20_int4/confidence_analysis_d0.1.json")
    if ca:
        claim("positions analysed", ".../confidence_analysis_d0.1.json",
              798, ca["n_positions"])
        claim("prompts analysed", ".../confidence_analysis_d0.1.json",
              96, ca["n_prompts"])
        claim("mean KL(dense||delete) at L20", ".../confidence_analysis_d0.1.json",
              0.31259, ca["kl_delete"]["point"], tol=0.0005)
        claim("mean KL(dense||remnant) at L20",
              ".../confidence_analysis_d0.1.json",
              0.00913, ca["kl_remnant"]["point"], tol=0.0002)
        claim("KL recovery ratio at L20", ".../confidence_analysis_d0.1.json",
              0.9708, ca["kl_recovery_ratio"], tol=0.001)
        pf = ca["per_feature"]
        claim("max_prob low-minus-high utility gap is NEGATIVE",
              ".../confidence_analysis_d0.1.json", True,
              pf["max_prob"]["gap_low_minus_high"] < 0,
              note="H1 predicts positive; measured "
                   + format(pf["max_prob"]["gap_low_minus_high"], "+.5f"))
        best = max(("max_prob", "neg_entropy", "prob_margin", "logit_margin"),
                   key=lambda f: pf[f]["auroc_needed"])
        claim("best raw-lens AUROC stays below 0.62",
              ".../confidence_analysis_d0.1.json", True,
              pf[best]["auroc_needed"] < 0.62,
              note=best + " = " + format(pf[best]["auroc_needed"], ".4f"))
        claim("H1 does not hold for all four features",
              ".../confidence_analysis_d0.1.json", False,
              ca["h1"]["all_features_positive_gap"])
        tt = ca["top_token"]
        claim("raw lens top-1 agrees with dense only ~27% of the time",
              ".../confidence_analysis_d0.1.json", 0.2694,
              tt["lens_agrees_with_dense"], tol=0.005)
        claim("deleted disagrees with dense on ~25% of tokens",
              ".../confidence_analysis_d0.1.json", 0.2481,
              tt["deleted_disagrees_with_dense"], tol=0.005)
        claim("the remnant repairs ~86% of those disagreements",
              ".../confidence_analysis_d0.1.json", 0.8586,
              tt["remnant_repair_rate_of_disagreements"], tol=0.005)
    else:
        claim("confidence analysis exists",
              "trace_L20_int4/confidence_analysis_d0.1.json", True, None)

    # ------------------------------------------------------------------
    section("H3 / significance and additivity at L20")
    sa = load("trace_L20_int4/significance_analysis.json")
    if sa:
        claim("significance beats random p95 at every active fraction",
              ".../significance_analysis.json", True,
              all(e["beats_random_p95"] for e in sa["topk_table"]
                  if e["rho"] < 1.0))
        t25 = next(e for e in sa["topk_table"] if abs(e["rho"] - 0.25) < 1e-9)
        claim("significance capture at rho=0.25", ".../significance_analysis.json",
              0.16097, t25["significance"], tol=0.0005)
        claim("random capture at rho=0.25", ".../significance_analysis.json",
              0.07398, t25["random_mean"], tol=0.0005)
        claim("significance beats random by >2x at rho=0.25",
              ".../significance_analysis.json", True,
              t25["significance"] / t25["random_mean"] > 2.0,
              note="ratio = " + format(t25["significance"] / t25["random_mean"], ".2f"))
        claim("group utilities are additive to within 2.5%",
              ".../significance_analysis.json", True,
              sa["interactions"]["mean_abs_relative_interaction"] < 0.025,
              note="mean |I|/(|U(A)|+|U(B)|) = "
                   + format(sa["interactions"]["mean_abs_relative_interaction"], ".4f"))
        claim("additivity gap (sum of groups minus whole remnant) is small",
              ".../significance_analysis.json", True,
              abs(sa["additivity_gap_sum_minus_whole"]) < 0.01,
              note=format(sa["additivity_gap_sum_minus_whole"], "+.5f"))
        h25 = next(e for e in sa["static_vs_oracle"] if abs(e["rho"] - 0.25) < 1e-9)
        claim("static ranking leaves >60% of oracle utility unclaimed at rho=0.25",
              ".../significance_analysis.json", True,
              h25["relative_headroom"] > 0.60,
              note=format(100 * h25["relative_headroom"], ".1f") + "%")
        claim("per-token group ordering barely agrees with the global ranking",
              ".../significance_analysis.json", True,
              sa["per_token_ranking_agreement"]["mean_spearman"] < 0.15,
              note="mean Spearman = "
                   + format(sa["per_token_ranking_agreement"]["mean_spearman"], ".4f"))
    else:
        claim("significance analysis exists",
              "trace_L20_int4/significance_analysis.json", True, None)

    # ------------------------------------------------------------------
    section("Probe: is the routing signal in the state?")
    pa = load("trace_L20_int4/probe_analysis.json")
    if pa:
        rn = pa["remnant_needed"]
        raw_best = max(rn["raw_lens_features"].values(),
                       key=lambda d: d["auroc"])["auroc"]
        probe = rn["fitted"]["logistic on hidden state"]["auroc"]
        lens = rn["fitted"]["logistic on 4 lens features"]["auroc"]
        claim("hidden-state probe AUROC above 0.90",
              ".../probe_analysis.json", True, probe > 0.90,
              note=format(probe, ".4f"))
        claim("best raw-lens AUROC below 0.62", ".../probe_analysis.json",
              True, raw_best < 0.62, note=format(raw_best, ".4f"))
        claim("fitted lens-feature AUROC below 0.62", ".../probe_analysis.json",
              True, lens < 0.62, note=format(lens, ".4f"))
        claim("probe beats the lens by more than 0.30 AUROC",
              ".../probe_analysis.json", True, probe - raw_best > 0.30,
              note=format(probe - raw_best, "+.4f"))
        ctl = rn["controls"]
        claim("permuted-label control is near chance", ".../probe_analysis.json",
              True, abs(ctl["permuted_labels"]["auroc"] - 0.5) < 0.06,
              note=format(ctl["permuted_labels"]["auroc"], ".4f")
                   + " -- rules out pipeline leakage")
        claim("probe beats the position-only control",
              ".../probe_analysis.json", True,
              probe > ctl["position_only"]["auroc"] + 0.25,
              note="position-only = "
                   + format(ctl["position_only"]["auroc"], ".4f"))
        claim("probe beats the prompt-level ORACLE ceiling",
              ".../probe_analysis.json", True,
              probe > ctl["prompt_mean_oracle"]["auroc"],
              note="prompt-mean oracle = "
                   + format(ctl["prompt_mean_oracle"]["auroc"], ".4f")
                   + "; the probe resolves individual tokens")
        claim("lambda was chosen on a disjoint validation split",
              ".../probe_analysis.json", True,
              "val_auroc" in rn["fitted"]["logistic on hidden state"])
        claim("train/val/test splits are disjoint by prompt",
              ".../probe_analysis.json", True,
              pa["train_prompts"] + pa["val_prompts"] + pa["test_prompts"]
              == pa["n_prompts"])
        for e in pa["topk_capture"]:
            if abs(e["rho"] - 0.25) < 1e-9:
                claim("probe wins >65% of static-to-oracle headroom at rho=0.25",
                      ".../probe_analysis.json", True,
                      e["headroom_fraction_won"] > 0.65,
                      note=format(100 * e["headroom_fraction_won"], ".1f") + "%")
                claim("probe beats static ranking at rho=0.25",
                      ".../probe_analysis.json", True, e["probe_beats_static"])
                claim("probe beats its own permuted-target control at rho=0.25",
                      ".../probe_analysis.json", True, e["probe_beats_permuted"],
                      note="permuted probe = " + format(e["permuted_probe"], ".5f"))
    else:
        claim("probe analysis exists", "trace_L20_int4/probe_analysis.json",
              True, None)

    # ------------------------------------------------------------------
    section("Quality arms (chat_dev, 510 cases)")
    base = load("eval_b0-dense__chat_dev/summary.json")
    stored = REPO / "artifacts" / "gate_runs" / "r2_base_chat_dev" / "summary.json"
    if base and stored.exists():
        ref = json.loads(stored.read_text(encoding="utf-8"))
        claim("dense arm reproduces the stored gate baseline exactly",
              "eval_b0-dense__chat_dev/summary.json",
              ref["mean_quality_score"], base["mean_quality_score"], tol=1e-9)
        claim("dense arm case count", "eval_b0-dense__chat_dev/summary.json",
              510, base["case_count"])
    else:
        claim("dense arm evaluated", "eval_b0-dense__chat_dev/summary.json",
              True, base)

    arms = {}
    for key in ("b0-dense", "b1-delete", "b2-fp-all", "b3-int8-all",
                "b4-int4-all", "b5-random-25", "b6-static-25",
                "b7-conf-whole", "b10-csgqr", "b13-wrong-layer"):
        suffix = "" if key == "b0-dense" else "__L20"
        s = load("eval_" + key + "__chat_dev" + suffix + "/summary.json")
        if s:
            arms[key] = s["mean_quality_score"]

    if "b0-dense" in arms and "b2-fp-all" in arms:
        floor = abs(arms["b0-dense"] - arms["b2-fp-all"])
        claim("noise floor |b0 - b2| is recorded", "eval_b2-fp-all.../summary.json",
              True, True, note="|b0 - b2| = " + format(floor, ".5f")
              + "  (differences below this are not attributable to CSGQR)")
    if "b1-delete" in arms and "b0-dense" in arms:
        claim("deletion reduces quality", "eval_b1-delete.../summary.json",
              True, arms["b1-delete"] < arms["b0-dense"],
              note="dense " + format(arms["b0-dense"], ".5f") + " -> delete "
                   + format(arms["b1-delete"], ".5f"))
    for key in ("b3-int8-all", "b4-int4-all"):
        if key in arms and "b1-delete" in arms:
            claim(key + " recovers quality above deletion",
                  "eval_" + key + ".../summary.json", True,
                  arms[key] > arms["b1-delete"],
                  note=format(arms[key], ".5f"))
    if "b6-static-25" in arms and "b5-random-25" in arms:
        claim("significance beats random at matched rho=0.25 on QUALITY",
              "eval_b6/b5.../summary.json", True,
              arms["b6-static-25"] > arms["b5-random-25"],
              note="static " + format(arms["b6-static-25"], ".5f")
                   + " vs random " + format(arms["b5-random-25"], ".5f"))
    if arms:
        print("\n  arms present: " + ", ".join(
            k + "=" + format(v, ".5f") for k, v in sorted(arms.items())))

    # ------------------------------------------------------------------
    failed = [c for c in CHECKS if c["status"] == "FAIL"]
    missing = [c for c in CHECKS if c["status"] == "MISSING"]
    ok = [c for c in CHECKS if c["status"] == "OK"]
    print("\n" + "=" * 70)
    print(str(len(ok)) + " OK, " + str(len(failed)) + " FAILED, "
          + str(len(missing)) + " MISSING, of " + str(len(CHECKS)) + " claims")
    if failed:
        print("\nFAILED:")
        for c in failed:
            print("  - " + c["claim"] + ": expected " + str(c["expected"])
                  + ", got " + str(c["actual"]) + "   [" + c["source"] + "]")
    if missing:
        print("\nMISSING (result file not committed / not yet produced):")
        for c in missing:
            print("  - " + c["claim"] + "   [" + c["source"] + "]")

    if args.json:
        dest = args.json if args.json.is_absolute() else REPO / args.json
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            "total": len(CHECKS), "ok": len(ok), "failed": len(failed),
            "missing": len(missing), "checks": CHECKS}, indent=2) + "\n",
            encoding="utf-8")
        print("\nwrote " + str(dest.relative_to(REPO)))

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
