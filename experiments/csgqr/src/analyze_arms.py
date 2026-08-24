#!/usr/bin/env python3
"""Assemble the arm comparison table with paired statistics.

Every arm saw the same 510 prompts in the same order under the same greedy
protocol, so all comparisons here are **paired** at the case level. That is
strictly tighter than treating arms as independent samples, and it is the design
the evaluation already provides for free.

Three things this is careful about.

**Recovery Ratio is unstable when the denominator is small.** RR is
``(Q_arm - Q_delete) / (Q_ref - Q_delete)``. When ``Q_ref - Q_delete`` is a
fraction of a quality point, RR amplifies noise without bound, so the raw
denominator is printed beside every RR and a flag is raised when it is below a
threshold. Absolute quality is always reported too.

**Two reference points, deliberately.** RR is computed against ``b0-dense`` (the
true ceiling) *and* ``b2-fp-all`` (the same weights through the remnant code
path). The second is the achievable ceiling for any quantized arm, because
cuBLAS picks different bf16 GEMM tiles for some shapes and the remnant path is
therefore not bit-identical to ``nn.Linear``. ``|b0 - b2|`` is the noise floor.

**Multiple comparisons.** Holm-Bonferroni across the planned contrasts. The
contrast list is fixed in ``CONTRASTS`` rather than chosen after looking at the
numbers.
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

from stats import (cluster_bootstrap_pair, holm_bonferroni,        # noqa: E402
                   mcnemar_exact, wilcoxon_signed_rank)

RESULTS = REPO / "experiments" / "csgqr" / "results"

# Fixed before the numbers were seen. Each pair is (arm, baseline, why).
CONTRASTS = [
    ("b1-delete", "b0-dense", "deletion damage"),
    ("b2-fp-all", "b0-dense", "NOISE FLOOR: identical weights, remnant path"),
    ("b3-int8-all", "b1-delete", "does INT8 recover deletion damage"),
    ("b4-int4-all", "b1-delete", "does INT4 recover deletion damage"),
    ("b4-int4-all", "b2-fp-all", "quantization cost vs the achievable ceiling"),
    ("b6-static-25", "b5-random-25", "H3: significance vs random at matched rho"),
    ("b7-conf-whole", "b4-int4-all", "H2: does confidence gating cost quality"),
    ("b10-csgqr", "b4-int4-all", "does full CSGQR hold the always-on ceiling"),
    ("b10-csgqr", "b5-random-25", "CSGQR vs random"),
    ("b13-wrong-layer", "b4-int4-all", "H4: does donor identity matter"),
]


def arm_dir(key: str, dataset: str, layer: int) -> Path:
    suffix = "" if key == "b0-dense" else "__L" + str(layer)
    return RESULTS / ("eval_" + key + "__" + dataset + suffix)


def load_arm(key: str, dataset: str, layer: int) -> dict | None:
    d = arm_dir(key, dataset, layer)
    s = d / "summary.json"
    if not s.exists():
        return None
    out = {"key": key,
           "summary": json.loads(s.read_text(encoding="utf-8"))}
    m = d / "run_manifest.json"
    if m.exists():
        out["manifest"] = json.loads(m.read_text(encoding="utf-8"))
    r = d / "results.jsonl"
    if r.exists():
        rows = [json.loads(l) for l in r.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        out["cases"] = {str(x["id"]): x for x in rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chat_dev")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--rr-denominator-floor", type=float, default=0.005)
    ap.add_argument("--out", default="experiments/csgqr/results/arms_comparison.json")
    args = ap.parse_args()

    order = ["b0-dense", "b1-delete", "b2-fp-all", "b3-int8-all", "b4-int4-all",
             "b5-random-25", "b6-static-25", "b7-conf-whole", "b10-csgqr",
             "b13-wrong-layer"]
    arms = {}
    for k in order:
        a = load_arm(k, args.dataset, args.layer)
        if a:
            arms[k] = a
    if "b0-dense" not in arms:
        raise SystemExit("b0-dense is required as the reference; run it first")
    missing = [k for k in order if k not in arms]
    if missing:
        print("NOT YET RUN: " + ", ".join(missing) + "\n")

    dense_q = arms["b0-dense"]["summary"]["mean_quality_score"]
    delete_q = (arms["b1-delete"]["summary"]["mean_quality_score"]
                if "b1-delete" in arms else None)
    fp_q = (arms["b2-fp-all"]["summary"]["mean_quality_score"]
            if "b2-fp-all" in arms else None)

    noise_floor = abs(dense_q - fp_q) if fp_q is not None else None

    # -- the main table ----------------------------------------------------
    rows = []
    for k in order:
        if k not in arms:
            continue
        s = arms[k]["summary"]
        man = arms[k].get("manifest") or {}
        inst = man.get("install") or {}
        routing = man.get("routing") or {}
        q = s["mean_quality_score"]

        def rr(ref: float | None) -> float | None:
            if ref is None or delete_q is None:
                return None
            den = ref - delete_q
            if abs(den) < 1e-12:
                return None
            return (q - delete_q) / den

        packed = (inst.get("packed_bytes") or {}).get("total")
        rows.append({
            "arm": k,
            "label": man.get("label", ""),
            "mean_quality_score": q,
            "quality_pass_rate": s.get("quality_pass_rate"),
            "intelligibility_pass_rate": s.get("intelligibility_pass_rate"),
            "critical_defect_case_rate": s.get("critical_defect_case_rate"),
            "recovery_ratio_vs_dense": rr(dense_q),
            "recovery_ratio_vs_fp_remnant": rr(fp_q),
            "active_fraction": routing.get("active_fraction"),
            "invocation_rate": routing.get("invocation_rate"),
            "skip_rate": routing.get("skip_rate"),
            "packed_bytes": packed,
            "compression_vs_dense": inst.get("compression_vs_dense"),
            "peak_reserved_mib": man.get("peak_reserved_mib"),
            "peak_allocated_mib": man.get("peak_allocated_mib"),
            "seconds_per_case": man.get("seconds_per_case"),
            "spec": inst.get("spec"),
            "donor_layer": inst.get("donor_layer"),
            "case_count": s.get("case_count"),
        })

    print("Arm table  (" + args.dataset + ", layer " + str(args.layer) + ")")
    hdr = ("  {:<16s} {:>9s} {:>9s} {:>9s} {:>8s} {:>11s} {:>10s} {:>8s}")
    print(hdr.format("arm", "quality", "RR/dense", "RR/fp", "rho", "packed KiB",
                     "peak MiB", "s/case"))
    for r in rows:
        print("  {:<16s} {:>9.5f} {:>9s} {:>9s} {:>8s} {:>11s} {:>10s} {:>8s}".format(
            r["arm"], r["mean_quality_score"],
            format(r["recovery_ratio_vs_dense"], ".4f")
            if r["recovery_ratio_vs_dense"] is not None else "-",
            format(r["recovery_ratio_vs_fp_remnant"], ".4f")
            if r["recovery_ratio_vs_fp_remnant"] is not None else "-",
            format(r["active_fraction"], ".3f")
            if r["active_fraction"] is not None else "-",
            format(r["packed_bytes"] / 1024, ",.0f")
            if r["packed_bytes"] else "-",
            format(r["peak_reserved_mib"], ",.0f")
            if r["peak_reserved_mib"] else "-",
            format(r["seconds_per_case"], ".3f")
            if r["seconds_per_case"] else "-"))

    if noise_floor is not None:
        print("\n  NOISE FLOOR |b0 - b2| = " + format(noise_floor, ".5f")
              + "   -- differences smaller than this are cuBLAS tile selection, "
              "not CSGQR")
    if delete_q is not None:
        den_dense = dense_q - delete_q
        print("  RR denominator (dense - delete) = " + format(den_dense, ".5f")
              + ("   *** SMALL: RR is unstable, read absolute quality ***"
                 if abs(den_dense) < args.rr_denominator_floor else ""))

    # -- paired contrasts --------------------------------------------------
    print("\nPaired contrasts (same 510 cases, greedy, case-level pairing)")
    contrasts = []
    pvals: dict[str, float] = {}
    for a, b, why in CONTRASTS:
        if a not in arms or b not in arms:
            continue
        ca, cb = arms[a].get("cases"), arms[b].get("cases")
        if not ca or not cb:
            continue
        ids = [i for i in ca if i in cb]
        qa = torch.tensor([float(ca[i].get("quality_score") or 0.0) for i in ids],
                          dtype=torch.float64)
        qb = torch.tensor([float(cb[i].get("quality_score") or 0.0) for i in ids],
                          dtype=torch.float64)
        # One case is one cluster here: the quality score is already a
        # per-prompt aggregate, so no within-prompt token correlation remains.
        clusters = list(range(len(ids)))
        ci = cluster_bootstrap_pair(qa, qb, clusters, n_boot=args.n_boot)
        wil = wilcoxon_signed_rank(qa - qb)

        # McNemar on the binary pass outcome.
        pa = [bool(ca[i].get("quality_pass")) for i in ids]
        pb = [bool(cb[i].get("quality_pass")) for i in ids]
        both = sum(1 for x, y in zip(pa, pb) if x and y)
        only_a = sum(1 for x, y in zip(pa, pb) if x and not y)
        only_b = sum(1 for x, y in zip(pa, pb) if y and not x)
        neither = sum(1 for x, y in zip(pa, pb) if not x and not y)
        mc = mcnemar_exact(both, only_a, only_b, neither)

        identical = sum(1 for i in ids if ca[i].get("reply") == cb[i].get("reply"))
        entry = {"arm": a, "baseline": b, "why": why, "n": len(ids),
                 "mean_delta": ci["point"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                 "excludes_zero": ci["excludes_zero"],
                 "cohens_d": ci["cohens_d"],
                 "wilcoxon_p": wil.get("p_value"),
                 "mcnemar": mc,
                 "replies_identical": identical,
                 "replies_identical_frac": identical / max(len(ids), 1)}
        contrasts.append(entry)
        pvals[a + " vs " + b] = wil.get("p_value")

        print("  {:<16s} vs {:<14s} delta {:>+9.5f}  [{:>+8.5f}, {:>+8.5f}]"
              "  {:>5s}  identical {:>5.1f}%".format(
                  a, b, ci["point"], ci["lo"], ci["hi"],
                  "SIG" if ci["excludes_zero"] else "ns",
                  100 * entry["replies_identical_frac"]))
        print("  {:<34s} {:s}".format("", why))

    corrected = holm_bonferroni(pvals) if pvals else None
    if corrected:
        print("\nHolm-Bonferroni across " + str(corrected["n_tests"])
              + " planned contrasts (alpha 0.05):")
        for k, v in sorted(corrected["results"].items(),
                           key=lambda kv: (kv[1]["adjusted_p"] != kv[1]["adjusted_p"],
                                           kv[1]["adjusted_p"])):
            if v["p_value"] is None or v["p_value"] != v["p_value"]:
                continue
            print("  {:<36s} p={:.3e}  adj={:.3e}  {}".format(
                k, v["p_value"], v["adjusted_p"],
                "reject H0" if v["reject"] else "retain H0"))

    out = {
        "dataset": args.dataset, "layer": args.layer,
        "n_boot": args.n_boot,
        "reference_dense": dense_q,
        "reference_delete": delete_q,
        "reference_fp_remnant": fp_q,
        "noise_floor_dense_vs_fp": noise_floor,
        "rr_denominator_dense_minus_delete": (dense_q - delete_q
                                              if delete_q is not None else None),
        "rr_denominator_floor": args.rr_denominator_floor,
        "arms_present": list(arms),
        "arms_missing": missing,
        "table": rows,
        "contrasts": contrasts,
        "holm_bonferroni": corrected,
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
