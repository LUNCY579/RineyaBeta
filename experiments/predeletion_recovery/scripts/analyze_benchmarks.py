#!/usr/bin/env python3
"""Paired confirmatory analysis for the real PDRR benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))

from pdrr.analysis import PRIMARY_COMPARATORS  # noqa: E402
from pdrr.metrics import recovery_ratio  # noqa: E402
from pdrr.provenance import atomic_write_json, atomic_write_text  # noqa: E402
from pdrr.stats import finalize_family, paired_contrast  # noqa: E402


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="causal_state_position_recovery")
    parser.add_argument("--practical-delta", type=float, default=0.005)
    parser.add_argument("--rr-advantage", type=float, default=0.10)
    parser.add_argument("--category-tolerance", type=float, default=0.02)
    parser.add_argument("--mechanistic", action="append", default=[])
    args = parser.parse_args()

    index = json.loads(resolve(args.index).read_text(encoding="utf-8"))
    rows = []
    missing = []
    for cell in index["cells"]:
        path = Path(cell["benchmark_rows"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            missing.append(str(path))
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row.update({"target_layer": int(cell["target_layer"]),
                            "seed": int(cell["seed"]), "arm": str(cell["arm"])})
                rows.append(row)
    if missing:
        raise FileNotFoundError("missing benchmark rows:\n  " + "\n  ".join(missing[:20]))

    keyed = {}
    for row in rows:
        key = (row["target_layer"], row["seed"], row["dataset"], str(row["id"]), row["arm"])
        if key in keyed:
            raise ValueError(f"duplicate benchmark row {key}")
        keyed[key] = row

    cell_scores: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        cell_scores[(row["target_layer"], row["seed"], row["arm"])].append(float(row["quality_score"]))

    rr_rows = []
    for (target, seed, arm), values in sorted(cell_scores.items()):
        if arm in {"full", "hard_prune"}:
            continue
        full = np.mean(cell_scores[(target, seed, "full")])
        hard = np.mean(cell_scores[(target, seed, "hard_prune")])
        method = np.mean(values)
        rr_rows.append({
            "target_layer": target, "seed": seed, "arm": arm,
            "quality": float(method),
            "recovery_ratio": recovery_ratio(float(full), float(hard), float(method)),
        })

    raw_contrasts = []
    diagnostics = {}
    for comparator in PRIMARY_COMPARATORS:
        clusters: dict[str, list[float]] = defaultdict(list)
        paired_method = []
        paired_comparator = []
        categories: dict[str, list[float]] = defaultdict(list)
        for (target, seed, dataset, example_id, arm), method_row in keyed.items():
            if arm != args.method:
                continue
            other_key = (target, seed, dataset, example_id, comparator)
            if other_key not in keyed:
                continue
            difference = float(method_row["quality_score"]) - float(keyed[other_key]["quality_score"])
            clusters[f"target={target}|seed={seed}"].append(difference)
            paired_method.append(float(method_row["quality_score"]))
            paired_comparator.append(float(keyed[other_key]["quality_score"]))
            categories[str(method_row["category"])].append(difference)
        if not clusters:
            continue
        result = paired_contrast(
            f"{args.method}__vs__{comparator}", clusters,
            practical_threshold=args.practical_delta,
            bootstrap_samples=20_000, permutation_samples=100_000,
            seed=20260823 + len(raw_contrasts) * 101)
        raw_contrasts.append(result)
        method_rr = [row["recovery_ratio"] for row in rr_rows if row["arm"] == args.method]
        comparator_rr = [row["recovery_ratio"] for row in rr_rows if row["arm"] == comparator]
        rr_advantage = float(np.mean(method_rr) - np.mean(comparator_rr))
        category_means = {name: float(np.mean(values)) for name, values in categories.items()}
        max_regression = max([0.0] + [-value for value in category_means.values() if value < 0])
        diagnostics[comparator] = {
            "paired_examples": len(paired_method),
            "clusters": len(clusters),
            "mean_method_quality": float(np.mean(paired_method)),
            "mean_comparator_quality": float(np.mean(paired_comparator)),
            "rr_advantage": rr_advantage,
            "category_mean_differences": category_means,
            "max_category_regression": max_regression,
        }
    contrasts = finalize_family(raw_contrasts)

    claim_gates = {}
    for result in contrasts:
        comparator = result.name.split("__vs__")[-1]
        diag = diagnostics[comparator]
        claim_gates[comparator] = {
            "statistical": result.statistical_pass,
            "practical": result.practical_pass,
            "recovery_ratio": diag["rr_advantage"] >= args.rr_advantage,
            "category_preservation": diag["max_category_regression"] <= args.category_tolerance,
        }
        claim_gates[comparator]["all"] = all(claim_gates[comparator].values())

    mechanism_records = [json.loads(resolve(path).read_text(encoding="utf-8"))
                         for path in args.mechanistic]
    mechanism_pass = bool(mechanism_records) and all(
        bool(record.get("mechanistic_gate_pass")) for record in mechanism_records)
    supported = {
        "recovery_effect": claim_gates.get("hard_prune_kd", {}).get("all", False),
        "predeletion_advantage": (
            claim_gates.get("hard_prune_kd", {}).get("all", False)
            and claim_gates.get("gradual_kd", {}).get("all", False)),
        "component_specific_transfer": (
            claim_gates.get("matrix_no_recovery", {}).get("all", False)
            and claim_gates.get("random_matrix_recovery", {}).get("all", False)),
        "causal_routing": claim_gates.get("state_position_no_causal", {}).get("all", False),
        "mechanistic_migration": mechanism_pass,
    }

    analysis = {
        "schema_version": 1,
        "method": args.method,
        "rows": len(rows),
        "cells": len(cell_scores),
        "recovery_ratio_cells": rr_rows,
        "contrasts": [result.__dict__ for result in contrasts],
        "diagnostics": diagnostics,
        "claim_gates_by_comparator": claim_gates,
        "supported_claims": supported,
        "mechanistic_records": mechanism_records,
        "thresholds": {
            "practical_quality_delta": args.practical_delta,
            "recovery_ratio_advantage": args.rr_advantage,
            "category_regression_tolerance": args.category_tolerance,
            "familywise_alpha": 0.05,
        },
    }
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "analysis.json", analysis)
    atomic_write_text(out / "report.md", render_report(analysis))
    print(f"wrote {out / 'analysis.json'}")


def render_report(analysis: dict) -> str:
    lines = [
        "# PDRR confirmatory benchmark analysis",
        "",
        f"Paired benchmark rows: **{analysis['rows']:,}**",
        "",
        "| Comparator | Δ quality | 95% CI | Holm p | RR advantage | Max category regression | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    by_name = {result["name"].split("__vs__")[-1]: result for result in analysis["contrasts"]}
    for comparator, diag in analysis["diagnostics"].items():
        result = by_name[comparator]
        gate = analysis["claim_gates_by_comparator"][comparator]["all"]
        lines.append(
            f"| `{comparator}` | {result['mean_difference']:+.5f} | "
            f"[{result['confidence_low']:+.5f}, {result['confidence_high']:+.5f}] | "
            f"{result['adjusted_p']:.4g} | {diag['rr_advantage']:+.3f} | "
            f"{diag['max_category_regression']:.3f} | {'PASS' if gate else 'FAIL'} |"
        )
    lines += ["", "## Supported claim levels", ""]
    for claim, passed in analysis["supported_claims"].items():
        lines.append(f"- {'SUPPORTED' if passed else 'NOT SUPPORTED'} — `{claim}`")
    lines += [
        "",
        "A behavioral advantage without a passing mechanistic probe may be described only as quality preservation. It must not be described as migration of causal job responsibility.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
