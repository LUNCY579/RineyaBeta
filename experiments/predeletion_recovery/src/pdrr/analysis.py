"""Aggregate CPU or real-model PDRR result records."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .stats import ContrastResult, finalize_family, paired_contrast


PRIMARY_COMPARATORS = (
    "hard_prune_kd",
    "gradual_kd",
    "matrix_no_recovery",
    "random_matrix_recovery",
    "state_position_no_causal",
)


def aggregate_arm_results(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["arm"])].append(record)
    out: dict[str, dict[str, Any]] = {}
    for arm, rows in sorted(grouped.items()):
        metrics = [row["metrics"] for row in rows]
        def mean(path: str) -> float:
            values = [float(item[path]) for item in metrics]
            return float(np.mean(values))
        payload: dict[str, Any] = {
            "n": len(rows),
            "mean_accuracy": mean("accuracy"),
            "mean_kl": mean("mean_kl"),
            "mean_quality": mean("quality"),
        }
        for key in ("recovery_ratio", "handoff_completeness", "route_alignment_spearman"):
            values = [float(row[key]) for row in rows if row.get(key) is not None
                      and math.isfinite(float(row[key]))]
            if values:
                payload[f"mean_{key}"] = float(np.mean(values))
        matches = [bool(row["top_route_top_gain_match"]) for row in rows
                   if row.get("top_route_top_gain_match") is not None]
        if matches:
            payload["top_route_top_gain_match_rate"] = float(np.mean(matches))
        out[arm] = payload
    return out


def paired_quality_family(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str = "causal_state_position_recovery",
    practical_threshold: float = 0.005,
    bootstrap_samples: int = 5000,
    permutation_samples: int = 10000,
    seed: int = 0,
) -> list[ContrastResult]:
    index = {(str(row["arm"]), int(row["seed"])): row for row in records}
    results = []
    for comparator in PRIMARY_COMPARATORS:
        differences: dict[str, list[float]] = {}
        seeds = sorted({seed_value for arm, seed_value in index if arm == method}
                       & {seed_value for arm, seed_value in index if arm == comparator})
        for seed_value in seeds:
            method_q = float(index[(method, seed_value)]["metrics"]["quality"])
            comparator_q = float(index[(comparator, seed_value)]["metrics"]["quality"])
            differences[str(seed_value)] = [method_q - comparator_q]
        if differences:
            results.append(paired_contrast(
                f"{method}__vs__{comparator}", differences,
                practical_threshold=practical_threshold,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=seed + len(results) * 101))
    return finalize_family(results)


def claim_support(contrasts: Sequence[ContrastResult], aggregate: Mapping[str, Any]) -> dict[str, bool]:
    passed = {result.name.split("__vs__")[-1]:
              result.statistical_pass and result.practical_pass for result in contrasts}
    full = aggregate.get("causal_state_position_recovery", {})
    mechanism = (
        float(full.get("mean_route_alignment_spearman", 0.0)) > 0.0
        and float(full.get("top_route_top_gain_match_rate", 0.0)) >= 0.5
    )
    return {
        "recovery_effect": bool(passed.get("hard_prune_kd", False)),
        "predeletion_advantage": bool(passed.get("hard_prune_kd", False)
                                      and passed.get("gradual_kd", False)),
        "component_specific_transfer": bool(passed.get("matrix_no_recovery", False)
                                             and passed.get("random_matrix_recovery", False)),
        "causal_routing": bool(passed.get("state_position_no_causal", False)),
        "mechanistic_migration": bool(mechanism),
    }
