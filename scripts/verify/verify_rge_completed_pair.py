#!/usr/bin/env python3
"""Re-audit completed RGE pair artifacts without launching model processes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiment"))
from rge_optimizer_pair import (  # noqa: E402
    atomic_json,
    compare_completed_runs,
    summarize_run,
    validate_pair_manifest,
)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--pair-summary", default=None)
    parser.add_argument("--external-timing-contention", action="store_true")
    args = parser.parse_args()
    pair_path = repo_path(args.pair)
    pair, _, problems = validate_pair_manifest(pair_path)
    if problems:
        raise SystemExit(f"pair manifest failed preflight: {problems}")
    control = summarize_run(repo_path(args.control_run))
    candidate = summarize_run(repo_path(args.candidate_run))
    comparison = compare_completed_runs(control, candidate, require_budget=True)
    if comparison["status"] != "PASS":
        raise SystemExit(json.dumps(comparison["mismatches"], indent=2))
    differences = [float(row["candidate_minus_control"])
                   for row in comparison["pointwise_loss_curve"]]
    wins = sum(value < 0.0 for value in differences)
    ties = sum(value == 0.0 for value in differences)
    losses = sum(value > 0.0 for value in differences)
    compact = {
        "schema_version": 1,
        "status": "PASS",
        "experiment_id": pair["experiment_id"],
        "pair_manifest": pair_path.relative_to(ROOT).as_posix(),
        "requested_supervised_tokens": pair["fixed_budget"][
            "requested_supervised_tokens"],
        "control": {
            "run_id": control["run_id"],
            "source_commit": control["comparability"]["git_commit"],
            "actual_supervised_tokens": control["actual_supervised_tokens"],
            "steps": control["steps"],
            "final_loss": control["final_loss"],
            "token_weighted_mean_loss": control["token_weighted_mean_loss"],
            "active_gpu_hours": control["active_gpu_hours"],
            "peak_reserved_mib": control["peak_reserved_mib"],
            "optimizer_state_tensor_bytes": control["optimizer_state_tensor_bytes"],
        },
        "candidate": {
            "run_id": candidate["run_id"],
            "source_commit": candidate["comparability"]["git_commit"],
            "actual_supervised_tokens": candidate["actual_supervised_tokens"],
            "steps": candidate["steps"],
            "final_loss": candidate["final_loss"],
            "token_weighted_mean_loss": candidate["token_weighted_mean_loss"],
            "active_gpu_hours": candidate["active_gpu_hours"],
            "peak_reserved_mib": candidate["peak_reserved_mib"],
            "optimizer_state_tensor_bytes": candidate["optimizer_state_tensor_bytes"],
        },
        "comparison": {
            "mismatches": {},
            **comparison["screening_metrics"],
            "paired_curve_points": len(differences),
            "candidate_lower_equal_higher_loss": [wins, ties, losses],
            "mean_pointwise_loss_delta": statistics.fmean(differences),
            "median_pointwise_loss_delta": statistics.median(differences),
        },
        "timing_interpretation": (
            "INVALID_EXTERNAL_GPU_CONTENTION" if args.external_timing_contention
            else "EXPLORATORY_SINGLE_ORDER_ONLY"),
        "maximum_claim": pair["claim_boundary"]["maximum_claim"],
        "sealed_data_accessed": False,
    }
    evidence_path = repo_path(args.evidence)
    atomic_json(evidence_path, compact)
    if args.pair_summary:
        atomic_json(repo_path(args.pair_summary), {
            "schema_version": 1,
            "status": "COMPLETE_REAUDITED",
            "arms": {"control": control, "candidate": candidate},
            "comparison": comparison,
            "timing_interpretation": compact["timing_interpretation"],
            "sealed_data_accessed": False,
        })
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
