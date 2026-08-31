#!/usr/bin/env python3
"""Paired retention comparison for frozen RGE development gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_rows(path: Path) -> dict[str, dict]:
    return {str(row["id"]): row for row in (
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip())}


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, samples: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def compare_dataset(name: str, control_dir: Path, candidate_dir: Path,
                    *, seed: int) -> dict:
    control = load_rows(control_dir / "results.jsonl")
    candidate = load_rows(candidate_dir / "results.jsonl")
    if set(control) != set(candidate):
        raise SystemExit(f"{name} case IDs differ")
    ids = sorted(control)
    differences = np.asarray([
        float(candidate[case]["quality_score"]) - float(control[case]["quality_score"])
        for case in ids], dtype=np.float64)
    by_category: dict[str, list[float]] = {}
    for case, difference in zip(ids, differences.tolist()):
        category = str(control[case]["category"])
        by_category.setdefault(category, []).append(difference)
    category_deltas = {category: statistics.fmean(values)
                       for category, values in sorted(by_category.items())}
    control_summary = json.loads((control_dir / "summary.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    wins = int(np.sum(differences > 0.0))
    ties = int(np.sum(differences == 0.0))
    losses = int(np.sum(differences < 0.0))
    mean_delta = float(differences.mean())
    critical_delta = (float(candidate_summary["critical_defect_case_rate"])
                      - float(control_summary["critical_defect_case_rate"]))
    contract_delta = (float(candidate_summary["contract_violation_case_rate"])
                      - float(control_summary["contract_violation_case_rate"]))
    return {
        "dataset": name,
        "cases": len(ids),
        "control_mean_quality": float(control_summary["mean_quality_score"]),
        "candidate_mean_quality": float(candidate_summary["mean_quality_score"]),
        "mean_quality_delta": mean_delta,
        "median_quality_delta": float(np.median(differences)),
        "paired_bootstrap_95_ci": bootstrap_mean_ci(differences, seed=seed),
        "candidate_win_tie_loss": [wins, ties, losses],
        "minimum_case_delta": float(differences.min()),
        "p10_case_delta": float(np.quantile(differences, 0.10)),
        "critical_defect_rate_control": float(control_summary[
            "critical_defect_case_rate"]),
        "critical_defect_rate_candidate": float(candidate_summary[
            "critical_defect_case_rate"]),
        "critical_defect_rate_delta": critical_delta,
        "contract_violation_rate_control": float(control_summary[
            "contract_violation_case_rate"]),
        "contract_violation_rate_candidate": float(candidate_summary[
            "contract_violation_case_rate"]),
        "contract_violation_rate_delta": contract_delta,
        "intelligibility_pass_rate_control": float(control_summary[
            "intelligibility_pass_rate"]),
        "intelligibility_pass_rate_candidate": float(candidate_summary[
            "intelligibility_pass_rate"]),
        "mean_noninferiority_pass": mean_delta >= -0.01,
        "critical_regression_present": critical_delta > 0.0,
        "worst_category_deltas": sorted(category_deltas.items(),
                                        key=lambda item: item[1])[:8],
        "best_category_deltas": sorted(category_deltas.items(),
                                       key=lambda item: item[1], reverse=True)[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-chat", required=True)
    parser.add_argument("--candidate-chat", required=True)
    parser.add_argument("--control-rag", required=True)
    parser.add_argument("--candidate-rag", required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    datasets = {
        "chat_dev": compare_dataset(
            "chat_dev", ROOT / args.control_chat, ROOT / args.candidate_chat,
            seed=20260831),
        "rag_dev": compare_dataset(
            "rag_dev", ROOT / args.control_rag, ROOT / args.candidate_rag,
            seed=20260832),
    }
    passes = all(payload["mean_noninferiority_pass"]
                 and not payload["critical_regression_present"]
                 for payload in datasets.values())
    result = {
        "schema_version": 1,
        "status": "PASS" if passes else "REJECT_MUON_RETENTION",
        "noninferiority_margin": -0.01,
        "datasets": datasets,
        "decision": (f"retain {args.candidate_label} for the next evidence tier"
                     if passes else
                     f"reject {args.candidate_label}; do not tune on retention"),
        "sealed_data_accessed": False,
    }
    evidence = ROOT / args.evidence
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
