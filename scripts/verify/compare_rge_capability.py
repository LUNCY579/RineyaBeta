#!/usr/bin/env python3
"""Paired audit for two frozen RGE capability-validation runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_rows(directory: Path) -> tuple[dict[str, dict], dict]:
    manifest = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or not manifest.get("full_frozen_bank"):
        raise ValueError(f"not a complete full-bank evaluation: {directory}")
    if manifest.get("sealed_data_accessed") or manifest.get("test_and_challenge_accessed"):
        raise ValueError(f"forbidden asset access recorded: {directory}")
    rows: dict[str, dict] = {}
    for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = str(row["id"])
        if rid in rows:
            raise ValueError(f"duplicate result ID {rid}: {directory}")
        rows[rid] = row
    if len(rows) != int(manifest["case_count"]):
        raise ValueError(f"coverage mismatch: {directory}")
    return rows, manifest


def mcnemar_exact(control_only: int, candidate_only: int) -> float:
    discordant = control_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(control_only, candidate_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def paired_bootstrap_ci(differences: np.ndarray, *, seed: int = 20260831,
                        samples: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 250
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        means[start:start + count] = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def compare(control_rows: dict[str, dict], candidate_rows: dict[str, dict]) -> dict:
    if set(control_rows) != set(candidate_rows):
        raise ValueError("control/candidate case ID sets differ")
    ids = sorted(control_rows)
    differences = []
    cells = defaultdict(int)
    families: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for rid in ids:
        control, candidate = control_rows[rid], candidate_rows[rid]
        for field in ("family", "difficulty", "answer_type"):
            if control[field] != candidate[field]:
                raise ValueError(f"case metadata mismatch for {rid}: {field}")
        c0, c1 = int(bool(control["correct"])), int(bool(candidate["correct"]))
        differences.append(c1 - c0)
        cells[(c0, c1)] += 1
        bucket = families[str(control["family"])]
        bucket[0] += c0
        bucket[1] += c1
        bucket[2] += 1
    array = np.asarray(differences, dtype=np.float64)
    interval = paired_bootstrap_ci(array)
    control_only, candidate_only = cells[(1, 0)], cells[(0, 1)]
    delta = float(array.mean())
    clear_improvement = delta > 0 and interval[0] > 0 and mcnemar_exact(
        control_only, candidate_only) < 0.05
    return {
        "schema_version": 1,
        "cases": len(ids),
        "control_accuracy": sum(int(bool(control_rows[rid]["correct"])) for rid in ids) / len(ids),
        "candidate_accuracy": sum(int(bool(candidate_rows[rid]["correct"])) for rid in ids) / len(ids),
        "accuracy_delta": delta,
        "paired_bootstrap_95_ci": interval,
        "bootstrap_seed": 20260831,
        "bootstrap_samples": 10_000,
        "win_tie_loss": [candidate_only, cells[(0, 0)] + cells[(1, 1)], control_only],
        "paired_cells": {
            "both_wrong": cells[(0, 0)],
            "candidate_only_correct": candidate_only,
            "control_only_correct": control_only,
            "both_correct": cells[(1, 1)],
        },
        "mcnemar_exact_two_sided_p": mcnemar_exact(control_only, candidate_only),
        "clear_improvement_rule": "delta>0, paired bootstrap 95% CI lower>0, exact McNemar p<0.05",
        "clear_improvement": clear_improvement,
        "by_family": {
            family: {
                "control_correct": values[0],
                "candidate_correct": values[1],
                "total": values[2],
                "control_accuracy": values[0] / values[2],
                "candidate_accuracy": values[1] / values[2],
                "accuracy_delta": (values[1] - values[0]) / values[2],
            }
            for family, values in sorted(families.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    control_dir = (ROOT / args.control).resolve()
    candidate_dir = (ROOT / args.candidate).resolve()
    control, control_manifest = load_rows(control_dir)
    candidate, candidate_manifest = load_rows(candidate_dir)
    for key in ("dataset_sha256", "manifest_sha256", "model_fingerprint", "protocol"):
        if control_manifest[key] != candidate_manifest[key]:
            raise SystemExit(f"matched capability protocol mismatch: {key}")
    result = compare(control, candidate)
    result.update({
        "status": "PASS" if result["clear_improvement"] else "FAIL",
        "decision": (
            "candidate unlocks confirmatory escalation"
            if result["clear_improvement"]
            else "candidate does not demonstrate a clear targeted-capability improvement"
        ),
        "control_adapter": control_manifest["adapter"],
        "candidate_adapter": candidate_manifest["adapter"],
        "dataset_sha256": control_manifest["dataset_sha256"],
        "sealed_data_accessed": False,
        "test_and_challenge_accessed": False,
    })
    evidence = (ROOT / args.evidence).resolve()
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
