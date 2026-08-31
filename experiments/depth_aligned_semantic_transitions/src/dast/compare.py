"""Paired arm comparisons with deterministic bootstrap and McNemar evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dast.artifacts import atomic_write_json


@dataclass(frozen=True)
class PairedResult:
    problem_id: str
    baseline_correct: bool
    candidate_correct: bool


def _read_predictions(path: str | Path) -> dict[str, bool]:
    rows: dict[str, bool] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            problem_id = str(value["problem_id"])
            if problem_id in rows:
                raise ValueError(f"duplicate problem_id {problem_id!r} in {path}:{line_number}")
            rows[problem_id] = bool(value["correct"])
    return rows


def _exact_two_sided_binomial(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    observed = math.comb(n, k) / (2**n)
    probability = 0.0
    for value in range(n + 1):
        term = math.comb(n, value) / (2**n)
        if term <= observed + 1e-15:
            probability += term
    return min(1.0, probability)


def compare_prediction_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 17,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline = _read_predictions(baseline_path)
    candidate = _read_predictions(candidate_path)
    common = sorted(set(baseline) & set(candidate))
    if set(baseline) != set(candidate):
        missing_baseline = sorted(set(candidate) - set(baseline))[:10]
        missing_candidate = sorted(set(baseline) - set(candidate))[:10]
        raise ValueError(
            "prediction sets differ; paired inference would be invalid. "
            f"missing baseline={missing_baseline}, missing candidate={missing_candidate}"
        )
    pairs = [PairedResult(key, baseline[key], candidate[key]) for key in common]
    base_values = np.array([item.baseline_correct for item in pairs], dtype=np.float64)
    candidate_values = np.array([item.candidate_correct for item in pairs], dtype=np.float64)
    differences = candidate_values - base_values
    rng = np.random.default_rng(seed)
    if len(pairs):
        indices = rng.integers(0, len(pairs), size=(bootstrap_samples, len(pairs)))
        boot = differences[indices].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975]).tolist()
    else:
        low = high = 0.0
    baseline_only = int(np.sum((base_values == 1) & (candidate_values == 0)))
    candidate_only = int(np.sum((base_values == 0) & (candidate_values == 1)))
    report = {
        "paired_problems": len(pairs),
        "baseline_accuracy": float(base_values.mean()) if len(pairs) else 0.0,
        "candidate_accuracy": float(candidate_values.mean()) if len(pairs) else 0.0,
        "candidate_minus_baseline": float(differences.mean()) if len(pairs) else 0.0,
        "bootstrap_95_ci": [float(low), float(high)],
        "discordant": {
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
        },
        "mcnemar_exact_two_sided_p": _exact_two_sided_binomial(
            min(baseline_only, candidate_only), baseline_only + candidate_only
        ),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "claim_boundary": "A single paired comparison is not confirmatory; aggregate preregistered seeds and controls.",
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report
