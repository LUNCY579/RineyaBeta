"""The paired interval SSOT section 10 actually promises.

The plan asks for "problem-cluster bootstrap intervals". compare.py resampled
problems i.i.d., which is anti-conservative because problems sharing a generator
share a template and are not independent. Measured on the real G1 pilot predictions
(800 DEV problems, 30 generators), the clustered interval is 22% wider than the
i.i.d. one, so the difference is not academic.
"""

from __future__ import annotations

import json
from pathlib import Path

from dast.compare import compare_prediction_files


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _pair(tmp_path: Path, *, with_generator: bool, correlated: bool):
    """Two prediction files over the same problems.

    When ``correlated`` the outcome is decided by the CLUSTER, which is exactly the
    dependence an i.i.d. bootstrap ignores.
    """

    baseline, candidate = [], []
    for cluster in range(10):
        cluster_wins = cluster % 2 == 0
        for member in range(20):
            problem_id = f"p{cluster:02d}_{member:02d}"
            row = {"problem_id": problem_id, "category": "c", "answer_type": "integer"}
            if with_generator:
                row["generator"] = f"gen_{cluster:02d}"
            base_ok = (member % 2 == 0)
            cand_ok = cluster_wins if correlated else (member % 3 == 0)
            baseline.append({**row, "correct": base_ok})
            candidate.append({**row, "correct": cand_ok})
    return (
        _write(tmp_path / "baseline.jsonl", baseline),
        _write(tmp_path / "candidate.jsonl", candidate),
    )


def test_clustered_interval_is_reported_when_the_cluster_unit_is_present(tmp_path):
    base, cand = _pair(tmp_path, with_generator=True, correlated=True)
    report = compare_prediction_files(base, cand, bootstrap_samples=400, seed=17)
    clustered = report["bootstrap_95_ci_clustered"]
    assert clustered["available"] is True
    assert clustered["unit"] == "generator"
    assert clustered["clusters"] == 10
    low, high = clustered["bootstrap_95_ci"]
    assert low <= high


def test_missing_generator_falls_back_and_says_so(tmp_path):
    """Older prediction files predate the field; they must degrade loudly, not
    silently report an interval that assumes independence."""

    base, cand = _pair(tmp_path, with_generator=False, correlated=True)
    report = compare_prediction_files(base, cand, bootstrap_samples=400, seed=17)
    clustered = report["bootstrap_95_ci_clustered"]
    assert clustered["available"] is False
    assert "generator" in clustered["reason"]
    # The i.i.d. interval is still produced, so old artifacts stay interpretable.
    assert len(report["bootstrap_95_ci"]) == 2


def test_clustered_interval_is_wider_when_outcomes_are_cluster_driven(tmp_path):
    """The whole point: ignoring cluster structure understates uncertainty."""

    base, cand = _pair(tmp_path, with_generator=True, correlated=True)
    report = compare_prediction_files(base, cand, bootstrap_samples=2000, seed=17)
    iid_low, iid_high = report["bootstrap_95_ci"]
    cl_low, cl_high = report["bootstrap_95_ci_clustered"]["bootstrap_95_ci"]
    assert (cl_high - cl_low) > (iid_high - iid_low)


def test_clustered_bootstrap_is_deterministic(tmp_path):
    base, cand = _pair(tmp_path, with_generator=True, correlated=True)
    first = compare_prediction_files(base, cand, bootstrap_samples=500, seed=29)
    second = compare_prediction_files(base, cand, bootstrap_samples=500, seed=29)
    assert (
        first["bootstrap_95_ci_clustered"]["bootstrap_95_ci"]
        == second["bootstrap_95_ci_clustered"]["bootstrap_95_ci"]
    )


def test_generation_record_carries_the_cluster_unit():
    """Without this field on the record, the clustered interval can never be
    computed from a real run."""

    from dast.evaluation import GenerationRecord

    record = GenerationRecord(
        problem_id="p1", category="c", generator="gen_a", answer_type="integer",
        expected="1", extracted="1", raw_response="the answer is 1",
        correct=True, extraction_route="labelled",
    )
    assert record.as_dict()["generator"] == "gen_a"
