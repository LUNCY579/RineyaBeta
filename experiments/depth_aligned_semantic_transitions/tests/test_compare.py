import json
from pathlib import Path

import pytest

from dast.compare import compare_prediction_files


def _write(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for problem_id, correct in rows:
            handle.write(json.dumps({"problem_id": problem_id, "correct": correct}) + "\n")


def test_paired_comparison(tmp_path: Path):
    baseline = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [("a", False), ("b", True), ("c", False)])
    _write(candidate, [("a", True), ("b", True), ("c", False)])
    report = compare_prediction_files(baseline, candidate, bootstrap_samples=100, seed=3)
    assert report["paired_problems"] == 3
    assert report["candidate_minus_baseline"] > 0


def test_unpaired_files_are_rejected(tmp_path: Path):
    baseline = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [("a", True)])
    _write(candidate, [("b", True)])
    with pytest.raises(ValueError, match="paired inference"):
        compare_prediction_files(baseline, candidate)
