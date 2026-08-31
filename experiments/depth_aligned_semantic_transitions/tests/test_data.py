import json
from pathlib import Path

from dast.config import ARM_SPECS
from dast.data import ReasoningRecord, build_examples, load_records, validate_dataset
from dast.smoke import fixture_path


def test_direct_and_paired_records_parse_identically():
    direct = {
        "problem_id": "p",
        "split": "train",
        "problem": "q",
        "answer": "a",
        "full_response": "r",
        "reasoning_steps": ["one"],
        "compact_state_trace": ["initial", "final"],
    }
    paired = {"full_context_record": direct}
    assert ReasoningRecord.from_mapping(direct) == ReasoningRecord.from_mapping(paired)


def test_fixture_validation_and_splits():
    report = validate_dataset(fixture_path())
    assert report.passed
    assert report.split_counts == {"dev": 4, "test": 1, "train": 8}


def test_compact_targets_exclude_initial_state():
    record = load_records(fixture_path(), split="train", limit=1)[0]
    examples = build_examples([record], arm=ARM_SPECS["state_ordered"], num_stages=4, seed=17)
    assert "No derived facts" not in examples[0].trajectory.canonical_texts[0]


def test_cross_split_duplicate_is_rejected(tmp_path: Path):
    base = {
        "problem_id": "duplicate",
        "problem": "q",
        "answer": "a",
        "full_response": "r",
        "reasoning_steps": ["s"],
        "compact_state_trace": ["i", "f"],
    }
    path = tmp_path / "dup.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for split in ("train", "dev"):
            handle.write(json.dumps({**base, "split": split}) + "\n")
    report = validate_dataset(path)
    assert not report.passed
    assert any("duplicate problem_id" in error for error in report.errors)
