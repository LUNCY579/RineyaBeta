#!/usr/bin/env python3
"""CPU proof obligations for the frozen RGE capability pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "verify"))

from compare_rge_capability import compare, mcnemar_exact  # noqa: E402
from run_rge_capability import (  # noqa: E402
    extract_final, normalize_answer, reject_nonvalidation_path, score_prediction,
    sha256_file, tree_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m6_capability_unit_cpu.json")
    args = parser.parse_args()
    checks: list[str] = []

    assert extract_final("<think>x</think><final>Blue.</final>") == "Blue."
    checks.append("final_tag_extraction")
    assert extract_final("<think>x</think> Yes.") == "Yes."
    checks.append("think_suffix_extraction")
    assert extract_final("work Final: 3 moves") == "3 moves"
    checks.append("final_prefix_extraction")
    assert normalize_answer("  A → B! ") == "a -> b"
    checks.append("frozen_normalization")
    row = {"accepted_answers": ["Knight.", "knight"]}
    assert score_prediction(row, "Final: KNIGHT!")["correct"]
    checks.append("accepted_answer_exact_match")
    assert not score_prediction(row, "probably knight")["correct"]
    checks.append("verbosity_not_falsely_accepted")

    reject_nonvalidation_path(Path("asset/validation.jsonl"))
    checks.append("validation_path_accepted")
    for name in ("test.jsonl", "challenge.jsonl", "rag_SEALED.jsonl"):
        try:
            reject_nonvalidation_path(Path("asset") / name)
        except ValueError:
            checks.append(f"forbidden_{name}_rejected")
        else:
            raise AssertionError(f"forbidden path accepted: {name}")

    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) < 0.05
    assert mcnemar_exact(5, 5) == 1.0
    checks.append("exact_mcnemar_reference_cases")

    control = {
        "a": {"correct": False, "family": "x", "difficulty": "easy", "answer_type": "z"},
        "b": {"correct": False, "family": "x", "difficulty": "easy", "answer_type": "z"},
        "c": {"correct": True, "family": "y", "difficulty": "medium", "answer_type": "z"},
    }
    candidate = {
        "a": {"correct": True, "family": "x", "difficulty": "easy", "answer_type": "z"},
        "b": {"correct": False, "family": "x", "difficulty": "easy", "answer_type": "z"},
        "c": {"correct": True, "family": "y", "difficulty": "medium", "answer_type": "z"},
    }
    paired = compare(control, candidate)
    assert paired["win_tie_loss"] == [1, 2, 0]
    assert abs(paired["accuracy_delta"] - 1 / 3) < 1e-12
    checks.append("paired_case_accounting")
    assert paired["by_family"]["x"]["accuracy_delta"] == 0.5
    checks.append("family_delta_accounting")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "a.txt").write_text("a", encoding="utf-8")
        first = tree_fingerprint(root)
        assert sha256_file(root / "a.txt")
        (root / "a.txt").write_text("b", encoding="utf-8")
        assert tree_fingerprint(root) != first
    checks.append("recursive_fingerprint_detects_change")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "checks": checks,
        "check_count": len(checks),
        "sealed_data_accessed": False,
        "test_and_challenge_accessed": False,
    }
    evidence = (ROOT / args.evidence).resolve()
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
