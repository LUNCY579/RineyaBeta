from __future__ import annotations

from rineya_lab.evaluation import failures
from rineya_lab.evaluation.aggregate import (
    audit_sample,
    free_rollout_gap,
    state_sufficiency_contrast,
    summarize_protocol,
)
from rineya_lab.evaluation.protocols import EvalRecord


def record(**kwargs) -> EvalRecord:
    defaults = dict(
        record_id="r", protocol="oracle_transition", problem_id="p1", category="geometry",
        domain="mathematics", answer_type="integer", total_steps=4,
    )
    defaults.update(kwargs)
    return EvalRecord(**defaults)


def test_denominator_includes_unscoreable_items():
    records = [
        record(record_id="a", strict_correct=True, lenient_correct=True, failure_type=failures.CORRECT),
        record(record_id="b", failure_type=failures.EMPTY_OUTPUT, parse_status="empty"),
        record(record_id="c", failure_type=failures.REFUSAL, parse_status="refusal"),
        record(record_id="d", failure_type=failures.BACKEND_FAILURE, parse_status="backend_failure"),
    ]
    summary = summarize_protocol("oracle_transition", records)
    assert summary["total_items"] == summary["scored_items"] == 4
    assert summary["strict"]["accuracy"] == 0.25
    assert sum(summary["failure_distribution"].values()) == 4
    assert summary["failure_groups"]["format"] == 3


def test_track_disagreement_is_reported():
    records = [
        record(record_id="a", strict_correct=False, lenient_correct=True),
        record(record_id="b", strict_correct=True, lenient_correct=True),
    ]
    summary = summarize_protocol("full_direct", records)
    assert summary["track_disagreement"]["lenient_only_correct"] == 1
    assert summary["track_disagreement"]["disagreement_rate"] == 0.5


def test_stratification_by_category_and_depth():
    records = [
        record(record_id="a", category="geometry", total_steps=3, strict_correct=True),
        record(record_id="b", category="finance", total_steps=7, strict_correct=False),
    ]
    summary = summarize_protocol("full_direct", records)
    assert summary["by_category"]["geometry"]["strict_accuracy"] == 1.0
    assert summary["by_total_steps"]["7"]["strict_accuracy"] == 0.0


def test_transition_components_are_summarised():
    records = [
        record(record_id="a", components={"step_match": True, "updated_state_valid": True, "done_correct": True}),
        record(record_id="b", components={"step_match": True, "updated_state_valid": False, "done_correct": True}),
    ]
    summary = summarize_protocol("oracle_transition", records)
    components = summary["transition_components"]
    assert components["step_match"]["rate"] == 1.0
    assert components["updated_state_valid"]["rate"] == 0.5


def test_rollout_statistics():
    records = [
        record(record_id="a", components={"steps_taken": 4, "first_error_depth": None,
                                          "termination_reason": "done"}, strict_correct=True),
        record(record_id="b", components={"steps_taken": 9, "first_error_depth": 2,
                                          "termination_reason": "nontermination"}),
    ]
    summary = summarize_protocol("generated_rollout", records)
    rollout = summary["rollout"]
    assert rollout["mean_steps"] == 6.5
    assert rollout["completion_rate"] == 0.5
    assert rollout["first_error_depth"]["items_with_error"] == 1
    assert rollout["first_error_depth"]["mean_depth"] == 2.0


def test_free_rollout_gap_arithmetic():
    summaries = {
        "oracle_transition": {"strict": {"accuracy": 0.9}},
        "generated_rollout": {"strict": {"accuracy": 0.6}},
        "oracle_trajectory": {"strict": {"accuracy": 0.85}},
    }
    gap = free_rollout_gap(summaries)
    assert gap["gap"] == 0.3
    assert gap["teacher_forcing_vs_free_rollout_gap"] == 0.25


def test_free_rollout_gap_absent_without_both_protocols():
    assert free_rollout_gap({"oracle_transition": {"strict": {"accuracy": 0.9}}}) is None


def test_state_sufficiency_contrast():
    summaries = {
        "oracle_transition": {"strict": {"accuracy": 0.90}},
        "full_history_transition": {"strict": {"accuracy": 0.92}},
        "no_state_transition": {"strict": {"accuracy": 0.31}},
    }
    contrast = state_sufficiency_contrast(summaries)
    assert contrast["compact_minus_history"] == -0.02
    assert contrast["state_information_gain"] == 0.59


def test_audit_sample_prefers_disagreements_and_is_deterministic():
    records = [record(record_id=f"ok{i}", strict_correct=True, lenient_correct=True,
                      failure_type=failures.CORRECT) for i in range(20)]
    records += [record(record_id=f"odd{i}", strict_correct=False, lenient_correct=True,
                       failure_type=failures.MALFORMED_STRUCTURE) for i in range(4)]
    first = audit_sample(records, 8, seed=1)
    second = audit_sample(records, 8, seed=1)
    assert first == second
    assert sum(1 for item in first if item["record_id"].startswith("odd")) >= 4


def test_failure_taxonomy_partitions_cleanly():
    groups = [failures.FORMAT_FAILURES, failures.CONTROL_FAILURES, failures.REASONING_FAILURES]
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            assert not (left & right)
    assert failures.CORRECT not in set().union(*groups)


def test_classify_from_status_maps_every_parse_status():
    for status in ["empty", "refusal", "malformed", "missing_fields", "invalid_completion_flag",
                   "ambiguous", "not_found", "overlong", "backend_failure"]:
        assert failures.classify_from_status(status, correct=False) in failures.ALL_FAILURES
    assert failures.classify_from_status("empty", correct=True) == failures.CORRECT
