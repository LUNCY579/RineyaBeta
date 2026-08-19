"""Tests that run against the real bundled dataset."""

from __future__ import annotations

import json

import pytest

from rineya_lab.config import ScoringConfig
from rineya_lab.data.records import FullRecord, RecordError, TransitionRecord
from rineya_lab.data.validate import check_causal_safeguard, normalize_problem, validate_dataset
from rineya_lab.evaluation.backends import _perturb_answer
from rineya_lab.evaluation.scoring import score_answer

pytestmark = pytest.mark.usefixtures("dataset")


def test_dev_split_passes_strict_validation(dataset):
    report = validate_dataset(dataset, splits=("dev",))
    assert report.status == "PASS", report.errors[:10]
    assert report.counts["full_records"]["dev"] == 2400
    assert report.counts["adjacent_same_problem_transitions"]["dev"] == 0


def test_transition_chains_reconstruct_full_trajectories(dev_trajectories):
    issues: list[str] = []
    for trajectory in dev_trajectories.values():
        issues.extend(trajectory.check_chain())
    assert not issues, issues[:10]


def test_causal_safeguard_holds_on_dev(dev_trajectories):
    violations: list[str] = []
    for trajectory in dev_trajectories.values():
        violations.extend(check_causal_safeguard(trajectory))
    assert not violations, violations[:10]


def test_causal_safeguard_catches_injected_leakage(dev_trajectories):
    """The safeguard must actually fire; a check that never fails proves nothing."""
    import copy

    trajectory = copy.deepcopy(next(iter(dev_trajectories.values())))
    first = trajectory.transitions[0]
    leaked = dict(first.raw)
    leaked["current_state"] = f"Final answer: {trajectory.full.answer}"
    trajectory.transitions[0] = TransitionRecord(leaked)
    assert check_causal_safeguard(trajectory)


def test_every_dev_answer_scores_correct_against_itself(dataset):
    cfg = ScoringConfig()
    failures = []
    for record in dataset.iter_full("dev"):
        result = score_answer(record.answer, record.answer, answer_type=record.answer_type, cfg=cfg)
        if not result.correct:
            failures.append((record.problem_id, record.answer_type, record.answer))
    assert not failures, failures[:10]


def test_perturbed_answers_are_rejected(dataset):
    cfg = ScoringConfig()
    false_positives = []
    for record in dataset.iter_full("dev", limit=800):
        perturbed = _perturb_answer(record.answer)
        if perturbed == record.answer:
            continue
        if score_answer(perturbed, record.answer, answer_type=record.answer_type, cfg=cfg).correct:
            false_positives.append((record.problem_id, record.answer_type, perturbed, record.answer))
    assert not false_positives, false_positives[:10]


def test_dev_and_test_problem_sets_are_disjoint(dataset):
    dev = {r.problem_id for r in dataset.iter_full("dev")}
    test = {r.problem_id for r in dataset.iter_full("test")}
    assert not dev & test


def test_normalized_problem_text_is_unique_within_dev(dataset):
    seen: dict[str, str] = {}
    for record in dataset.iter_full("dev"):
        key = normalize_problem(record.problem)
        assert key not in seen, (record.problem_id, seen.get(key))
        seen[key] = record.problem_id


def test_stateful_rollout_target_matches_its_fields(dataset):
    for transition in dataset.iter_transitions("dev", limit=500):
        payload = json.loads(transition.stateful_rollout_target)
        assert payload["next_reasoning_step"] == transition.target_step
        assert payload["updated_compact_state"] == transition.next_state
        assert payload["done"] == transition.done
        assert payload["final_answer"] == transition.final_answer


def test_record_wrappers_reject_incomplete_rows():
    with pytest.raises(RecordError):
        FullRecord({"problem_id": "x"})
    with pytest.raises(RecordError):
        TransitionRecord({"problem_id": "x"})


def test_file_hashes_are_stable(dataset):
    first = dataset.file_hashes(splits=("dev",))
    second = dataset.file_hashes(splits=("dev",))
    assert first == second and all(len(v) == 64 for v in first.values())
