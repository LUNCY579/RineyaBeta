from __future__ import annotations

import pytest

from rineya_lab.config import BackendConfig, RolloutConfig, ScoringConfig
from rineya_lab.evaluation import failures
from rineya_lab.evaluation.backends import ReferenceBackend, _perturb_answer, build_backend
from rineya_lab.evaluation.rollout import max_steps_for, run_oracle_trajectory, run_rollout

pytestmark = pytest.mark.usefixtures("dataset")


def backend(mode: str) -> ReferenceBackend:
    return ReferenceBackend(BackendConfig(kind="reference", reference_mode=mode))


@pytest.fixture
def trajectory(dev_trajectories):
    # a multi-step trajectory makes the detectors meaningful
    return next(t for t in dev_trajectories.values() if t.length >= 4)


def test_oracle_rollout_reaches_the_right_answer(trajectory):
    result = run_rollout(backend("oracle"), trajectory, RolloutConfig(), scoring_cfg=ScoringConfig())
    assert result.terminated and result.termination_reason == "done"
    assert result.final_answer == trajectory.full.answer
    assert result.first_error_depth is None
    assert result.steps_taken == trajectory.length


def test_loop_is_detected(trajectory):
    result = run_rollout(backend("looping"), trajectory, RolloutConfig())
    assert result.control_failure == failures.LOOP


def test_stagnation_is_detected(trajectory):
    result = run_rollout(backend("stagnant"), trajectory, RolloutConfig(stagnation_patience=2))
    assert result.control_failure == failures.STAGNATION


def test_nontermination_is_detected(trajectory):
    result = run_rollout(backend("nonterminating"), trajectory, RolloutConfig())
    assert result.control_failure == failures.NONTERMINATION
    assert not result.terminated
    assert result.steps_taken == result.max_steps_allowed


def test_premature_completion_is_detected(trajectory):
    result = run_rollout(backend("premature"), trajectory, RolloutConfig())
    assert result.control_failure == failures.PREMATURE_COMPLETION
    assert result.final_answer is None


def test_refusal_and_empty_keep_their_own_labels(trajectory):
    assert run_rollout(backend("refusal"), trajectory, RolloutConfig()).control_failure == failures.REFUSAL
    assert run_rollout(backend("empty"), trajectory, RolloutConfig()).control_failure == failures.EMPTY_OUTPUT


def test_wrong_mode_diverges_at_the_first_step(trajectory):
    result = run_rollout(backend("wrong"), trajectory, RolloutConfig(), scoring_cfg=ScoringConfig())
    assert result.first_error_depth == 1


def test_step_budget_respects_dataset_bound_and_absolute_cap(trajectory):
    cfg = RolloutConfig(max_steps_absolute=100, max_steps_margin=3)
    assert max_steps_for(trajectory, cfg) == trajectory.length + 3
    capped = RolloutConfig(max_steps_absolute=3, max_steps_margin=10)
    assert max_steps_for(trajectory, capped) == 3
    unbounded = RolloutConfig(max_steps_absolute=9, use_dataset_step_bound=False)
    assert max_steps_for(trajectory, unbounded) == 9


def test_oracle_trajectory_is_teacher_forced(trajectory):
    result = run_oracle_trajectory(backend("oracle"), trajectory, RolloutConfig())
    assert result.final_answer == trajectory.full.answer
    assert [step.state_in for step in result.steps] == [t.current_state for t in trajectory.transitions]


def test_oracle_trajectory_survives_a_wrong_model(trajectory):
    """Teacher forcing must not stop early just because the model is wrong."""
    result = run_oracle_trajectory(backend("wrong"), trajectory, RolloutConfig())
    assert len(result.steps) == trajectory.length
    assert result.final_answer != trajectory.full.answer


def test_noisy_backend_is_deterministic(trajectory):
    cfg = BackendConfig(kind="reference", reference_mode="noisy", reference_noise=0.5, reference_seed=3)
    first = run_rollout(ReferenceBackend(cfg), trajectory, RolloutConfig())
    second = run_rollout(ReferenceBackend(cfg), trajectory, RolloutConfig())
    assert [s.raw_response for s in first.steps] == [s.raw_response for s in second.steps]


def test_unknown_reference_mode_rejected():
    with pytest.raises(ValueError, match="reference_mode"):
        ReferenceBackend(BackendConfig(kind="reference", reference_mode="telepathy"))


def test_build_backend_returns_reference(base_config):
    base_config.evaluation.backend.kind = "reference"
    assert build_backend(base_config).is_reference


def test_perturb_answer_changes_every_kind_of_answer():
    for answer in ["568", "$12.50", "True", "x = 6", "{1, 2}", "abc"]:
        assert _perturb_answer(answer) != answer
