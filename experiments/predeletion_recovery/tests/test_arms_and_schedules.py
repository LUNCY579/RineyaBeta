from pdrr.arms import arm_registry, get_arm
from pdrr.schedules import RecoverySchedule


def test_registry_contains_all_preregistered_arms():
    registry = arm_registry()
    assert len(registry) == 13
    assert "matrix_no_recovery" in registry
    assert "random_matrix_recovery" in registry
    assert "causal_state_position_recovery" in registry


def test_decisive_ablations_are_semantically_distinct():
    no_loss = get_arm("matrix_no_recovery")
    random = get_arm("random_matrix_recovery")
    assert no_loss.routing == "state_position"
    assert no_loss.recovery_loss is False
    assert random.routing == "random"
    assert random.recovery_loss is True
    assert random.router_trainable is False


def test_schedule_has_notice_withdrawal_and_nonzero_consolidation():
    schedule = RecoverySchedule(100)
    assert schedule.phase(0) == "notice"
    assert schedule.doomed_gate(0) == 1.0
    assert schedule.phase(50) == "withdrawal"
    assert 0.0 < schedule.doomed_gate(50) < 1.0
    assert schedule.phase(99) == "consolidation"
    assert schedule.doomed_gate(99) == 0.0
    assert schedule.recovery_weight(99) == 0.0
    assert schedule.compact_only(99)


def test_schedule_endpoints_are_exact_for_all_shapes():
    for kind in ("linear", "cosine", "sigmoid", "staircase"):
        schedule = RecoverySchedule(120, kind=kind)
        assert schedule.doomed_gate(0) == 1.0
        assert schedule.doomed_gate(schedule.withdrawal_end_step) == 0.0
