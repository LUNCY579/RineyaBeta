from pdrr.toy import ToyConfig, run_reference


def test_small_reference_executes_distinct_arms_and_parameter_frontier():
    config = ToyConfig(train_examples=64, eval_examples=32, steps=2, batch_size=8)
    results = run_reference(
        config, [17],
        arms=["full", "hard_prune", "gradual_kd", "random_matrix_recovery",
              "causal_state_position_recovery"],
    )
    by_arm = {result.arm: result for result in results}
    assert by_arm["full"].metrics.quality == 1.0
    assert by_arm["hard_prune"].metrics.quality < 1.0
    for arm in ("hard_prune", "gradual_kd", "random_matrix_recovery",
                "causal_state_position_recovery"):
        assert by_arm[arm].final_parameter_count == by_arm[arm].hard_parameter_count
        assert by_arm[arm].gate_zero_physical_max_abs <= 1e-5
