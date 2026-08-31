from dast.alignment import align_trajectory, resolve_checkpoint_layers
from dast.config import ARM_SPECS


def test_resolve_qwen_28_layer_quarters():
    assert resolve_checkpoint_layers(28, (0.25, 0.5, 0.75, 1.0)) == (7, 14, 21, 28)


def test_variable_trace_alignment_pins_endpoints():
    result = align_trajectory(
        ["s1", "s2", "s3", "s4", "s5"],
        num_stages=4,
        arm=ARM_SPECS["state_ordered"],
        problem_id="p",
        seed=17,
    )
    assert result.canonical_texts == ("s1", "s2", "s4", "s5")
    assert result.canonical_source_indices == (0, 1, 3, 4)


def test_short_trace_repeats_are_explicit():
    result = align_trajectory(
        ["early", "late"],
        num_stages=4,
        arm=ARM_SPECS["state_ordered"],
        problem_id="p",
        seed=17,
    )
    assert result.canonical_texts == ("early", "early", "late", "late")
    assert result.repeated_stage_mask == (False, True, False, True)


def test_shuffled_control_is_deterministic_and_not_identity():
    kwargs = dict(
        raw_targets=["a", "b", "c", "d"],
        num_stages=4,
        arm=ARM_SPECS["state_shuffled"],
        problem_id="same",
        seed=9,
    )
    first = align_trajectory(**kwargs)
    second = align_trajectory(**kwargs)
    assert first.assignment == second.assignment
    assert first.assignment != (0, 1, 2, 3)
    assert sorted(first.training_texts) == sorted(first.canonical_texts)


def test_reverse_and_final_repeated_controls():
    targets = ["a", "b", "c", "d"]
    reverse = align_trajectory(
        targets, num_stages=4, arm=ARM_SPECS["state_reverse"], problem_id="p", seed=1
    )
    repeated = align_trajectory(
        targets, num_stages=4, arm=ARM_SPECS["state_final_repeated"], problem_id="p", seed=1
    )
    assert reverse.training_texts == ("d", "c", "b", "a")
    assert repeated.training_texts == ("d", "d", "d", "d")
