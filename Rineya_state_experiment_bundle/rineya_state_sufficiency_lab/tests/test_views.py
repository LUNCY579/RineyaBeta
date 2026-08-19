"""The order ablation lives or dies on these tests."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from rineya_lab.config import load_config
from rineya_lab.data.views import build_view_plan, iter_examples, order_diagnostics

pytestmark = pytest.mark.usefixtures("dataset")


def make_cfg(dataset_root, *, condition, view, order, **overrides):
    data = {
        "dataset": {"root": str(dataset_root), "verify_hashes": False, "train_split": "dev"},
        "experiment": {"name": "views_test", "condition": condition, "seed": 20260818},
        "data_view": {"view": view, "order": order, "max_records": 400, **overrides.pop("data_view", {})},
        "model": {"name_or_path": "builtin:byte", "tokenizer_name_or_path": "builtin:byte"},
        "training": {"epochs": 1, "max_seq_len": 256},
    }
    for key, value in overrides.items():
        data.setdefault(key, {}).update(value)
    return load_config(None, extra=data)


def test_shuffled_order_scatters_same_problem_rows(dataset, dataset_root):
    """Condition C must destroy chronology -- but not by being a constrained shuffle.

    This previously asserted both counts were exactly 0, which held on this
    fixture and failed on the real TRAIN epoch (5 and 1 observed against 3.32
    and 0.75 expected). Zero is not the invariant; staying at the
    uniform-shuffle scale is.
    """
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled")
    plan = build_view_plan(dataset, cfg)
    diagnostics = order_diagnostics(plan)["epoch_0"]
    ref = diagnostics["uniform_shuffle_reference"]
    assert diagnostics["adjacent_same_problem"] <= ref["adjacent_same_problem_upper"]
    assert diagnostics["adjacent_consecutive_steps"] <= ref["adjacent_consecutive_steps_upper"]
    # And the rows really are spread out, not merely un-adjacent.
    assert diagnostics["mean_gap_between_same_problem_rows"] > 1.0


def test_uniform_shuffle_reference_matches_the_closed_form(dataset, dataset_root):
    """The gate ceiling is only meaningful if the expectation is computed right."""
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled")
    plan = build_view_plan(dataset, cfg)
    rows = plan.epochs[0].rows
    sizes = Counter(row.problem_id for row in rows)
    n = len(rows)
    ref = order_diagnostics(plan)["epoch_0"]["uniform_shuffle_reference"]
    assert ref["adjacent_same_problem_expected"] == pytest.approx(
        sum(k * (k - 1) for k in sizes.values()) / n
    )
    assert ref["adjacent_consecutive_steps_expected"] == pytest.approx(
        sum(k - 1 for k in sizes.values()) / n
    )
    assert ref["consecutive_pairs_available"] == sum(k - 1 for k in sizes.values())
    # A trajectory-ordered plan must blow straight through the shuffle ceiling.
    ordered_cfg = make_cfg(
        dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="trajectory"
    )
    ordered = order_diagnostics(build_view_plan(dataset, ordered_cfg))["epoch_0"]
    assert ordered["adjacent_consecutive_steps"] > ref["adjacent_consecutive_steps_upper"]


def test_trajectory_order_keeps_steps_contiguous_and_ascending(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="trajectory")
    plan = build_view_plan(dataset, cfg)
    rows = plan.epochs[0].rows
    grouped = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[row.problem_id].append((position, row.step_index))
    for problem_id, entries in grouped.items():
        positions = [p for p, _ in entries]
        steps = [s for _, s in entries]
        assert positions == list(range(positions[0], positions[0] + len(positions))), problem_id
        assert steps == sorted(steps), problem_id


def test_reverse_order_descends_within_each_problem(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="reverse")
    plan = build_view_plan(dataset, cfg)
    grouped = defaultdict(list)
    for row in plan.epochs[0].rows:
        grouped[row.problem_id].append(row.step_index)
    assert any(len(steps) > 1 for steps in grouped.values())
    for steps in grouped.values():
        assert steps == sorted(steps, reverse=True)


def test_curriculum_order_is_non_decreasing_in_step_index(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="curriculum")
    plan = build_view_plan(dataset, cfg)
    steps = [row.step_index for row in plan.epochs[0].rows]
    assert steps == sorted(steps)


def test_ordering_is_deterministic_across_rebuilds(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled")
    first = [r.row_index for r in build_view_plan(dataset, cfg).epochs[0].rows]
    second = [r.row_index for r in build_view_plan(dataset, cfg).epochs[0].rows]
    assert first == second


def test_seed_changes_the_shuffle(dataset, dataset_root):
    a = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled")
    b = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled",
                 experiment={"seed": 999})
    assert [r.row_index for r in build_view_plan(dataset, a).epochs[0].rows] != [
        r.row_index for r in build_view_plan(dataset, b).epochs[0].rows
    ]


def test_ordered_and_unordered_conditions_use_the_same_record_set(dataset, dataset_root):
    """B vs C must differ only in order -- otherwise the ablation is confounded."""
    ordered = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="trajectory",
                       data_view={"max_records": None})
    unordered = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled",
                         data_view={"max_records": None})
    left = {r.row_index for r in build_view_plan(dataset, ordered).epochs[0].rows}
    right = {r.row_index for r in build_view_plan(dataset, unordered).epochs[0].rows}
    assert left == right


def test_one_transition_per_problem_matches_problem_count(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled",
                   data_view={"one_transition_per_problem": True, "max_records": None})
    plan = build_view_plan(dataset, cfg)
    rows = plan.epochs[0].rows
    assert len(rows) == len({r.problem_id for r in rows}) == 2400


def test_one_transition_per_problem_rotates_across_epochs(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled",
                   data_view={"one_transition_per_problem": True, "max_records": None},
                   training={"epochs": 2})
    plan = build_view_plan(dataset, cfg)
    first = {r.problem_id: r.step_index for r in plan.epochs[0].rows}
    second = {r.problem_id: r.step_index for r in plan.epochs[1].rows}
    changed = sum(1 for pid in first if first[pid] != second[pid])
    assert changed > 0


def test_inverse_transition_weighting_normalises_problem_mass(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled",
                   data_view={"max_records": None}, training={"loss_weighting": "inverse_transitions"})
    plan = build_view_plan(dataset, cfg)
    totals = defaultdict(float)
    for example in iter_examples(dataset, plan, cfg):
        totals[example.problem_id] += example.weight
    assert all(abs(value - 1.0) < 1e-9 for value in totals.values())


def test_full_view_yields_one_record_per_problem(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="FULL", view="full_context", order="shuffled",
                   data_view={"max_records": None})
    plan = build_view_plan(dataset, cfg)
    assert plan.total_rows == plan.unique_problems == 2400


def test_stateful_and_next_step_views_read_different_targets(dataset, dataset_root):
    stateful = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="trajectory")
    next_step = make_cfg(dataset_root, condition="SPLIT_NEXT_STEP", view="split_next_step", order="trajectory")
    a = next(iter_examples(dataset, build_view_plan(dataset, stateful), stateful))
    b = next(iter_examples(dataset, build_view_plan(dataset, next_step), next_step))
    assert a.example_id == b.example_id
    assert a.target.startswith("{") and not b.target.startswith("{")
    assert "Return a JSON object" in a.prompt and "Return a JSON object" not in b.prompt


def test_category_filter_applies(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="FULL", view="full_context", order="shuffled",
                   data_view={"categories": ["geometry"], "max_records": None})
    plan = build_view_plan(dataset, cfg)
    assert all(row.category == "geometry" for row in plan.epochs[0].rows)


def test_resume_offset_skips_exactly(dataset, dataset_root):
    cfg = make_cfg(dataset_root, condition="SPLIT_STATEFUL", view="split_stateful", order="shuffled")
    plan = build_view_plan(dataset, cfg)
    full = [e.example_id for e in iter_examples(dataset, plan, cfg)]
    resumed = [e.example_id for e in iter_examples(dataset, plan, cfg, start_epoch=0, start_position=10)]
    assert resumed == full[10:]
