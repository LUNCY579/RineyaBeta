"""Training views: turn the two dataset variants into ordered example streams.

This module owns the experimental controls that decide *what the optimizer
sees*, which is the entire independent variable of the study:

* ``view``  -- full trajectory, stateful transition, or next-step-only.
* ``order`` -- shuffled (condition C), trajectory (condition B), reverse,
  curriculum.
* ``one_transition_per_problem`` -- source-problem-matched epochs.

Ordering is produced by sorting on salted SHA-256 keys rather than
``random.shuffle`` so an epoch plan is byte-identical on any machine, which is
what makes the B-versus-C order ablation a clean single-variable contrast.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from rineya_lab.config import DataViewConfig, LabConfig
from rineya_lab.data.loading import JsonlIndex, RineyaDataset, RowMeta
from rineya_lab.utils.hashing import stable_shuffle_key
from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("data.views")


@dataclass
class TrainingExample:
    """One optimizer-visible record."""

    example_id: str
    problem_id: str
    messages: list[dict[str, str]]
    category: str
    step_index: int
    total_steps: int
    weight: float = 1.0
    epoch: int = 0
    position: int = 0

    @property
    def prompt(self) -> str:
        return self.messages[0]["content"]

    @property
    def target(self) -> str:
        return self.messages[-1]["content"]


@dataclass
class EpochPlan:
    """The exact row order for one epoch, plus provenance for the manifest."""

    epoch: int
    rows: list[RowMeta]
    order: str
    salt: str

    def __len__(self) -> int:
        return len(self.rows)


@dataclass
class ViewPlan:
    """A complete, reproducible description of the training stream."""

    view: str
    order: str
    variant: str
    split: str
    epochs: list[EpochPlan] = field(default_factory=list)
    unique_problems: int = 0
    total_rows: int = 0
    one_transition_per_problem: bool = False
    filters_applied: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "order": self.order,
            "variant": self.variant,
            "split": self.split,
            "epochs_planned": len(self.epochs),
            "rows_per_epoch": [len(e) for e in self.epochs],
            "total_rows": self.total_rows,
            "unique_problems": self.unique_problems,
            "one_transition_per_problem": self.one_transition_per_problem,
            "filters_applied": self.filters_applied,
            "epoch_order_salts": [e.salt for e in self.epochs],
        }


VIEW_TO_VARIANT = {
    "full_context": "full_context",
    "split_stateful": "split_transitions",
    "split_next_step": "split_transitions",
}

VIEW_TO_MESSAGE_FIELD = {
    "full_context": "messages",
    "split_stateful": "stateful_rollout_messages",
    "split_next_step": "messages",
}


def _row_identity(row: RowMeta) -> str:
    return f"{row.problem_id}::step_{row.step_index:02d}" if row.step_index else row.problem_id


def _apply_filters(rows: list[RowMeta], cfg: DataViewConfig) -> tuple[list[RowMeta], dict[str, Any]]:
    applied: dict[str, Any] = {"input_rows": len(rows)}
    if cfg.categories:
        allowed = set(cfg.categories)
        rows = [r for r in rows if r.category in allowed]
        applied["categories"] = sorted(allowed)
    if cfg.step_index_min is not None:
        rows = [r for r in rows if r.step_index >= cfg.step_index_min]
        applied["step_index_min"] = cfg.step_index_min
    if cfg.step_index_max is not None:
        rows = [r for r in rows if r.step_index <= cfg.step_index_max]
        applied["step_index_max"] = cfg.step_index_max
    if cfg.max_unique_problems is not None:
        # Deterministically choose a problem subset, then keep all of their rows.
        problem_ids = sorted({r.problem_id for r in rows})
        problem_ids.sort(key=lambda pid: stable_shuffle_key(pid, "problem-subset"))
        keep = set(problem_ids[: cfg.max_unique_problems])
        rows = [r for r in rows if r.problem_id in keep]
        applied["max_unique_problems"] = cfg.max_unique_problems
    applied["kept_rows"] = len(rows)
    return rows, applied


def _select_one_per_problem(rows: list[RowMeta], epoch: int, seed: int) -> list[RowMeta]:
    """Pick exactly one transition per source problem, rotating across epochs.

    Matches ``scripts/build_epoch_view.py`` in the dataset repository: the
    starting offset is a hash of the problem id, so different problems begin at
    different steps and the whole trajectory is covered over enough epochs.
    """
    grouped: dict[str, list[RowMeta]] = defaultdict(list)
    for row in rows:
        grouped[row.problem_id].append(row)
    selected: list[RowMeta] = []
    for problem_id, group in grouped.items():
        group.sort(key=lambda r: r.step_index)
        offset = int(stable_shuffle_key(problem_id, f"epoch-offset|{seed}")[:16], 16) % len(group)
        selected.append(group[(offset + epoch) % len(group)])
    return selected


def _order_rows(rows: list[RowMeta], order: str, salt: str) -> list[RowMeta]:
    if order == "shuffled":
        # Global shuffle: rows of one problem are scattered across the epoch.
        return sorted(rows, key=lambda r: stable_shuffle_key(_row_identity(r), salt))
    if order in ("trajectory", "reverse"):
        grouped: dict[str, list[RowMeta]] = defaultdict(list)
        for row in rows:
            grouped[row.problem_id].append(row)
        problem_ids = sorted(grouped, key=lambda pid: stable_shuffle_key(pid, salt))
        out: list[RowMeta] = []
        for problem_id in problem_ids:
            group = sorted(grouped[problem_id], key=lambda r: r.step_index, reverse=(order == "reverse"))
            out.extend(group)
        return out
    if order == "curriculum":
        # All first steps, then all second steps, ... ; ties broken deterministically.
        return sorted(rows, key=lambda r: (r.step_index, stable_shuffle_key(_row_identity(r), salt)))
    raise ValueError(f"Unknown order {order!r}")


def build_view_plan(dataset: RineyaDataset, cfg: LabConfig, *, split: str | None = None) -> ViewPlan:
    """Materialise the full epoch-by-epoch row order for a training run."""
    view_cfg = cfg.data_view
    variant = VIEW_TO_VARIANT[view_cfg.view]
    split = split or cfg.dataset.train_split
    index = dataset.index(variant, split)
    rows, applied = _apply_filters(list(index.rows), view_cfg)
    if not rows:
        raise ValueError("Training view is empty after filters; check data_view.categories / step_index bounds")

    plan = ViewPlan(
        view=view_cfg.view,
        order=view_cfg.order,
        variant=variant,
        split=split,
        one_transition_per_problem=view_cfg.one_transition_per_problem,
        filters_applied=applied,
    )

    whole_epochs = int(cfg.training.epochs)
    fractional = cfg.training.epochs - whole_epochs
    epoch_count = whole_epochs + (1 if fractional > 1e-9 else 0)

    for epoch in range(epoch_count):
        epoch_rows = rows
        if view_cfg.one_transition_per_problem and variant == "split_transitions":
            epoch_rows = _select_one_per_problem(rows, epoch, cfg.experiment.seed)
        salt_epoch = epoch if view_cfg.reshuffle_each_epoch else 0
        salt = f"{cfg.experiment.name}|{cfg.experiment.seed}|{view_cfg.order}|epoch{salt_epoch}"
        ordered = _order_rows(epoch_rows, view_cfg.order, salt)
        if epoch == epoch_count - 1 and fractional > 1e-9:
            ordered = ordered[: max(1, int(round(len(ordered) * fractional)))]
        if view_cfg.max_records is not None:
            ordered = ordered[: view_cfg.max_records]
        plan.epochs.append(EpochPlan(epoch=epoch, rows=ordered, order=view_cfg.order, salt=salt))

    plan.total_rows = sum(len(e) for e in plan.epochs)
    plan.unique_problems = len({r.problem_id for e in plan.epochs for r in e.rows})
    LOGGER.info(
        "View plan: view=%s order=%s epochs=%d rows/epoch=%s unique_problems=%d",
        plan.view,
        plan.order,
        len(plan.epochs),
        [len(e) for e in plan.epochs],
        plan.unique_problems,
    )
    return plan


def _weight_for(row: RowMeta, weighting: str) -> float:
    if weighting == "inverse_transitions" and row.total_steps > 0 and row.step_index > 0:
        # Each source problem contributes total weight 1.0 regardless of how many
        # transitions it expanded into, so SPLIT and FULL weight problems equally.
        return 1.0 / float(row.total_steps)
    return 1.0


def iter_examples(
    dataset: RineyaDataset,
    plan: ViewPlan,
    cfg: LabConfig,
    *,
    start_epoch: int = 0,
    start_position: int = 0,
) -> Iterator[TrainingExample]:
    """Stream the planned rows as training examples.

    ``start_epoch`` / ``start_position`` implement exact checkpoint resume: the
    stream is skipped forward without changing the ordering.
    """
    index: JsonlIndex = dataset.index(plan.variant, plan.split)
    message_field = VIEW_TO_MESSAGE_FIELD[plan.view]
    weighting = cfg.training.loss_weighting

    for epoch_plan in plan.epochs:
        if epoch_plan.epoch < start_epoch:
            continue
        rows = epoch_plan.rows
        offset = start_position if epoch_plan.epoch == start_epoch else 0
        rows = rows[offset:]
        for position, (row, record) in enumerate(zip(rows, index.read_many(rows)), start=offset):
            messages = record.get(message_field)
            if not messages:
                raise KeyError(
                    f"Row {record.get('problem_id')} has no '{message_field}' field required by view {plan.view!r}"
                )
            yield TrainingExample(
                example_id=record.get("transition_id") or record["problem_id"],
                problem_id=record["problem_id"],
                messages=list(messages),
                category=record.get("category", ""),
                step_index=int(record.get("step_index", 0) or 0),
                total_steps=int(record.get("total_steps", record.get("step_count", 0)) or 0),
                weight=_weight_for(row, weighting),
                epoch=epoch_plan.epoch,
                position=position,
            )


def order_diagnostics(plan: ViewPlan) -> dict[str, Any]:
    """Evidence that the requested order really was produced.

    For the shuffled condition we report how often two adjacent rows come from
    the same source problem. The dataset ships with zero adjacent same-problem
    rows; the training view must not reintroduce them.
    """
    out: dict[str, Any] = {}
    for epoch_plan in plan.epochs:
        rows = epoch_plan.rows
        adjacent_same_problem = sum(1 for a, b in zip(rows, rows[1:]) if a.problem_id == b.problem_id)
        consecutive_steps = sum(
            1
            for a, b in zip(rows, rows[1:])
            if a.problem_id == b.problem_id and b.step_index == a.step_index + 1
        )
        mean_gap = None
        positions: dict[str, list[int]] = defaultdict(list)
        for position, row in enumerate(rows):
            positions[row.problem_id].append(position)
        gaps = [b - a for values in positions.values() for a, b in zip(values, values[1:])]
        if gaps:
            mean_gap = sum(gaps) / len(gaps)
        out[f"epoch_{epoch_plan.epoch}"] = {
            "rows": len(rows),
            "adjacent_same_problem": adjacent_same_problem,
            "adjacent_consecutive_steps": consecutive_steps,
            "mean_gap_between_same_problem_rows": mean_gap,
        }
    return out
