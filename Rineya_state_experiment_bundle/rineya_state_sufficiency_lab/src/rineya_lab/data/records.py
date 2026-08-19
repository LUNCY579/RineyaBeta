"""Typed views over the two dataset variants.

These wrappers are deliberately thin: they never mutate the underlying row and
never invent fields. Anything not present in the source JSON is ``None`` so that
a schema drift shows up as a missing value rather than a plausible default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_FULL_FIELDS = (
    "variant",
    "problem_id",
    "split",
    "domain",
    "category",
    "problem",
    "answer",
    "full_response",
    "reasoning_steps",
    "compact_state_trace",
    "step_count",
    "messages",
)

REQUIRED_TRANSITION_FIELDS = (
    "variant",
    "transition_id",
    "problem_id",
    "split",
    "domain",
    "category",
    "step_index",
    "total_steps",
    "problem",
    "current_state",
    "target_step",
    "next_state",
    "done",
    "messages",
    "stateful_rollout_messages",
)


class RecordError(ValueError):
    """Raised when a dataset row does not satisfy the documented contract."""


@dataclass(frozen=True)
class FullRecord:
    """One problem with one uninterrupted reasoning target (variant A)."""

    raw: dict[str, Any]

    def __post_init__(self) -> None:
        missing = [f for f in REQUIRED_FULL_FIELDS if f not in self.raw]
        if missing:
            raise RecordError(f"full_context record missing fields: {missing}")

    @property
    def problem_id(self) -> str:
        return self.raw["problem_id"]

    @property
    def split(self) -> str:
        return self.raw["split"]

    @property
    def category(self) -> str:
        return self.raw["category"]

    @property
    def domain(self) -> str:
        return self.raw["domain"]

    @property
    def difficulty(self) -> int | None:
        return self.raw.get("difficulty")

    @property
    def answer_type(self) -> str | None:
        return self.raw.get("answer_type")

    @property
    def problem(self) -> str:
        return self.raw["problem"]

    @property
    def answer(self) -> str:
        return self.raw["answer"]

    @property
    def full_response(self) -> str:
        return self.raw["full_response"]

    @property
    def reasoning_steps(self) -> list[str]:
        return list(self.raw["reasoning_steps"])

    @property
    def compact_state_trace(self) -> list[str]:
        return list(self.raw["compact_state_trace"])

    @property
    def initial_state(self) -> str:
        return self.compact_state_trace[0]

    @property
    def step_count(self) -> int:
        return int(self.raw["step_count"])

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self.raw["messages"])

    def history_prefix(self, step_index: int) -> list[str]:
        """The completed reasoning steps strictly before 1-based ``step_index``."""
        if step_index < 1:
            raise RecordError(f"step_index must be >= 1, got {step_index}")
        return self.reasoning_steps[: step_index - 1]


@dataclass(frozen=True)
class TransitionRecord:
    """One compact-state transition (variant B).

    ``current_state`` is derived from the complete canonical trajectory but, by
    the dataset's causal safeguard, contains only information available at this
    step. Nothing downstream of ``step_index`` may appear in it.
    """

    raw: dict[str, Any]

    def __post_init__(self) -> None:
        missing = [f for f in REQUIRED_TRANSITION_FIELDS if f not in self.raw]
        if missing:
            raise RecordError(f"split_transition record missing fields: {missing}")

    @property
    def transition_id(self) -> str:
        return self.raw["transition_id"]

    @property
    def problem_id(self) -> str:
        return self.raw["problem_id"]

    @property
    def split(self) -> str:
        return self.raw["split"]

    @property
    def category(self) -> str:
        return self.raw["category"]

    @property
    def domain(self) -> str:
        return self.raw["domain"]

    @property
    def answer_type(self) -> str | None:
        return self.raw.get("answer_type")

    @property
    def step_index(self) -> int:
        return int(self.raw["step_index"])

    @property
    def total_steps(self) -> int:
        return int(self.raw["total_steps"])

    @property
    def problem(self) -> str:
        return self.raw["problem"]

    @property
    def current_state(self) -> str:
        return self.raw["current_state"]

    @property
    def target_step(self) -> str:
        return self.raw["target_step"]

    @property
    def next_state(self) -> str:
        return self.raw["next_state"]

    @property
    def done(self) -> bool:
        return bool(self.raw["done"])

    @property
    def final_answer(self) -> str | None:
        return self.raw.get("final_answer")

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self.raw["messages"])

    @property
    def stateful_rollout_messages(self) -> list[dict[str, str]]:
        return list(self.raw["stateful_rollout_messages"])

    @property
    def stateful_rollout_target(self) -> str | None:
        return self.raw.get("stateful_rollout_target")

    def messages_for_view(self, view: str) -> list[dict[str, str]]:
        if view == "split_stateful":
            return self.stateful_rollout_messages
        if view == "split_next_step":
            return self.messages
        raise RecordError(f"transition record has no message view named {view!r}")


@dataclass
class Trajectory:
    """A full-context record paired with its ordered transition chain."""

    full: FullRecord
    transitions: list[TransitionRecord]

    @property
    def problem_id(self) -> str:
        return self.full.problem_id

    @property
    def length(self) -> int:
        return len(self.transitions)

    def state_at(self, step_index: int) -> str:
        """Canonical compact state entering 1-based ``step_index``."""
        return self.transitions[step_index - 1].current_state

    def check_chain(self) -> list[str]:
        """Return a list of contract violations; empty means the chain is sound."""
        problems: list[str] = []
        ordered = sorted(self.transitions, key=lambda t: t.step_index)
        indices = [t.step_index for t in ordered]
        if indices != list(range(1, len(ordered) + 1)):
            problems.append(f"{self.problem_id}: step indices are not contiguous 1..n: {indices}")
        if ordered and ordered[0].current_state != self.full.initial_state:
            problems.append(f"{self.problem_id}: first transition state does not match compact_state_trace[0]")
        for previous, following in zip(ordered, ordered[1:]):
            if previous.next_state != following.current_state:
                problems.append(
                    f"{self.problem_id}: chain break between step {previous.step_index} and {following.step_index}"
                )
        for transition in ordered[:-1]:
            if transition.done:
                problems.append(f"{self.problem_id}: non-final step {transition.step_index} has done=true")
        if ordered:
            last = ordered[-1]
            if not last.done:
                problems.append(f"{self.problem_id}: final step {last.step_index} has done=false")
            if last.final_answer != self.full.answer:
                problems.append(
                    f"{self.problem_id}: final_answer {last.final_answer!r} != full answer {self.full.answer!r}"
                )
        if ordered and len(ordered) != self.full.step_count:
            problems.append(f"{self.problem_id}: {len(ordered)} transitions but step_count={self.full.step_count}")
        if [t.target_step for t in ordered] != self.full.reasoning_steps:
            problems.append(f"{self.problem_id}: transition targets do not reproduce reasoning_steps")
        return problems
