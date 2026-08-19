"""Prompt construction for every evaluation protocol.

The transition prompts reproduce the dataset's training prompts *exactly* so
that the oracle-state protocol is in-distribution for the SPLIT adapter. The
perturbation protocols are then defined as minimal edits of that same string:
each one removes or replaces exactly one component, so a drop in accuracy is
attributable to that component and not to a reworded instruction.

Any prompt that is knowingly out of distribution for one arm is marked with
``in_distribution_for`` so the report can say so rather than implying a fair
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NEXT_STEP_INSTRUCTION = "Give exactly the next correct reasoning step. Do not replay the omitted reasoning history."
STATEFUL_INSTRUCTION = "Return a JSON object with next_reasoning_step, updated_compact_state, done, and final_answer."
DIRECT_ANSWER_INSTRUCTION = "Give only the final answer, with no explanation."


@dataclass
class BuiltPrompt:
    messages: list[dict[str, str]]
    protocol: str
    expects_json: bool
    in_distribution_for: tuple[str, ...]
    meta: dict[str, Any]

    @property
    def text(self) -> str:
        return self.messages[0]["content"]


def _user(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def transition_prompt_text(
    problem: str | None,
    state: str | None,
    *,
    stateful: bool,
    history: list[str] | None = None,
) -> str:
    """The canonical transition prompt, with optional components omitted."""
    blocks: list[str] = []
    if problem is not None:
        blocks.append(f"Original problem:\n{problem}")
    if history:
        numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(history, 1))
        blocks.append(f"Reasoning so far:\n{numbered}")
    if state is not None:
        blocks.append(f"Current compact state:\n{state}")
    instruction = NEXT_STEP_INSTRUCTION + (f"\n{STATEFUL_INSTRUCTION}" if stateful else "")
    blocks.append(instruction)
    return "\n\n".join(blocks)


def full_direct(problem: str) -> BuiltPrompt:
    """FULL's native prompt: the bare problem, exactly as trained."""
    return BuiltPrompt(
        messages=_user(problem),
        protocol="full_direct",
        expects_json=False,
        in_distribution_for=("FULL",),
        meta={"prompt_style": "bare_problem"},
    )


def direct_answer(problem: str) -> BuiltPrompt:
    return BuiltPrompt(
        messages=_user(f"{problem}\n\n{DIRECT_ANSWER_INSTRUCTION}"),
        protocol="direct_answer",
        expects_json=False,
        in_distribution_for=(),
        meta={"prompt_style": "answer_only", "note": "out of distribution for both arms; diagnostic only"},
    )


def oracle_transition(problem: str, state: str, *, stateful: bool) -> BuiltPrompt:
    return BuiltPrompt(
        messages=_user(transition_prompt_text(problem, state, stateful=stateful)),
        protocol="oracle_transition",
        expects_json=stateful,
        in_distribution_for=("SPLIT_STATEFUL",) if stateful else ("SPLIT_NEXT_STEP",),
        meta={"prompt_style": "problem_plus_state"},
    )


def full_history_transition(problem: str, history: list[str], *, stateful: bool) -> BuiltPrompt:
    """The state-sufficiency contrast: complete history instead of a compact state."""
    return BuiltPrompt(
        messages=_user(transition_prompt_text(problem, None, stateful=stateful, history=history)),
        protocol="full_history_transition",
        expects_json=stateful,
        in_distribution_for=(),
        meta={
            "prompt_style": "problem_plus_complete_history",
            "history_steps": len(history),
            "note": "tests whether the compact state loses information the full history retains",
        },
    )


def no_state_transition(problem: str, *, stateful: bool) -> BuiltPrompt:
    return BuiltPrompt(
        messages=_user(transition_prompt_text(problem, None, stateful=stateful)),
        protocol="no_state_transition",
        expects_json=stateful,
        in_distribution_for=(),
        meta={"prompt_style": "problem_only", "note": "lower bound: the state carries no information if this ties"},
    )


def last_state_only(state: str, *, stateful: bool) -> BuiltPrompt:
    return BuiltPrompt(
        messages=_user(transition_prompt_text(None, state, stateful=stateful)),
        protocol="last_state_only",
        expects_json=stateful,
        in_distribution_for=(),
        meta={"prompt_style": "state_only", "note": "tests whether the state alone is sufficient without the problem"},
    )


def shuffled_state(problem: str, wrong_state: str, *, stateful: bool, source_step: int) -> BuiltPrompt:
    return BuiltPrompt(
        messages=_user(transition_prompt_text(problem, wrong_state, stateful=stateful)),
        protocol="shuffled_state",
        expects_json=stateful,
        in_distribution_for=(),
        meta={
            "prompt_style": "problem_plus_wrong_step_state",
            "source_step_index": source_step,
            "note": "state comes from a different step of the same problem",
        },
    )


def distractor_state(problem: str, foreign_state: str, *, stateful: bool, source_problem_id: str) -> BuiltPrompt:
    return BuiltPrompt(
        messages=_user(transition_prompt_text(problem, foreign_state, stateful=stateful)),
        protocol="distractor_state",
        expects_json=stateful,
        in_distribution_for=(),
        meta={
            "prompt_style": "problem_plus_foreign_state",
            "source_problem_id": source_problem_id,
            "note": "state comes from an unrelated problem; measures blind following vs detection",
        },
    )


def rollout_step(problem: str, state: str, *, stateful: bool = True) -> BuiltPrompt:
    return BuiltPrompt(
        messages=_user(transition_prompt_text(problem, state, stateful=stateful)),
        protocol="generated_rollout",
        expects_json=stateful,
        in_distribution_for=("SPLIT_STATEFUL",),
        meta={"prompt_style": "problem_plus_generated_state"},
    )
