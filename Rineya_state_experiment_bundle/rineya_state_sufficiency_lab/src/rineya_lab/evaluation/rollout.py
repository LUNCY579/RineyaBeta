"""Autonomous generated-state rollout.

This is the protocol that separates the two conclusions the study must not
conflate:

    "The compact state is sufficient when a correct state is supplied."
        -- measured by the oracle-state transition protocol.

    "The model can construct and maintain sufficient states on its own."
        -- measured here, where every step after the first consumes a state the
           model itself produced.

The gap between the two is the free-rollout gap, and it is the primary evidence
about error propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rineya_lab.config import RolloutConfig, ScoringConfig
from rineya_lab.data.records import Trajectory
from rineya_lab.evaluation import failures
from rineya_lab.evaluation.backends import Backend
from rineya_lab.evaluation.parsing import TransitionParse, parse_transition_response
from rineya_lab.evaluation.prompts import rollout_step
from rineya_lab.evaluation.scoring import normalize_text, states_equivalent


@dataclass
class RolloutStep:
    depth: int
    state_in: str
    prompt: str
    raw_response: str | None
    parse: TransitionParse
    state_out: str | None
    done: bool | None
    canonical_state_out: str | None = None
    canonical_step: str | None = None
    state_matches_canonical: bool | None = None
    backend_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "state_in": self.state_in,
            "raw_response": self.raw_response,
            "parse": self.parse.as_dict(),
            "state_out": self.state_out,
            "done": self.done,
            "canonical_state_out": self.canonical_state_out,
            "canonical_step": self.canonical_step,
            "state_matches_canonical": self.state_matches_canonical,
            "backend_error": self.backend_error,
        }


@dataclass
class RolloutResult:
    problem_id: str
    steps: list[RolloutStep] = field(default_factory=list)
    final_answer: str | None = None
    terminated: bool = False
    termination_reason: str = "unknown"
    control_failure: str | None = None
    first_error_depth: int | None = None
    canonical_length: int = 0
    max_steps_allowed: int = 0
    generated_tokens: int = 0

    @property
    def steps_taken(self) -> int:
        return len(self.steps)

    def as_dict(self, *, include_steps: bool = True) -> dict[str, Any]:
        payload = {
            "problem_id": self.problem_id,
            "steps_taken": self.steps_taken,
            "final_answer": self.final_answer,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "control_failure": self.control_failure,
            "first_error_depth": self.first_error_depth,
            "canonical_length": self.canonical_length,
            "max_steps_allowed": self.max_steps_allowed,
            "generated_tokens": self.generated_tokens,
            "step_overrun": max(0, self.steps_taken - self.canonical_length),
        }
        if include_steps:
            payload["steps"] = [step.as_dict() for step in self.steps]
        return payload


def max_steps_for(trajectory: Trajectory, cfg: RolloutConfig) -> int:
    """Step budget for one rollout.

    Defaults to the constant ``max_steps_absolute``, deliberately.

    ``use_dataset_step_bound`` derives the budget from ``trajectory.length``,
    which is *gold* information. Enabling it gives the SPLIT arm a per-problem
    hint the FULL arm never receives -- 5 steps on a 2-step problem, 12 on a
    9-step one -- so it tells the model roughly how long the problem is and
    makes a premature ``done`` likelier to be right. ``full_direct`` gets one
    fixed-length call and no such hint. It was the default, and it is not
    defensible on the primary endpoint.

    The constant is generous: 16 against a dataset whose longest trajectory is
    9 steps, so nothing correct is truncated and every problem is judged under
    the same budget. Set ``use_dataset_step_bound: true`` only as a declared
    secondary analysis, and report it as such.

    ``oracle_trajectory`` is unaffected -- there the step count *is* the gold
    trajectory by construction, which is what teacher forcing means.
    """
    if cfg.use_dataset_step_bound and trajectory is not None:
        return max(2, min(cfg.max_steps_absolute, trajectory.length + cfg.max_steps_margin))
    return cfg.max_steps_absolute


@dataclass
class _LiveRollout:
    """Mutable per-problem state for one in-flight rollout."""

    trajectory: Trajectory
    result: RolloutResult
    state: str
    budget: int
    seen_states: set[str] = field(default_factory=set)
    unchanged_streak: int = 0
    invalid_streak: int = 0
    active: bool = True
    depth: int = 0

    @classmethod
    def start(cls, trajectory: Trajectory, cfg: RolloutConfig) -> _LiveRollout:
        budget = max_steps_for(trajectory, cfg)
        state = trajectory.full.initial_state
        return cls(
            trajectory=trajectory,
            result=RolloutResult(
                problem_id=trajectory.problem_id,
                canonical_length=trajectory.length,
                max_steps_allowed=budget,
            ),
            state=state,
            budget=budget,
            seen_states={normalize_text(state)},
        )

    def canonical_at(self, depth: int):
        return self.trajectory.transitions[depth - 1] if depth <= self.trajectory.length else None

    def oracle_context(self, depth: int) -> dict[str, Any]:
        canonical = self.canonical_at(depth)
        return {
            "problem_id": self.trajectory.problem_id,
            "transition_id": canonical.transition_id if canonical else f"{self.trajectory.problem_id}::overrun",
            "current_state": self.state,
            "target_step": canonical.target_step if canonical else "No further step is required.",
            "next_state": canonical.next_state if canonical else self.state,
            "done": canonical.done if canonical else True,
            "final_answer": self.trajectory.full.answer,
            "answer": self.trajectory.full.answer,
        }

    def finish(self, reason: str, control_failure: str | None, *, terminated: bool = True) -> None:
        self.result.terminated = terminated
        self.result.termination_reason = reason
        self.result.control_failure = control_failure
        self.active = False


def _apply_step(
    live: _LiveRollout,
    depth: int,
    prompt_text: str,
    generated,
    cfg: RolloutConfig,
    scoring_cfg: ScoringConfig,
) -> None:
    """Consume one generation for one rollout, updating state and detectors.

    Shared by the sequential and batched drivers so both cannot drift apart.
    """
    parse = parse_transition_response(generated.text)
    canonical = live.canonical_at(depth)
    live.result.generated_tokens += generated.completion_tokens

    step = RolloutStep(
        depth=depth,
        state_in=live.state,
        prompt=prompt_text,
        raw_response=generated.text,
        parse=parse,
        state_out=parse.updated_compact_state,
        done=parse.done,
        canonical_state_out=canonical.next_state if canonical else None,
        canonical_step=canonical.target_step if canonical else None,
        backend_error=generated.error,
    )
    if canonical is not None and parse.updated_compact_state is not None:
        step.state_matches_canonical = states_equivalent(
            parse.updated_compact_state, canonical.next_state, scoring_cfg
        )
        if not step.state_matches_canonical and live.result.first_error_depth is None:
            live.result.first_error_depth = depth
    live.result.steps.append(step)

    if generated.error is not None:
        live.finish("backend_failure", failures.BACKEND_FAILURE)
        return

    # Unusable for continuing: no next state and no completion claim, even if prose
    # could be salvaged from the response.
    if parse.updated_compact_state is None and not parse.done:
        live.invalid_streak += 1
        if cfg.stop_on_invalid or live.invalid_streak > cfg.max_invalid_retries:
            live.finish(
                "invalid_output",
                failures.classify_from_status(parse.status, correct=False, default=failures.MALFORMED_STRUCTURE),
            )
            return
        if cfg.carry_last_valid_state:
            return
    else:
        live.invalid_streak = 0

    if parse.done:
        live.result.final_answer = parse.final_answer
        if parse.final_answer:
            live.finish("done", None)
        else:
            live.finish("premature_completion", failures.PREMATURE_COMPLETION)
        return

    new_state = parse.updated_compact_state if parse.updated_compact_state is not None else live.state
    normalized = normalize_text(new_state)

    if normalized == normalize_text(live.state):
        live.unchanged_streak += 1
        if live.unchanged_streak >= cfg.stagnation_patience:
            live.finish("stagnation", failures.STAGNATION)
            return
    else:
        live.unchanged_streak = 0
        if cfg.loop_detection and normalized in live.seen_states:
            live.finish("loop", failures.LOOP)
            return

    live.seen_states.add(normalized)
    live.state = new_state

    if depth >= live.budget:
        live.finish("max_steps", failures.NONTERMINATION, terminated=False)


def run_rollouts_batched(
    backend: Backend,
    trajectories: list[Trajectory],
    cfg: RolloutConfig,
    *,
    scoring_cfg: ScoringConfig | None = None,
    stateful: bool = True,
    batch_size: int = 8,
    generation_overrides: dict[str, Any] | None = None,
) -> list[RolloutResult]:
    """Run many autonomous rollouts step-synchronously.

    Every rollout still consumes only its own generated states -- batching groups
    the *depth-t* calls of different problems, never mixes their states. Rollouts
    that terminate early drop out, so the batch shrinks as the run proceeds.
    """
    scoring_cfg = scoring_cfg or ScoringConfig()
    lives = [_LiveRollout.start(trajectory, cfg) for trajectory in trajectories]
    batch_size = max(1, batch_size)

    depth = 0
    while any(live.active for live in lives):
        depth += 1
        pending = [live for live in lives if live.active and depth <= live.budget]
        for live in lives:
            if live.active and depth > live.budget:
                live.finish("max_steps", failures.NONTERMINATION, terminated=False)
        if not pending:
            break

        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            built = [rollout_step(live.trajectory.full.problem, live.state, stateful=stateful) for live in chunk]
            oracles = [live.oracle_context(depth) for live in chunk]
            generations = backend.generate_batch(
                [item.messages for item in built], oracles=oracles, **(generation_overrides or {})
            )
            for live, item, generated in zip(chunk, built, generations):
                _apply_step(live, depth, item.text, generated, cfg, scoring_cfg)

    return [live.result for live in lives]


def run_rollout(
    backend: Backend,
    trajectory: Trajectory,
    cfg: RolloutConfig,
    *,
    scoring_cfg: ScoringConfig | None = None,
    stateful: bool = True,
    generation_overrides: dict[str, Any] | None = None,
) -> RolloutResult:
    """Run one autonomous rollout from the canonical initial state."""
    return run_rollouts_batched(
        backend,
        [trajectory],
        cfg,
        scoring_cfg=scoring_cfg,
        stateful=stateful,
        batch_size=1,
        generation_overrides=generation_overrides,
    )[0]


def run_oracle_trajectories_batched(
    backend: Backend,
    trajectories: list[Trajectory],
    cfg: RolloutConfig,
    *,
    stateful: bool = True,
    batch_size: int = 8,
    generation_overrides: dict[str, Any] | None = None,
) -> list[RolloutResult]:
    """Teacher-forced walks over the canonical states.

    Every prompt is known in advance -- no step depends on a previous generation --
    so the whole protocol is one flat batched pass.
    """
    requests: list[tuple[int, int, Any, Any]] = []
    results = [
        RolloutResult(
            problem_id=trajectory.problem_id,
            canonical_length=trajectory.length,
            max_steps_allowed=trajectory.length,
        )
        for trajectory in trajectories
    ]
    for index, trajectory in enumerate(trajectories):
        for depth, canonical in enumerate(trajectory.transitions, start=1):
            built = rollout_step(trajectory.full.problem, canonical.current_state, stateful=stateful)
            requests.append((index, depth, canonical, built))

    batch_size = max(1, batch_size)
    for start in range(0, len(requests), batch_size):
        chunk = requests[start : start + batch_size]
        oracles = [
            {
                "problem_id": trajectories[index].problem_id,
                "transition_id": canonical.transition_id,
                "current_state": canonical.current_state,
                "target_step": canonical.target_step,
                "next_state": canonical.next_state,
                "done": canonical.done,
                "final_answer": trajectories[index].full.answer,
                "answer": trajectories[index].full.answer,
            }
            for index, _, canonical, _ in chunk
        ]
        generations = backend.generate_batch(
            [built.messages for _, _, _, built in chunk], oracles=oracles, **(generation_overrides or {})
        )
        for (index, depth, canonical, built), generated in zip(chunk, generations):
            parse = parse_transition_response(generated.text)
            result = results[index]
            result.generated_tokens += generated.completion_tokens
            result.steps.append(
                RolloutStep(
                    depth=depth,
                    state_in=canonical.current_state,
                    prompt=built.text,
                    raw_response=generated.text,
                    parse=parse,
                    state_out=parse.updated_compact_state,
                    done=parse.done,
                    canonical_state_out=canonical.next_state,
                    canonical_step=canonical.target_step,
                    backend_error=generated.error,
                )
            )

    for result in results:
        result.steps.sort(key=lambda step: step.depth)
        # The declared answer is the one from the first step that claims completion;
        # teacher forcing never stops early, so later steps cannot overwrite it.
        for step in result.steps:
            if step.parse.done and step.parse.final_answer:
                result.final_answer = step.parse.final_answer
                result.terminated = True
                result.termination_reason = "done"
                break
        else:
            last = result.steps[-1] if result.steps else None
            result.final_answer = last.parse.final_answer if last else None
            result.terminated = bool(last and last.parse.done)
            result.termination_reason = "canonical_chain_exhausted"
            if not result.final_answer:
                result.control_failure = failures.NO_ANSWER_FOUND
    return results


def run_oracle_trajectory(
    backend: Backend,
    trajectory: Trajectory,
    cfg: RolloutConfig,
    *,
    stateful: bool = True,
    generation_overrides: dict[str, Any] | None = None,
) -> RolloutResult:
    """Walk the canonical states end-to-end (teacher forcing) and take the final answer.

    Contrasted against ``run_rollout``, this isolates how much capability is lost
    purely to consuming self-generated states.
    """
    return run_oracle_trajectories_batched(
        backend,
        [trajectory],
        cfg,
        stateful=stateful,
        batch_size=1,
        generation_overrides=generation_overrides,
    )[0]
