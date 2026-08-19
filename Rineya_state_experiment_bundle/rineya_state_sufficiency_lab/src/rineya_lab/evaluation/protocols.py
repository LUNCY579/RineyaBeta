"""Protocol implementations.

Each protocol answers one question and only one. The four questions the runbook
insists on keeping separate are:

1. local transition competence   -> ``oracle_transition``
2. state sufficiency             -> ``oracle_transition`` vs ``full_history_transition``
                                    (with ``no_state_transition`` / ``last_state_only``
                                    as the lower-bound controls)
3. error propagation             -> ``oracle_trajectory`` vs ``generated_rollout``
4. end-to-end capability         -> ``full_direct`` / ``generated_rollout``

Every record keeps the raw response, the extraction route and the scorer used.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from rineya_lab.config import LabConfig
from rineya_lab.data.records import Trajectory, TransitionRecord
from rineya_lab.evaluation import failures, prompts
from rineya_lab.evaluation.backends import Backend
from rineya_lab.evaluation.parsing import (
    AnswerParse,
    TransitionParse,
    extract_final_answer,
    lenient_answer_of,
    parse_next_step_response,
    parse_transition_response,
)
from rineya_lab.evaluation.rollout import (
    RolloutResult,
    run_oracle_trajectories_batched,
    run_rollouts_batched,
)
from rineya_lab.evaluation.scoring import ScoreResult, score_answer, states_equivalent
from rineya_lab.utils.hashing import stable_shuffle_key
from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("evaluation.protocols")

ANSWER_PROTOCOLS = frozenset({"full_direct", "direct_answer", "oracle_trajectory", "generated_rollout"})
TRANSITION_PROTOCOLS = frozenset(
    {
        "oracle_transition",
        "full_history_transition",
        "no_state_transition",
        "last_state_only",
        "shuffled_state",
        "distractor_state",
    }
)


@dataclass
class EvalRecord:
    """One scored evaluation item. Serialised verbatim to ``records.jsonl``."""

    record_id: str
    protocol: str
    problem_id: str
    category: str
    domain: str
    answer_type: str | None
    total_steps: int
    transition_id: str | None = None
    step_index: int | None = None
    prompt: str = ""
    raw_response: str | None = None
    prediction: str | None = None
    gold: str | None = None
    strict_correct: bool = False
    lenient_correct: bool = False
    extraction_route: str = "none"
    parse_status: str = "not_found"
    scorer: str | None = None
    failure_type: str = failures.NO_ANSWER_FOUND
    components: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_raw: bool = True) -> dict[str, Any]:
        payload = {
            "record_id": self.record_id,
            "protocol": self.protocol,
            "problem_id": self.problem_id,
            "transition_id": self.transition_id,
            "category": self.category,
            "domain": self.domain,
            "answer_type": self.answer_type,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "prediction": self.prediction,
            "gold": self.gold,
            "strict_correct": self.strict_correct,
            "lenient_correct": self.lenient_correct,
            "extraction_route": self.extraction_route,
            "parse_status": self.parse_status,
            "scorer": self.scorer,
            "failure_type": self.failure_type,
            "components": self.components,
            "meta": self.meta,
        }
        if include_raw:
            payload["prompt"] = self.prompt
            payload["raw_response"] = self.raw_response
        return payload


@dataclass
class EvalSet:
    """The deterministic set of problems every protocol is evaluated on.

    All protocols share one problem list so results pair cleanly by
    ``problem_id`` in the comparison stage.
    """

    trajectories: dict[str, Trajectory]
    order: list[str]

    def __len__(self) -> int:
        return len(self.order)

    def iter_trajectories(self) -> Iterator[Trajectory]:
        for problem_id in self.order:
            yield self.trajectories[problem_id]

    def iter_transitions(self) -> Iterator[tuple[Trajectory, TransitionRecord]]:
        for trajectory in self.iter_trajectories():
            for transition in trajectory.transitions:
                yield trajectory, transition


def build_eval_set(dataset, cfg: LabConfig) -> EvalSet:
    split = cfg.evaluation.split
    trajectories = dataset.trajectories(split)
    if cfg.evaluation.categories:
        allowed = set(cfg.evaluation.categories)
        trajectories = {k: v for k, v in trajectories.items() if v.full.category in allowed}
    order = sorted(trajectories)
    if cfg.evaluation.shuffle_eval_order:
        # Keyed on evaluation.subset_seed, NOT experiment.seed: every training seed
        # must be scored on the identical held-out subset or the paired comparison
        # silently degrades to the intersection of two different samples.
        order.sort(key=lambda pid: stable_shuffle_key(pid, f"eval|{cfg.evaluation.subset_seed}|{split}"))
    if cfg.evaluation.limit is not None:
        order = order[: cfg.evaluation.limit]
    LOGGER.info("Evaluation set: %d problems from split %s", len(order), split)
    return EvalSet(trajectories={pid: trajectories[pid] for pid in order}, order=order)


# --------------------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------------------


def _score_answer_record(
    record: EvalRecord,
    parse: AnswerParse,
    gold: str,
    cfg: LabConfig,
) -> EvalRecord:
    scoring_cfg = cfg.evaluation.scoring
    record.extraction_route = parse.route
    record.parse_status = parse.status
    record.prediction = parse.answer
    record.gold = gold

    strict_result = ScoreResult(False, "none")
    if parse.strict_ok and parse.answer is not None:
        strict_result = score_answer(parse.answer, gold, answer_type=record.answer_type, cfg=scoring_cfg)
    record.strict_correct = bool(strict_result.correct)
    record.scorer = strict_result.scorer

    if scoring_cfg.lenient_track_enabled:
        lenient_value = lenient_answer_of(parse)
        if parse.lenient_ok and lenient_value is not None:
            lenient_result = score_answer(lenient_value, gold, answer_type=record.answer_type, cfg=scoring_cfg)
            record.lenient_correct = bool(lenient_result.correct)
            record.components["lenient"] = lenient_result.as_dict()
            if record.scorer in (None, "none"):
                record.scorer = lenient_result.scorer

    record.components["strict"] = strict_result.as_dict()
    record.components["parse"] = parse.as_dict()
    record.failure_type = failures.classify_from_status(
        parse.status, correct=record.strict_correct, default=failures.WRONG_ANSWER
    )
    return record


def _score_transition_record(
    record: EvalRecord,
    parse: TransitionParse,
    canonical: TransitionRecord,
    cfg: LabConfig,
    *,
    stateful: bool,
) -> EvalRecord:
    scoring_cfg = cfg.evaluation.scoring
    record.extraction_route = parse.route
    record.parse_status = parse.status
    record.prediction = parse.next_reasoning_step
    record.gold = canonical.target_step

    step_match = states_equivalent(parse.next_reasoning_step, canonical.target_step, scoring_cfg)
    state_match = states_equivalent(parse.updated_compact_state, canonical.next_state, scoring_cfg)
    done_correct = parse.done == canonical.done if parse.done is not None else False
    answer_equivalent = None
    if canonical.done and canonical.final_answer:
        answer_result = score_answer(
            parse.final_answer, canonical.final_answer, answer_type=record.answer_type, cfg=scoring_cfg
        )
        answer_equivalent = bool(answer_result.correct)
        record.components["final_answer_score"] = answer_result.as_dict()

    record.components.update(
        {
            "step_match": step_match,
            "updated_state_valid": state_match,
            "done_correct": done_correct,
            "answer_equivalent": answer_equivalent,
            "predicted_state": parse.updated_compact_state,
            "canonical_state": canonical.next_state,
            "predicted_done": parse.done,
            "canonical_done": canonical.done,
            "parse": parse.as_dict(),
        }
    )

    if stateful:
        strict = bool(parse.strict_ok and step_match and state_match and done_correct)
        lenient = bool(parse.lenient_ok and step_match)
    else:
        # The next-step-only contract asks for a step and nothing else: no state to
        # validate and no completion flag to classify.
        strict = bool(parse.strict_ok and step_match)
        lenient = bool(parse.lenient_ok and step_match)

    record.strict_correct = strict
    record.lenient_correct = lenient or strict
    record.scorer = "transition_match"

    if strict:
        record.failure_type = failures.CORRECT
    elif parse.status in ("empty", "refusal", "malformed", "missing_fields", "invalid_completion_flag",
                          "backend_failure", "not_found", "overlong"):
        record.failure_type = failures.classify_from_status(parse.status, correct=False,
                                                            default=failures.WRONG_TRANSITION)
    elif not step_match:
        record.failure_type = failures.WRONG_TRANSITION
    elif stateful and not state_match:
        record.failure_type = failures.STATE_DRIFT
    elif stateful and not done_correct:
        record.failure_type = failures.INVALID_COMPLETION_FLAG
    else:  # pragma: no cover - defensive
        record.failure_type = failures.WRONG_TRANSITION
    return record


def _base_record(protocol: str, trajectory: Trajectory, transition: TransitionRecord | None = None) -> EvalRecord:
    full = trajectory.full
    return EvalRecord(
        record_id=f"{protocol}::{transition.transition_id if transition else full.problem_id}",
        protocol=protocol,
        problem_id=full.problem_id,
        category=full.category,
        domain=full.domain,
        answer_type=full.answer_type,
        total_steps=full.step_count,
        transition_id=transition.transition_id if transition else None,
        step_index=transition.step_index if transition else None,
    )


# --------------------------------------------------------------------------------------
# Answer-level protocols
# --------------------------------------------------------------------------------------


def _generate_in_batches(backend: Backend, requests: list[tuple[Any, Any, dict[str, Any]]], cfg: LabConfig):
    """Run (record, built_prompt, oracle) triples through the backend in batches.

    Batching is what makes the full DEV suite affordable: it is ~80,500 generations
    per adapter, and one-at-a-time decoding leaves the accelerator mostly idle.
    """
    batch_size = max(1, cfg.evaluation.generation.batch_size)
    for start in range(0, len(requests), batch_size):
        chunk = requests[start : start + batch_size]
        generations = backend.generate_batch(
            [built.messages for _, built, _ in chunk],
            oracles=[oracle for _, _, oracle in chunk],
        )
        for (record, built, _), generated in zip(chunk, generations):
            yield record, built, generated


def run_answer_protocol(backend: Backend, eval_set: EvalSet, cfg: LabConfig, protocol: str) -> Iterator[EvalRecord]:
    stateful = cfg.experiment.condition != "SPLIT_NEXT_STEP"
    strip_thinking = cfg.evaluation.strip_reasoning_blocks
    trajectories = list(eval_set.iter_trajectories())

    if protocol in ("full_direct", "direct_answer"):
        requests = []
        for trajectory in trajectories:
            full = trajectory.full
            record = _base_record(protocol, trajectory)
            built = (
                prompts.full_direct(full.problem)
                if protocol == "full_direct"
                else prompts.direct_answer(full.problem)
            )
            oracle = {
                "problem_id": full.problem_id,
                "answer": full.answer,
                "full_response": full.full_response,
                "done": True,
                "final_answer": full.answer,
            }
            requests.append((record, built, oracle))

        for record, built, generated in _generate_in_batches(backend, requests, cfg):
            trajectory = eval_set.trajectories[record.problem_id]
            record.prompt = built.text
            record.raw_response = generated.text
            record.meta.update({"in_distribution_for": list(built.in_distribution_for), **built.meta})
            if generated.error:
                record.failure_type = failures.BACKEND_FAILURE
                record.parse_status = "backend_failure"
                record.meta["backend_error"] = generated.error
                yield record
                continue
            parse = extract_final_answer(generated.text, strip_thinking=strip_thinking)
            record.meta["completion_tokens"] = generated.completion_tokens
            yield _score_answer_record(record, parse, trajectory.full.answer, cfg)
        return

    # Trajectory-level protocols: batched across problems, never across states.
    batch_size = max(1, cfg.evaluation.generation.batch_size)
    if protocol == "oracle_trajectory":
        results = run_oracle_trajectories_batched(
            backend, trajectories, cfg.evaluation.rollout, stateful=stateful, batch_size=batch_size
        )
    else:
        results = run_rollouts_batched(
            backend,
            trajectories,
            cfg.evaluation.rollout,
            scoring_cfg=cfg.evaluation.scoring,
            stateful=stateful,
            batch_size=batch_size,
        )
    for trajectory, result in zip(trajectories, results):
        record = _base_record(protocol, trajectory)
        yield _score_rollout_record(record, result, trajectory, cfg, protocol)


def _score_rollout_record(
    record: EvalRecord,
    result: RolloutResult,
    trajectory: Trajectory,
    cfg: LabConfig,
    protocol: str,
) -> EvalRecord:
    gold = trajectory.full.answer
    record.gold = gold
    record.prediction = result.final_answer
    record.prompt = result.steps[0].prompt if result.steps else ""
    record.raw_response = result.steps[-1].raw_response if result.steps else None
    record.meta.update(
        {
            "rollout": result.as_dict(include_steps=cfg.evaluation.save_raw_generations),
            "in_distribution_for": ["SPLIT_STATEFUL"],
        }
    )
    record.components["first_error_depth"] = result.first_error_depth
    record.components["steps_taken"] = result.steps_taken
    record.components["termination_reason"] = result.termination_reason
    record.components["state_match_by_depth"] = [
        step.state_matches_canonical for step in result.steps
    ]

    if result.final_answer:
        strict_score = score_answer(result.final_answer, gold, answer_type=record.answer_type, cfg=cfg.evaluation.scoring)
        record.strict_correct = bool(strict_score.correct) and result.termination_reason == "done"
        record.lenient_correct = bool(strict_score.correct)
        record.scorer = strict_score.scorer
        record.components["strict"] = strict_score.as_dict()
        record.extraction_route = result.steps[-1].parse.route if result.steps else "none"
        record.parse_status = result.steps[-1].parse.status if result.steps else "not_found"
    else:
        # No answer was produced: recover one leniently from the last response.
        last_raw = result.steps[-1].raw_response if result.steps else None
        parse = extract_final_answer(last_raw, strip_thinking=cfg.evaluation.strip_reasoning_blocks)
        record.extraction_route = parse.route
        record.parse_status = parse.status
        lenient_value = lenient_answer_of(parse)
        if lenient_value:
            lenient_score = score_answer(lenient_value, gold, answer_type=record.answer_type, cfg=cfg.evaluation.scoring)
            record.lenient_correct = bool(lenient_score.correct)
            record.components["lenient"] = lenient_score.as_dict()
            record.prediction = lenient_value

    if record.strict_correct:
        record.failure_type = failures.CORRECT
    elif result.control_failure:
        record.failure_type = result.control_failure
    elif result.final_answer:
        record.failure_type = failures.WRONG_ANSWER
    else:
        record.failure_type = failures.classify_from_status(
            record.parse_status, correct=False, default=failures.NO_ANSWER_FOUND
        )
    record.protocol = protocol
    return record


# --------------------------------------------------------------------------------------
# Transition-level protocols
# --------------------------------------------------------------------------------------


def _perturbation_state(
    protocol: str,
    trajectory: Trajectory,
    transition: TransitionRecord,
    eval_set: EvalSet,
    seed: int,
) -> tuple[str | None, dict[str, Any]]:
    """The state actually shown to the model for a perturbation protocol."""
    if protocol == "shuffled_state":
        others = [t for t in trajectory.transitions if t.step_index != transition.step_index]
        if not others:
            return transition.current_state, {"perturbation": "skipped", "reason": "single-step trajectory"}
        pick = others[int(stable_shuffle_key(transition.transition_id, f"shuffle|{seed}")[:8], 16) % len(others)]
        return pick.current_state, {"perturbation": "shuffled_state", "source_step_index": pick.step_index}
    if protocol == "distractor_state":
        candidates = [pid for pid in eval_set.order if pid != trajectory.problem_id]
        if not candidates:
            return transition.current_state, {"perturbation": "skipped", "reason": "single-problem eval set"}
        index = int(stable_shuffle_key(transition.transition_id, f"distractor|{seed}")[:8], 16) % len(candidates)
        donor = eval_set.trajectories[candidates[index]]
        donor_step = donor.transitions[index % donor.length]
        return donor_step.current_state, {
            "perturbation": "distractor_state",
            "source_problem_id": donor.problem_id,
            "source_step_index": donor_step.step_index,
        }
    return transition.current_state, {}


def run_transition_protocol(
    backend: Backend, eval_set: EvalSet, cfg: LabConfig, protocol: str
) -> Iterator[EvalRecord]:
    stateful = cfg.experiment.condition != "SPLIT_NEXT_STEP"
    strip_thinking = cfg.evaluation.strip_reasoning_blocks
    seed = cfg.experiment.seed

    requests: list[tuple[Any, Any, dict[str, Any]]] = []
    canonical_by_record: dict[str, Any] = {}
    for trajectory, transition in eval_set.iter_transitions():
        record = _base_record(protocol, trajectory, transition)
        problem = trajectory.full.problem

        if protocol == "oracle_transition":
            built = prompts.oracle_transition(problem, transition.current_state, stateful=stateful)
            extra: dict[str, Any] = {}
        elif protocol == "full_history_transition":
            built = prompts.full_history_transition(
                problem, trajectory.full.history_prefix(transition.step_index), stateful=stateful
            )
            extra = {"history_steps": transition.step_index - 1}
        elif protocol == "no_state_transition":
            built = prompts.no_state_transition(problem, stateful=stateful)
            extra = {}
        elif protocol == "last_state_only":
            built = prompts.last_state_only(transition.current_state, stateful=stateful)
            extra = {}
        elif protocol in ("shuffled_state", "distractor_state"):
            state, extra = _perturbation_state(protocol, trajectory, transition, eval_set, seed)
            if protocol == "shuffled_state":
                built = prompts.shuffled_state(
                    problem, state, stateful=stateful, source_step=extra.get("source_step_index", -1)
                )
            else:
                built = prompts.distractor_state(
                    problem, state, stateful=stateful, source_problem_id=extra.get("source_problem_id", "")
                )
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unsupported transition protocol {protocol!r}")

        record.meta.update({"in_distribution_for": list(built.in_distribution_for), **built.meta, **extra})
        canonical_by_record[record.record_id] = transition
        requests.append(
            (
                record,
                built,
                {
                    "problem_id": trajectory.problem_id,
                    "transition_id": transition.transition_id,
                    "current_state": transition.current_state,
                    "target_step": transition.target_step,
                    "next_state": transition.next_state,
                    "done": transition.done,
                    "final_answer": transition.final_answer or trajectory.full.answer,
                    "answer": trajectory.full.answer,
                },
            )
        )

    for record, built, generated in _generate_in_batches(backend, requests, cfg):
        transition = canonical_by_record[record.record_id]
        record.prompt = built.text
        record.raw_response = generated.text
        record.meta["completion_tokens"] = generated.completion_tokens

        if generated.error:
            record.failure_type = failures.BACKEND_FAILURE
            record.parse_status = "backend_failure"
            record.meta["backend_error"] = generated.error
            yield record
            continue

        parse = (
            parse_transition_response(generated.text, strip_thinking=strip_thinking)
            if stateful
            else parse_next_step_response(generated.text, strip_thinking=strip_thinking)
        )
        yield _score_transition_record(record, parse, transition, cfg, stateful=stateful)


def run_protocol(backend: Backend, eval_set: EvalSet, cfg: LabConfig, protocol: str) -> Iterator[EvalRecord]:
    if protocol in ANSWER_PROTOCOLS:
        return run_answer_protocol(backend, eval_set, cfg, protocol)
    if protocol in TRANSITION_PROTOCOLS:
        return run_transition_protocol(backend, eval_set, cfg, protocol)
    raise ValueError(f"Unknown protocol {protocol!r}")
