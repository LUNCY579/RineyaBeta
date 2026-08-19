"""Aggregation of per-example evaluation records into protocol reports.

Rules that matter for the integrity of the primary metric:

* Unparseable, refused, empty and backend-failed items stay in the denominator.
  Nothing is silently dropped; ``scored_items == total_items`` always.
* Strict is primary. Lenient is reported beside it, and the disagreement between
  the two tracks is reported explicitly, because a large disagreement means the
  comparison is partly measuring formatting compliance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from rineya_lab.evaluation import failures
from rineya_lab.evaluation.protocols import EvalRecord


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _stratify(records: list[EvalRecord], key) -> dict[str, dict[str, Any]]:
    buckets: dict[Any, list[EvalRecord]] = defaultdict(list)
    for record in records:
        buckets[key(record)].append(record)
    out: dict[str, dict[str, Any]] = {}
    for bucket_key in sorted(buckets, key=lambda value: (value is None, value)):
        bucket = buckets[bucket_key]
        strict = sum(1 for r in bucket if r.strict_correct)
        lenient = sum(1 for r in bucket if r.lenient_correct)
        out[str(bucket_key)] = {
            "n": len(bucket),
            "strict_correct": strict,
            "strict_accuracy": _rate(strict, len(bucket)),
            "lenient_correct": lenient,
            "lenient_accuracy": _rate(lenient, len(bucket)),
        }
    return out


def summarize_protocol(protocol: str, records: list[EvalRecord]) -> dict[str, Any]:
    total = len(records)
    strict = sum(1 for r in records if r.strict_correct)
    lenient = sum(1 for r in records if r.lenient_correct)

    failure_counts = Counter(r.failure_type for r in records)
    route_counts = Counter(r.extraction_route for r in records)
    status_counts = Counter(r.parse_status for r in records)
    scorer_counts = Counter(r.scorer or "none" for r in records)

    format_failures = sum(count for name, count in failure_counts.items() if failures.is_format_failure(name))
    control_failures = sum(count for name, count in failure_counts.items() if name in failures.CONTROL_FAILURES)
    reasoning_failures = sum(count for name, count in failure_counts.items() if name in failures.REASONING_FAILURES)

    lenient_only = sum(1 for r in records if r.lenient_correct and not r.strict_correct)
    strict_only = sum(1 for r in records if r.strict_correct and not r.lenient_correct)

    summary: dict[str, Any] = {
        "protocol": protocol,
        "total_items": total,
        "scored_items": total,
        "strict": {"correct": strict, "accuracy": _rate(strict, total)},
        "lenient": {"correct": lenient, "accuracy": _rate(lenient, total)},
        "track_disagreement": {
            "lenient_only_correct": lenient_only,
            "strict_only_correct": strict_only,
            "disagreement_rate": _rate(lenient_only + strict_only, total),
        },
        "failure_distribution": dict(failure_counts.most_common()),
        "failure_groups": {
            "format": format_failures,
            "control": control_failures,
            "reasoning": reasoning_failures,
            "format_rate": _rate(format_failures, total),
            "control_rate": _rate(control_failures, total),
            "reasoning_rate": _rate(reasoning_failures, total),
        },
        "extraction_routes": dict(route_counts.most_common()),
        "parse_statuses": dict(status_counts.most_common()),
        "scorers_used": dict(scorer_counts.most_common()),
        "by_category": _stratify(records, lambda r: r.category),
        "by_total_steps": _stratify(records, lambda r: r.total_steps),
        "by_answer_type": _stratify(records, lambda r: r.answer_type),
        "unique_problems": len({r.problem_id for r in records}),
    }

    if any(r.step_index for r in records):
        summary["by_step_index"] = _stratify(records, lambda r: r.step_index)

    component_keys = ("step_match", "updated_state_valid", "done_correct", "answer_equivalent")
    component_summary: dict[str, Any] = {}
    for key in component_keys:
        present = [r.components.get(key) for r in records if key in r.components]
        present = [value for value in present if value is not None]
        if present:
            component_summary[key] = {
                "n": len(present),
                "true": sum(1 for value in present if value),
                "rate": _rate(sum(1 for value in present if value), len(present)),
            }
    if component_summary:
        summary["transition_components"] = component_summary

    rollout_records = [r for r in records if "steps_taken" in r.components]
    if rollout_records:
        steps = [r.components["steps_taken"] for r in rollout_records]
        depths = [r.components.get("first_error_depth") for r in rollout_records]
        clean = [d for d in depths if d is not None]
        terminations = Counter(r.components.get("termination_reason") for r in rollout_records)
        summary["rollout"] = {
            "mean_steps": round(sum(steps) / len(steps), 4),
            "max_steps": max(steps),
            "completion_rate": _rate(
                sum(1 for r in rollout_records if r.components.get("termination_reason") == "done"),
                len(rollout_records),
            ),
            "termination_reasons": dict(terminations.most_common()),
            "first_error_depth": {
                "items_with_error": len(clean),
                "items_without_error": len(depths) - len(clean),
                "mean_depth": round(sum(clean) / len(clean), 4) if clean else None,
                "histogram": dict(sorted(Counter(clean).items())),
            },
            "overrun_rate": _rate(
                sum(1 for r in rollout_records if r.components.get("steps_taken", 0) > r.total_steps),
                len(rollout_records),
            ),
        }

    completion_tokens = [r.meta.get("completion_tokens", 0) for r in records if "completion_tokens" in r.meta]
    if completion_tokens:
        summary["generation"] = {
            "total_completion_tokens": sum(completion_tokens),
            "mean_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 3),
        }
    return summary


def free_rollout_gap(summaries: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """oracle-state transition success minus free-rollout final success.

    A large gap means local transitions were learned but errors compound once the
    model consumes its own states.
    """
    oracle = summaries.get("oracle_transition")
    rollout = summaries.get("generated_rollout")
    if not oracle or not rollout:
        return None
    teacher = summaries.get("oracle_trajectory")
    payload = {
        "oracle_transition_strict_accuracy": oracle["strict"]["accuracy"],
        "generated_rollout_strict_accuracy": rollout["strict"]["accuracy"],
        "gap": None,
        "teacher_forced_final_accuracy": teacher["strict"]["accuracy"] if teacher else None,
        "teacher_forcing_vs_free_rollout_gap": None,
    }
    if oracle["strict"]["accuracy"] is not None and rollout["strict"]["accuracy"] is not None:
        payload["gap"] = round(oracle["strict"]["accuracy"] - rollout["strict"]["accuracy"], 6)
    if teacher and teacher["strict"]["accuracy"] is not None and rollout["strict"]["accuracy"] is not None:
        payload["teacher_forcing_vs_free_rollout_gap"] = round(
            teacher["strict"]["accuracy"] - rollout["strict"]["accuracy"], 6
        )
    return payload


def state_sufficiency_contrast(summaries: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Compact state versus complete history, with the lower-bound controls."""
    oracle = summaries.get("oracle_transition")
    if not oracle:
        return None
    def acc(name: str) -> float | None:
        summary = summaries.get(name)
        return summary["strict"]["accuracy"] if summary else None

    compact = oracle["strict"]["accuracy"]
    history = acc("full_history_transition")
    payload: dict[str, Any] = {
        "compact_state_accuracy": compact,
        "full_history_accuracy": history,
        "compact_minus_history": round(compact - history, 6) if compact is not None and history is not None else None,
        "no_state_accuracy": acc("no_state_transition"),
        "last_state_only_accuracy": acc("last_state_only"),
        "shuffled_state_accuracy": acc("shuffled_state"),
        "distractor_state_accuracy": acc("distractor_state"),
    }
    if compact is not None and payload["no_state_accuracy"] is not None:
        payload["state_information_gain"] = round(compact - payload["no_state_accuracy"], 6)
    return payload


def audit_sample(records: Iterable[EvalRecord], size: int, seed: int) -> list[dict[str, Any]]:
    """A deterministic sample of records for manual inspection.

    Deliberately over-samples disagreements and format failures: those are where
    an automated verdict is most likely to be wrong.
    """
    from rineya_lab.utils.hashing import stable_shuffle_key

    records = list(records)
    if not records or size <= 0:
        return []
    interesting = [
        r for r in records
        if (r.lenient_correct != r.strict_correct) or failures.is_format_failure(r.failure_type)
    ]
    ordinary = [r for r in records if r not in interesting]
    interesting.sort(key=lambda r: stable_shuffle_key(r.record_id, f"audit|{seed}"))
    ordinary.sort(key=lambda r: stable_shuffle_key(r.record_id, f"audit|{seed}"))
    take_interesting = min(len(interesting), max(1, size // 2))
    chosen = interesting[:take_interesting] + ordinary[: size - take_interesting]
    return [
        {
            "record_id": r.record_id,
            "protocol": r.protocol,
            "problem_id": r.problem_id,
            "gold": r.gold,
            "prediction": r.prediction,
            "strict_correct": r.strict_correct,
            "lenient_correct": r.lenient_correct,
            "failure_type": r.failure_type,
            "extraction_route": r.extraction_route,
            "raw_response": (r.raw_response or "")[:1200],
        }
        for r in chosen[:size]
    ]
