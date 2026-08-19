"""Strict validation of a located dataset against the documented contract.

This is deliberately stronger than "does it parse". The experiment's conclusion
depends on FULL and SPLIT being exactly paired representations of the same
problems, so the validator re-derives that pairing rather than trusting the
released report.

Checks
------
1. Schema conformance of every row (jsonschema when available, structural
   fallback otherwise).
2. Unique problem ids and unique normalised problem text.
3. TRAIN / DEV / TEST disjoint at the problem level.
4. Every full-context record has a complete, contiguous transition chain whose
   targets reproduce ``reasoning_steps`` and whose states chain end-to-end.
5. Exactly one ``done=true`` transition per problem, carrying the final answer.
6. Causal safeguard: no compact state entering step ``t`` may contain the final
   answer before it has been derived.
7. Global shuffle really is shuffled (no adjacent rows from one problem).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rineya_lab.data.loading import SPLITS, RineyaDataset
from rineya_lab.data.records import FullRecord, Trajectory, TransitionRecord
from rineya_lab.utils.io import read_json
from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("data.validate")

MAX_REPORTED_ISSUES = 50


@dataclass
class ValidationReport:
    status: str = "PASS"
    checks: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.status = "FAIL"
        if len(self.errors) < MAX_REPORTED_ISSUES:
            self.errors.append(message)
        elif len(self.errors) == MAX_REPORTED_ISSUES:
            self.errors.append(f"... additional errors suppressed after {MAX_REPORTED_ISSUES}")

    def warn(self, message: str) -> None:
        if len(self.warnings) < MAX_REPORTED_ISSUES:
            self.warnings.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": self.checks,
            "counts": self.counts,
            "error_count": len(self.errors),
            "errors": self.errors,
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
        }


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _load_schema_validators(schema_dir: Path) -> dict[str, Any] | None:
    try:
        import jsonschema
    except ImportError:
        LOGGER.warning("jsonschema is not installed; falling back to structural field checks")
        return None
    validators: dict[str, Any] = {}
    for name, filename in (("full_context", "full_context.schema.json"),
                           ("split_transition", "split_transition.schema.json")):
        path = schema_dir / filename
        if path.exists():
            validators[name] = jsonschema.Draft202012Validator(read_json(path))
    return validators or None


def validate_dataset(
    dataset: RineyaDataset,
    *,
    splits: tuple[str, ...] = SPLITS,
    schema_check: bool = True,
    limit: int | None = None,
) -> ValidationReport:
    report = ValidationReport()
    validators = _load_schema_validators(dataset.paths.root / "schemas") if schema_check else None
    report.checks["schema_validation"] = "jsonschema" if validators else "structural-fallback"

    problems_by_split: dict[str, set[str]] = {}
    normalized_seen: dict[str, str] = {}
    total_full = 0
    total_transitions = 0
    category_counts: Counter[str] = Counter()
    step_histogram: Counter[int] = Counter()

    for split in splits:
        LOGGER.info("Validating split %s", split)
        full_records: dict[str, FullRecord] = {}
        for row in _iter_raw(dataset, "full_context", split, limit):
            total_full += 1
            _schema_check(row, "full_context", validators, report)
            record = FullRecord(row)
            if record.split != split:
                report.fail(f"{record.problem_id}: split field {record.split!r} != file split {split!r}")
            if record.problem_id in full_records:
                report.fail(f"Duplicate problem_id within {split}: {record.problem_id}")
            full_records[record.problem_id] = record
            key = normalize_problem(record.problem)
            if key in normalized_seen and normalized_seen[key] != record.problem_id:
                report.fail(
                    f"Duplicate normalised problem text: {record.problem_id} and {normalized_seen[key]}"
                )
            normalized_seen[key] = record.problem_id
            category_counts[record.category] += 1
            step_histogram[record.step_count] += 1

        problems_by_split[split] = set(full_records)

        grouped: dict[str, list[TransitionRecord]] = defaultdict(list)
        previous_problem_id: str | None = None
        adjacent_same_problem = 0
        transition_ids: set[str] = set()
        for row in _iter_raw(dataset, "split_transitions", split, limit):
            total_transitions += 1
            _schema_check(row, "split_transition", validators, report)
            transition = TransitionRecord(row)
            if transition.split != split:
                report.fail(f"{transition.transition_id}: split field mismatch in {split}")
            if transition.transition_id in transition_ids:
                report.fail(f"Duplicate transition_id: {transition.transition_id}")
            transition_ids.add(transition.transition_id)
            if transition.problem_id not in full_records and limit is None:
                report.fail(f"{transition.transition_id}: no matching full_context record")
            if previous_problem_id == transition.problem_id:
                adjacent_same_problem += 1
            previous_problem_id = transition.problem_id
            if row.get("sequence_free_training") is not True:
                report.fail(f"{transition.transition_id}: sequence_free_training flag is not true")
            grouped[transition.problem_id].append(transition)

        report.counts.setdefault("adjacent_same_problem_transitions", {})[split] = adjacent_same_problem
        if adjacent_same_problem and limit is None:
            report.fail(
                f"{split}: {adjacent_same_problem} adjacent transition rows share a problem_id; "
                "the released file is supposed to be globally shuffled"
            )

        if limit is None:
            _validate_chains(full_records, grouped, report)

        report.counts.setdefault("full_records", {})[split] = len(full_records)
        report.counts.setdefault("transition_records", {})[split] = sum(len(v) for v in grouped.values())

    # cross-split disjointness
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1:]:
            overlap = problems_by_split.get(left, set()) & problems_by_split.get(right, set())
            if overlap:
                report.fail(
                    f"{len(overlap)} problem ids appear in both {left} and {right} "
                    f"(e.g. {sorted(overlap)[:3]}); splits must be disjoint before transition expansion"
                )
    report.checks["cross_split_disjoint"] = not any(
        problems_by_split.get(a, set()) & problems_by_split.get(b, set())
        for i, a in enumerate(splits)
        for b in splits[i + 1:]
    )

    report.counts["total_full_records"] = total_full
    report.counts["total_transition_records"] = total_transitions
    report.counts["unique_normalized_problems"] = len(normalized_seen)
    report.counts["categories"] = len(category_counts)
    report.counts["step_count_histogram"] = dict(sorted(step_histogram.items()))

    _compare_with_released_manifest(dataset, report, limited=limit is not None)

    if report.status == "PASS":
        LOGGER.info("Dataset validation PASSED (%d full, %d transitions)", total_full, total_transitions)
    else:
        LOGGER.error("Dataset validation FAILED with %d errors", len(report.errors))
    return report


def _iter_raw(dataset: RineyaDataset, variant: str, split: str, limit: int | None):
    from rineya_lab.utils.io import iter_jsonl

    return iter_jsonl(dataset.path_for(variant, split), limit=limit)


def _schema_check(row: dict[str, Any], kind: str, validators: dict[str, Any] | None, report: ValidationReport) -> None:
    if validators and kind in validators:
        for error in validators[kind].iter_errors(row):
            report.fail(f"{row.get('problem_id', '?')}: schema violation at {list(error.path)}: {error.message}")
        return
    required = {
        "full_context": ("problem_id", "problem", "answer", "full_response", "reasoning_steps", "compact_state_trace"),
        "split_transition": ("transition_id", "problem_id", "current_state", "target_step", "next_state", "done"),
    }[kind]
    missing = [f for f in required if f not in row]
    if missing:
        report.fail(f"{row.get('problem_id', '?')}: missing required fields {missing}")


def _validate_chains(
    full_records: dict[str, FullRecord],
    grouped: dict[str, list[TransitionRecord]],
    report: ValidationReport,
) -> None:
    for problem_id, record in full_records.items():
        steps = grouped.get(problem_id, [])
        if not steps:
            report.fail(f"{problem_id}: no transitions expanded from this problem")
            continue
        trajectory = Trajectory(full=record, transitions=sorted(steps, key=lambda t: t.step_index))
        for issue in trajectory.check_chain():
            report.fail(issue)
        for issue in check_causal_safeguard(trajectory):
            report.fail(issue)


def check_causal_safeguard(trajectory: Trajectory) -> list[str]:
    """Detect future information leaking backwards into an earlier state.

    The dataset builder is allowed to *inspect* the finished trajectory when it
    constructs ``s_t``; it is not allowed to *insert* anything only knowable
    after step ``t``. The strongest cheap test is: the literal final answer must
    not appear in a state that precedes the step which derives it.
    """
    issues: list[str] = []
    answer = trajectory.full.answer.strip()
    if not answer or len(answer) < 2:
        return issues
    # Index of the first step whose target legitimately produces the answer.
    reveal_index = next(
        (t.step_index for t in trajectory.transitions if answer in t.target_step),
        len(trajectory.transitions),
    )
    for transition in trajectory.transitions:
        if transition.step_index <= reveal_index and answer in transition.current_state:
            # A trivially short answer (e.g. "5") can appear as an intermediate
            # value by coincidence, so only flag the explicit "final answer" form.
            state = transition.current_state.lower()
            if "final answer" in state or "answer is" in state:
                issues.append(
                    f"{trajectory.problem_id}: state entering step {transition.step_index} already states the "
                    f"final answer, which is only derived at step {reveal_index}"
                )
    return issues


def _compare_with_released_manifest(dataset: RineyaDataset, report: ValidationReport, *, limited: bool) -> None:
    released = dataset.released_validation_report()
    stats = dataset.stats()
    if not released and not stats:
        report.warn("No released validation_report.json / dataset_stats.json found to cross-check against")
        return
    report.checks["released_report_status"] = released.get("status")
    if limited:
        report.warn("Row limit in effect; skipping count comparison against the released manifest")
        return
    expected_full = stats.get("problem_splits", {})
    expected_transitions = stats.get("transition_splits", {})
    for split, expected in expected_full.items():
        actual = report.counts.get("full_records", {}).get(split)
        if actual is not None and actual != expected:
            report.fail(f"full_context/{split}: {actual} rows but manifest declares {expected}")
    for split, expected in expected_transitions.items():
        actual = report.counts.get("transition_records", {}).get(split)
        if actual is not None and actual != expected:
            report.fail(f"split_transitions/{split}: {actual} rows but manifest declares {expected}")
