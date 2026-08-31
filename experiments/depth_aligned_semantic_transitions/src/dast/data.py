"""Dataset loading, validation, and arm-specific semantic views."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from dast.alignment import AlignedTrajectory, align_trajectory
from dast.config import ArmSpec


@dataclass(frozen=True)
class ReasoningRecord:
    problem_id: str
    split: str
    problem: str
    answer: str
    full_response: str
    reasoning_steps: tuple[str, ...]
    compact_state_trace: tuple[str, ...]
    answer_type: str = "text"
    category: str = "unknown"
    domain: str = "unknown"
    generator: str = "unknown"
    source_pair_sha256: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ReasoningRecord":
        record = value.get("full_context_record", value)
        if not isinstance(record, dict):
            raise TypeError("full_context_record must be a mapping")
        required = ("problem_id", "split", "problem", "answer", "full_response", "reasoning_steps", "compact_state_trace")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"record is missing required fields: {', '.join(missing)}")
        return cls(
            problem_id=str(record["problem_id"]),
            split=str(record["split"]).lower(),
            problem=str(record["problem"]),
            answer=str(record["answer"]),
            full_response=str(record["full_response"]),
            reasoning_steps=tuple(str(item) for item in record["reasoning_steps"]),
            compact_state_trace=tuple(str(item) for item in record["compact_state_trace"]),
            answer_type=str(record.get("answer_type", "text")),
            category=str(record.get("category", "unknown")),
            domain=str(record.get("domain", "unknown")),
            generator=str(record.get("generator", "unknown")),
            source_pair_sha256=str(record.get("source_pair_sha256", "")),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.problem_id:
            errors.append("problem_id is empty")
        if self.split not in {"train", "dev", "test"}:
            errors.append(f"unknown split {self.split!r}")
        if not self.problem.strip() or not self.full_response.strip() or not self.answer.strip():
            errors.append("problem, full_response, and answer must be non-empty")
        if not self.reasoning_steps:
            errors.append("reasoning_steps is empty")
        if len(self.compact_state_trace) != len(self.reasoning_steps) + 1:
            errors.append(
                "compact_state_trace must contain the initial state plus one post-step state "
                f"({len(self.compact_state_trace)} != {len(self.reasoning_steps) + 1})"
            )
        return errors

    def raw_targets(self, target_kind: str) -> tuple[str, ...]:
        if target_kind == "compact_state":
            # Exclude the pre-reasoning initial state: checkpoint 1 should be a
            # completed semantic milestone, not an unsolved duplicate of input.
            return self.compact_state_trace[1:]
        if target_kind == "reasoning_step":
            return self.reasoning_steps
        if target_kind == "none":
            return ()
        raise ValueError(f"unsupported target kind: {target_kind}")


@dataclass(frozen=True)
class DASTExample:
    record: ReasoningRecord
    trajectory: AlignedTrajectory


@dataclass(frozen=True)
class DatasetValidationReport:
    path: str
    sha256: str
    records: int
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "records": self.records,
            "split_counts": self.split_counts,
            "category_counts": self.category_counts,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "status": "PASS" if self.passed else "FAIL",
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record at {path}:{line_number} is not an object")
            yield value


def load_records(
    path: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
) -> list[ReasoningRecord]:
    records: list[ReasoningRecord] = []
    for raw in iter_jsonl(path):
        record = ReasoningRecord.from_mapping(raw)
        if split is not None and record.split != split.lower():
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def validate_dataset(path: str | Path, *, max_errors: int = 100) -> DatasetValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    count = 0

    try:
        raw_iterable: Iterable[dict[str, Any]] = iter_jsonl(path)
        for raw in raw_iterable:
            count += 1
            try:
                record = ReasoningRecord.from_mapping(raw)
            except Exception as exc:  # noqa: BLE001 - validation must continue across malformed rows
                errors.append(f"record {count}: {exc}")
                if len(errors) >= max_errors:
                    break
                continue
            row_errors = record.validate()
            errors.extend(f"{record.problem_id}: {message}" for message in row_errors)
            previous_split = seen_ids.get(record.problem_id)
            if previous_split is not None:
                errors.append(
                    f"duplicate problem_id {record.problem_id!r} appears in {previous_split!r} and {record.split!r}"
                )
            else:
                seen_ids[record.problem_id] = record.split
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
            category_counts[record.category] = category_counts.get(record.category, 0) + 1
            if len(errors) >= max_errors:
                break
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    if not split_counts.get("train"):
        warnings.append("no train records were found")
    if not split_counts.get("dev"):
        warnings.append("no dev records were found")
    if split_counts.get("test") and not split_counts.get("dev"):
        warnings.append("TEST exists without a DEV split; TEST must remain sealed")

    return DatasetValidationReport(
        path=str(Path(path).resolve()),
        sha256=sha256_file(path) if Path(path).is_file() else "",
        records=count,
        split_counts=dict(sorted(split_counts.items())),
        category_counts=dict(sorted(category_counts.items())),
        errors=tuple(errors[:max_errors]),
        warnings=tuple(warnings),
    )


def build_examples(
    records: Sequence[ReasoningRecord],
    *,
    arm: ArmSpec,
    num_stages: int,
    seed: int,
) -> list[DASTExample]:
    examples: list[DASTExample] = []
    for record in records:
        canonical_kind = arm.target_kind if arm.target_kind != "none" else "compact_state"
        raw_targets = record.raw_targets(canonical_kind)
        trajectory = align_trajectory(
            raw_targets,
            num_stages=num_stages,
            arm=arm,
            problem_id=record.problem_id,
            seed=seed,
        )
        examples.append(DASTExample(record=record, trajectory=trajectory))
    return examples
