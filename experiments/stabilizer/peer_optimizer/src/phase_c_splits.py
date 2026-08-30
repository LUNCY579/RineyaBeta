#!/usr/bin/env python3
"""Leakage-resistant grouped splits and deterministic Phase-C row selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

SPLIT_NAMES = ("META_TRAIN", "MASK_CAL", "ONE_STEP_DEV", "SEALED")
_WS = re.compile(r"\s+")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    pieces: list[str] = []
    for message in messages[:-1]:
        role = str(message.get("role", "")).strip().lower()
        content = _WS.sub(" ", str(message.get("content", "")).strip().lower())
        pieces.append(f"{role}:{content}")
    return stable_hash("\n".join(pieces))


def _nested_get(row: Mapping[str, Any], dotted: str) -> Any:
    current: Any = row
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def family_key(row: Mapping[str, Any], group_fields: Sequence[str], source_label: str) -> str:
    for field in group_fields:
        value = _nested_get(row, field)
        if value is not None and str(value).strip():
            return f"{source_label}|{field}={str(value).strip()}"
    raise ValueError(
        f"row {row.get('id', '<missing-id>')!r} has none of required group fields {list(group_fields)}"
    )


def row_id(row: Mapping[str, Any], source_label: str) -> str:
    explicit = row.get("id") or row.get("source_id")
    if explicit is not None and str(explicit).strip():
        return f"{source_label}|{str(explicit).strip()}"
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row without id must contain messages")
    return f"{source_label}|content:{stable_hash(json.dumps(messages, sort_keys=True, ensure_ascii=False))}"


def assign_split(group: str, *, seed: int, ratios: Mapping[str, float]) -> str:
    missing = [name for name in SPLIT_NAMES if name not in ratios]
    if missing:
        raise ValueError(f"missing split ratios: {missing}")
    total = sum(float(ratios[name]) for name in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    value = int(stable_hash("split", seed, group)[:16], 16) / float(16**16)
    cumulative = 0.0
    for name in SPLIT_NAMES:
        cumulative += float(ratios[name])
        if value < cumulative:
            return name
    return SPLIT_NAMES[-1]


@dataclass(frozen=True)
class IndexedRow:
    source_path: str
    source_label: str
    row_id: str
    family: str
    split: str
    prompt_hash: str
    messages: tuple[dict[str, str], ...]

    def public_manifest(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_label": self.source_label,
            "row_id_sha256": stable_hash("row", self.row_id),
            "family_sha256": stable_hash("family", self.family),
            "split": self.split,
            "prompt_sha256": self.prompt_hash,
        }


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield row


def index_sources(
    source_paths: Sequence[str | Path],
    *,
    root: str | Path,
    group_fields: Sequence[str],
    split_seed: int,
    ratios: Mapping[str, float],
    expected_sha256: Mapping[str, str] | None = None,
) -> tuple[list[IndexedRow], dict[str, Any]]:
    """Index only metadata/messages needed for Phase C and fail closed on leakage."""
    root_path = Path(root)
    indexed: list[IndexedRow] = []
    source_ledger: list[dict[str, Any]] = []
    family_to_split: dict[str, str] = {}
    prompt_to_split: dict[str, str] = {}
    prompt_collisions: list[dict[str, str]] = []

    for source in source_paths:
        relative = str(source).replace("\\", "/")
        path = root_path / source
        if not path.exists():
            raise FileNotFoundError(
                f"required Phase-C source is missing: {path}. Rebuild the audited corpus before running."
            )
        digest = sha256_file(path)
        if expected_sha256 and relative in expected_sha256:
            expected = str(expected_sha256[relative]).lower()
            if digest.lower() != expected:
                raise RuntimeError(
                    f"source hash mismatch for {relative}: expected {expected}, got {digest}"
                )
        source_label = Path(relative).stem
        rows_seen = 0
        for row in read_jsonl(path):
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
                continue
            target = str(messages[-1].get("content", "")).strip()
            if not target:
                continue

            family = family_key(row, group_fields, source_label)
            split = assign_split(family, seed=split_seed, ratios=ratios)
            previous_split = family_to_split.setdefault(family, split)
            if previous_split != split:
                raise RuntimeError(f"family {family!r} mapped to multiple splits")

            prompt_hash = normalized_prompt_hash(messages)
            earlier = prompt_to_split.get(prompt_hash)
            if earlier is not None and earlier != split:
                prompt_collisions.append(
                    {"prompt_sha256": prompt_hash, "first_split": earlier, "second_split": split}
                )
            else:
                prompt_to_split[prompt_hash] = split

            clean_messages: list[dict[str, str]] = []
            valid = True
            for message in messages:
                if not isinstance(message, dict):
                    valid = False
                    break
                role = str(message.get("role", ""))
                content = str(message.get("content", ""))
                if role not in {"system", "user", "assistant"} or not content:
                    valid = False
                    break
                clean_messages.append({"role": role, "content": content})
            if not valid:
                continue

            indexed.append(
                IndexedRow(
                    source_path=relative,
                    source_label=source_label,
                    row_id=row_id(row, source_label),
                    family=family,
                    split=split,
                    prompt_hash=prompt_hash,
                    messages=tuple(clean_messages),
                )
            )
            rows_seen += 1
        source_ledger.append(
            {"path": relative, "sha256": digest, "eligible_rows": rows_seen}
        )

    if prompt_collisions:
        sample = prompt_collisions[:5]
        raise RuntimeError(
            "normalized prompt leakage crosses grouped splits; "
            f"{len(prompt_collisions)} collisions found, sample={sample}"
        )

    counts = {name: 0 for name in SPLIT_NAMES}
    families = {name: set() for name in SPLIT_NAMES}
    for row in indexed:
        counts[row.split] += 1
        families[row.split].add(row.family)
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise RuntimeError(f"grouped split produced empty partitions: {empty}")

    audit = {
        "split_seed": split_seed,
        "group_fields": list(group_fields),
        "ratios": {name: float(ratios[name]) for name in SPLIT_NAMES},
        "source_ledger": source_ledger,
        "row_counts": counts,
        "family_counts": {name: len(families[name]) for name in SPLIT_NAMES},
        "family_overlap": 0,
        "normalized_prompt_overlap": 0,
    }
    audit["audit_sha256"] = stable_hash(json.dumps(audit, sort_keys=True))
    return indexed, audit


def select_rows(
    indexed: Sequence[IndexedRow],
    *,
    split: str,
    count: int,
    seed: int,
) -> list[IndexedRow]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}")
    if count <= 0:
        raise ValueError("selection count must be positive")
    candidates = [row for row in indexed if row.split == split]
    if len(candidates) < count:
        raise RuntimeError(
            f"{split} has only {len(candidates)} eligible rows, need {count}"
        )

    # Round-robin across families before taking a second row from any family.
    by_family: dict[str, list[IndexedRow]] = {}
    for row in candidates:
        by_family.setdefault(row.family, []).append(row)
    for family, rows in by_family.items():
        rows.sort(key=lambda row: stable_hash("row-order", seed, family, row.row_id))
    family_order = sorted(
        by_family, key=lambda family: stable_hash("family-order", seed, split, family)
    )

    selected: list[IndexedRow] = []
    depth = 0
    while len(selected) < count:
        progressed = False
        for family in family_order:
            rows = by_family[family]
            if depth < len(rows):
                selected.append(rows[depth])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        depth += 1
    if len(selected) != count:
        raise RuntimeError(f"could not select {count} rows from {split}")
    return selected


def selection_manifest(rows: Iterable[IndexedRow]) -> list[dict[str, Any]]:
    return [row.public_manifest() for row in rows]
