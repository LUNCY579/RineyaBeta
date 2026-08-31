"""Leakage-resistant grouped train/dev split construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import forbidden_data_marker
from .evidence import atomic_write_json, sha256_file


def _guard_path(path: Path) -> None:
    marker = forbidden_data_marker(path.as_posix())
    if marker is not None:
        raise ValueError(
            f"refusing split path containing sealed/hidden marker {marker!r}: {path}"
        )


def _group_value(row: dict[str, Any], fields: Iterable[str]) -> tuple[str, str]:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return field, str(value)
    raise ValueError(
        "row has no explicit source-family field; refusing a random row split. "
        f"Tried {list(fields)}"
    )


def _bucket(seed: int, group: str) -> float:
    digest = hashlib.sha256(f"{seed}|{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_grouped_split(
    *,
    input_path: str | Path,
    train_path: str | Path,
    dev_path: str | Path,
    manifest_path: str | Path,
    group_fields: list[str],
    dev_fraction: float,
    seed: int,
    category_field: str = "category",
) -> dict[str, Any]:
    if not group_fields:
        raise ValueError("at least one group field is required")
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    source = Path(input_path)
    train = Path(train_path)
    dev = Path(dev_path)
    manifest = Path(manifest_path)
    for path in (source, train, dev, manifest):
        _guard_path(path)
    if train.resolve() == dev.resolve():
        raise ValueError("train and dev paths must differ")

    train_rows: list[str] = []
    dev_rows: list[str] = []
    train_groups: set[str] = set()
    dev_groups: set[str] = set()
    field_usage: Counter[str] = Counter()
    train_categories: Counter[str] = Counter()
    dev_categories: Counter[str] = Counter()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} is not an object")
            field, raw_group = _group_value(row, group_fields)
            group = f"{field}:{raw_group}"
            field_usage[field] += 1
            category = str(row.get(category_field, "?"))
            normalized_line = line if line.endswith("\n") else line + "\n"
            if _bucket(seed, group) < dev_fraction:
                dev_rows.append(normalized_line)
                dev_groups.add(group)
                dev_categories[category] += 1
            else:
                train_rows.append(normalized_line)
                train_groups.add(group)
                train_categories[category] += 1

    overlap = train_groups & dev_groups
    if overlap:
        raise AssertionError(f"group leakage detected: {sorted(overlap)[:5]}")
    if not train_rows or not dev_rows:
        raise ValueError(
            "hash split produced an empty arm; use more groups or a different preregistered seed"
        )

    def write_lines(path: Path, rows: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text("".join(rows), encoding="utf-8")
        temporary.replace(path)

    write_lines(train, train_rows)
    write_lines(dev, dev_rows)
    record = {
        "schema_version": 1,
        "seed": seed,
        "dev_fraction_requested": dev_fraction,
        "group_fields_precedence": group_fields,
        "category_field": category_field,
        "input": {
            "path": source.as_posix(),
            "rows": len(train_rows) + len(dev_rows),
            "sha256": sha256_file(source),
        },
        "train": {
            "path": train.as_posix(),
            "rows": len(train_rows),
            "groups": len(train_groups),
            "categories": dict(sorted(train_categories.items())),
            "sha256": sha256_file(train),
        },
        "dev": {
            "path": dev.as_posix(),
            "rows": len(dev_rows),
            "groups": len(dev_groups),
            "categories": dict(sorted(dev_categories.items())),
            "sha256": sha256_file(dev),
        },
        "group_overlap": 0,
        "field_usage_rows": dict(sorted(field_usage.items())),
        "claim_boundary": (
            "This manifest proves deterministic group separation only. It does not "
            "establish that the chosen groups cover the target capability."
        ),
    }
    atomic_write_json(manifest, record)
    return record
