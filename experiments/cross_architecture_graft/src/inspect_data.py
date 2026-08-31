"""Inspect an unsealed JSONL corpus for fields needed by a leakage-safe split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import forbidden_data_marker
from .evidence import atomic_write_json, sha256_file


DEFAULT_GROUP_FIELDS = (
    "template_id",
    "generator_id",
    "group_id",
    "cag_group_id",
    "source_family",
)


def inspect_jsonl(
    path: str | Path,
    *,
    group_fields: tuple[str, ...] = DEFAULT_GROUP_FIELDS,
    max_rows: int | None = None,
) -> dict[str, Any]:
    source = Path(path)
    marker = forbidden_data_marker(source.as_posix())
    if marker is not None:
        raise ValueError(f"refusing corpus path containing {marker!r}: {source}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or null")

    rows = 0
    field_presence: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    group_values: dict[str, set[str]] = {field: set() for field in group_fields}
    content_hashes: Counter[str] = Counter()
    malformed_messages = 0

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
            rows += 1
            for key in row:
                field_presence[str(key)] += 1
            categories[str(row.get("category", "?"))] += 1
            sources[str(row.get("source", "?"))] += 1
            for field in group_fields:
                value = row.get(field)
                if value is not None and str(value).strip():
                    group_values[field].add(str(value))
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                malformed_messages += 1
            canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
            content_hashes[hashlib.sha256(canonical.encode("utf-8")).hexdigest()] += 1
            if max_rows is not None and rows >= max_rows:
                break

    if rows == 0:
        raise ValueError("corpus is empty")
    duplicate_rows = sum(count - 1 for count in content_hashes.values() if count > 1)
    candidates = []
    for field in group_fields:
        present = field_presence[field]
        unique = len(group_values[field])
        candidates.append(
            {
                "field": field,
                "rows_present": present,
                "coverage": present / rows,
                "unique_values": unique,
                "eligible_for_primary_split": present == rows and unique >= 10,
            }
        )
    eligible = [item["field"] for item in candidates if item["eligible_for_primary_split"]]
    status = "READY_FOR_GROUPED_SPLIT" if eligible else "BLOCKED_NO_COMPLETE_GROUP_KEY"
    return {
        "schema_version": 1,
        "status": status,
        "path": source.as_posix(),
        "sha256": sha256_file(source),
        "rows_inspected": rows,
        "max_rows_limit": max_rows,
        "malformed_message_rows": malformed_messages,
        "duplicate_rows": duplicate_rows,
        "top_categories": dict(categories.most_common(50)),
        "top_sources": dict(sources.most_common(50)),
        "group_field_candidates": candidates,
        "recommended_group_field_precedence": eligible,
        "interpretation": (
            "A primary split requires a provenance field present on every row with "
            "enough independent groups. If status is blocked, add generator/template "
            "provenance; do not fall back to a random row split."
        ),
        "sealed_evaluation_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--group-fields",
        default=",".join(DEFAULT_GROUP_FIELDS),
        help="comma-separated candidate provenance fields",
    )
    args = parser.parse_args()
    fields = tuple(field.strip() for field in args.group_fields.split(",") if field.strip())
    result = inspect_jsonl(args.input, group_fields=fields, max_rows=args.max_rows)
    if args.out:
        atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_GROUPED_SPLIT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
