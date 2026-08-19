"""JSON / JSONL IO helpers with strict UTF-8 and newline handling."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def json_dumps(obj: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def iter_jsonl(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file one record at a time.

    The TRAIN split of the split-transition dataset is ~385 MB; nothing in this
    repository loads it eagerly unless a caller explicitly materialises it.
    """
    path = Path(path)
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            yield record
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, limit=limit))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, sort_keys: bool = True) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json_dumps(row, sort_keys=sort_keys))
            handle.write("\n")
            count += 1
    return count


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, obj: Any, *, indent: int = 2, sort_keys: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    atomic_write_text(path, text + "\n")
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write via a temporary file + rename so an interrupted run never leaves a
    half-written manifest that later looks like a complete record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def count_lines(path: str | Path) -> int:
    total = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            total += chunk.count(b"\n")
    return total
