#!/usr/bin/env python3
"""Shared contracts for the catastrophic-removal critical-core experiment.

The helpers in this module are deliberately model-light.  They keep provenance,
path safety, deterministic case selection, and layer mappings testable without
loading torch or a checkpoint.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "crcc.v1"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


class ContractError(ValueError):
    """Raised when an experiment artifact would be ambiguous or incomparable."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def assert_unsealed_path(path: Path, *, purpose: str) -> None:
    """Refuse obvious held-out assets before opening them.

    The repository's sealed assets intentionally carry ``SEALED`` in their name.
    We fail on any path component containing that marker rather than relying on a
    particular filename.  This is a guardrail, not a substitute for dataset-role
    metadata in the final evaluation harness.
    """

    marked = [part for part in path.parts if "SEALED" in part.upper()]
    if marked:
        raise ContractError(
            f"refusing sealed asset for {purpose}: {path} (marked component(s): {marked})"
        )


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("PyYAML is required for CRCC configuration files") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected a YAML mapping in {path}")
    return payload


def dump_yaml(value: Mapping[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("PyYAML is required for CRCC configuration files") from exc
    return yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def _record_messages(
    row: Mapping[str, Any], *, source: str, line_number: int
) -> tuple[list[dict[str, str]], str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContractError(f"{source}:{line_number}: expected a non-empty messages list")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ContractError(f"{source}:{line_number}: message {index} is not an object")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ContractError(
                f"{source}:{line_number}: unsupported role {role!r} at message {index}"
            )
        normalized.append({"role": role, "content": content})

    explicit_reference = row.get("reference")
    if normalized[-1]["role"] == "assistant":
        reference = normalized[-1]["content"]
        prompt_messages = normalized[:-1]
    elif explicit_reference is not None:
        reference = str(explicit_reference)
        prompt_messages = normalized
    else:
        raise ContractError(
            f"{source}:{line_number}: diagnostic rows need a final assistant message "
            "or an explicit reference"
        )
    if not prompt_messages or not reference.strip():
        raise ContractError(f"{source}:{line_number}: empty prompt or reference")
    return prompt_messages, reference


def _selection_key(seed: int, source: str, line_number: int, row: Mapping[str, Any]) -> int:
    stable_id = row.get("id") or row.get("case_id") or row.get("uid") or line_number
    material = f"{seed}\0{source}\0{stable_id}\0{line_number}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def select_jsonl_records(
    path: Path,
    *,
    take: int,
    seed: int,
    source_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the smallest deterministic hashes without loading a whole corpus.

    Selection is independent of file traversal order after ``line_number`` is
    included in the key and uses only training/diagnostic assets accepted by the
    caller.  A bounded max-heap keeps memory proportional to ``take``.
    """

    if take <= 0:
        raise ContractError(f"take must be positive for {source_label}")
    assert_unsealed_path(path, purpose="critical-layer discovery")

    heap: list[tuple[int, int, dict[str, Any]]] = []
    total_rows = 0
    usable_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ContractError(f"{path}:{line_number}: expected a JSON object")
            try:
                prompt_messages, reference = _record_messages(
                    row, source=repo_relative(path), line_number=line_number
                )
            except ContractError:
                # A corpus may contain metadata rows.  They are not silently used,
                # but unusable rows do not make an otherwise valid source unreadable.
                continue
            usable_rows += 1
            key = _selection_key(seed, source_label, line_number, row)
            candidate = {
                "selection_key": f"{key:064x}",
                "line_number": line_number,
                "source": source_label,
                "source_path": repo_relative(path),
                "source_id": str(
                    row.get("id") or row.get("case_id") or row.get("uid") or line_number
                ),
                "messages": prompt_messages,
                "reference": reference,
                "category": str(
                    row.get("category")
                    or (row.get("metadata") or {}).get("category")
                    or (row.get("metadata") or {}).get("reply_class")
                    or "uncategorized"
                ),
            }
            # Python's heap is a min-heap.  Negative keys make heap[0] the worst
            # (largest real key) retained item.
            item = (-key, -line_number, candidate)
            if len(heap) < take:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    if usable_rows < take:
        raise ContractError(
            f"{path} supplied only {usable_rows} usable rows, fewer than requested {take}"
        )
    selected = [item[2] for item in heap]
    selected.sort(key=lambda row: (row["selection_key"], row["line_number"]))
    for row in selected:
        row["id"] = f"{source_label}:{row['source_id']}:{row['line_number']}"
    provenance = {
        "label": source_label,
        "path": repo_relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "total_nonempty_rows": total_rows,
        "usable_rows": usable_rows,
        "selected_rows": len(selected),
        "selection_seed": seed,
    }
    return selected, provenance


def load_diagnostic_records(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ContractError("config.data must be a mapping")
    sources = data.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ContractError("config.data.sources must be a non-empty list")
    seed = int(config.get("seed", 0))

    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    labels: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ContractError("each data source must be a mapping")
        label = str(source.get("label", "")).strip()
        if not label or label in labels:
            raise ContractError(f"source labels must be non-empty and unique: {label!r}")
        labels.add(label)
        path = resolve_repo_path(str(source["path"]), must_exist=True)
        selected, source_provenance = select_jsonl_records(
            path,
            take=int(source["take"]),
            seed=seed,
            source_label=label,
        )
        records.extend(selected)
        provenance.append(source_provenance)

    ids = [str(row["id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ContractError("diagnostic record ids are not unique")
    records.sort(key=lambda row: row["id"])
    minimum = int(data.get("minimum_cases", 1))
    if len(records) < minimum:
        raise ContractError(f"diagnostic set has {len(records)} rows; minimum is {minimum}")
    return records, provenance


def parse_layer_selection(value: Any, *, num_layers: int) -> list[int]:
    if value in (None, "all"):
        return list(range(num_layers))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError("scan.layers must be 'all' or a list of integers")
    layers = [int(item) for item in value]
    if len(layers) != len(set(layers)):
        raise ContractError("layer selection contains duplicates")
    if any(layer < 0 or layer >= num_layers for layer in layers):
        raise ContractError(f"layer selection must lie in [0, {num_layers - 1}]")
    return sorted(layers)


def validate_layer_indices(indices: Sequence[int], *, num_layers: int, label: str) -> list[int]:
    normalized = [int(value) for value in indices]
    if not normalized:
        raise ContractError(f"{label} layer list is empty")
    if normalized != sorted(normalized):
        raise ContractError(f"{label} layers must preserve source order")
    if len(normalized) != len(set(normalized)):
        raise ContractError(f"{label} layers contain duplicates")
    if normalized[0] < 0 or normalized[-1] >= num_layers:
        raise ContractError(f"{label} layers must lie in [0, {num_layers - 1}]")
    return normalized


def layer_mapping(indices: Sequence[int]) -> list[dict[str, int]]:
    return [
        {"reduced_layer": reduced, "source_layer": int(source)}
        for reduced, source in enumerate(indices)
    ]


def mapping_source_layers(mapping: Sequence[Mapping[str, Any]]) -> list[int]:
    expected = list(range(len(mapping)))
    reduced = [int(entry["reduced_layer"]) for entry in mapping]
    if reduced != expected:
        raise ContractError(f"reduced layer mapping must be contiguous {expected}, got {reduced}")
    source = [int(entry["source_layer"]) for entry in mapping]
    if source != sorted(source) or len(source) != len(set(source)):
        raise ContractError("source layer mapping must be strictly increasing")
    return source


def checkpoint_inventory(path: Path, *, hash_weights: bool) -> dict[str, Any]:
    """Create a stable local-checkpoint identity without ever copying weights."""

    if not path.is_dir():
        raise ContractError(f"checkpoint must be a local directory: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ContractError(f"checkpoint lacks config.json: {path}")
    files: list[dict[str, Any]] = []
    for candidate in sorted(path.iterdir(), key=lambda p: p.name):
        if not candidate.is_file():
            continue
        item: dict[str, Any] = {"name": candidate.name, "bytes": candidate.stat().st_size}
        metadata_names = {
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        }
        if candidate.name in metadata_names or (
            hash_weights and candidate.suffix.lower() in WEIGHT_SUFFIXES
        ):
            item["sha256"] = sha256_file(candidate)
        files.append(item)
    identity = {
        "path": repo_relative(path),
        "config_sha256": sha256_file(config_path),
        "files": files,
        "weights_hashed": bool(hash_weights),
    }
    identity["inventory_sha256"] = object_sha256(identity)
    return identity


def deep_without_paths(value: Mapping[str, Any], dotted_paths: Sequence[str]) -> dict[str, Any]:
    """Return a deep copy with declared dotted keys removed for matched-arm checks."""

    copied: MutableMapping[str, Any] = deepcopy(dict(value))
    for dotted in dotted_paths:
        parts = dotted.split(".")
        cursor: MutableMapping[str, Any] | None = copied
        for part in parts[:-1]:
            next_value = cursor.get(part) if cursor is not None else None
            if not isinstance(next_value, MutableMapping):
                cursor = None
                break
            cursor = next_value
        if cursor is not None:
            cursor.pop(parts[-1], None)
    return dict(copied)
