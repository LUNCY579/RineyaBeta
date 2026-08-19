"""Locating, verifying and streaming the Rineya sequence-independence dataset.

The SPLIT TRAIN file is ~385 MB of JSON. Materialising it as Python dicts costs
several gigabytes, so this module builds a byte-offset index once and reads rows
on demand. Only the small metadata needed for ordering and budgeting
(``problem_id``, ``step_index``, ``total_steps``, ``category``) is held in RAM.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rineya_lab.config import DatasetConfig
from rineya_lab.data.bundle import extract_bundle, verify_checksums
from rineya_lab.data.records import FullRecord, Trajectory, TransitionRecord
from rineya_lab.utils.hashing import sha256_file
from rineya_lab.utils.io import iter_jsonl, read_json
from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("data.loading")

SPLITS = ("train", "dev", "test")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset cannot be located from the configured hints."""


class DatasetIntegrityError(RuntimeError):
    """Raised when the located dataset fails checksum or structural checks."""


@dataclass
class DatasetPaths:
    root: Path
    full_context: dict[str, Path]
    split_transitions: dict[str, Path]
    manifests: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "full_context": {k: str(v) for k, v in self.full_context.items()},
            "split_transitions": {k: str(v) for k, v in self.split_transitions.items()},
            "manifests": str(self.manifests),
        }


DEFAULT_DATASET_DIRNAME = "rineya_sequence_independence_dataset_v1"


def _is_explicit(hint: str | None) -> bool:
    """Whether the caller named a specific dataset location.

    An explicit hint disables the convenience search. Silently loading a
    *different* dataset than the one the operator named would invalidate every
    provenance record downstream, so an explicit-but-missing root is an error.
    """
    if not hint:
        return False
    return hint != DatasetConfig().root


def _candidate_roots(hint: str | None) -> list[Path]:
    """Search order for the dataset directory.

    A relative hint is tried against the working directory, the repository root
    and the repository's parent, so ``rineya-lab`` works both from the repo and
    from an installed console script started elsewhere. Convenience fallbacks are
    appended only when no explicit hint was supplied.
    """
    candidates: list[Path] = []
    repo_root = Path(__file__).resolve().parents[3]
    anchors = [Path.cwd(), repo_root, repo_root.parent]
    env_hint = os.environ.get("RINEYA_DATASET_ROOT")
    # Precedence: an explicit config/CLI root outranks the environment variable,
    # which in turn outranks the convenience search. A misspelled --dataset-root
    # must fail loudly rather than silently resolving to whatever RINEYA_DATASET_ROOT
    # happens to point at.
    if _is_explicit(hint):
        sources, explicit = [hint], True
    elif env_hint:
        sources, explicit = [env_hint], True
    else:
        sources, explicit = [hint], False
    for raw in sources:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend(anchor / path for anchor in anchors)
    if not explicit:
        candidates.extend(anchor / DEFAULT_DATASET_DIRNAME for anchor in anchors)
    # de-duplicate, keep order
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _looks_like_dataset(root: Path) -> bool:
    return (root / "data" / "full_context" / "train.jsonl").exists() and (
        root / "data" / "split_transitions" / "train.jsonl"
    ).exists()


def resolve_dataset(cfg: DatasetConfig) -> DatasetPaths:
    """Find the dataset directory, extracting the bundle ZIP if needed."""
    for candidate in _candidate_roots(cfg.root):
        if _looks_like_dataset(candidate):
            LOGGER.info("Using dataset root %s", candidate)
            return _paths_for(candidate)
        # tolerate an extra nesting level created by unzipping into a fresh dir
        if candidate.is_dir():
            for child in sorted(candidate.iterdir()):
                if child.is_dir() and _looks_like_dataset(child):
                    LOGGER.info("Using nested dataset root %s", child)
                    return _paths_for(child)

    if cfg.bundle_zip:
        destination = Path(cfg.extract_dir).expanduser()
        LOGGER.info("Dataset directory not found; extracting bundle %s", cfg.bundle_zip)
        extract_bundle(cfg.bundle_zip, destination)
        for candidate in [destination, *(p for p in sorted(destination.iterdir()) if p.is_dir())]:
            if _looks_like_dataset(candidate):
                return _paths_for(candidate)
        raise DatasetNotFoundError(f"Bundle {cfg.bundle_zip} did not contain a recognisable dataset layout")

    tried = "\n  ".join(str(p) for p in _candidate_roots(cfg.root))
    explicit_note = (
        "\nAn explicit dataset root was supplied, so the default search locations were NOT tried: "
        "loading a different dataset than the one you named would corrupt run provenance."
        if _is_explicit(cfg.root) or os.environ.get("RINEYA_DATASET_ROOT")
        else ""
    )
    raise DatasetNotFoundError(
        "Could not locate the Rineya sequence-independence dataset.\n"
        f"Tried:\n  {tried}{explicit_note}\n"
        "Set dataset.root in the config, pass --dataset-root, set RINEYA_DATASET_ROOT, "
        "or provide dataset.bundle_zip."
    )


def _paths_for(root: Path) -> DatasetPaths:
    root = root.resolve()
    return DatasetPaths(
        root=root,
        full_context={s: root / "data" / "full_context" / f"{s}.jsonl" for s in SPLITS},
        split_transitions={s: root / "data" / "split_transitions" / f"{s}.jsonl" for s in SPLITS},
        manifests=root / "data" / "manifests",
    )


@dataclass
class RowMeta:
    """Per-row metadata retained in memory for ordering and budgeting."""

    offset: int
    length: int
    problem_id: str
    step_index: int = 0
    total_steps: int = 0
    category: str = ""
    row_index: int = 0


@dataclass
class JsonlIndex:
    """Byte-offset index over a JSONL file plus lightweight per-row metadata."""

    path: Path
    rows: list[RowMeta] = field(default_factory=list)
    file_sha256: str | None = None

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def build(cls, path: str | Path, *, with_hash: bool = False) -> JsonlIndex:
        path = Path(path)
        index = cls(path=path)
        offset = 0
        row_index = 0
        with path.open("rb") as handle:
            for raw in handle:
                length = len(raw)
                stripped = raw.strip()
                if stripped:
                    record = json.loads(stripped.decode("utf-8"))
                    index.rows.append(
                        RowMeta(
                            offset=offset,
                            length=length,
                            problem_id=record.get("problem_id", ""),
                            step_index=int(record.get("step_index", 0) or 0),
                            total_steps=int(record.get("total_steps", record.get("step_count", 0)) or 0),
                            category=record.get("category", ""),
                            row_index=row_index,
                        )
                    )
                    row_index += 1
                offset += length
        if with_hash:
            index.file_sha256 = sha256_file(path)
        return index

    def read(self, row: RowMeta) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(row.offset)
            payload = handle.read(row.length)
        return json.loads(payload.decode("utf-8"))

    def read_many(self, rows: list[RowMeta]) -> Iterator[dict[str, Any]]:
        """Read rows in the given (arbitrary) order using one open handle.

        The global shuffle means access is random, not sequential; a single
        handle with seeks is dramatically cheaper than reopening per row.
        """
        with self.path.open("rb") as handle:
            for row in rows:
                handle.seek(row.offset)
                payload = handle.read(row.length)
                yield json.loads(payload.decode("utf-8"))


class RineyaDataset:
    """High-level accessor over one located dataset directory."""

    def __init__(self, paths: DatasetPaths, *, verify_hashes: bool = True, require_manifest: bool = True) -> None:
        self.paths = paths
        self._index_cache: dict[tuple[str, str], JsonlIndex] = {}
        self._integrity: dict[str, Any] = {"verified": False}
        self._check_structure(require_manifest=require_manifest)
        if verify_hashes:
            self.verify_integrity()

    # -- construction helpers -----------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DatasetConfig) -> RineyaDataset:
        return cls(resolve_dataset(cfg), verify_hashes=cfg.verify_hashes, require_manifest=cfg.require_manifest)

    def _check_structure(self, *, require_manifest: bool) -> None:
        missing = [str(p) for group in (self.paths.full_context, self.paths.split_transitions)
                   for p in group.values() if not p.exists()]
        if missing:
            raise DatasetIntegrityError(f"Dataset is missing expected files: {missing}")
        if require_manifest and not (self.paths.manifests / "dataset_stats.json").exists():
            raise DatasetIntegrityError(f"Dataset manifest missing: {self.paths.manifests / 'dataset_stats.json'}")

    def verify_integrity(self) -> dict[str, Any]:
        sums = self.paths.manifests / "sha256sums.txt"
        if not sums.exists():
            LOGGER.warning("No sha256sums.txt found under %s; skipping checksum verification", self.paths.manifests)
            self._integrity = {"verified": False, "reason": "sha256sums.txt not present"}
            return self._integrity
        report = verify_checksums(self.paths.root, sums, strict=False)
        if report.hash_mismatches:
            raise DatasetIntegrityError(
                "Dataset checksum verification failed:\n  " + "\n  ".join(report.hash_mismatches)
            )
        self._integrity = {
            "verified": True,
            "files_checked": report.verified_files,
            "missing_from_manifest": report.missing_from_manifest,
        }
        LOGGER.info("Verified %d dataset files against sha256sums.txt", report.verified_files)
        return self._integrity

    # -- accessors ----------------------------------------------------------------

    def path_for(self, variant: str, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        if variant == "full_context":
            return self.paths.full_context[split]
        if variant == "split_transitions":
            return self.paths.split_transitions[split]
        raise ValueError(f"Unknown variant {variant!r}")

    def index(self, variant: str, split: str) -> JsonlIndex:
        key = (variant, split)
        if key not in self._index_cache:
            path = self.path_for(variant, split)
            LOGGER.info("Indexing %s", path)
            self._index_cache[key] = JsonlIndex.build(path)
        return self._index_cache[key]

    def iter_full(self, split: str, *, limit: int | None = None) -> Iterator[FullRecord]:
        for row in iter_jsonl(self.path_for("full_context", split), limit=limit):
            yield FullRecord(row)

    def iter_transitions(self, split: str, *, limit: int | None = None) -> Iterator[TransitionRecord]:
        for row in iter_jsonl(self.path_for("split_transitions", split), limit=limit):
            yield TransitionRecord(row)

    def full_by_problem(self, split: str, *, limit: int | None = None) -> dict[str, FullRecord]:
        return {record.problem_id: record for record in self.iter_full(split, limit=limit)}

    def trajectories(self, split: str, *, problem_ids: set[str] | None = None) -> dict[str, Trajectory]:
        """Reassemble ordered trajectories from the globally shuffled rows."""
        full = {
            record.problem_id: record
            for record in self.iter_full(split)
            if problem_ids is None or record.problem_id in problem_ids
        }
        grouped: dict[str, list[TransitionRecord]] = {pid: [] for pid in full}
        for transition in self.iter_transitions(split):
            bucket = grouped.get(transition.problem_id)
            if bucket is not None:
                bucket.append(transition)
        out: dict[str, Trajectory] = {}
        for pid, record in full.items():
            steps = sorted(grouped[pid], key=lambda t: t.step_index)
            out[pid] = Trajectory(full=record, transitions=steps)
        return out

    # -- metadata -----------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        path = self.paths.manifests / "dataset_stats.json"
        return read_json(path) if path.exists() else {}

    def released_validation_report(self) -> dict[str, Any]:
        path = self.paths.manifests / "validation_report.json"
        return read_json(path) if path.exists() else {}

    def file_hashes(self, *, splits: tuple[str, ...] = SPLITS) -> dict[str, str]:
        """SHA-256 of every data file actually used. Recorded in the run manifest."""
        out: dict[str, str] = {}
        for split in splits:
            for variant in ("full_context", "split_transitions"):
                path = self.path_for(variant, split)
                out[f"{variant}/{split}.jsonl"] = sha256_file(path)
        return out

    def provenance(self, *, splits: tuple[str, ...] = SPLITS, include_hashes: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "paths": self.paths.as_dict(),
            "integrity": self._integrity,
            "released_stats": self.stats(),
            "released_validation_report": self.released_validation_report(),
        }
        if include_hashes:
            record["file_sha256"] = self.file_hashes(splits=splits)
        return record
