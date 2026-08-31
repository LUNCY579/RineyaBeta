"""Machine-checkable evidence manifests for CAG runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import ExperimentConfig

MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def git_fingerprint(cwd: str | Path | None = None) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD") or os.environ.get("CAG_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA")
    branch = run("rev-parse", "--abbrev-ref", "HEAD") or os.environ.get("CAG_SOURCE_BRANCH") or os.environ.get("GITHUB_REF_NAME")
    status = run("status", "--porcelain")
    if status is not None:
        dirty: bool | None = bool(status)
    else:
        raw_dirty = os.environ.get("CAG_SOURCE_DIRTY")
        if raw_dirty is None:
            dirty = None
        elif raw_dirty.strip().lower() in {"0", "false", "no", "clean"}:
            dirty = False
        elif raw_dirty.strip().lower() in {"1", "true", "yes", "dirty"}:
            dirty = True
        else:
            raise ValueError("CAG_SOURCE_DIRTY must be true/false when supplied")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def runtime_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def fingerprint_paths(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def fingerprint_model_dir(path: str | Path) -> dict[str, Any]:
    """Hash every model/config/tokenizer file in a local checkpoint directory."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    allowed_suffixes = {".json", ".model", ".txt", ".safetensors", ".bin"}
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and (candidate.suffix.lower() in allowed_suffixes or candidate.name.endswith(".tiktoken"))
    )
    if not files:
        raise ValueError(f"no checkpoint files found under {root}")
    records = []
    for file_path in files:
        records.append(
            {
                "path": file_path.relative_to(root).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return {
        "path": root.as_posix(),
        "files": records,
        "aggregate_sha256": sha256_json(records),
    }


def build_manifest(
    *,
    config: ExperimentConfig,
    status: str,
    phase: str,
    results: dict[str, Any],
    arms: list[dict[str, Any]],
    invariants: dict[str, Any],
    data_assets: list[dict[str, Any]] | None = None,
    model_assets: list[dict[str, Any]] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config.experiment_id,
        "phase": phase,
        "status": status,
        "claim_level": config.raw["claim_level"],
        "claim_boundary": (
            "Infrastructure and synthetic-mechanism evidence only. This manifest "
            "does not establish a Rineya capability, quality, efficiency, or "
            "cross-family knowledge-transfer improvement. Sealed evaluation remained closed."
        ),
        "config": {
            "path": config.source.as_posix(),
            "canonical_sha256": config.sha256,
            "schema_version": config.raw["schema_version"],
            "snapshot": config.raw,
        },
        "git": git_fingerprint(cwd),
        "runtime": runtime_fingerprint(),
        "seed": config.seed,
        "sealed_evaluation_opened": False,
        "arms": arms,
        "invariants": invariants,
        "results": results,
        "data_assets": data_assets or [],
        "model_assets": model_assets or [],
    }
    manifest["manifest_payload_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "generated_at_utc"}
    )
    return manifest


def verify_manifest(manifest: dict[str, Any]) -> None:
    """Reject a malformed, drifted, or claim-escalated CAG evidence manifest."""

    if int(manifest.get("manifest_schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported CAG evidence manifest schema")
    if manifest.get("claim_level") != "infrastructure_only":
        raise ValueError("CAG v1 evidence may not exceed infrastructure_only")
    if manifest.get("sealed_evaluation_opened") is not False:
        raise ValueError("manifest reports sealed evaluation as opened")
    invariants = manifest.get("invariants", {})
    if invariants.get("sealed_evaluation_opened") is not False:
        raise ValueError("manifest invariant does not prove sealed evaluation stayed closed")
    config = manifest.get("config", {})
    snapshot = config.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("manifest is missing its canonical config snapshot")
    expected_config_hash = sha256_json(snapshot)
    if config.get("canonical_sha256") != expected_config_hash:
        raise ValueError("config snapshot hash mismatch")
    claimed = manifest.get("manifest_payload_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("manifest payload hash is missing or invalid")
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"generated_at_utc", "manifest_payload_sha256"}
    }
    actual = sha256_json(payload)
    if actual != claimed:
        raise ValueError(f"manifest payload hash mismatch: expected {claimed}, got {actual}")


def load_verified_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest root must be an object: {source}")
    verify_manifest(value)
    return value
