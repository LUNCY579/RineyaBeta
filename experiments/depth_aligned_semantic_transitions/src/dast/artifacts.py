"""Reproducible run artifacts and checkpoint state."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dast.config import DASTConfig
from dast.data import sha256_file
from dast.model import DASTModel


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def _git_revision() -> str | None:
    explicit = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def environment_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_revision": _git_revision(),
    }
    if torch.cuda.is_available():
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(0)
    for package in ("transformers", "peft", "accelerate", "safetensors"):
        try:
            module = __import__(package)
            result[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            result[package] = None
    return result


def build_run_manifest(
    config: DASTConfig,
    *,
    data_path: str | Path,
    checkpoint_layers: tuple[int, ...],
    model_info: dict[str, Any],
    status: str = "BUILT_UNRUN",
) -> dict[str, Any]:
    split = config.data.eval_split.lower()
    return {
        "schema_version": "dast-run-manifest-v1",
        "status": status,
        "experiment": config.experiment.name,
        "arm": config.experiment.arm,
        "seed": config.experiment.seed,
        "config_sha256": config.canonical_sha256(),
        "config": config.as_dict(),
        "data": {
            "path": str(Path(data_path).resolve()),
            "sha256": sha256_file(data_path),
            "train_split": config.data.train_split,
            "eval_split": config.data.eval_split,
            "sealed_test_opened": split == "test",
        },
        "checkpoint_layers": list(checkpoint_layers),
        "model": model_info,
        "environment": environment_manifest(),
        "claim_boundary": (
            "Engineering and DEV evidence only. A run manifest does not establish a Rineya quality claim; "
            "TEST remains sealed unless the frozen preregistration hash explicitly opens it."
        ),
    }


def save_rng_state(path: str | Path) -> None:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_rng_state(path: str | Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def save_checkpoint(
    output_dir: str | Path,
    *,
    model: DASTModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None = None,
    epoch: int,
    global_step: int,
    config_sha256: str,
) -> Path:
    root = Path(output_dir) / f"checkpoint-{global_step:08d}"
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.auxiliary_state_dict(), root / "dast_auxiliary.pt")
    base = model.base_model
    if hasattr(base, "save_pretrained"):
        base.save_pretrained(root / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "config_sha256": config_sha256,
        },
        root / "training_state.pt",
    )
    save_rng_state(root / "rng_state.pt")
    checksums: dict[str, str] = {}
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        checksums[str(file_path.relative_to(root))] = sha256_file(file_path)
    atomic_write_json(root / "checksums.json", checksums)
    return root


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
