"""Capture software and hardware environment for the run manifest."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any


def _safe(callable_):
    try:
        return callable_()
    except Exception as exc:  # pragma: no cover - defensive
        return f"<unavailable: {exc}>"


def _package_versions() -> dict[str, str]:
    names = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "bitsandbytes",
        "deepspeed",
        "numpy",
        "yaml",
        "sympy",
        "jsonschema",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            out[name] = "not-installed"
    return out


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "devices": []}
    try:
        import torch

        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:  # pragma: no cover - GPU only
            info["device_count"] = torch.cuda.device_count()
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                info["devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": props.total_memory,
                        "capability": f"{props.major}.{props.minor}",
                    }
                )
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    if not info["devices"] and shutil.which("nvidia-smi"):  # pragma: no cover - host dependent
        info["nvidia_smi"] = _safe(
            lambda: subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout.strip()
        )
    return info


def _git_info() -> dict[str, Any]:  # pragma: no cover - depends on checkout
    if not shutil.which("git"):
        return {"available": False}
    def run(args: list[str]) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run(["rev-parse", "HEAD"])
    if commit is None:
        return {"available": False}
    return {
        "available": True,
        "commit": commit,
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
    }


def collect_environment() -> dict[str, Any]:
    """Full environment snapshot recorded with every run."""
    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "gpu": _gpu_info(),
        "git": _git_info(),
        "env_vars": {
            key: os.environ.get(key)
            for key in [
                "PYTHONHASHSEED",
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_OFFLINE",
                "TOKENIZERS_PARALLELISM",
                "OMP_NUM_THREADS",
            ]
        },
        "argv": list(sys.argv),
    }
