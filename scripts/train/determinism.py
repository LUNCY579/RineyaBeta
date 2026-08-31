#!/usr/bin/env python3
"""Bit-exact reproducibility controls for the healing runs.

The guarantee this module underwrites: **for a fixed plan, a resumed run produces
byte-identical weights to an uninterrupted one.** That requires four things, and
all four are enforced rather than hoped for.

  1. Every RNG stream is seeded and, at checkpoint time, captured and restored --
     Python, NumPy, torch CPU, and every CUDA device.
  2. Nondeterministic kernels are refused, not tolerated. `use_deterministic_
     algorithms(True)` raises on any op lacking a deterministic implementation,
     so a silent source of drift becomes a loud crash at step 0.
  3. cuBLAS needs `CUBLAS_WORKSPACE_CONFIG` set *before* CUDA initialises. It
     cannot be fixed from inside a running process, so this module checks it and
     refuses to start without it.
  4. Attention must not use the flash or mem-efficient SDPA backends: their
     backward passes accumulate nondeterministically. `math` is slower and needs
     more memory (it materialises the score matrix) and is the only bit-exact
     option. This is the deliberate speed-for-exactness trade.

What it cannot promise: identity across different GPUs, driver versions, or torch
builds. Reduction order is a property of the kernel that ran. `fingerprint()`
records the environment so a changed one is detected instead of silently
producing a different model.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REQUIRED_ENV = {
    # ":4096:8" makes cuBLAS use a fixed workspace, which is what makes its
    # reductions reproducible. Must precede the first CUDA call.
    "CUBLAS_WORKSPACE_CONFIG": (":4096:8", ":16:8"),
    # Not strictly needed for weights, but keeps set/dict iteration stable in any
    # helper that touches them.
    "PYTHONHASHSEED": ("0",),
}


@dataclass
class RngState:
    python: Any
    numpy: Any
    torch_cpu: torch.Tensor
    torch_cuda: list = field(default_factory=list)

    @staticmethod
    def capture() -> "RngState":
        return RngState(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else [],
        )

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch_cpu)
        if self.torch_cuda and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda)


def check_env(strict: bool = True) -> list[str]:
    """Verify the env vars that cannot be set after CUDA starts."""
    problems = []
    for key, accepted in REQUIRED_ENV.items():
        value = os.environ.get(key)
        if value not in accepted:
            problems.append(
                f"{key}={value!r} but must be one of {accepted}. "
                f"Set it before launching python (see scripts/train/run_heal.ps1)."
            )
    if problems and strict:
        raise SystemExit("determinism preconditions not met:\n  - "
                         + "\n  - ".join(problems))
    return problems


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable(seed: int, strict: bool = True) -> None:
    """Put the process into bit-exact mode. Call before building the model."""
    check_env(strict=strict)
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Refuse the reduced-precision reduction shortcuts; they are not reproducible
    # across launch configurations.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "matmul"):
        for attr in ("allow_fp16_reduced_precision_reduction",
                     "allow_bf16_reduced_precision_reduction"):
            if hasattr(torch.backends.cuda.matmul, attr):
                setattr(torch.backends.cuda.matmul, attr, False)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def fingerprint() -> dict:
    """Environment identity. A change here invalidates bit-exactness claims."""
    fp = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            fp["devices"].append({
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_mib": props.total_memory // (1 << 20),
            })
    return fp


def code_fingerprint(module_dir: Path | None = None) -> dict:
    """Hash of the training source itself.

    The plan hash covers *what* the optimizer sees; it says nothing about the code
    that consumes it. Editing the loss, the collate function, or the optimizer
    setup leaves every other guard satisfied while silently changing the numerics,
    so a resumed run would diverge from its own checkpoint undetected.

    Line endings are normalised before hashing. `.gitattributes` declares
    `*.py text eol=lf`, so a checkout can rewrite CRLF to LF in the working copy
    without changing a single token -- and a byte-level hash would then refuse a
    perfectly valid resume. Newlines are the one difference that is provably inert
    for Python semantics, so they are the one difference excluded.

    Everything else still trips this, including comment-only edits. That is the
    conservative direction: a hash cannot tell a docstring from a reduction order.
    Override with `resume.allow_code_drift` when the change is known to be inert.
    """
    directory = (module_dir or Path(__file__).resolve().parent).resolve()
    files = {}
    # Training-critical modules now live below scripts/train/optim as well as at
    # the top level.  A shallow glob would let an optimizer change escape the
    # exact-resume guard.  Keep the traversal deliberately narrow (Python source
    # below the training package), deterministic, and independent of checkout
    # path.  Generated bytecode and cache directories are excluded by selecting
    # source files and filtering path components explicitly.
    paths = [
        path for path in directory.rglob("*.py")
        if not any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"}
                   for part in path.relative_to(directory).parts)
    ]
    for path in sorted(paths, key=lambda item: item.relative_to(directory).as_posix()):
        source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        name = path.relative_to(directory).as_posix()
        files[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    combined = hashlib.sha256(
        "".join(f"{name}:{digest}" for name, digest in sorted(files.items())).encode()
    ).hexdigest()
    return {"schema_version": 2, "files": files, "combined": combined}


def compare_code_fingerprint(saved: dict, current: dict) -> list[str]:
    """Modules whose source changed since the checkpoint was written."""
    if not saved:
        return ["checkpoint predates code fingerprinting"]
    if saved.get("combined") == current.get("combined"):
        return []
    old, new = saved.get("files", {}), current.get("files", {})
    drift = []
    for name in sorted(set(old) | set(new)):
        if name not in old:
            drift.append(f"{name}: added")
        elif name not in new:
            drift.append(f"{name}: removed")
        elif old[name] != new[name]:
            drift.append(f"{name}: modified")
    return drift


def compare_fingerprint(saved: dict, current: dict) -> list[str]:
    """Fields whose change breaks bit-exact replay."""
    drift = []
    for key in ("torch", "cuda_runtime", "cudnn", "cublas_workspace_config"):
        if saved.get(key) != current.get(key):
            drift.append(f"{key}: {saved.get(key)!r} -> {current.get(key)!r}")
    old = [d.get("name") for d in saved.get("devices", [])]
    new = [d.get("name") for d in current.get("devices", [])]
    if old != new:
        drift.append(f"devices: {old} -> {new}")
    return drift
