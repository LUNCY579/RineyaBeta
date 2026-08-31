"""Shared subprocess and exact-comparison helpers for RGE verification scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RGE = ROOT / "scripts" / "train" / "rge_peft.py"
HEAL = ROOT / "scripts" / "train" / "heal.py"
ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONIOENCODING": "utf-8",
}


def run_entry(entry: Path, config: Path, out: Path, steps: int,
              resume: str, *, geometry: str | None = None) -> None:
    command = [sys.executable, str(entry), "--config", str(config), "--out", str(out),
               "--max-steps", str(steps), "--resume", resume]
    if geometry is not None:
        command += ["--geometry", geometry]
    proc = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENV},
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    label = entry.name + (f" geometry={geometry}" if geometry else "")
    print(f"\n$ {label} --max-steps {steps} --resume {resume} --out {out.name}")
    for line in [line for line in proc.stdout.splitlines() if line.strip()][-10:]:
        print("    " + line)
    if proc.returncode != 0:
        print(proc.stdout[-6000:])
        print(proc.stderr[-6000:])
        raise SystemExit(f"{label} exited {proc.returncode}")


def load_latest(run_dir: Path) -> dict:
    checkpoint_dir = run_dir / "checkpoints"
    pointer = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    return torch.load(checkpoint_dir / pointer["checkpoint"], map_location="cpu",
                      weights_only=False)


def exact_problems(a, b, path: str = "root") -> list[str]:
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return [f"{path}: tensor/non-tensor mismatch"]
        if a.shape != b.shape or a.dtype != b.dtype:
            return [f"{path}: shape/dtype {tuple(a.shape)}/{a.dtype} != "
                    f"{tuple(b.shape)}/{b.dtype}"]
        if torch.equal(a, b):
            return []
        delta = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
        return [f"{path}: tensor differs max_abs={delta:.9g}"]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return [] if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) \
            and a.dtype == b.dtype and np.array_equal(a, b) else [f"{path}: ndarray differs"]
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return [f"{path}: mapping/type mismatch"]
        problems = []
        if set(a) != set(b):
            problems.append(f"{path}: keys differ +{sorted(set(b)-set(a), key=str)} "
                            f"-{sorted(set(a)-set(b), key=str)}")
        for key in sorted(set(a) & set(b), key=str):
            problems += exact_problems(a[key], b[key], f"{path}.{key}")
        return problems
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{path}: sequence type/length differs"]
        problems = []
        for index, (left, right) in enumerate(zip(a, b)):
            problems += exact_problems(left, right, f"{path}[{index}]")
        return problems
    try:
        equal = a == b
    except Exception:
        equal = False
    return [] if isinstance(equal, (bool, np.bool_)) and bool(equal) else \
        [f"{path}: {a!r} != {b!r}"]


def metric_sequence(run_dir: Path) -> list[tuple[int, float, float]]:
    rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return [(int(row["step"]), float(row["loss"]), float(row["lr"])) for row in rows]


def geometry_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "geometry.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_evidence(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8", newline="\n")
