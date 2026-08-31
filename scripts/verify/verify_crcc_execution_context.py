#!/usr/bin/env python3
"""Fail closed on runtime or Git drift before CRCC scientific inference."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG_HEAD = "242613d1b4b5e0a5a4a7154c60337115ca394bd7"
EXPECTED_BRANCH = "research/catastrophic-removal-core-v1"
EXPECTED = {
    "python": "3.11",
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "accelerate": "1.10.1",
    "PyYAML": "6.0.2",
}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True, capture_output=True
    )


def base_version(value: str) -> str:
    return value.split("+", 1)[0]


def main() -> None:
    python_version = ".".join(platform.python_version_tuple()[:2])
    if python_version != EXPECTED["python"]:
        raise RuntimeError(f"CRCC requires Python {EXPECTED['python']}; found {platform.python_version()}")

    try:
        import accelerate
        import torch
        import transformers
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "CRCC runtime dependencies are incomplete; use bootstrap_and_run_crcc.ps1"
        ) from exc

    observed = {
        "python": python_version,
        "torch": base_version(torch.__version__),
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "PyYAML": yaml.__version__,
    }
    drift = {
        key: {"expected": EXPECTED[key], "observed": value}
        for key, value in observed.items()
        if value != EXPECTED[key]
    }
    if drift:
        raise RuntimeError(
            "CRCC pinned runtime drifted: "
            + ", ".join(
                f"{key} expected {row['expected']} found {row['observed']}"
                for key, row in drift.items()
            )
        )

    branch = git("branch", "--show-current").stdout.strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"CRCC execution must run on {EXPECTED_BRANCH}; found {branch!r}")

    head = git("rev-parse", "HEAD").stdout.strip()
    ancestry = git("merge-base", "--is-ancestor", PREREG_HEAD, head, check=False)
    if ancestry.returncode != 0:
        raise RuntimeError(f"HEAD {head} is not descended from CRCC preregistration {PREREG_HEAD}")

    dirty = git("status", "--porcelain", "--untracked-files=all").stdout.strip()
    if dirty:
        preview = "\n".join(dirty.splitlines()[:20])
        raise RuntimeError(
            "CRCC scientific inference requires a clean tracked/untracked worktree. "
            "Commit, stash, or remove local changes first.\n" + preview
        )

    print("CRCC execution context: PASS")
    print(f"branch: {branch}")
    print(f"head: {head}")
    for key in ("python", "torch", "transformers", "accelerate", "PyYAML"):
        print(f"{key}: {observed[key]}")


if __name__ == "__main__":
    main()
