"""Authoritative local qualification gate for DAST.

This replaces GitHub Actions as the qualification record. It is deliberately
STRONGER than the workflow it replaces, in one specific way that matters: it runs
the test suite BOTH with the accelerator visible and with it hidden.

That is not a nicety. The 2026-08-30 qualification record said "pytest PASS (37
tests)" without naming a host, and it was reproducible only on a CPU-only machine:
a CPU/CUDA device comparison in the smoke crashed on every GPU host. A CPU-only
hosted runner could never have caught it, because the defect only exists when a
GPU is present. Any gate that reports one number for one device configuration can
hide that class of bug again.

Usage
-----
    python scripts/qualify.py                       # full gate, writes the artifact
    python scripts/qualify.py --fast                # skip the smoke and the CPU-only pass
    python scripts/qualify.py --output some.json    # alternate artifact path

Exit code is 0 only if every step passed in every device configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "qualification" / "latest.json"


def _run(command: list[str], *, env: dict[str, str], cwd: Path) -> tuple[bool, str, float]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True, timeout=3600
        )
        ok = completed.returncode == 0
        tail = (completed.stdout + completed.stderr).strip().splitlines()
        return ok, "\n".join(tail[-12:]), time.monotonic() - started
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT after 3600s", time.monotonic() - started


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _environment() -> dict[str, object]:
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the artifact does not describe any committed state.
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    for package in ("torch", "transformers", "peft", "accelerate", "numpy", "pytest", "ruff"):
        try:
            module = __import__(package)
            info[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[package] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["cuda_available"] = None
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local qualification gate for DAST")
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--fast", action="store_true", help="skip the smoke and CPU-only pass")
    args = parser.parse_args(argv)

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = str(ROOT / "src")
    base_env["PYTHONHASHSEED"] = "0"
    base_env["PYTHONIOENCODING"] = "utf-8"
    cpu_env = dict(base_env)
    cpu_env["CUDA_VISIBLE_DEVICES"] = "-1"

    python = sys.executable
    steps: list[tuple[str, list[str], dict[str, str]]] = [
        ("compileall", [python, "-m", "compileall", "-q", "src", "tests"], base_env),
        ("ruff", [python, "-m", "ruff", "check", "src", "tests"], base_env),
        ("pytest_device_visible", [python, "-m", "pytest", "-q"], base_env),
    ]
    if not args.fast:
        steps += [
            # The pass a CPU-only hosted runner would have given us, kept so a
            # regression that only shows on one device configuration cannot hide.
            ("pytest_cpu_only", [python, "-m", "pytest", "-q"], cpu_env),
            (
                "validate_fixture",
                [python, "-m", "dast.cli", "validate-data", "fixtures/tiny_reasoning.jsonl"],
                base_env,
            ),
            (
                "smoke",
                [python, "-m", "dast.cli", "smoke", "--output-dir", "runs/qualify-smoke"],
                base_env,
            ),
        ]

    print(f"DAST local qualification  commit={_git('rev-parse', '--short', 'HEAD')}")
    results: list[dict[str, object]] = []
    ok_all = True
    for name, command, env in steps:
        print(f"  {name:24s} ", end="", flush=True)
        ok, tail, seconds = _run(command, env=env, cwd=ROOT)
        print(f"{'PASS' if ok else 'FAIL'}  ({seconds:.1f}s)")
        if not ok:
            print("\n".join(f"      {line}" for line in tail.splitlines()))
        ok_all = ok_all and ok
        results.append(
            {
                "step": name,
                "passed": ok,
                "seconds": round(seconds, 2),
                "cuda_visible": env.get("CUDA_VISIBLE_DEVICES") != "-1",
                "output_tail": tail if not ok else "",
            }
        )

    environment = _environment()
    artifact = {
        "schema_version": "dast-local-qualification-v1",
        "verdict": "PASS" if ok_all else "FAIL",
        "mode": "fast" if args.fast else "full",
        "environment": environment,
        "steps": results,
        "claim_boundary": (
            "Engineering evidence only. A qualification pass does not establish any "
            "Rineya capability claim, and it does not validate the GitHub Actions "
            "workflow file itself - only the commands that workflow would invoke."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nVERDICT: {artifact['verdict']}   artifact: {destination}")
    if environment["git_dirty"]:
        print("WARNING: working tree is dirty, so this artifact describes no committed state.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
