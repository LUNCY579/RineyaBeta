#!/usr/bin/env python3
"""Prove that a *real* signal mid-run still yields a byte-identical result.

verify_resume.py stops cleanly at a step boundary via --max-steps, which tests the
resume path but not the interrupt path. This tests the thing that will actually
happen: a training process gets Ctrl-C'd (or the machine needs the GPU back),
partway through, at whatever step it happens to be on.

  run A : uninterrupted reference, N steps
  run B : launched with no step cap, signalled once it has logged >= K steps,
          then resumed in a new process and run to N

The signal arrives asynchronously, so run B's stopping point is whatever step it
reached -- which is the realistic case. Weights and Adam moments must still match
run A exactly at step N.

On Windows only SIGBREAK is reliably deliverable to a child process group, so the
child is spawned in its own group and sent CTRL_BREAK; POSIX gets SIGINT.

Usage:
  python verify_interrupt.py --config configs/heal_smoke.yaml --steps 8 --signal-after 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HEAL = ROOT / "scripts" / "train" / "heal.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_resume import ENV, adam_moments, compare_tensors, load_latest, run  # noqa: E402

IS_WINDOWS = os.name == "nt"


def count_logged_steps(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    try:
        with log_path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def launch_and_signal(config: Path, out: Path, signal_after: int,
                      timeout_s: float = 3600.0) -> int:
    cmd = [sys.executable, str(HEAL), "--config", str(config), "--out", str(out),
           "--resume", "none"]
    env = {**os.environ, **ENV}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    print(f"\n$ heal.py --out {out.name} (uncapped), will signal after "
          f"{signal_after} logged steps")
    proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT), creationflags=flags,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    log_path = out / "train_log.jsonl"
    sent = False
    deadline = time.monotonic() + timeout_s
    while proc.poll() is None:
        if not sent and count_logged_steps(log_path) >= signal_after:
            time.sleep(0.05)  # let the step finish flushing its log line
            print(f"  -> sending {'CTRL_BREAK' if IS_WINDOWS else 'SIGINT'} "
                  f"at {count_logged_steps(log_path)} logged steps")
            proc.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGINT)
            sent = True
        if time.monotonic() > deadline:
            proc.kill()
            raise SystemExit("timed out waiting for the child to stop")
        time.sleep(0.2)
    out_text = proc.stdout.read() if proc.stdout else ""
    for line in [l for l in out_text.splitlines() if l.strip()][-8:]:
        print("    " + line)
    if not sent:
        raise SystemExit("child finished before the signal could be delivered; "
                         "raise --steps or lower --signal-after")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/heal_smoke.yaml")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--signal-after", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    config = ROOT / args.config
    base = ROOT / "artifacts" / "verify_interrupt"
    run_a, run_b = base / "run_a", base / "run_b"
    for path in (run_a, run_b):
        shutil.rmtree(path, ignore_errors=True)

    print(f"config={args.config}  steps={args.steps}  "
          f"signal_after={args.signal_after}")
    print("=" * 72)
    print("A: uninterrupted reference")
    run(config, run_a, args.steps, "none")

    print("\n" + "=" * 72)
    print("B: signalled mid-run, then resumed")
    code = launch_and_signal(config, run_b, args.signal_after)
    pointer = run_b / "checkpoints" / "latest.json"
    if not pointer.exists():
        raise SystemExit("interrupted run left no checkpoint pointer")
    stopped_at = json.loads(pointer.read_text(encoding="utf-8"))["global_step"]
    print(f"  child exited {code}, checkpointed at step {stopped_at}")
    if code != 0:
        raise SystemExit(f"expected a clean exit 0 after the signal, got {code}")
    if stopped_at >= args.steps:
        raise SystemExit(f"signal landed too late (step {stopped_at} >= "
                         f"{args.steps}); nothing was left to resume")

    run(config, run_b, args.steps, "auto")

    pa, pb = load_latest(run_a), load_latest(run_b)
    print("\n" + "=" * 72)
    print(f"A ended at {pa['position']['global_step']}, "
          f"B at {pb['position']['global_step']} "
          f"(interrupted at {stopped_at})")

    problems = []
    if pa["position"]["global_step"] != pb["position"]["global_step"]:
        problems.append("global_step differs")
    same_w, diff_w, p1 = compare_tensors(pa["adapter_state"], pb["adapter_state"], "lora")
    same_o, diff_o, p2 = compare_tensors(adam_moments(pa), adam_moments(pb), "adam")
    problems += p1 + p2

    print(f"\nLoRA tensors  : {same_w} identical, {diff_w} differing")
    print(f"Adam moments  : {same_o} identical, {diff_o} differing")
    if problems:
        print("\nFAIL -- interrupt+resume is not bit-exact:")
        for line in problems[:20]:
            print(f"  - {line}")
        raise SystemExit(1)

    print(f"\nPASS: a run signalled at step {stopped_at} and resumed is "
          f"byte-identical to one that ran straight through")
    if not args.keep:
        for path in (run_a, run_b):
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
