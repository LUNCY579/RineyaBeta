#!/usr/bin/env python3
"""Prove that an interrupted-and-resumed run equals an uninterrupted one, bitwise.

The claim in heal.py's docstring is only worth something if it is tested, so this
runs the real entrypoint in real subprocesses -- not an in-process simulation that
would share RNG state and hide exactly the bugs being looked for.

  run A :  fresh -> N steps
  run B :  fresh -> K steps  |  new process, --resume auto -> N steps

Then every LoRA tensor and every Adam moment from both runs is compared with
`torch.equal`. Any difference at all is a failure: "close" is not the claim.

Adam moments are checked as well as weights because optimizer state is the usual
place resume bugs hide. Weights can coincide for a step or two while `exp_avg`
has already diverged, and the drift only becomes visible in the loss much later.

Usage:
  python verify_resume.py --config configs/heal_smoke.yaml --steps 6 --break-at 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HEAL = ROOT / "scripts" / "train" / "heal.py"

ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONIOENCODING": "utf-8",
}


def run(config: Path, out: Path, max_steps: int, resume: str) -> None:
    cmd = [sys.executable, str(HEAL), "--config", str(config), "--out", str(out),
           "--max-steps", str(max_steps), "--resume", resume]
    env = {**os.environ, **ENV}
    print(f"\n$ heal.py --out {out.name} --max-steps {max_steps} --resume {resume}")
    # Explicit utf-8: the Windows default is cp1252 and mangles the child's output.
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    tail = [l for l in proc.stdout.splitlines() if l.strip()][-6:]
    for line in tail:
        print("    " + line)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit(f"heal.py exited {proc.returncode}")


def load_latest(run_dir: Path) -> dict:
    import json
    ckpt_dir = run_dir / "checkpoints"
    name = json.loads((ckpt_dir / "latest.json").read_text(encoding="utf-8"))["checkpoint"]
    return torch.load(ckpt_dir / name, map_location="cpu", weights_only=False)


def compare_tensors(a: dict, b: dict, label: str) -> tuple[int, int, list[str]]:
    same = diff = 0
    problems = []
    keys_a, keys_b = set(a), set(b)
    if keys_a != keys_b:
        problems.append(f"{label}: key sets differ "
                        f"(+{sorted(keys_b - keys_a)[:3]} -{sorted(keys_a - keys_b)[:3]})")
    for key in sorted(keys_a & keys_b):
        ta, tb = a[key], b[key]
        if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
            continue
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            diff += 1
            problems.append(f"{label}:{key} shape/dtype {tuple(ta.shape)}/{ta.dtype}"
                            f" vs {tuple(tb.shape)}/{tb.dtype}")
        elif torch.equal(ta, tb):
            same += 1
        else:
            diff += 1
            delta = (ta.float() - tb.float()).abs().max().item()
            problems.append(f"{label}:{key} differs, max|delta|={delta:.3e}")
    return same, diff, problems


def adam_moments(payload: dict) -> dict:
    out = {}
    state = payload["optimizer_state"].get("state", {})
    for pid, entry in state.items():
        for field in ("exp_avg", "exp_avg_sq"):
            if field in entry and torch.is_tensor(entry[field]):
                out[f"p{pid}.{field}"] = entry[field]
        if "step" in entry:
            step = entry["step"]
            out[f"p{pid}.step"] = (step if torch.is_tensor(step)
                                   else torch.tensor(float(step)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/heal_smoke.yaml")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--break-at", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="do not delete run dirs")
    args = ap.parse_args()

    if not 0 < args.break_at < args.steps:
        raise SystemExit("--break-at must be strictly between 0 and --steps")

    config = ROOT / args.config
    base = ROOT / "artifacts" / "verify_resume"
    run_a, run_b = base / "run_a", base / "run_b"
    for path in (run_a, run_b):
        shutil.rmtree(path, ignore_errors=True)
    # Share the tokenizer cache so neither run pays for it twice.
    (base / "cache").mkdir(parents=True, exist_ok=True)

    print(f"config={args.config}  steps={args.steps}  break_at={args.break_at}")
    print("=" * 72)
    print("A: uninterrupted reference")
    run(config, run_a, args.steps, "none")

    print("\n" + "=" * 72)
    print(f"B: interrupted at {args.break_at}, then resumed in a new process")
    run(config, run_b, args.break_at, "none")
    run(config, run_b, args.steps, "auto")

    pa, pb = load_latest(run_a), load_latest(run_b)
    print("\n" + "=" * 72)
    print(f"A ended at global_step {pa['position']['global_step']}, "
          f"B at {pb['position']['global_step']}")

    problems = []
    if pa["position"]["global_step"] != pb["position"]["global_step"]:
        problems.append("global_step differs; runs are not comparable")
    if pa["plan_hash"] != pb["plan_hash"]:
        problems.append("plan hash differs between runs")

    same_w, diff_w, p1 = compare_tensors(pa["adapter_state"], pb["adapter_state"],
                                         "lora")
    same_o, diff_o, p2 = compare_tensors(adam_moments(pa), adam_moments(pb), "adam")
    problems += p1 + p2

    print(f"\nLoRA tensors  : {same_w} identical, {diff_w} differing")
    print(f"Adam moments  : {same_o} identical, {diff_o} differing")

    if problems:
        print("\nFAIL -- resume is not bit-exact:")
        for line in problems[:20]:
            print(f"  - {line}")
        raise SystemExit(1)

    print("\nPASS: resumed run is byte-identical to the uninterrupted run")
    print(f"  ({same_w} LoRA tensors and {same_o} optimizer tensors compared)")
    if not args.keep:
        for path in (run_a, run_b):
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
