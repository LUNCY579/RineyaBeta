#!/usr/bin/env python3
"""Prove that the same seed twice gives the same weights.

This tests a different property from verify_resume.py, and the distinction matters
for diagnosis. Resume can be perfect while the pipeline is nondeterministic, and
the pipeline can be deterministic while resume is broken. Testing only resume
conflates the two: if resume "passes" because both arms are equally random, the
test is worthless.

  run 1 : fresh, N steps
  run 2 : fresh, N steps, separate process, separate output directory

Nothing is restored between them -- the only thing they share is the seed and the
config. Every LoRA tensor and Adam moment must match exactly.

Usage:
  python verify_determinism.py --config configs/heal_smoke.yaml --steps 5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_resume import adam_moments, compare_tensors, load_latest, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/heal_smoke.yaml")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    config = ROOT / args.config
    base = ROOT / "artifacts" / "verify_determinism"
    one, two = base / "run_1", base / "run_2"
    for path in (one, two):
        shutil.rmtree(path, ignore_errors=True)

    print(f"config={args.config}  steps={args.steps}")
    print("=" * 72)
    print("run 1: fresh")
    run(config, one, args.steps, "none")
    print("\nrun 2: fresh, identical seed and config, separate process")
    run(config, two, args.steps, "none")

    p1, p2 = load_latest(one), load_latest(two)
    problems = []
    if p1["plan_hash"] != p2["plan_hash"]:
        problems.append("plan hash differs between two identical configs")
    same_w, diff_w, e1 = compare_tensors(p1["adapter_state"], p2["adapter_state"], "lora")
    same_o, diff_o, e2 = compare_tensors(adam_moments(p1), adam_moments(p2), "adam")
    problems += e1 + e2

    print("\n" + "=" * 72)
    print(f"LoRA tensors  : {same_w} identical, {diff_w} differing")
    print(f"Adam moments  : {same_o} identical, {diff_o} differing")
    if problems:
        print("\nFAIL -- the pipeline is not seed-deterministic:")
        for line in problems[:20]:
            print(f"  - {line}")
        raise SystemExit(1)

    print(f"\nPASS: two independent runs at seed {p1['config']['seed']} are "
          f"byte-identical")
    if not args.keep:
        for path in (one, two):
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
