#!/usr/bin/env python3
"""Prove exact RGE resume, including generic optimizer and geometry state."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rge_verify_common import (  # noqa: E402
    RGE, artifact_rows, exact_problems, geometry_rows, load_latest, metric_sequence, run_entry,
    write_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rge_peft/smoke_adamw.yaml")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--break-at", type=int, default=2)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()
    if not 0 < args.break_at < args.steps:
        raise SystemExit("--break-at must be strictly between 0 and --steps")
    config = ROOT / args.config
    base = ROOT / "artifacts" / "verify_rge_resume" / config.stem
    straight, resumed = base / "straight", base / "resumed"
    for path in (straight, resumed):
        shutil.rmtree(path, ignore_errors=True)

    run_entry(RGE, config, straight, args.steps, "none", geometry="on")
    run_entry(RGE, config, resumed, args.break_at, "none", geometry="on")
    # Simulate an asynchronously killed process that flushed a newer log row but
    # did not reach the next atomic checkpoint. Exact resume must discard it.
    log_path = resumed / "train_log.jsonl"
    stale = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    stale["step"] = args.break_at + 1
    stale["loss"] = 12345.0
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(stale) + "\n")
    run_entry(RGE, config, resumed, args.steps, "auto", geometry="on")
    left, right = load_latest(straight), load_latest(resumed)
    problems = []
    for key in ("adapter_state", "optimizer_state", "position", "plan_hash",
                "mix_sha256", "rng", "extra_state"):
        problems += exact_problems(left[key], right[key], f"resume.{key}")
    problems += exact_problems(metric_sequence(straight), metric_sequence(resumed),
                               "resume.loss_lr_sequence")
    problems += exact_problems(geometry_rows(straight), geometry_rows(resumed),
                               "resume.geometry_artifact")
    projection_left = artifact_rows(straight, "deltaball_projection.jsonl")
    projection_right = artifact_rows(resumed, "deltaball_projection.jsonl")
    problems += exact_problems(projection_left, projection_right,
                               "resume.deltaball_projection_artifact")
    manifest = json.loads((resumed / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("resume_events"):
        problems.append("resumed manifest contains no resume event")
    evidence = {
        "schema_version": 1, "status": "FAIL" if problems else "PASS",
        "config": args.config, "steps": args.steps, "break_at": args.break_at,
        "final_position": right["position"], "plan_hash": right["plan_hash"],
        "adapter_tensor_count": len(right["adapter_state"]),
        "optimizer_state_entries": len(right["optimizer_state"]["state"]),
        "geometry_record_count": len(geometry_rows(resumed)),
        "deltaball_projection_record_count": len(projection_right),
        "generic_optimizer_state_compared": True,
        "geometry_resume_state_compared": True,
        "stale_train_log_tail_injected_and_removed": True,
        "bit_exact": not problems, "problems": problems[:100],
        "sealed_data_accessed": False,
    }
    evidence_path = (ROOT / args.evidence if args.evidence else
                     ROOT / "experiments/rge_peft/evidence" /
                     f"m1_resume_{config.stem}.json")
    write_evidence(evidence_path, evidence)
    print(json.dumps(evidence, indent=2))
    if problems:
        raise SystemExit(1)
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
