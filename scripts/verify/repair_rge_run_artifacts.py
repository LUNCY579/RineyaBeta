#!/usr/bin/env python3
"""Repair checkpoint-inconsistent RGE evidence tails without touching tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from optim.delta_geometry import reconcile_geometry_log  # noqa: E402
from rge_peft import (  # noqa: E402
    _atomic_json,
    logged_training_seconds,
    reconcile_train_log,
)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    pointer = json.loads((run_dir / "checkpoints/latest.json").read_text(
        encoding="utf-8"))
    checkpoint = torch.load(run_dir / "checkpoints" / pointer["checkpoint"],
                            map_location="cpu", weights_only=False)
    position = checkpoint["position"]
    through_step = int(position["global_step"])
    train_log = run_dir / "train_log.jsonl"
    geometry_log = run_dir / "geometry.jsonl"
    before_train_sha = sha256_file(train_log)
    before_geometry_sha = sha256_file(geometry_log)
    reconciliation = reconcile_train_log(train_log, through_step=through_step)
    reconcile_geometry_log(geometry_log, through_step=through_step)
    active_gpu_hours = logged_training_seconds(train_log) / 3600.0

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repair = {
        "schema_version": 1,
        "kind": "checkpoint_consistent_evidence_tail_reconciliation",
        "checkpoint": pointer["checkpoint"],
        "through_step": through_step,
        "train_log": reconciliation,
        "train_log_sha256_before": before_train_sha,
        "train_log_sha256_after": sha256_file(train_log),
        "geometry_sha256_before": before_geometry_sha,
        "geometry_sha256_after": sha256_file(geometry_log),
        "model_or_optimizer_tensors_touched": False,
    }
    manifest.setdefault("artifact_repairs", []).append(repair)
    manifest["active_gpu_hours"] = active_gpu_hours
    _atomic_json(manifest_path, manifest)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["active_gpu_hours"] = active_gpu_hours
    summary.setdefault("artifact_repairs", []).append(repair)
    _atomic_json(summary_path, summary)

    evidence = {
        "schema_version": 1,
        "status": "PASS",
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "position": position,
        "active_gpu_hours_from_reconciled_log": active_gpu_hours,
        "repair": repair,
        "sealed_data_accessed": False,
    }
    evidence_path = Path(args.evidence)
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    _atomic_json(evidence_path, evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
