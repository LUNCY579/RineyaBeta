#!/usr/bin/env python3
"""Validate the cross-process RGE run-manifest, checkpoint, and geometry schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from rge_peft import REQUIRED_SECTIONS  # noqa: E402

MANIFEST_FIELDS = {
    "schema_version", "run_id", "program_version", "designation", "git",
    "model", "tokenizer", "data", "evaluation_assets", "seed", "environment",
    "precision", "lora", "trainable_initial_state", "optimizer", "geometry",
    "schedule", "curriculum", "cma", "supervised_token_budget", "plan_hash",
    "plan_summaries", "first_batches_sha256", "memory", "source_fingerprint",
    "wall_time_s", "active_gpu_hours", "peak_allocated_mib",
    "peak_reserved_mib", "exit_reason", "final_position",
}
GEOMETRY_FIELDS = {
    "schema_version", "run_id", "step", "previous_measurement_step", "step_span",
    "module", "base_norm", "delta_norm", "relative_delta",
    "effective_step_norm", "base_relative_effective_step", "lora_scale",
    "implementation", "cadence_steps",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    problems = []
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    missing = sorted(MANIFEST_FIELDS - set(manifest))
    if missing:
        problems.append(f"manifest missing fields: {missing}")
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    missing_cfg = sorted(REQUIRED_SECTIONS - set(config))
    if missing_cfg:
        problems.append(f"resolved config missing sections: {missing_cfg}")
    if config.get("resume", {}).get("allow_code_drift") is not False or \
            config.get("resume", {}).get("allow_env_drift") is not False:
        problems.append("resolved config does not enforce strict resume drift")
    pointer = json.loads((run_dir / "checkpoints/latest.json").read_text(encoding="utf-8"))
    if not (run_dir / "summary.json").exists():
        problems.append("run summary.json is missing")
    if not (run_dir / "checkpoint_ledger.jsonl").exists():
        problems.append("checkpoint ledger is missing")
    checkpoint = torch.load(run_dir / "checkpoints" / pointer["checkpoint"],
                            map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2:
        problems.append("checkpoint is not format version 2")
    rge = checkpoint.get("extra_state", {}).get("rge")
    if not rge:
        problems.append("checkpoint is missing RGE extra state")
    else:
        for field in ("optimizer_contract", "geometry_contract", "geometry_state",
                      "curriculum", "cma"):
            if field not in rge:
                problems.append(f"RGE checkpoint state missing {field}")
    geometry_path = run_dir / str(config["geometry"]["artifact"])
    rows = []
    if geometry_path.exists():
        for line_number, line in enumerate(geometry_path.read_text(
                encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            absent = sorted(GEOMETRY_FIELDS - set(row))
            if absent:
                problems.append(f"geometry line {line_number} missing {absent}")
            rows.append(row)
    if config["geometry"]["enabled"] and checkpoint["position"]["global_step"] >= \
            int(config["geometry"]["cadence_steps"]) and not rows:
        problems.append("geometry enabled and due, but artifact contains no records")
    result = {"status": "FAIL" if problems else "PASS", "run_dir": str(run_dir),
              "manifest_fields": len(manifest), "geometry_records": len(rows),
              "checkpoint_format_version": checkpoint.get("format_version"),
              "problems": problems}
    print(json.dumps(result, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
