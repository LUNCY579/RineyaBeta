#!/usr/bin/env python3
"""Matched training benchmark for RGE geometry cadence and implementation cost."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rge_verify_common import RGE, run_entry, write_evidence  # noqa: E402


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in (path / "train_log.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    times = [float(row["step_time_s"]) for row in rows]
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    geometry_path = path / "geometry.jsonl"
    geometry_records = (len(geometry_path.read_text(encoding="utf-8").splitlines())
                        if geometry_path.exists() else 0)
    return {
        "steps": len(times), "mean_step_time_s": statistics.fmean(times),
        "median_step_time_s": statistics.median(times),
        "total_step_time_s": sum(times),
        "peak_reserved_mib": max((float(row.get("peak_reserved_mib", 0.0))
                                  for row in rows), default=0.0),
        "geometry_records": geometry_records,
        "active_gpu_hours": manifest["active_gpu_hours"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rge_peft/m1_principal_smoke.yaml")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cadence", type=int, default=3)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--evidence", default="experiments/rge_peft/evidence/m1_overhead.json")
    args = parser.parse_args()
    source = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    base = ROOT / "artifacts" / "verify_rge_overhead"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    modes = {
        "reference_geometry_disabled": {"enabled": False, "cadence_steps": 1,
                                         "include_radius_angle": False},
        "geometry_every_step_full": {"enabled": True, "cadence_steps": 1,
                                      "include_radius_angle": True},
        "geometry_every_step_low_rank_norms": {"enabled": True, "cadence_steps": 1,
                                                "include_radius_angle": False},
        "geometry_every_n_full": {"enabled": True, "cadence_steps": args.cadence,
                                   "include_radius_angle": True},
    }
    summaries = {}
    for name, geometry in modes.items():
        cfg = yaml.safe_load(yaml.safe_dump(source))
        cfg["geometry"].update(geometry)
        cfg["run"]["id"] = f"{source['run']['id']}__overhead__{name}"
        cfg_path = base / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
                            newline="\n")
        out = base / name
        run_entry(RGE, cfg_path, out, args.steps, "none", geometry="config")
        summaries[name] = summarize(out)
    baseline = summaries["reference_geometry_disabled"]["mean_step_time_s"]
    for payload in summaries.values():
        payload["overhead_fraction_vs_reference"] = (
            payload["mean_step_time_s"] / baseline - 1.0)
    cadence_overhead = summaries["geometry_every_n_full"]["overhead_fraction_vs_reference"]
    evidence = {
        "schema_version": 1,
        "status": "PASS" if cadence_overhead <= 0.10 else "SCREENING_REQUIRED",
        "config": args.config, "steps": args.steps, "cadence": args.cadence,
        "preferred_overhead_target": 0.10, "per_step_hard_threshold": 0.20,
        "modes": summaries,
        "selected_policy": (f"full metrics every {args.cadence} steps"
                            if cadence_overhead <= 0.10 else
                            "cadence must be reduced or dense interactions disabled"),
        "sealed_data_accessed": False,
    }
    write_evidence(ROOT / args.evidence, evidence)
    print(json.dumps(evidence, indent=2))
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
