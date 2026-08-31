#!/usr/bin/env python3
"""Matched training benchmark for RGE geometry cadence and implementation cost."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
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
        "step_times_s": times,
        "peak_reserved_mib": max((float(row.get("peak_reserved_mib", 0.0))
                                  for row in rows), default=0.0),
        "geometry_records": geometry_records,
        "active_gpu_hours": manifest["active_gpu_hours"],
    }


def gpu_telemetry() -> dict | None:
    proc = subprocess.run([
        "nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    values = [value.strip() for value in proc.stdout.splitlines()[0].split(",")]
    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None
    return {"temperature_c": number(values[0]), "power_w": number(values[1]),
            "memory_used_mib": number(values[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rge_peft/m1_principal_smoke.yaml")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cadence", type=int, default=3)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--analyze-existing", action="store_true",
                        help="reuse completed run directories without launching trainers")
    parser.add_argument("--evidence", default="experiments/rge_peft/evidence/m1_overhead.json")
    args = parser.parse_args()
    source = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    base = ROOT / "artifacts" / "verify_rge_overhead"
    if not args.analyze_existing:
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
    modes = {
        "reference_geometry_disabled_start": {"enabled": False, "cadence_steps": 1,
                                               "include_radius_angle": False},
        "geometry_every_step_full": {"enabled": True, "cadence_steps": 1,
                                      "include_radius_angle": True},
        "geometry_every_step_low_rank_norms": {"enabled": True, "cadence_steps": 1,
                                                "include_radius_angle": False},
        "geometry_every_n_full": {"enabled": True, "cadence_steps": args.cadence,
                                   "include_radius_angle": True},
        "reference_geometry_disabled_end": {"enabled": False, "cadence_steps": 1,
                                             "include_radius_angle": False},
    }
    summaries = {}
    prior = None
    evidence_path = ROOT / args.evidence
    if args.analyze_existing and evidence_path.exists():
        prior = json.loads(evidence_path.read_text(encoding="utf-8"))
    for name, geometry in modes.items():
        out = base / name
        before = None
        if not args.analyze_existing:
            cfg = yaml.safe_load(yaml.safe_dump(source))
            cfg["geometry"].update(geometry)
            cfg["run"]["id"] = f"{source['run']['id']}__overhead__{name}"
            cfg_path = base / f"{name}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
                                newline="\n")
            before = gpu_telemetry()
            run_entry(RGE, cfg_path, out, args.steps, "none", geometry="config")
        elif not out.is_dir():
            raise SystemExit(f"missing completed overhead arm: {out}")
        summaries[name] = summarize(out)
        old_mode = (prior or {}).get("modes", {}).get(name, {})
        summaries[name]["gpu_before"] = before or old_mode.get("gpu_before")
        summaries[name]["gpu_after"] = (gpu_telemetry() if not args.analyze_existing
                                         else old_mode.get("gpu_after"))
    reference_start = summaries["reference_geometry_disabled_start"]["mean_step_time_s"]
    reference_end = summaries["reference_geometry_disabled_end"]["mean_step_time_s"]
    baseline = statistics.fmean((reference_start, reference_end))
    for payload in summaries.values():
        payload["overhead_fraction_vs_reference"] = (
            payload["mean_step_time_s"] / baseline - 1.0)
    cadence_overhead = summaries["geometry_every_n_full"]["overhead_fraction_vs_reference"]
    cadence_times = summaries["geometry_every_n_full"]["step_times_s"]
    measured = [value for index, value in enumerate(cadence_times, 1)
                if index % args.cadence == 0]
    unmeasured = [value for index, value in enumerate(cadence_times, 1)
                  if index % args.cadence != 0]
    if not measured or not unmeasured:
        raise SystemExit("cadence benchmark needs at least one measured and unmeasured step")
    within_baseline = statistics.fmean(unmeasured)
    within_overhead = sum(cadence_times) / (within_baseline * len(cadence_times)) - 1.0
    reference_drift = reference_end / reference_start - 1.0
    stable_reference = abs(reference_drift) <= 0.10
    status = ("PASS" if within_overhead <= 0.10 else "SCREENING_REQUIRED")
    evidence = {
        "schema_version": 1,
        "status": status,
        "config": args.config, "steps": args.steps, "cadence": args.cadence,
        "preferred_overhead_target": 0.10, "per_step_hard_threshold": 0.20,
        "reference_mean_step_time_s": baseline,
        "reference_repeat_drift_fraction": reference_drift,
        "cross_arm_reference_status": ("STABLE" if stable_reference else
                                       "INCONCLUSIVE_THERMAL_DRIFT"),
        "within_cadence_arm": {
            "nonmeasurement_mean_step_time_s": within_baseline,
            "measurement_mean_step_time_s": statistics.fmean(measured),
            "measurement_steps": [index for index in range(1, len(cadence_times) + 1)
                                  if index % args.cadence == 0],
            "amortized_overhead_fraction": within_overhead,
            "rationale": "paired measured/unmeasured steps in one process and trajectory",
        },
        "modes": summaries,
        "selected_policy": (f"full metrics every {args.cadence} steps"
                            if within_overhead <= 0.10 else
                            "cadence must be reduced or dense interactions disabled"),
        "sealed_data_accessed": False,
    }
    write_evidence(evidence_path, evidence)
    print(json.dumps(evidence, indent=2))
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
