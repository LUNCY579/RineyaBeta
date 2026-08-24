#!/usr/bin/env python3
"""Sequential target x seed x arm orchestration with resumable cell manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))

from pdrr.arms import arm_registry  # noqa: E402
from pdrr.provenance import atomic_write_json, config_hash  # noqa: E402


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="confirmatory config with base_config, seeds, and arms")
    parser.add_argument("--targets", default=None,
                        help="comma-separated targets; otherwise use --calibration low/medium/high")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    confirm_path = Path(args.config).resolve()
    confirm = yaml.safe_load(confirm_path.read_text(encoding="utf-8"))
    base_path = resolve(str(confirm["base_config"]))
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if args.cpu:
        base["device"] = "cpu"
        base["model"]["teacher_device"] = "cpu"
    resolved_base = resolve(args.out) / "resolved_base_config.yaml"
    resolved_base.parent.mkdir(parents=True, exist_ok=True)
    resolved_base.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    if args.targets:
        targets = [int(value) for value in args.targets.split(",") if value.strip()]
    elif args.calibration:
        calibration = json.loads(resolve(args.calibration).read_text(encoding="utf-8"))
        targets = [int(calibration["selection"][name]) for name in ("low", "medium", "high")]
    else:
        raise ValueError("provide --targets or --calibration")
    seeds = [int(value) for value in confirm.get("seeds", [20260823])]
    arms = [str(value) for value in confirm.get("arms", arm_registry().keys())]
    unknown = set(arms) - arm_registry().keys()
    if unknown:
        raise ValueError(f"unknown arms in confirmatory config: {sorted(unknown)}")
    max_steps = int(confirm.get("compute_match", {}).get(
        "optimizer_steps", base["training"].get("steps", 300)))

    output = resolve(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for target in targets:
        for seed in seeds:
            for arm in arms:
                cell = output / f"target_{target:02d}" / f"seed_{seed}" / arm
                cell.mkdir(parents=True, exist_ok=True)
                done = cell / "cell_complete.json"
                model_dir = cell / ("export_full" if arm == "full" else "export_compact")
                train_command = [
                    sys.executable, str(HERE / "scripts" / "train_hf.py"),
                    "--config", str(resolved_base), "--arm", arm,
                    "--target-layer", str(target), "--seed", str(seed),
                    "--out", str(cell), "--max-steps", str(max_steps), "--resume",
                ]
                eval_dir = cell / "evaluation"
                eval_command = [
                    sys.executable, str(HERE / "scripts" / "evaluate_export.py"),
                    "--model", str(model_dir), "--out", str(eval_dir),
                    "--run-manifest", str(cell / "run_manifest.json"),
                ]
                if args.cpu:
                    eval_command.append("--cpu")
                if args.eval_limit is not None:
                    eval_command += ["--limit", str(args.eval_limit)]
                manifest = {
                    "target_layer": target,
                    "seed": seed,
                    "arm": arm,
                    "cell": str(cell),
                    "model": str(model_dir),
                    "benchmark_rows": str(eval_dir / "benchmark_rows.jsonl"),
                    "train_command": train_command,
                    "eval_command": eval_command,
                }
                atomic_write_json(cell / "cell_manifest.json", manifest)
                cells.append(manifest)
                if args.dry_run or done.exists():
                    print(("SKIP complete " if done.exists() else "DRY-RUN ") + str(cell))
                    continue
                subprocess.run(train_command, cwd=ROOT, check=True)
                if not args.skip_eval:
                    subprocess.run(eval_command, cwd=ROOT, check=True)
                atomic_write_json(done, {
                    "train_complete": True,
                    "evaluation_complete": args.skip_eval or (eval_dir / "benchmark_rows.jsonl").exists(),
                    "cell_manifest_hash": config_hash(manifest),
                })

    atomic_write_json(output / "experiment_index.json", {
        "schema_version": 1,
        "confirmatory_config": str(confirm_path),
        "resolved_base_config": str(resolved_base),
        "targets": targets,
        "seeds": seeds,
        "arms": arms,
        "optimizer_steps": max_steps,
        "cells": cells,
        "dry_run": args.dry_run,
    })
    print(f"indexed {len(cells)} cells at {output / 'experiment_index.json'}")


if __name__ == "__main__":
    main()
