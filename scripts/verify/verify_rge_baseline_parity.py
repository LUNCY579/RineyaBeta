#!/usr/bin/env python3
"""Prove legacy AdamW parity and geometry instrumentation side-effect freedom."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rge_peft import as_legacy_config, load_rge_config  # noqa: E402
from rge_verify_common import (  # noqa: E402
    HEAL, RGE, exact_problems, geometry_rows, load_latest, metric_sequence,
    run_entry, write_evidence,
)


def compare_training(left: dict, right: dict, label: str,
                     left_dir: Path, right_dir: Path, *, compare_rng: bool) -> list[str]:
    problems = []
    for key in ("adapter_state", "optimizer_state", "position", "plan_hash",
                "mix_sha256"):
        problems += exact_problems(left[key], right[key], f"{label}.{key}")
    problems += exact_problems(metric_sequence(left_dir), metric_sequence(right_dir),
                               f"{label}.loss_lr_sequence")
    if compare_rng:
        problems += exact_problems(left["rng"], right["rng"], f"{label}.rng")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rge_peft/smoke_adamw.yaml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--cache-from", default=None,
                        help="seed each run from an existing content-addressed tokenizer cache")
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()
    config = ROOT / args.config
    base = ROOT / "artifacts" / "verify_rge_parity" / config.stem
    legacy_dir, off_dir, on_dir = base / "legacy", base / "rge_off", base / "rge_on"
    for path in (legacy_dir, off_dir, on_dir):
        shutil.rmtree(path, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    if args.cache_from:
        cache_source = ROOT / args.cache_from
        if not cache_source.is_dir():
            raise SystemExit(f"--cache-from directory does not exist: {cache_source}")
        for run_dir in (legacy_dir, off_dir, on_dir):
            destination = run_dir / "cache"
            destination.mkdir(parents=True, exist_ok=True)
            for source in cache_source.iterdir():
                if not source.is_file():
                    continue
                target = destination / source.name
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)

    rge_cfg = load_rge_config(config)
    legacy_cfg = as_legacy_config(rge_cfg)
    legacy_cfg.pop("_config_path", None)
    legacy_config_path = base / "legacy_config.yaml"
    legacy_config_path.write_text(yaml.safe_dump(legacy_cfg, sort_keys=False),
                                  encoding="utf-8", newline="\n")

    run_entry(HEAL, legacy_config_path, legacy_dir, args.steps, "none")
    run_entry(RGE, config, off_dir, args.steps, "none", geometry="off")
    run_entry(RGE, config, on_dir, args.steps, "none", geometry="on")
    legacy, off, on = load_latest(legacy_dir), load_latest(off_dir), load_latest(on_dir)
    problems = compare_training(legacy, off, "legacy_vs_rge_off", legacy_dir,
                                off_dir, compare_rng=True)
    problems += compare_training(off, on, "geometry_off_vs_on", off_dir, on_dir,
                                 compare_rng=True)

    off_manifest = json.loads((off_dir / "run_manifest.json").read_text(encoding="utf-8"))
    on_manifest = json.loads((on_dir / "run_manifest.json").read_text(encoding="utf-8"))
    for key in ("plan_hash", "first_batches_sha256", "trainable_initial_state"):
        problems += exact_problems(off_manifest[key], on_manifest[key], f"manifest.{key}")
    if off_manifest["optimizer"]["parameter_groups"] != \
            on_manifest["optimizer"]["parameter_groups"]:
        problems.append("optimizer routing differs between geometry arms")
    rows = geometry_rows(on_dir)
    expected_modules = on_manifest["geometry"]["target_count"]
    expected_measurements = args.steps // int(on_manifest["geometry"]["cadence_steps"])
    if len(rows) != expected_modules * expected_measurements:
        problems.append(f"geometry row count {len(rows)} != "
                        f"{expected_modules * expected_measurements}")
    if geometry_rows(off_dir):
        problems.append("geometry-off arm unexpectedly emitted geometry records")

    evidence = {
        "schema_version": 1, "status": "FAIL" if problems else "PASS",
        "config": args.config, "steps": args.steps,
        "run_device": on_manifest["environment"]["run_device"],
        "available_devices": on_manifest["environment"]["devices"],
        "plan_hash": on["plan_hash"], "mix_sha256": on["mix_sha256"],
        "lora_tensor_count": len(on["adapter_state"]),
        "optimizer_state_tensor_groups": len(on["optimizer_state"]["state"]),
        "trainable_parameter_count": on_manifest["trainable_initial_state"]["parameter_count"],
        "geometry_target_count": expected_modules, "geometry_record_count": len(rows),
        "legacy_rge_bit_exact": not bool(compare_training(
            legacy, off, "summary", legacy_dir, off_dir, compare_rng=True)),
        "geometry_side_effect_bit_exact": not bool(compare_training(
            off, on, "summary", off_dir, on_dir, compare_rng=True)),
        "problems": problems[:100], "sealed_data_accessed": False,
    }
    evidence_path = (ROOT / args.evidence if args.evidence else
                     ROOT / "experiments/rge_peft/evidence" /
                     f"m1_parity_{config.stem}.json")
    write_evidence(evidence_path, evidence)
    print(json.dumps(evidence, indent=2))
    if problems:
        raise SystemExit(1)
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
