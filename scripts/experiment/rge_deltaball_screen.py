#!/usr/bin/env python3
"""Run and audit the preregistered AdamW DeltaBall radius screen."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
RGE = ROOT / "scripts" / "train" / "rge_peft.py"
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from optim.delta_geometry import reconcile_geometry_log  # noqa: E402
from rge_peft import reconcile_train_log  # noqa: E402
ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONIOENCODING": "utf-8",
}


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def scientific_base(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("run")
    value["geometry"].pop("deltaball")
    return value


def seed_cache(source: Path, run_dir: Path) -> None:
    destination = run_dir / "cache"
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if not item.is_file():
            continue
        target = destination / item.name
        try:
            os.link(item, target)
        except OSError:
            import shutil
            shutil.copy2(item, target)


def run_config(config: Path, run_dir: Path) -> None:
    command = [sys.executable, str(RGE), "--config", str(config),
               "--out", str(run_dir), "--resume", "none"]
    print("$ " + " ".join(command))
    process = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENV})
    if process.returncode:
        raise SystemExit(f"DeltaBall arm failed: {config}")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summarize(config: dict, run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    final_position = manifest.get("final_position")
    artifact_reconciliation = None
    if not final_position:
        pointer = json.loads((run_dir / "checkpoints/latest.json").read_text(
            encoding="utf-8"))
        checkpoint = torch.load(run_dir / "checkpoints" / pointer["checkpoint"],
                                map_location="cpu", weights_only=False)
        final_position = checkpoint["position"]
        through_step = int(final_position["global_step"])
        artifact_reconciliation = reconcile_train_log(
            run_dir / "train_log.jsonl", through_step=through_step)
        reconcile_geometry_log(run_dir / "geometry.jsonl", through_step=through_step)
        reconcile_geometry_log(
            run_dir / config["geometry"]["deltaball"]["artifact"],
            through_step=through_step)
    rows = jsonl(run_dir / "train_log.jsonl")
    tokens = sum(int(row["supervised_tokens"]) for row in rows)
    weighted_loss = sum(float(row["loss"]) * int(row["supervised_tokens"])
                        for row in rows) / tokens
    db = config["geometry"]["deltaball"]
    projection = jsonl(run_dir / db["artifact"])
    post = [row for row in projection if int(row["step"]) > int(db["warmin_steps"])]
    hit_rows = [row for row in post if bool(row["boundary_hit"])]
    post_steps = sorted({int(row["step"]) for row in post})
    hit_steps = {int(row["step"]) for row in post if bool(row["boundary_hit"])}
    by_module: dict[str, list[bool]] = {}
    for row in post:
        by_module.setdefault(str(row["module"]), []).append(bool(row["boundary_hit"]))
    module_rates = {name: sum(values) / len(values)
                    for name, values in by_module.items()}
    return {
        "run_id": manifest["run_id"],
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "source_commit": manifest["git"]["commit"],
        "source_fingerprint": manifest["source_fingerprint"]["combined"],
        "model_sha256": manifest["model"]["combined_sha256"],
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "mix_sha256": manifest["data"]["mix_sha256"],
        "plan_hash": manifest["plan_hash"],
        "first_batches_sha256": manifest["first_batches_sha256"],
        "initial_state_sha256": manifest["trainable_initial_state"]["combined_sha256"],
        "actual_supervised_tokens": int(final_position["supervised_tokens_seen"]),
        "steps": len(rows),
        "token_weighted_mean_loss": weighted_loss,
        "final_loss": float(rows[-1]["loss"]),
        "active_gpu_hours": sum(float(row["step_time_s"]) for row in rows) / 3600.0,
        "peak_reserved_mib": max(float(row.get("peak_reserved_mib", 0.0))
                                 for row in rows),
        "deltaball": manifest["deltaball"],
        "post_warmin_matrix_checks": len(post),
        "post_warmin_matrix_hits": len(hit_rows),
        "post_warmin_matrix_hit_rate": len(hit_rows) / len(post) if post else 0.0,
        "post_warmin_step_hit_rate": len(hit_steps) / len(post_steps) if post_steps else 0.0,
        "median_projection_c_on_hits": (statistics.median(
            float(row["projection_c"]) for row in hit_rows) if hit_rows else None),
        "most_projected_modules": sorted(module_rates.items(),
                                         key=lambda item: item[1], reverse=True)[:12],
        "stable": all(math.isfinite(float(row["loss"])) for row in rows),
        "completed_budget": int(final_position["supervised_tokens_seen"]) >= int(
            config["train"]["supervised_token_budget"]),
        "artifact_reconciliation": artifact_reconciliation,
        "loss_sequence": [float(row["loss"]) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=[
        "configs/rge_peft/m3_g_a_adamw.yaml",
        "configs/rge_peft/m3_g_ad_eps05.yaml",
        "configs/rge_peft/m3_g_ad_eps10.yaml",
        "configs/rge_peft/m3_g_ad_eps20.yaml",
    ])
    parser.add_argument("--cache-from", default="artifacts/heal_runs/b0/cache")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m3_radius_screen.json")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--external-timing-contention", action="store_true")
    args = parser.parse_args()
    config_paths = [ROOT / value for value in args.configs]
    configs = [load_yaml(path) for path in config_paths]
    base = scientific_base(configs[0])
    if any(scientific_base(config) != base for config in configs[1:]):
        raise SystemExit("DeltaBall screen configs differ outside run/intervention")
    if configs[0]["geometry"]["deltaball"]["enabled"]:
        raise SystemExit("first screen arm must be the disabled control")
    cache = ROOT / args.cache_from
    if not cache.is_dir():
        raise SystemExit(f"cache source does not exist: {cache}")
    run_dirs = [ROOT / config["run"]["dir"] for config in configs]
    if not args.analyze_existing:
        for path in run_dirs:
            if path.exists():
                raise SystemExit(f"screen run directory already exists: {path}")
        for config_path, run_dir in zip(config_paths, run_dirs):
            seed_cache(cache, run_dir)
            run_config(config_path, run_dir)

    arms = [summarize(config, path) for config, path in zip(configs, run_dirs)]
    comparable_fields = ("source_commit", "source_fingerprint", "model_sha256",
                         "tokenizer_sha256", "mix_sha256", "plan_hash",
                         "first_batches_sha256", "initial_state_sha256",
                         )
    mismatches = {}
    for field in comparable_fields:
        values = [arm[field] for arm in arms]
        if any(value != values[0] for value in values[1:]):
            mismatches[field] = values
    if mismatches:
        raise SystemExit(json.dumps(mismatches, indent=2))
    completed = [arm for arm in arms if arm["completed_budget"]]
    if len(completed) < 3:
        raise SystemExit("control and both potentially active radius arms must complete")
    for field in ("actual_supervised_tokens", "steps"):
        values = [arm[field] for arm in completed]
        if any(value != values[0] for value in values[1:]):
            raise SystemExit(f"completed-arm comparability mismatch: {field}={values}")
    control_losses = arms[0]["loss_sequence"]
    for arm in arms[1:]:
        arm["early_stopped_inactive"] = (
            not arm["completed_budget"] and arm["post_warmin_matrix_hits"] == 0)
        arm["prefix_matches_control"] = (
            arm["loss_sequence"] == control_losses[:arm["steps"]])
        if not arm["completed_budget"] and not (
                arm["early_stopped_inactive"] and arm["prefix_matches_control"]):
            raise SystemExit(f"invalid incomplete screen arm: {arm['run_id']}")
    eligible = [arm for arm in arms[1:]
                if arm["completed_budget"] and arm["stable"]
                and 0.01 <= arm["post_warmin_matrix_hit_rate"] <= 0.50]
    selected = (min(eligible, key=lambda arm: (
        arm["token_weighted_mean_loss"], -float(arm["deltaball"]["epsilon_max"])))
                if eligible else None)
    result = {
        "schema_version": 1,
        "status": "PASS" if selected else "NO_ELIGIBLE_RADIUS",
        "configs": [path.relative_to(ROOT).as_posix() for path in config_paths],
        "comparability_mismatches": {},
        "arms": [{key: value for key, value in arm.items()
                  if key != "loss_sequence"} for arm in arms],
        "selected_run_id": selected["run_id"] if selected else None,
        "selected_epsilon_max": (selected["deltaball"]["epsilon_max"]
                                 if selected else None),
        "selection_rule": "eligible stable arm with lowest token-weighted loss; ties choose larger radius",
        "timing_interpretation": ("INVALID_EXTERNAL_GPU_CONTENTION"
                                  if args.external_timing_contention else
                                  "EXPLORATORY_SINGLE_ORDER_ONLY"),
        "sealed_data_accessed": False,
    }
    evidence = ROOT / args.evidence
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": result["status"],
        "selected_epsilon_max": result["selected_epsilon_max"],
        "arms": [{key: arm[key] for key in (
            "run_id", "token_weighted_mean_loss", "final_loss",
            "post_warmin_matrix_hit_rate", "post_warmin_step_hit_rate",
            "active_gpu_hours", "peak_reserved_mib", "completed_budget")}
                 | ({"early_stopped_inactive": arm.get("early_stopped_inactive")}
                    if "early_stopped_inactive" in arm else {})
                 for arm in arms],
        "sealed_data_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
