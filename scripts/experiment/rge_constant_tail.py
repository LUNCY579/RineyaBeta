#!/usr/bin/env python3
"""Create and execute an audited constant-LR phase-transition checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts" / "train"
RGE = TRAIN / "rge_peft.py"
sys.path.insert(0, str(TRAIN))
import determinism  # noqa: E402

ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONIOENCODING": "utf-8",
}


def latest_checkpoint(run_dir: Path) -> Path:
    pointer = json.loads((run_dir / "checkpoints/latest.json").read_text(
        encoding="utf-8"))
    return run_dir / "checkpoints" / pointer["checkpoint"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    process = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENV})
    if process.returncode:
        raise SystemExit(f"command failed with exit code {process.returncode}")


def atomic_torch_save(path: Path, payload: dict) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(buffer.getvalue())
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run",
                        default="artifacts/rge_peft/m4_c_c_curriculum_r16_subset10")
    parser.add_argument("--config",
                        default="configs/rge_peft/m4_c_ct_constant_tail.yaml")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m4_constant_tail_transition.json")
    args = parser.parse_args()
    source_run, config_path = ROOT / args.source_run, ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.steps != int(config["cma"]["continuation_steps"]):
        raise SystemExit("CLI steps differ from frozen continuation_steps")
    tail_run = ROOT / config["run"]["dir"]
    if tail_run.exists():
        raise SystemExit(f"tail run directory already exists: {tail_run}")
    source_path = latest_checkpoint(source_run)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    current_code = determinism.code_fingerprint()
    source_drift = determinism.compare_code_fingerprint(
        source["code_fingerprint"], current_code)
    if source_drift not in ([], ["cma.py: added"]):
        raise SystemExit(f"training source changed since endpoint: {source_drift}")
    if source["config"]["curriculum"]["membership_sha256"] != config[
            "curriculum"]["membership_sha256"]:
        raise SystemExit("tail membership differs from source endpoint")
    if source["mix_sha256"] != hashlib.sha256((ROOT / config["data"]["mix"])
                                               .read_bytes()).hexdigest():
        raise SystemExit("tail mix differs from source endpoint")

    run([sys.executable, str(RGE), "--config", str(config_path),
         "--out", str(tail_run), "--dry-run", "--resume", "none"])
    dry_manifest = json.loads((tail_run / "run_manifest.json").read_text(
        encoding="utf-8"))
    tail_contract = {
        "implementation": "rge-deltaball-v1",
        "enabled": True,
        "epsilon_max": float(config["geometry"]["deltaball"]["epsilon_max"]),
        "warmin_steps": 0,
        "schedule": "linear",
        "schedule_unit": "steps",
        "warmin_supervised_tokens": None,
        "target_count": int(source["extra_state"]["rge"][
            "deltaball_contract"]["target_count"]),
        "optimizer_state_projection": "none",
        "factor_projection": "symmetric_sqrt_c",
    }
    db_source = source["extra_state"]["rge"]["deltaball_state"]
    optimizer_contract = copy.deepcopy(
        source["extra_state"]["rge"]["optimizer_contract"])
    optimizer_contract["numerics"]["lr"] = float(config["optimizer"]["lr"])
    geometry_contract = copy.deepcopy(
        source["extra_state"]["rge"]["geometry_contract"])
    geometry_contract["cadence_steps"] = int(config["geometry"]["cadence_steps"])
    geometry_state = copy.deepcopy(
        source["extra_state"]["rge"]["geometry_state"])
    geometry_state["cadence_steps"] = int(config["geometry"]["cadence_steps"])
    transition = dict(source)
    transition["position"] = {
        "epoch": 0, "step_in_epoch": 0, "global_step": 0,
        "rows_seen": 0, "supervised_tokens_seen": 0}
    transition["plan_hash"] = dry_manifest["plan_hash"]
    transition["plan_summary"] = {
        "epochs": dry_manifest["plan_summaries"],
        "total_steps": sum(item["steps"] for item in dry_manifest["plan_summaries"])}
    transition["config"] = config
    transition["fingerprint"] = determinism.fingerprint()
    transition["code_fingerprint"] = current_code
    transition["metrics"] = {"transition": "m4_constant_tail"}
    transition["extra_state"] = {"rge": {
        **source["extra_state"]["rge"],
        "program_version": config["program_version"],
        "optimizer_contract": optimizer_contract,
        "geometry_contract": geometry_contract,
        "geometry_state": geometry_state,
        "deltaball_contract": tail_contract,
        "deltaball_state": {
            "implementation": "rge-deltaball-v1",
            "contract": tail_contract,
            "steps_projected": 0,
            "matrix_checks": 0,
            "boundary_hits": 0,
            "base_norms": db_source["base_norms"],
        },
        "curriculum": config["curriculum"],
        "cma": config["cma"],
    }}
    transition["transition"] = {
        "source_checkpoint": source_path.relative_to(ROOT).as_posix(),
        "source_checkpoint_sha256": sha256_file(source_path),
        "adapter_optimizer_rng_retained": True,
        "plan_position_reset": True,
        "projection_counters_reset": True,
        "averaging_performed": False,
        "allowed_inert_source_drift": source_drift,
    }
    transition_path = tail_run / "transition_step_000000000.pt"
    atomic_torch_save(transition_path, transition)
    evidence = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_TAIL_RUN",
        **transition["transition"],
        "tail_config": args.config,
        "tail_config_sha256": sha256_file(config_path),
        "tail_plan_hash": dry_manifest["plan_hash"],
        "constant_lr": config["optimizer"]["lr"],
        "steps": args.steps,
        "checkpoint_steps": config["cma"]["checkpoint_steps"],
        "sealed_data_accessed": False,
    }
    evidence_path = ROOT / args.evidence
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    run([sys.executable, str(RGE), "--config", str(config_path),
         "--out", str(tail_run), "--resume", str(transition_path),
         "--max-steps", str(args.steps)])
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
