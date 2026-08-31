#!/usr/bin/env python3
"""Run and audit the frozen same-membership M4 curriculum pair."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RGE = ROOT / "scripts" / "train" / "rge_peft.py"
sys.path.insert(0, str(ROOT / "scripts" / "train"))
import heal  # noqa: E402

ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONIOENCODING": "utf-8",
}


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def scientific_base(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("run")
    value["data"].pop("mix")
    value["data"].pop("order_mode")
    value["curriculum"].pop("mode")
    value["curriculum"].pop("order_sha256")
    return value


def run(config_path: Path, run_dir: Path) -> None:
    command = [sys.executable, str(RGE), "--config", str(config_path),
               "--out", str(run_dir), "--resume", "none"]
    print("$ " + " ".join(command))
    process = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENV})
    if process.returncode:
        raise SystemExit(f"M4 arm failed: {config_path}")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summarize(config: dict, run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    log = rows(run_dir / "train_log.jsonl")
    cumulative = 0
    total = int(manifest["final_position"]["supervised_tokens_seen"])
    warmup = int(config["schedule"]["warmup_supervised_tokens"])
    for row in log:
        cumulative += int(row["supervised_tokens"])
        expected = heal.lr_at(
            cumulative - 1, base_lr=float(config["optimizer"]["lr"]),
            total=total, warmup=warmup, schedule=str(config["schedule"]["name"]),
            min_ratio=float(config["schedule"]["min_lr_ratio"]))
        if float(row["lr"]) != expected:
            raise SystemExit(f"token-based LR mismatch at cumulative token {cumulative}")
    weighted = sum(float(row["loss"]) * int(row["supervised_tokens"])
                   for row in log) / total
    return {
        "run_id": manifest["run_id"],
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "source_commit": manifest["git"]["commit"],
        "source_fingerprint": manifest["source_fingerprint"]["combined"],
        "model_sha256": manifest["model"]["combined_sha256"],
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "membership_sha256": config["curriculum"]["membership_sha256"],
        "order_sha256": config["curriculum"]["order_sha256"],
        "mix_sha256": manifest["data"]["mix_sha256"],
        "plan_hash": manifest["plan_hash"],
        "initial_state_sha256": manifest["trainable_initial_state"]["combined_sha256"],
        "actual_supervised_tokens": total,
        "rows_seen": int(manifest["final_position"]["rows_seen"]),
        "steps": len(log),
        "token_weighted_mean_loss": weighted,
        "final_loss": float(log[-1]["loss"]),
        "active_gpu_hours": float(manifest["active_gpu_hours"]),
        "peak_reserved_mib": float(manifest["peak_reserved_mib"]),
        "token_schedule_verified": True,
        "curriculum": manifest["curriculum"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform", default="configs/rge_peft/m4_c_u_uniform.yaml")
    parser.add_argument("--curriculum",
                        default="configs/rge_peft/m4_c_c_curriculum.yaml")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m4_curriculum_pair.json")
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    paths = [ROOT / args.uniform, ROOT / args.curriculum]
    configs = [load_yaml(path) for path in paths]
    if scientific_base(configs[0]) != scientific_base(configs[1]):
        raise SystemExit("M4 configs differ outside intended order variables")
    if configs[0]["curriculum"]["membership_sha256"] != configs[1][
            "curriculum"]["membership_sha256"]:
        raise SystemExit("M4 membership hashes differ")
    run_dirs = [ROOT / config["run"]["dir"] for config in configs]
    if not args.analyze_existing:
        for run_dir in run_dirs:
            if run_dir.exists():
                raise SystemExit(f"M4 run directory exists: {run_dir}")
        for path, run_dir in zip(paths, run_dirs):
            run(path, run_dir)
    arms = [summarize(config, run_dir) for config, run_dir in zip(configs, run_dirs)]
    for field in ("source_commit", "source_fingerprint", "model_sha256",
                  "tokenizer_sha256", "membership_sha256", "initial_state_sha256",
                  "actual_supervised_tokens", "rows_seen"):
        if arms[0][field] != arms[1][field]:
            raise SystemExit(f"M4 comparability mismatch {field}")
    if arms[0]["plan_hash"] == arms[1]["plan_hash"]:
        raise SystemExit("M4 intended order intervention did not change plan hash")
    delta_weighted = (arms[1]["token_weighted_mean_loss"]
                      - arms[0]["token_weighted_mean_loss"])
    delta_final = arms[1]["final_loss"] - arms[0]["final_loss"]
    result = {
        "schema_version": 1,
        "status": "PASS",
        "arms": arms,
        "comparison": {
            "token_weighted_loss_delta_curriculum_minus_uniform": delta_weighted,
            "final_loss_delta_curriculum_minus_uniform": delta_final,
            "curriculum_training_signal_positive": delta_weighted < 0.0,
            "membership_and_effective_tokens_identical": True,
            "token_schedule_verified": True,
        },
        "selected_order": "component_local" if delta_weighted < 0.0 else "reference",
        "selection_rule": "lower token-weighted training loss before retention evaluation",
        "timing_interpretation": "EXPLORATORY_SINGLE_ORDER_ONLY",
        "sealed_data_accessed": False,
    }
    evidence = ROOT / args.evidence
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
