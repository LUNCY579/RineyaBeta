#!/usr/bin/env python3
"""Negative-control tests for the RGE AdamW/Muon pair contract."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiment"))

from rge_optimizer_pair import (  # noqa: E402
    compare_completed_runs,
    summarize_run,
    validate_pair_manifest,
)


PAIR = ROOT / "manifests" / "rge_peft" / "m2_optimizer_pair.yaml"


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")




def make_synthetic_run(path: Path, *, run_id: str, optimizer_name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = path / "checkpoints"
    checkpoint_dir.mkdir()
    manifest = {
        "run_id": run_id,
        "program_version": "rge-peft-v1.1",
        "optimizer": {"name": optimizer_name},
        "model": {"combined_sha256": "model"},
        "tokenizer": {"sha256": "tokenizer"},
        "data": {"mix_sha256": "mix"},
        "seed": 7,
        "environment": {"torch": "test", "devices": [{"name": "consumer-gpu"}]},
        "precision": "bfloat16",
        "lora": {"r": 16, "alpha": 32},
        "geometry": {"cadence_steps": 10},
        "schedule": {"name": "cosine"},
        "curriculum": {"mode": "reference"},
        "cma": {"enabled": False},
        "supervised_token_budget": 128,
        "plan_hash": "plan",
        "first_batches_sha256": "batches",
        "trainable_initial_state": {
            "combined_sha256": "initial", "parameter_count": 24
        },
        "memory": {"cap_mib": 5120},
        "source_fingerprint": {"combined": "source"},
        "git": {"commit": "commit"},
        "evaluation_assets": {"manifest_sha256": "eval"},
        "active_gpu_hours": 0.01,
        "wall_time_s": 12.0,
        "peak_allocated_mib": 900.0,
        "peak_reserved_mib": 1000.0,
        "exit_reason": "supervised_token_budget",
        "final_position": {"supervised_tokens_seen": 128},
    }
    (path / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8", newline="\n"
    )
    (path / "summary.json").write_text(
        json.dumps({"sealed_data_accessed": False}), encoding="utf-8", newline="\n"
    )
    rows = [
        {"step": 1, "supervised_tokens": 64, "loss": 2.4, "lr": 1e-4,
         "step_time_s": 5.0},
        {"step": 2, "supervised_tokens": 64, "loss": 2.0, "lr": 9e-5,
         "step_time_s": 4.0},
    ]
    (path / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8", newline="\n"
    )
    checkpoint_name = "step_000002.pt"
    torch.save(
        {
            "position": {"supervised_tokens_seen": 128},
            "optimizer_state": {
                "state": {0: {"momentum_buffer": torch.zeros(2, 3)}}
            },
        },
        checkpoint_dir / checkpoint_name,
    )
    (checkpoint_dir / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint_name}),
        encoding="utf-8", newline="\n"
    )


def expect_problem(pair: dict, configs: dict[str, dict], mutate, needle: str,
                   directory: Path, checks: list[str]) -> None:
    local_pair = copy.deepcopy(pair)
    local_configs = copy.deepcopy(configs)
    mutate(local_pair, local_configs)
    for arm in ("control", "candidate"):
        config_path = directory / f"{arm}.yaml"
        write_yaml(config_path, local_configs[arm])
        local_pair["arms"][arm]["config"] = str(config_path)
    pair_path = directory / "pair.yaml"
    write_yaml(pair_path, local_pair)
    _, _, problems = validate_pair_manifest(pair_path)
    if not any(needle in problem for problem in problems):
        raise AssertionError(f"expected problem containing {needle!r}; got {problems}")
    checks.append(f"negative control rejected: {needle}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()
    pair, configs, problems = validate_pair_manifest(PAIR)
    if problems:
        raise AssertionError(f"registered pair is invalid: {problems}")
    checks = ["registered pair passes strict preflight"]

    with tempfile.TemporaryDirectory(prefix="rge_pair_verify_") as tmp:
        directory = Path(tmp)
        expect_problem(
            pair, configs,
            lambda _p, c: c["candidate"].__setitem__("seed", 7),
            "differ outside optimizer and run", directory, checks)
        expect_problem(
            pair, configs,
            lambda _p, c: c["candidate"]["lora"].__setitem__("r", 32),
            "differ outside optimizer and run", directory, checks)
        expect_problem(
            pair, configs,
            lambda _p, c: c["candidate"]["data"].__setitem__("mix", "other.jsonl"),
            "differ outside optimizer and run", directory, checks)
        expect_problem(
            pair, configs,
            lambda _p, c: c["candidate"]["schedule"].__setitem__("warmup_steps", 5),
            "differ outside optimizer and run", directory, checks)
        expect_problem(
            pair, configs,
            lambda _p, c: c["candidate"]["train"].__setitem__(
                "supervised_token_budget", 999),
            "differ outside optimizer and run", directory, checks)
        expect_problem(
            pair, configs,
            lambda p, _c: p["evaluation"]["datasets"].append(
                {"name": "rag_test_SEALED", "enabled": True}),
            "active evaluation datasets", directory, checks)
        expect_problem(
            pair, configs,
            lambda p, _c: p["claim_boundary"].__setitem__(
                "maximum_claim", "general_intelligence"),
            "unfrozen capability bank", directory, checks)
        expect_problem(
            pair, configs,
            lambda p, _c: p["dataset_roles"]["sealed"].__setitem__(
                "access", "allowed"),
            "forbidden_until_m7", directory, checks)
        expect_problem(
            pair, configs,
            lambda p, _c: p["execution_policy"].__setitem__(
                "concurrency", "parallel"),
            "sequential_only", directory, checks)
        expect_problem(
            pair, configs,
            lambda p, _c: p["execution_policy"].__setitem__(
                "reverse_order_repeat", "optional"),
            "reverse-order", directory, checks)
        expect_problem(
            pair, configs,
            lambda p, _c: p["optimizer_calibration"].__setitem__(
                "status", "result_driven"),
            "optimizer_calibration.status", directory, checks)

    with tempfile.TemporaryDirectory(prefix="rge_pair_summary_") as tmp:
        summary_root = Path(tmp)
        make_synthetic_run(summary_root / "control", run_id="control",
                           optimizer_name="adamw")
        synthetic = summarize_run(summary_root / "control")
        if synthetic["actual_supervised_tokens"] != 128:
            raise AssertionError("summarizer lost effective token count")
        if synthetic["environment"]["devices"][0]["name"] != "consumer-gpu":
            raise AssertionError("summarizer lost consumer-hardware provenance")
        if synthetic["optimizer_state_tensor_bytes"] != 24:
            raise AssertionError("summarizer miscounted optimizer tensor bytes")
        if synthetic["step_time_distribution"]["all_steps"]["median_s"] != 4.5:
            raise AssertionError("summarizer timing distribution is incorrect")
        checks.append("run summarizer preserves tokens, hardware, state bytes, and timing")

    base_comparability = {
        "model_sha256": "m", "tokenizer_sha256": "t", "mix_sha256": "d",
        "seed": 1, "precision": {"dtype": "bfloat16"}, "lora": {"r": 16},
        "schedule": {"name": "cosine"}, "curriculum": {"mode": "reference"},
        "cma": {"enabled": False}, "requested_supervised_token_budget": 100,
        "plan_hash": "p", "first_batches_sha256": "b",
        "trainable_initial_state_sha256": "i",
    }
    curve = [
        {"supervised_tokens_seen": 64, "loss": 2.4},
        {"supervised_tokens_seen": 128, "loss": 2.0},
    ]
    timing = {
        "all_steps": {"count": 2, "mean_s": 5.0, "median_s": 5.0, "p90_s": 5.5},
        "steady_state": {"count": 1, "mean_s": 5.0, "median_s": 5.0, "p90_s": 5.0},
        "steady_state_excludes_first_steps": 1,
    }
    control = {
        "comparability": copy.deepcopy(base_comparability),
        "actual_supervised_tokens": 128, "final_loss": 2.0,
        "token_weighted_mean_loss": 2.2, "step_time_s": 10.0,
        "step_time_distribution": copy.deepcopy(timing),
        "supervised_tokens_per_s": 12.8, "optimizer_state_tensor_bytes": 100,
        "peak_reserved_mib": 1000.0, "curve": copy.deepcopy(curve),
    }
    candidate = copy.deepcopy(control)
    candidate.update({"final_loss": 1.9, "token_weighted_mean_loss": 2.1,
                      "step_time_s": 11.0, "supervised_tokens_per_s": 11.64,
                      "optimizer_state_tensor_bytes": 50,
                      "peak_reserved_mib": 1004.0})
    candidate["curve"][0]["loss"] = 2.3
    candidate["curve"][1]["loss"] = 1.9
    candidate["step_time_distribution"]["steady_state"]["median_s"] = 5.5
    matched = compare_completed_runs(control, candidate)
    if matched["status"] != "PASS":
        raise AssertionError(matched)
    if len(matched["pointwise_loss_curve"]) != 2:
        raise AssertionError("pointwise token-aligned curve was not emitted")
    checks.append("post-run matcher accepts identical scientific substrate")
    checks.append("post-run matcher emits token-aligned loss curve")
    candidate["comparability"]["plan_hash"] = "changed"
    mismatched = compare_completed_runs(control, candidate)
    if mismatched["status"] != "FAIL" or "plan_hash" not in mismatched["mismatches"]:
        raise AssertionError(mismatched)
    checks.append("post-run matcher rejects plan-hash drift")
    candidate["comparability"]["plan_hash"] = "p"
    short_control = copy.deepcopy(control)
    short_candidate = copy.deepcopy(candidate)
    short_control["actual_supervised_tokens"] = 64
    short_candidate["actual_supervised_tokens"] = 64
    budget_failure = compare_completed_runs(short_control, short_candidate)
    if budget_failure["status"] != "FAIL" or "control_budget_not_reached" not in (
            budget_failure["mismatches"]):
        raise AssertionError(budget_failure)
    bounded_ok = compare_completed_runs(
        short_control, short_candidate, require_budget=False
    )
    if bounded_ok["status"] != "PASS":
        raise AssertionError(bounded_ok)
    checks.append("full pair requires budget while bounded smoke may stop early")

    # A blocked gate must fail before model loading even though the checked-in
    # manifest is now M1-authorized.
    with tempfile.TemporaryDirectory(prefix="rge_pair_gate_") as tmp:
        blocked_pair = copy.deepcopy(pair)
        blocked_pair["execution_gate"]["status"] = "BLOCKED"
        blocked_path = Path(tmp) / "blocked_pair.yaml"
        write_yaml(blocked_path, blocked_pair)
        process = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "experiment" / "rge_optimizer_pair.py"),
             "--pair", str(blocked_path), "--arm", "control", "--max-steps", "1"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace")
    if process.returncode == 0 or "execution gate is not AUTHORIZED" not in (
            process.stdout + process.stderr):
        raise AssertionError("blocked execution gate did not fail closed")
    checks.append("blocked execution gate fails closed before model loading")

    # The authorized manifest still blocks a full pair while LR/effective-step
    # calibration is provisional. Bounded positive-step smoke is the only path.
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "experiment" / "rge_optimizer_pair.py"),
         "--pair", str(PAIR), "--arm", "control"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode == 0 or "optimizer calibration is not frozen" not in (
            process.stdout + process.stderr):
        raise AssertionError("provisional calibration did not block full training")
    checks.append("provisional calibration blocks full pair before model loading")
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "experiment" / "rge_optimizer_pair.py"),
         "--pair", str(PAIR), "--arm", "control", "--max-steps", "0"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode == 0 or "--max-steps must be positive" not in (
            process.stdout + process.stderr):
        raise AssertionError("non-positive bounded smoke was accepted")
    checks.append("bounded smoke requires a positive step count")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "checks": len(checks),
        "check_names": checks,
        "registered_pair": str(PAIR.relative_to(ROOT)).replace("\\", "/"),
        "execution_gate": pair["execution_gate"],
        "maximum_claim": pair["claim_boundary"]["maximum_claim"],
        "sealed_data_accessed": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.evidence:
        path = Path(args.evidence)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
