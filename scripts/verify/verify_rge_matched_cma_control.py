#!/usr/bin/env python3
"""Prove that the M5 CMA interaction pair differs only in main-segment order."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def normalized_main_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("program_version", None)
    value.pop("run", None)
    value["data"]["mix"] = "<ORDER_SPECIFIC_MIX>"
    value["data"]["order_mode"] = "<ORDER_MODE>"
    value["curriculum"]["mode"] = "<ORDER_MODE>"
    value["curriculum"]["order_sha256"] = "<ORDER_HASH>"
    return value


def normalized_tail_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("program_version", None)
    value.pop("run", None)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-main",
                        default="artifacts/rge_peft/m4_c_u_uniform_r16_subset10")
    parser.add_argument("--candidate-main",
                        default="artifacts/rge_peft/m4_c_c_curriculum_r16_subset10")
    parser.add_argument("--control-tail",
                        default="artifacts/rge_peft/m5_c_utm_uniform_main_matched_tail_r16")
    parser.add_argument("--candidate-tail",
                        default="artifacts/rge_peft/m4_c_ct_constant_tail_r16")
    parser.add_argument("--control-cma",
                        default="experiments/rge_peft/evidence/m5_uniform_matched_cma_construction.json")
    parser.add_argument("--candidate-cma",
                        default="experiments/rge_peft/evidence/m5_cma_construction.json")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m5_corrected_matched_control_audit.json")
    args = parser.parse_args()
    control_main_dir, candidate_main_dir = ROOT / args.control_main, ROOT / args.candidate_main
    control_tail_dir, candidate_tail_dir = ROOT / args.control_tail, ROOT / args.candidate_tail
    control_main = load_json(f"{args.control_main}/run_manifest.json")
    candidate_main = load_json(f"{args.candidate_main}/run_manifest.json")
    control_tail = load_json(f"{args.control_tail}/run_manifest.json")
    candidate_tail = load_json(f"{args.candidate_tail}/run_manifest.json")
    control_cma, candidate_cma = load_json(args.control_cma), load_json(args.candidate_cma)
    checks: list[str] = []

    def require(condition: bool, name: str) -> None:
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    require(control_main["final_position"]["supervised_tokens_seen"]
            == candidate_main["final_position"]["supervised_tokens_seen"] == 105402,
            "main_supervised_tokens_exact")
    require(control_main["final_position"]["rows_seen"]
            == candidate_main["final_position"]["rows_seen"] == 12875,
            "main_membership_row_count_exact")
    require(control_main["curriculum"]["membership_sha256"]
            == candidate_main["curriculum"]["membership_sha256"],
            "main_membership_hash_exact")
    require(control_main["curriculum"]["order_sha256"]
            != candidate_main["curriculum"]["order_sha256"],
            "main_order_is_intended_difference")
    control_main_config = yaml.safe_load((control_main_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"))
    candidate_main_config = yaml.safe_load((candidate_main_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"))
    require(normalized_main_config(control_main_config)
            == normalized_main_config(candidate_main_config),
            "main_configs_equal_after_order_normalization")
    for key in ("model", "tokenizer", "lora", "optimizer", "schedule", "deltaball"):
        require(control_main[key] == candidate_main[key], f"main_{key}_exact")

    require(control_tail["plan_hash"] == candidate_tail["plan_hash"],
            "tail_plan_hash_exact")
    require(control_tail["data"]["mix_sha256"] == candidate_tail["data"]["mix_sha256"],
            "tail_mix_hash_exact")
    for key, expected in (("global_step", 12), ("rows_seen", 1536),
                          ("supervised_tokens_seen", 12713)):
        require(control_tail["final_position"][key]
                == candidate_tail["final_position"][key] == expected,
                f"tail_{key}_exact")
    control_tail_config = yaml.safe_load((control_tail_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"))
    candidate_tail_config = yaml.safe_load((candidate_tail_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"))
    require(normalized_tail_config(control_tail_config)
            == normalized_tail_config(candidate_tail_config),
            "tail_configs_exact")
    for key in ("model", "tokenizer", "lora", "optimizer", "schedule", "deltaball"):
        require(control_tail[key] == candidate_tail[key], f"tail_{key}_exact")

    for key in ("source_steps", "weights", "source_rank", "source_scale", "exact_rank",
                "rank_matched_rank", "module_count", "factor_averaging_used"):
        require(control_cma[key] == candidate_cma[key], f"cma_{key}_exact")
    control_adapter = load_json(
        f"{control_cma['rank_matched_adapter']}/adapter_config.json")
    candidate_adapter = load_json(
        f"{candidate_cma['rank_matched_adapter']}/adapter_config.json")
    for key in ("r", "lora_alpha", "task_type", "bias"):
        require(control_adapter[key] == candidate_adapter[key], f"adapter_{key}_exact")
    require(set(control_adapter["target_modules"])
            == set(candidate_adapter["target_modules"]),
            "adapter_target_modules_exact")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "checks": checks,
        "check_count": len(checks),
        "scientific_difference": "main-segment order only",
        "control_main_order_sha256": control_main["curriculum"]["order_sha256"],
        "candidate_main_order_sha256": candidate_main["curriculum"]["order_sha256"],
        "matched_main_supervised_tokens": 105402,
        "matched_tail_supervised_tokens": 12713,
        "matched_total_supervised_tokens": 118115,
        "matched_tail_plan_hash": control_tail["plan_hash"],
        "control_training_gpu_hours": (
            control_main["active_gpu_hours"] + control_tail["active_gpu_hours"]),
        "candidate_training_gpu_hours": (
            candidate_main["active_gpu_hours"] + candidate_tail["active_gpu_hours"]),
        "control_rank16_retained_energy_fraction": control_cma[
            "global_retained_energy_fraction"],
        "candidate_rank16_retained_energy_fraction": candidate_cma[
            "global_retained_energy_fraction"],
        "sealed_data_accessed": False,
        "test_and_challenge_accessed": False,
    }
    evidence = ROOT / args.evidence
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
