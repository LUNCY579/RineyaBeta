#!/usr/bin/env python3
"""Verify and materialize the clean-SHA M1 substrate identity freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
import determinism  # noqa: E402
from rge_peft import _combined_files, _sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/rge_peft/m1_substrate_freeze")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m1_substrate_freeze.json")
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    memory = json.loads((run_dir / "memory_profile.json").read_text(encoding="utf-8"))
    eval_manifest_path = ROOT / "manifests/eval_assets.json"
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    problems = []

    model_dir = ROOT / config["model"]["path"]
    current_model = _combined_files(model_dir)
    if current_model != {"combined_sha256": manifest["model"]["combined_sha256"],
                         "files": manifest["model"]["files"]}:
        problems.append("model directory content differs from run manifest")
    tokenizer = _sha256_file(model_dir / "tokenizer.json")
    if tokenizer != manifest["tokenizer"]["sha256"]:
        problems.append("tokenizer hash differs from run manifest")
    mix = _sha256_file(ROOT / config["data"]["mix"])
    if mix != manifest["data"]["mix_sha256"]:
        problems.append("training mix hash differs from run manifest")
    if _sha256_file(eval_manifest_path) != manifest["evaluation_assets"]["manifest_sha256"]:
        problems.append("evaluation asset manifest hash differs")
    expected_sets = {
        name: {"path": value["path"], "rows": value["rows"],
               "sha256": value["sha256"], "note": value.get("note")}
        for name, value in eval_manifest["eval_sets"].items()
    }
    if expected_sets != manifest["evaluation_assets"]["sets"]:
        problems.append("evaluation identities differ from committed manifest")
    if determinism.code_fingerprint() != manifest["source_fingerprint"]:
        problems.append("training source fingerprint differs from frozen run")
    if manifest["git"]["tracked_dirty"]:
        problems.append("frozen run had tracked worktree changes")
    if manifest["git"]["commit"] != manifest["git"]["origin_research"]:
        problems.append("frozen run commit was not the remote research head")
    if config["resume"] != {"allow_env_drift": False, "allow_code_drift": False}:
        problems.append("RGE resume drift policy is not strict")
    if config["lora"] != {
            "r": 16, "alpha": 32, "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"]}:
        problems.append("principal LoRA capacity/targets differ from frozen contract")
    if int(config["geometry"]["cadence_steps"]) != 10:
        problems.append("selected geometry cadence is not 10")
    if not memory["ok"] or float(manifest["peak_reserved_mib"]) < \
            float(memory["peak_reserved_mib"]):
        problems.append("manifest does not preserve the successful probe peak")

    evidence = {
        "schema_version": 1, "status": "FAIL" if problems else "PASS",
        "branch": manifest["git"]["branch"], "commit": manifest["git"]["commit"],
        "origin_main": manifest["git"]["origin_main"],
        "origin_research": manifest["git"]["origin_research"],
        "tracked_dirty": manifest["git"]["tracked_dirty"],
        "untracked_user_artifacts_present": manifest["git"]["dirty"],
        "model_combined_sha256": manifest["model"]["combined_sha256"],
        "model_tensor_sha256": manifest["model"]["files"]["model.safetensors"],
        "tokenizer_sha256": tokenizer, "mix_sha256": mix,
        "data_rows": manifest["data"]["rows"], "plan_hash": manifest["plan_hash"],
        "first_batches_sha256": manifest["first_batches_sha256"],
        "eval_sets": manifest["evaluation_assets"]["sets"],
        "environment": manifest["environment"],
        "source_fingerprint": manifest["source_fingerprint"],
        "trainable_parameter_count": manifest["trainable_initial_state"]["parameter_count"],
        "optimizer": {
            "name": manifest["optimizer"]["name"],
            "implementation": manifest["optimizer"]["implementation"],
            "version": manifest["optimizer"]["version"],
            "contract_version": manifest["optimizer"]["contract_version"],
            "routes": [{key: group[key] for key in
                        ("route", "parameter_count", "tensor_count")}
                       for group in manifest["optimizer"]["parameter_groups"]],
        },
        "geometry": {
            key: manifest["geometry"][key] for key in
            ("implementation", "enabled", "cadence_steps", "include_radius_angle",
             "target_count", "base_norm_cache", "successive_step_semantics")
        } | {"targets_sha256": hashlib.sha256(json.dumps(
            manifest["geometry"]["targets"], separators=(",", ":")).encode()).hexdigest()},
        "memory_probe": memory, "manifest_peak_allocated_mib": manifest["peak_allocated_mib"],
        "manifest_peak_reserved_mib": manifest["peak_reserved_mib"],
        "sealed_payload_read": False,
        "sealed_identity_source": "committed manifests/eval_assets.json only",
        "problems": problems,
    }
    path = ROOT / args.evidence
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8", newline="\n")
    print(json.dumps(evidence, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
