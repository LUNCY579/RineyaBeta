#!/usr/bin/env python3
"""Generate provably matched deterministic recovery configs for every arm.

This script reuses ``scripts/train/heal.py`` unchanged.  It changes only the
reduced checkpoint, run directory, and replication seed, forces the temporary
B0 code-drift waiver back off, and verifies that all other training fields are
identical within a replication.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    SCHEMA_VERSION,
    assert_unsealed_path,
    atomic_write_json,
    atomic_write_text,
    deep_without_paths,
    dump_yaml,
    load_yaml,
    object_sha256,
    read_json,
    repo_relative,
    resolve_repo_path,
    sha256_file,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arm_table(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    table = {str(arm["name"]): arm for arm in manifest["arms"]}
    if len(table) != len(manifest["arms"]):
        raise ContractError("arm manifest contains duplicate names")
    return table


def _path_under_root(root: Path, arm: str, seed: int) -> Path:
    return root / arm / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arms", default=None, help="override design.output")
    parser.add_argument(
        "--design-only",
        action="store_true",
        help="write configs and commands even before reduced checkpoints exist",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, must_exist=True)
    experiment = load_yaml(config_path)
    arm_manifest_path = resolve_repo_path(
        args.arms or experiment["design"]["output"], must_exist=True
    )
    arm_manifest = read_json(arm_manifest_path)
    if arm_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("arm manifest schema is not CRCC v1")
    arms_by_name = _arm_table(arm_manifest)

    recovery = experiment["recovery"]
    requested = list(recovery.get("arms", [])) + list(recovery.get("reference_arms", []))
    if not requested:
        raise ContractError("recovery.arms is empty")
    missing_names = sorted(set(requested) - set(arms_by_name))
    if missing_names:
        raise ContractError(f"unknown recovery arms: {missing_names}")

    template_path = resolve_repo_path(recovery["template"], must_exist=True)
    template = load_yaml(template_path)
    mix_path = resolve_repo_path(template["data"]["mix"], must_exist=True)
    assert_unsealed_path(mix_path, purpose="CRCC recovery training")
    if not bool(template.get("determinism", {}).get("strict", False)):
        raise ContractError("recovery template must enable strict determinism")

    model_root = resolve_repo_path(experiment["build"]["model_root"])
    config_root = resolve_repo_path(recovery["config_root"])
    run_root = resolve_repo_path(recovery["run_root"])
    adapter_root = resolve_repo_path(recovery["adapter_root"])
    matrix_path = resolve_repo_path(recovery["matrix_output"])
    seeds = [int(seed) for seed in recovery["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ContractError("recovery.seeds must be a non-empty unique list")

    entries: list[dict[str, Any]] = []
    contracts_by_seed: dict[int, str] = {}
    for seed in seeds:
        for arm_name in requested:
            arm = arms_by_name[arm_name]
            model_dir = model_root / arm_name
            model_ready = (model_dir / "config.json").is_file() and (
                model_dir / "layer_map.json"
            ).is_file()
            if not model_ready and not args.design_only:
                raise FileNotFoundError(
                    f"reduced model for {arm_name} is missing under {model_dir}; "
                    "run build_reduced_model.py first or use --design-only"
                )

            run_dir = _path_under_root(run_root, arm_name, seed)
            adapter_dir = _path_under_root(adapter_root, arm_name, seed)
            generated_path = config_root / arm_name / f"seed_{seed}.yaml"

            generated = deepcopy(template)
            generated["seed"] = seed
            generated["model"]["path"] = repo_relative(model_dir)
            generated["run"]["dir"] = repo_relative(run_dir)
            generated.setdefault("resume", {})["allow_code_drift"] = False
            generated["crcc_provenance"] = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": experiment["experiment_id"],
                "arm": arm_name,
                "arm_type": arm["arm_type"],
                "source_layers": arm["source_layers"],
                "arm_design_sha256": arm_manifest["design_sha256"],
                "template_path": repo_relative(template_path),
                "template_sha256": sha256_file(template_path),
            }

            contract = deep_without_paths(
                generated,
                [
                    "model.path",
                    "run.dir",
                    "crcc_provenance",
                ],
            )
            contract_sha = object_sha256(contract)
            prior = contracts_by_seed.setdefault(seed, contract_sha)
            if prior != contract_sha:
                raise ContractError(
                    f"training contract drifted across arms for seed {seed}: "
                    f"{prior} vs {contract_sha}"
                )

            atomic_write_text(generated_path, dump_yaml(generated))
            train_command = [
                "python",
                "scripts/train/heal.py",
                "--config",
                repo_relative(generated_path),
                "--resume",
                "auto",
            ]
            export_command = [
                "python",
                "scripts/train/export_adapter.py",
                repo_relative(run_dir),
                "--out",
                repo_relative(adapter_dir),
                "--base",
                repo_relative(model_dir),
            ]
            entries.append(
                {
                    "arm": arm_name,
                    "arm_type": arm["arm_type"],
                    "seed": seed,
                    "model_dir": repo_relative(model_dir),
                    "model_ready": model_ready,
                    "config": repo_relative(generated_path),
                    "config_sha256": sha256_file(generated_path),
                    "run_dir": repo_relative(run_dir),
                    "adapter_dir": repo_relative(adapter_dir),
                    "training_contract_sha256": contract_sha,
                    "train_command": train_command,
                    "export_command": export_command,
                }
            )

    matrix = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "CRCC matched recovery matrix",
        "status": (
            "CONFIGS_READY"
            if all(entry["model_ready"] for entry in entries)
            else "DESIGNED_MODELS_PENDING"
        ),
        "created_at": _utc_now(),
        "experiment_config": {
            "path": repo_relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "arm_manifest": {
            "path": repo_relative(arm_manifest_path),
            "sha256": sha256_file(arm_manifest_path),
            "design_sha256": arm_manifest["design_sha256"],
        },
        "template": {
            "path": repo_relative(template_path),
            "sha256": sha256_file(template_path),
            "data_mix": repo_relative(mix_path),
            "data_mix_sha256": sha256_file(mix_path),
        },
        "allowed_differences": ["seed", "model.path", "run.dir", "crcc_provenance"],
        "entries": entries,
    }
    matrix["matrix_sha256"] = object_sha256(
        {key: value for key, value in matrix.items() if key != "created_at"}
    )
    atomic_write_json(matrix_path, matrix)

    print(f"generated {len(entries)} matched recovery config(s)")
    for entry in entries:
        readiness = "ready" if entry["model_ready"] else "model pending"
        print(f"  {entry['arm']:28s} seed={entry['seed']}  {readiness}")
    print(f"wrote {repo_relative(matrix_path)}")


if __name__ == "__main__":
    main()
