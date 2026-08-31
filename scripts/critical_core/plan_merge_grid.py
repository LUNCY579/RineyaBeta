#!/usr/bin/env python3
"""Expand the preregistered CRCC donor/target coefficient matrix into plans."""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    SCHEMA_VERSION,
    atomic_write_json,
    load_yaml,
    object_sha256,
    read_json,
    repo_relative,
    resolve_repo_path,
    sha256_file,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coefficient_tag(value: float) -> str:
    if not math.isfinite(value):
        raise ContractError("coefficient must be finite")
    sign = "m" if value < 0 else ""
    text = f"{abs(value):.4f}".rstrip("0").rstrip(".")
    return sign + text.replace(".", "p")


def _lookup_recovery(
    entries: Sequence[Mapping[str, Any]], arm: str, seed: int
) -> Mapping[str, Any]:
    matches = [
        entry for entry in entries if entry["arm"] == arm and int(entry["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ContractError(
            f"expected exactly one recovery entry for arm={arm!r}, seed={seed}; got {len(matches)}"
        )
    return matches[0]


def _path_ready(path: str, *, kind: str) -> bool:
    resolved = resolve_repo_path(path)
    if kind == "checkpoint":
        return (resolved / "config.json").is_file()
    if kind == "adapter":
        return (resolved / "adapter_config.json").is_file() and (
            resolved / "adapter_model.safetensors"
        ).is_file()
    if kind == "layer_map":
        return resolved.is_file()
    raise AssertionError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, must_exist=True)
    config = load_yaml(config_path)
    recovery_path = resolve_repo_path(config["recovery_matrix"], must_exist=True)
    recovery = read_json(recovery_path)
    if recovery.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("recovery matrix schema is not CRCC v1")

    targets = config["targets"]
    donors_config = config["donors"]
    coefficient_sets = {
        name: [float(value) for value in values]
        for name, values in config["coefficient_sets"].items()
    }
    primary = float(config["primary_matched_coefficient"])
    plan_root = resolve_repo_path(config["paths"]["plan_root"])
    model_root = resolve_repo_path(config["paths"]["model_root"])
    matrix_path = resolve_repo_path(config["paths"]["matrix_output"])

    donor_sources: dict[str, dict[str, Any]] = {}
    for donor_id, donor_cfg in donors_config.items():
        arm = str(donor_cfg["arm"])
        seed = int(donor_cfg["seed"])
        entry = _lookup_recovery(recovery["entries"], arm, seed)
        layer_map = str(resolve_repo_path(entry["model_dir"]) / "layer_map.json")
        layer_map_display = repo_relative(Path(layer_map))
        donor_sources[donor_id] = {
            "id": donor_id,
            "arm": arm,
            "seed": seed,
            "base_model": entry["model_dir"],
            "adapter": entry["adapter_dir"],
            "layer_map": layer_map_display,
            "ready": (
                _path_ready(entry["model_dir"], kind="checkpoint")
                and _path_ready(entry["adapter_dir"], kind="adapter")
                and _path_ready(layer_map_display, kind="layer_map")
            ),
        }

    candidates: list[dict[str, Any]] = []
    for recipe in config["recipes"]:
        target_id = str(recipe["target"])
        if target_id not in targets:
            raise ContractError(f"recipe {recipe['id']} references unknown target {target_id}")
        target_path = str(targets[target_id]["path"])
        target_ready = _path_ready(target_path, kind="checkpoint")

        donor_specs = recipe["donors"]
        coefficient_lists: list[list[float]] = []
        for spec in donor_specs:
            coefficient_value = spec["coefficients"]
            values = (
                coefficient_sets[str(coefficient_value)]
                if isinstance(coefficient_value, str)
                else [float(value) for value in coefficient_value]
            )
            coefficient_lists.append(values)

        for coefficient_tuple in itertools.product(*coefficient_lists):
            donor_plans: list[dict[str, Any]] = []
            tags: list[str] = []
            donor_ready = True
            for spec, coefficient in zip(donor_specs, coefficient_tuple, strict=True):
                donor_id = str(spec["id"])
                if donor_id not in donor_sources:
                    raise ContractError(
                        f"recipe {recipe['id']} references unknown donor {donor_id}"
                    )
                source = donor_sources[donor_id]
                donor_ready = donor_ready and bool(source["ready"])
                donor_plans.append(
                    {
                        "id": donor_id,
                        "arm": source["arm"],
                        "seed": source["seed"],
                        "base_model": source["base_model"],
                        "adapter": source["adapter"],
                        "layer_map": source["layer_map"],
                        "coefficient": coefficient,
                    }
                )
                tags.append(f"{donor_id}_{_coefficient_tag(coefficient)}")

            candidate_id = f"{recipe['id']}__{'__'.join(tags)}"
            output_model = model_root / candidate_id
            plan_path = plan_root / f"{candidate_id}.json"
            matched_primary = (
                recipe["stage"] == "donor_qualification"
                and len(coefficient_tuple) == 1
                and math.isclose(coefficient_tuple[0], primary, rel_tol=0.0, abs_tol=1e-12)
            )
            plan = {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "experiment_id": config["experiment_id"],
                "stage": recipe["stage"],
                "analysis_role": (
                    "PRIMARY_MATCHED_CONTROL" if matched_primary else "EXPLORATORY"
                ),
                "target_id": target_id,
                "target_model": target_path,
                "output_model": repo_relative(output_model),
                "dtype": config.get("dtype", "bfloat16"),
                "donors": donor_plans,
                "safety": {"max_abs_coefficient": 1.5},
            }
            plan["plan_sha256"] = object_sha256(plan)
            atomic_write_json(plan_path, plan)
            ready = target_ready and donor_ready
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "recipe": recipe["id"],
                    "stage": recipe["stage"],
                    "analysis_role": plan["analysis_role"],
                    "target": target_id,
                    "plan": repo_relative(plan_path),
                    "plan_file_sha256": sha256_file(plan_path),
                    "plan_sha256": plan["plan_sha256"],
                    "output_model": repo_relative(output_model),
                    "ready": ready,
                    "target_ready": target_ready,
                    "donors_ready": donor_ready,
                }
            )

    matrix = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "CRCC layer-delta merge matrix",
        "status": "PLANNED_NOT_EVALUATED",
        "created_at": _utc_now(),
        "config": {
            "path": repo_relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "recovery_matrix": {
            "path": repo_relative(recovery_path),
            "sha256": sha256_file(recovery_path),
            "matrix_sha256": recovery.get("matrix_sha256"),
        },
        "primary_matched_coefficient": primary,
        "donor_sources": donor_sources,
        "development_evaluation": config["development_evaluation"],
        "candidate_count": len(candidates),
        "ready_candidate_count": sum(bool(row["ready"]) for row in candidates),
        "candidates": candidates,
    }
    matrix["matrix_sha256"] = object_sha256(
        {key: value for key, value in matrix.items() if key != "created_at"}
    )
    atomic_write_json(matrix_path, matrix)

    print(
        f"planned {len(candidates)} candidates; "
        f"{matrix['ready_candidate_count']} currently ready"
    )
    primary_count = sum(
        candidate["analysis_role"] == "PRIMARY_MATCHED_CONTROL"
        for candidate in candidates
    )
    print(f"primary matched candidates: {primary_count}")
    print(f"wrote {repo_relative(matrix_path)}")
    if args.require_ready and matrix["ready_candidate_count"] != len(candidates):
        raise SystemExit("one or more target/donor artifacts are not ready")


if __name__ == "__main__":
    main()
