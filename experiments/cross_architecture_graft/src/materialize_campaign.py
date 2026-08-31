"""Materialize matched real-run configs from the preregistered CAG arm plans."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import load_config, validate_config
from .evidence import atomic_write_json, sha256_file, sha256_json
from .real_runner import _verify_split_manifest


DEFAULT_SEEDS = (20260831, 20260832, 20260833, 20260834, 20260835)
ARM_PLANS = {
    "latent_kv": "latent_kv_primary_plan.json",
    "nope_gqa": "nope_attention_matched_control_plan.json",
    "swiglu": "swiglu_matched_control_plan.json",
}


def _frozen_contract(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": raw["schema_version"],
        "mode": raw["mode"],
        "claim_level": raw["claim_level"],
        "host": raw["host"],
        "training": raw["training"],
        "data": raw["data"],
        "evaluation": raw.get("evaluation", {}),
        "data_policy": raw["data_policy"],
        "memory": raw.get("memory", {}),
    }


def materialize_campaign(
    *,
    model_path: str | Path,
    train_mix: str | Path,
    dev_mix: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_train_rows: int | None = None,
    max_dev_rows: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be unique non-negative integers")
    root = Path(__file__).resolve().parents[1]
    model = Path(model_path)
    train = Path(train_mix)
    dev = Path(dev_mix)
    split = Path(split_manifest)
    destination = Path(output_dir)
    if not model.is_dir():
        raise FileNotFoundError(f"local host checkpoint not found: {model}")
    _verify_split_manifest(split, train, dev)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"campaign config directory is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    contract_hashes: set[str] = set()
    for seed in seeds:
        for arm_name, filename in ARM_PLANS.items():
            plan_path = root / "configs" / filename
            plan = load_config(plan_path)
            raw = deepcopy(plan.raw)
            raw["mode"] = "real"
            raw["seed"] = seed
            raw["experiment_id"] = f"cag-v1-{arm_name}-seed-{seed}"
            raw["host"]["model_path"] = model.as_posix()
            raw["data"] = {
                "train_mix": train.as_posix(),
                "dev_mix": dev.as_posix(),
                "split_manifest": split.as_posix(),
                "cache_dir": (destination / "token_cache").as_posix(),
            }
            if max_train_rows is not None:
                raw["data"]["max_train_rows"] = int(max_train_rows)
            if max_dev_rows is not None:
                raw["data"]["max_dev_rows"] = int(max_dev_rows)
            raw["evaluation"] = {
                "dev_batch_size": 1,
                "equivalence_probe_rows": 8,
                "required_categories": [],
            }
            raw["memory"] = {"cap_mib": 7000}
            raw["output"] = {
                "run_dir": (destination / "runs" / str(seed) / arm_name).as_posix(),
                "overwrite": False,
            }
            raw["notes"] = (
                f"Materialized from {filename}; arm={arm_name}; seed={seed}. "
                "This is an unsealed mechanism run and cannot promote the architecture."
            )
            validated = validate_config(raw)
            contract_hashes.add(sha256_json(_frozen_contract(validated)))
            config_path = destination / "configs" / str(seed) / f"{arm_name}.json"
            atomic_write_json(config_path, validated)
            loaded = load_config(config_path)
            generated.append(
                {
                    "arm": arm_name,
                    "seed": seed,
                    "path": config_path.as_posix(),
                    "canonical_sha256": loaded.sha256,
                    "plan_path": plan_path.relative_to(root).as_posix(),
                    "plan_sha256": plan.sha256,
                }
            )

    # Graft differs by design, so this contract hash includes all fields that
    # must stay identical across the three arms for each seed.
    if len(contract_hashes) != 1:
        raise AssertionError("materialized arms drifted outside the graft definition")
    manifest = {
        "schema_version": 1,
        "campaign_id": "cag-v1-primary-matched-arms",
        "status": "CONFIGS_MATERIALIZED_UNRUN",
        "seeds": list(seeds),
        "arms": list(ARM_PLANS),
        "frozen_contract_sha256": next(iter(contract_hashes)),
        "host_model_path": model.as_posix(),
        "data": {
            "train_mix": {"path": train.as_posix(), "sha256": sha256_file(train)},
            "dev_mix": {"path": dev.as_posix(), "sha256": sha256_file(dev)},
            "split_manifest": {"path": split.as_posix(), "sha256": sha256_file(split)},
        },
        "generated_configs": generated,
        "sealed_evaluation_opened": False,
        "claim_boundary": (
            "This manifest proves config materialization and matched run contracts only; "
            "all model runs and semantic gates remain unrun."
        ),
    }
    atomic_write_json(destination / "campaign_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-mix", required=True)
    parser.add_argument("--dev-mix", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated fixed seeds",
    )
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-dev-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = materialize_campaign(
        model_path=args.model_path,
        train_mix=args.train_mix,
        dev_mix=args.dev_mix,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value.strip()),
        max_train_rows=args.max_train_rows,
        max_dev_rows=args.max_dev_rows,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
