"""Verify the preregistered primary and matched-control plan contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config
from .evidence import sha256_json
from .factory import component_plan


PLAN_FILES = {
    "latent_kv": "latent_kv_primary_plan.json",
    "nope_gqa": "nope_attention_matched_control_plan.json",
    "swiglu": "swiglu_matched_control_plan.json",
}


def _shared_contract(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw.items()
        if key not in {"experiment_id", "graft", "notes"}
    }


def verify_plans(root: str | Path | None = None) -> dict[str, Any]:
    experiment_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    records: dict[str, Any] = {}
    contract_hashes: set[str] = set()
    for name, filename in PLAN_FILES.items():
        config = load_config(experiment_root / "configs" / filename)
        plan = component_plan(config)
        if not plan["initial_gate_exactly_zero"]:
            raise ValueError(f"{name} gate is not exactly zero")
        records[name] = {
            "path": f"configs/{filename}",
            "experiment_id": config.experiment_id,
            "config_sha256": config.sha256,
            "parameters": plan["branch_total"]["parameters"],
            "trainable_parameters": plan["branch_total"]["trainable_parameters"],
            "component_plan": plan,
        }
        contract_hashes.add(sha256_json(_shared_contract(config.raw)))
    if len(contract_hashes) != 1:
        raise ValueError("plan configs drift outside the intended graft definition")
    primary = int(records["latent_kv"]["trainable_parameters"])
    for name, record in records.items():
        gap = abs(int(record["trainable_parameters"]) - primary) / primary
        record["trainable_parameter_gap_vs_primary"] = gap
        if name != "latent_kv" and gap > 0.01:
            raise ValueError(f"{name} parameter gap {gap:.4%} exceeds 1%")
    return {
        "status": "PREFLIGHT_BUNDLE_PASS",
        "shared_contract_sha256": next(iter(contract_hashes)),
        "primary": "latent_kv",
        "arms": records,
        "sealed_evaluation_opened": False,
    }


def main() -> int:
    print(json.dumps(verify_plans(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
