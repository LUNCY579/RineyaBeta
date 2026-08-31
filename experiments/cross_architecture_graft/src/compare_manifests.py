"""Compare one-seed CAG arm manifests only after proving run comparability.

The output is a development-screening summary, not a model-quality verdict.
Semantic promotion remains governed by EXPERIMENT_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .evidence import load_verified_manifest, sha256_json


_ALLOWED_CONFIG_DIFFERENCES = {"experiment_id", "graft", "notes", "output"}


def _without_allowed_differences(snapshot: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(snapshot)
    for key in _ALLOWED_CONFIG_DIFFERENCES:
        value.pop(key, None)
    return value


def _asset_map(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        role = str(record.get("role") or record.get("path"))
        digest = record.get("sha256") or record.get("aggregate_sha256")
        if not isinstance(digest, str):
            raise ValueError(f"asset {role!r} has no content hash")
        if role in result:
            raise ValueError(f"duplicate asset role: {role}")
        result[role] = digest
    return result


def _model_hash(manifest: dict[str, Any]) -> str:
    assets = manifest.get("model_assets", [])
    if len(assets) != 1 or not isinstance(assets[0].get("aggregate_sha256"), str):
        raise ValueError("each real run must fingerprint exactly one host model directory")
    return str(assets[0]["aggregate_sha256"])


def _arm_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    arms = manifest.get("arms", [])
    if len(arms) != 1:
        raise ValueError("each real-run manifest must contain exactly one trained arm")
    arm = arms[0]
    results = manifest.get("results", {})
    before = float(results["dev_before"]["mean_nll"])
    after = float(results["dev_after"]["mean_nll"])
    snapshot = manifest["config"]["snapshot"]
    return {
        "experiment_id": manifest["experiment_id"],
        "seed": manifest["seed"],
        "donor_kind": snapshot["graft"]["donor"]["kind"],
        "graft_parameters": int(arm["graft_parameters"]),
        "graft_trainable_parameters": int(arm["graft_trainable_parameters"]),
        "dev_before_mean_nll": before,
        "dev_after_mean_nll": after,
        "delta_mean_nll": after - before,
        "train_seconds": results.get("train_seconds"),
        "peak_allocated_mib": results.get("peak_allocated_mib"),
    }


def compare_manifests(
    paths: list[str | Path],
    *,
    reference_id: str | None = None,
) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two arm manifests are required")
    manifests = [load_verified_manifest(path) for path in paths]
    for manifest in manifests:
        if manifest.get("phase") != "real_train_dev":
            raise ValueError("only real_train_dev manifests are comparable here")
        if manifest.get("status") != "UNSEALED_MECHANISM_RUN_COMPLETE":
            raise ValueError("all arms must have completed the unsealed mechanism run")
        invariants = manifest.get("invariants", {})
        required = (
            "zero_gate_bit_exact",
            "only_graft_parameters_trainable",
            "frozen_parameters_unchanged",
        )
        failed = [name for name in required if invariants.get(name) is not True]
        if failed:
            raise ValueError(f"arm {manifest.get('experiment_id')} failed invariants: {failed}")

    first = manifests[0]
    first_snapshot = first["config"]["snapshot"]
    first_contract = _without_allowed_differences(first_snapshot)
    first_data = _asset_map(first.get("data_assets", []))
    first_model = _model_hash(first)
    first_seed = int(first["seed"])
    first_baseline = float(first["results"]["dev_before"]["mean_nll"])

    for manifest in manifests[1:]:
        if int(manifest["seed"]) != first_seed:
            raise ValueError("one comparison bundle must use the same seed")
        contract = _without_allowed_differences(manifest["config"]["snapshot"])
        if sha256_json(contract) != sha256_json(first_contract):
            raise ValueError(
                "arm configs differ outside experiment_id/graft/notes/output; "
                "training, host, data, evaluation, and memory must be identical"
            )
        if _asset_map(manifest.get("data_assets", [])) != first_data:
            raise ValueError("arm data fingerprints differ")
        if _model_hash(manifest) != first_model:
            raise ValueError("arm host-model fingerprints differ")
        baseline = float(manifest["results"]["dev_before"]["mean_nll"])
        if baseline != first_baseline:
            raise ValueError(
                "host baseline NLL differs across arms; deterministic comparability failed"
            )

    summaries = [_arm_summary(manifest) for manifest in manifests]
    ids = {item["experiment_id"] for item in summaries}
    if len(ids) != len(summaries):
        raise ValueError("duplicate experiment_id in comparison bundle")
    if reference_id is None:
        reference_id = summaries[0]["experiment_id"]
    if reference_id not in ids:
        raise ValueError(f"reference id not present: {reference_id}")
    reference = next(item for item in summaries if item["experiment_id"] == reference_id)
    for item in summaries:
        item["trainable_parameter_gap_vs_reference"] = abs(
            item["graft_trainable_parameters"]
            - reference["graft_trainable_parameters"]
        ) / max(reference["graft_trainable_parameters"], 1)
        item["dev_after_nll_difference_vs_reference"] = (
            item["dev_after_mean_nll"] - reference["dev_after_mean_nll"]
        )

    return {
        "status": "COMPARABILITY_PASS",
        "interpretation": (
            "This compares deterministic unsealed NLL health screens only. It cannot "
            "promote an architecture or substitute for paired semantic DEV gates."
        ),
        "seed": first_seed,
        "reference_id": reference_id,
        "host_model_aggregate_sha256": first_model,
        "data_fingerprints": first_data,
        "frozen_contract_sha256": sha256_json(first_contract),
        "arms": sorted(summaries, key=lambda item: item["experiment_id"]),
        "sealed_evaluation_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--reference-id")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = compare_manifests(args.manifests, reference_id=args.reference_id)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
