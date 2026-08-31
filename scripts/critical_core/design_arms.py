#!/usr/bin/env python3
"""Turn a CRCC discovery scan into matched reduced-depth experiment arms.

The CRCC arm is matched against equal-depth random, least-critical, and (when a
layer profile is available) high-residual-rewrite controls.  Aegis's historical
19-layer mapping is recorded as a reference arm, never mislabeled as a matched
control.
"""

from __future__ import annotations

import argparse
import itertools
import random
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
    layer_mapping,
    load_yaml,
    object_sha256,
    read_json,
    repo_relative,
    resolve_repo_path,
    sha256_file,
    validate_layer_indices,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ranked_noncore(
    rows: Sequence[Mapping[str, Any]],
    core: set[int],
    *,
    key: str,
    reverse: bool,
    count: int,
) -> list[int]:
    eligible = [row for row in rows if int(row["layer"]) not in core]
    if any(row.get(key) is None for row in eligible):
        raise ContractError(f"scan rows do not all carry {key!r}")
    ranked = sorted(
        eligible,
        key=lambda row: (float(row[key]), int(row["layer"])),
        reverse=reverse,
    )
    return sorted(int(row["layer"]) for row in ranked[:count])


def _arm(
    name: str,
    layers: Sequence[int],
    *,
    arm_type: str,
    basis: str,
    match_group: str | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "arm_type": arm_type,
        "selection_basis": basis,
        "match_group": match_group,
        "depth": len(layers),
        "source_layers": list(layers),
        "layer_mapping": layer_mapping(layers),
    }


def _pairwise_overlap(arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left, right in itertools.combinations(arms, 2):
        lset, rset = set(left["source_layers"]), set(right["source_layers"])
        intersection = sorted(lset & rset)
        union = lset | rset
        rows.append(
            {
                "left": left["name"],
                "right": right["name"],
                "intersection": intersection,
                "intersection_count": len(intersection),
                "jaccard": len(intersection) / len(union) if union else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scan", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, must_exist=True)
    config = load_yaml(config_path)
    scan_path = resolve_repo_path(
        args.scan or config["discovery"]["scan_summary"], must_exist=True
    )
    scan = read_json(scan_path)
    if scan.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"unsupported scan schema: {scan.get('schema_version')!r}")

    num_layers = int(scan["num_hidden_layers"])
    evaluated = sorted(int(value) for value in scan["evaluated_layers"])
    if evaluated != list(range(num_layers)) and not bool(
        config.get("design", {}).get("allow_partial_scan", False)
    ):
        raise ContractError(
            "matched arm design requires a complete one-layer ablation scan; "
            "set allow_partial_scan only for software smoke tests"
        )

    automatic = set(int(value) for value in scan["automatic_critical_layers"])
    manual = set(int(value) for value in scan.get("manual_hypothesis_layers", []))
    include_manual = bool(config.get("design", {}).get("include_manual_hypothesis", False))
    core = sorted(automatic | manual) if include_manual else sorted(automatic)
    core = validate_layer_indices(core, num_layers=num_layers, label="CRCC core")

    minimum = int(config.get("design", {}).get("minimum_core_depth", 1))
    maximum = int(config.get("design", {}).get("maximum_core_depth", num_layers // 2))
    if not minimum <= len(core) <= maximum:
        raise ContractError(
            f"CRCC core depth {len(core)} is outside predeclared [{minimum}, {maximum}]"
        )
    noncore = sorted(set(range(num_layers)) - set(core))
    if len(noncore) < len(core):
        raise ContractError("not enough non-CRCC layers to form a matched control")

    rows = sorted(scan["layers"], key=lambda row: int(row["layer"]))
    match_group = f"matched_depth_{len(core)}"
    arms: list[dict[str, Any]] = [
        _arm(
            "crcc_monster",
            core,
            arm_type="hypothesis",
            basis=(
                "automatic catastrophic-removal classification"
                + (" plus explicitly separated manual hypothesis layers" if include_manual else "")
            ),
            match_group=match_group,
        )
    ]

    controls = set(config.get("design", {}).get(
        "controls", ["random", "low_criticality", "angular_high"]
    ))
    if "random" in controls:
        rng = random.Random(int(config.get("design", {}).get("random_seed", 20260831)))
        random_layers = sorted(rng.sample(noncore, len(core)))
        arms.append(
            _arm(
                "control_random",
                random_layers,
                arm_type="matched_control",
                basis="uniform sample without replacement from non-CRCC layers",
                match_group=match_group,
            )
        )
    if "low_criticality" in controls:
        arms.append(
            _arm(
                "control_low_criticality",
                _ranked_noncore(
                    rows,
                    set(core),
                    key="criticality_score",
                    reverse=False,
                    count=len(core),
                ),
                arm_type="matched_control",
                basis="lowest catastrophic-removal criticality scores outside CRCC",
                match_group=match_group,
            )
        )
    if "angular_high" in controls:
        arms.append(
            _arm(
                "control_angular_high",
                _ranked_noncore(
                    rows,
                    set(core),
                    key="single_layer_angular_distance",
                    reverse=True,
                    count=len(core),
                ),
                arm_type="matched_control",
                basis="largest residual-stream angular rewrite outside CRCC",
                match_group=match_group,
            )
        )

    historical = config.get("historical_reference", {})
    if bool(historical.get("include", True)):
        kept = validate_layer_indices(
            historical.get("aegis_19_source_layers", []),
            num_layers=num_layers,
            label="Aegis 19L reference",
        )
        if len(kept) != 19:
            raise ContractError(f"Aegis reference must contain 19 layers, got {len(kept)}")
        arms.append(
            _arm(
                "aegis_19_reference",
                kept,
                arm_type="historical_reference",
                basis="complement of the nine deleted layers recorded in MINI_MODEL_ROADMAP §2",
                match_group=None,
            )
        )

    matched = [arm for arm in arms if arm["match_group"] == match_group]
    if len(matched) < 3:
        raise ContractError("CRCC requires the hypothesis arm and at least two matched controls")
    if {arm["depth"] for arm in matched} != {len(core)}:
        raise AssertionError("matched arm depths drifted")
    layer_sets = [tuple(arm["source_layers"]) for arm in matched]
    if len(layer_sets) != len(set(layer_sets)):
        raise ContractError(
            "two matched arms selected the same layer set; freeze a different random seed "
            "before recovery rather than counting a duplicate control twice"
        )

    output = resolve_repo_path(args.out or config["design"]["output"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "catastrophic-removal critical-core matched-arm design",
        "status": "DESIGNED_NOT_TRAINED",
        "created_at": _utc_now(),
        "config": {
            "path": repo_relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "scan": {
            "path": repo_relative(scan_path),
            "sha256": sha256_file(scan_path),
            "run_identity_sha256": scan["run_identity_sha256"],
            "summary_sha256": scan.get("summary_sha256"),
        },
        "source_model": scan["model"],
        "source_depth": num_layers,
        "automatic_critical_layers": sorted(automatic),
        "manual_hypothesis_layers": sorted(manual),
        "manual_hypothesis_included": include_manual,
        "arms": arms,
        "matched_contract": {
            "group": match_group,
            "depth": len(core),
            "same_source_checkpoint": True,
            "same_training_data_required": True,
            "same_optimizer_and_token_budget_required": True,
            "same_seed_required_within_replication": True,
            "standalone_quality_is_not_donor_quality": True,
        },
        "pairwise_overlap": _pairwise_overlap(arms),
    }
    payload["design_sha256"] = object_sha256(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    atomic_write_json(output, payload)
    print(f"CRCC depth: {len(core)} / {num_layers}")
    for arm in arms:
        print(f"  {arm['name']:28s} {arm['source_layers']}")
    print(f"wrote {repo_relative(output)}")


if __name__ == "__main__":
    main()
