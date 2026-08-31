#!/usr/bin/env python3
"""Integrity checks for resumable CRCC discovery artifacts.

This module is intentionally model-light. It validates existing baseline/layer JSON
before the expensive scanner is allowed to reuse it. The final evidence assembler
still repeats its own independent checks; this is an earlier corruption/drift guard,
not a replacement for final assembly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import ContractError, object_sha256, read_json, resolve_repo_path
from scripts.critical_core.metrics import summarize_cases


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _cases(payload: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    raw = payload.get("cases")
    if not isinstance(raw, list):
        raise ContractError(f"{label}.cases must be a list")
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, value in enumerate(raw):
        row = _require_mapping(value, f"{label}.cases[{index}]")
        if "id" not in row:
            raise ContractError(f"{label}.cases[{index}] lacks id")
        rows.append(dict(row))
        ids.append(str(row["id"]))
    if len(ids) != len(set(ids)):
        raise ContractError(f"{label} contains duplicate case ids")
    return rows


def validate_result(
    path: Path,
    *,
    expected_ids: Sequence[str] | None = None,
    expected_condition: str | None = None,
    expected_layer: int | None = None,
) -> dict[str, Any]:
    """Validate one completed baseline or bypass result and return its payload."""

    payload = dict(_require_mapping(read_json(path), str(path)))
    rows = _cases(payload, str(path))
    ids = [str(row["id"]) for row in rows]
    if expected_ids is not None and ids != [str(value) for value in expected_ids]:
        raise ContractError(f"{path} case order differs from the frozen diagnostic set")

    if expected_condition is not None and str(payload.get("condition")) != expected_condition:
        raise ContractError(
            f"{path} condition={payload.get('condition')!r}; expected {expected_condition!r}"
        )
    if expected_layer is not None:
        try:
            layer = int(payload.get("layer", -1))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{path} has invalid layer metadata") from exc
        if layer != int(expected_layer):
            raise ContractError(f"{path} records layer {layer}; expected {expected_layer}")

    stored = _require_mapping(payload.get("summary"), f"{path}.summary")
    recomputed = summarize_cases(rows)
    if object_sha256(stored) != object_sha256(recomputed):
        raise ContractError(f"{path} summary does not reproduce from its raw cases")

    payload["resume_semantic_sha256"] = object_sha256(
        {key: value for key, value in payload.items() if key not in {"elapsed_s", "resume_semantic_sha256"}}
    )
    return payload


def validate_scan_directory(
    out_dir: Path,
    *,
    expected_ids: Sequence[str] | None = None,
    expected_layers: int = 28,
) -> dict[str, Any]:
    """Validate all currently present resumable artifacts without requiring completeness."""

    baseline_path = out_dir / "baseline.json"
    layers_dir = out_dir / "layers"
    present_layers: list[int] = []
    semantic_hashes: dict[str, str] = {}

    if baseline_path.exists():
        baseline = validate_result(
            baseline_path,
            expected_ids=expected_ids,
            expected_condition="intact",
        )
        if expected_ids is None:
            expected_ids = [str(row["id"]) for row in baseline["cases"]]
        semantic_hashes["baseline.json"] = str(baseline["resume_semantic_sha256"])

    if layers_dir.exists():
        for path in sorted(layers_dir.glob("layer_*.json")):
            stem = path.stem
            try:
                layer = int(stem.split("_")[-1])
            except ValueError as exc:
                raise ContractError(f"unexpected CRCC layer filename: {path.name}") from exc
            if layer < 0 or layer >= expected_layers:
                raise ContractError(f"{path} layer index is outside 0..{expected_layers - 1}")
            result = validate_result(
                path,
                expected_ids=expected_ids,
                expected_condition="single_layer_bypass",
                expected_layer=layer,
            )
            present_layers.append(layer)
            semantic_hashes[path.name] = str(result["resume_semantic_sha256"])

    if len(present_layers) != len(set(present_layers)):
        raise ContractError("duplicate CRCC layer results are present")

    return {
        "baseline_present": baseline_path.exists(),
        "present_layers": sorted(present_layers),
        "present_layer_count": len(present_layers),
        "expected_layer_count": expected_layers,
        "complete": baseline_path.exists() and sorted(present_layers) == list(range(expected_layers)),
        "semantic_hashes": semantic_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/critical_core/crcc-v1/scan")
    parser.add_argument("--expected-layers", type=int, default=28)
    args = parser.parse_args()
    out_dir = resolve_repo_path(args.out)
    report = validate_scan_directory(out_dir, expected_layers=args.expected_layers)
    print("CRCC resume integrity: PASS")
    print(f"baseline present: {report['baseline_present']}")
    print(f"layers present: {report['present_layer_count']}/{report['expected_layer_count']}")
    if report["present_layers"]:
        print("present layer indices:", report["present_layers"])


if __name__ == "__main__":
    main()
