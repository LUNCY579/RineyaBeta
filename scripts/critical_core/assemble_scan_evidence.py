#!/usr/bin/env python3
"""Assemble sharded CRCC discovery outputs into one audited evidence package.

The layer intervention, metrics, thresholds, and source-selection contract stay in
``scan_layer_ablation.py`` and the committed scan config. This helper performs no
model inference. It validates one intact baseline plus exactly one result for each
of the 28 one-layer bypasses, recomputes the preregistered classifier, and writes a
small append-only package suitable for committing while raw case outputs remain in
an external workflow artifact.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    SCHEMA_VERSION,
    WEIGHT_SUFFIXES,
    assert_unsealed_path,
    atomic_write_json,
    atomic_write_text,
    load_yaml,
    object_sha256,
    parse_layer_selection,
    read_json,
    repo_relative,
    resolve_repo_path,
    sha256_file,
)
from scripts.critical_core.metrics import classify_layer, summarize_cases  # noqa: E402

_LAYER_NAME = re.compile(r"^layer_(\d{2})\.json$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _case_ids(payload: Mapping[str, Any], label: str) -> list[str]:
    cases = payload.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise ContractError(f"{label}.cases must be a list")
    ids: list[str] = []
    for index, row in enumerate(cases):
        if not isinstance(row, Mapping) or "id" not in row:
            raise ContractError(f"{label}.cases[{index}] lacks an id")
        ids.append(str(row["id"]))
    if len(ids) != len(set(ids)):
        raise ContractError(f"{label} contains duplicate case ids")
    return ids


def _validated_summary(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise ContractError(f"{label}.cases must be a list")
    normalized = [dict(_require_mapping(row, f"{label}.cases[]")) for row in cases]
    recomputed = summarize_cases(normalized)
    stored = _require_mapping(payload.get("summary"), f"{label}.summary")
    if object_sha256(recomputed) != object_sha256(stored):
        raise ContractError(f"{label} stored summary does not reproduce from raw cases")
    return recomputed


def _validate_run_identity(manifest: Mapping[str, Any]) -> None:
    identity_keys = (
        "schema_version",
        "config_path",
        "config_sha256",
        "seed",
        "model",
        "sources",
        "diagnostic_cases",
        "generation",
        "criticality",
        "code",
    )
    missing = [key for key in identity_keys if key not in manifest]
    if missing:
        raise ContractError(f"run manifest lacks identity fields: {missing}")
    identity = {key: deepcopy(manifest[key]) for key in identity_keys}
    if object_sha256(identity) != str(manifest.get("run_identity_sha256")):
        raise ContractError("run manifest identity hash does not reproduce")


def _semantic_result_sha(payload: Mapping[str, Any]) -> str:
    """Hash measured semantics while excluding wall-clock duration only."""

    semantic = deepcopy(dict(payload))
    semantic.pop("elapsed_s", None)
    return object_sha256(semantic)


def _canonical_layer_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable scientific portion of a layer summary row."""

    return {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in {"result_path", "result_sha256"}
    }


def _load_layer_results(
    layers_dir: Path, *, expected_layers: int
) -> dict[int, tuple[Path, dict[str, Any]]]:
    found: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(layers_dir.glob("layer_*.json")):
        match = _LAYER_NAME.match(path.name)
        if not match:
            continue
        layer = int(match.group(1))
        if layer in found:
            raise ContractError(f"duplicate result for layer {layer}")
        payload = _require_mapping(read_json(path), str(path))
        if payload.get("condition") != "single_layer_bypass":
            raise ContractError(f"{path} is not a single-layer bypass result")
        if int(payload.get("layer", -1)) != layer:
            raise ContractError(f"{path} filename and payload layer disagree")
        found[layer] = (path, dict(payload))

    expected = set(range(expected_layers))
    observed = set(found)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ContractError(
            f"layer result coverage must be exactly 0..{expected_layers - 1}; "
            f"missing={missing}, extra={extra}"
        )
    return found


def _validate_sources(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    data = _require_mapping(config.get("data"), "config.data")
    configured = data.get("sources")
    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        raise ContractError("config.data.sources must be a list")

    labels: list[str] = []
    for source in configured:
        source_map = _require_mapping(source, "config.data.sources[]")
        path = resolve_repo_path(str(source_map["path"]), must_exist=True)
        assert_unsealed_path(path, purpose="CRCC discovery evidence assembly")
        labels.append(str(source_map["label"]))

    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, Sequence) or isinstance(
        manifest_sources, (str, bytes)
    ):
        raise ContractError("run manifest lacks source provenance")
    normalized = [
        dict(_require_mapping(item, "run_manifest.sources[]"))
        for item in manifest_sources
    ]
    if [str(item.get("label")) for item in normalized] != labels:
        raise ContractError("run-manifest source labels/order differ from the committed config")
    for item in normalized:
        path = resolve_repo_path(str(item["path"]), must_exist=True)
        assert_unsealed_path(path, purpose="CRCC discovery evidence assembly")
        if sha256_file(path) != str(item.get("sha256")):
            raise ContractError(f"diagnostic source hash drifted: {repo_relative(path)}")
    return normalized


def _join_layer_profile(summary: dict[str, Any], configured_path: str | None) -> None:
    if not configured_path:
        return
    path = resolve_repo_path(configured_path, must_exist=True)
    profile = _require_mapping(read_json(path), str(path))
    angular = _require_mapping(
        profile.get("angular_distance"), "layer profile angular_distance"
    )
    distances = angular.get("1")
    if not isinstance(distances, list) or len(distances) != int(
        summary["num_hidden_layers"]
    ):
        raise ContractError(f"layer profile {path} does not match the scanned model depth")
    for row in summary["layers"]:
        row["single_layer_angular_distance"] = float(distances[int(row["layer"])])
    summary["layer_profile"] = {
        "path": repo_relative(path),
        "sha256": sha256_file(path),
    }


def _g0_checks(
    *,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    automatic = [int(value) for value in summary["automatic_critical_layers"]]
    model = _require_mapping(manifest.get("model"), "run_manifest.model")
    diagnostics = manifest.get("diagnostic_cases")
    diagnostic_count = len(diagnostics) if isinstance(diagnostics, list) else 0
    minimum_cases = int(
        _require_mapping(config.get("data"), "config.data").get("minimum_cases", 1)
    )
    baseline_nll = _require_mapping(summary.get("baseline"), "summary.baseline").get(
        "mean_reference_nll"
    )
    finite_baseline = baseline_nll is not None and math.isfinite(float(baseline_nll))
    complete = list(summary["evaluated_layers"]) == list(
        range(int(summary["num_hidden_layers"]))
    )
    no_manual_promotion = bool(summary.get("manual_layers_are_not_automatic_evidence"))
    within_depth = 1 <= len(automatic) <= 14
    files = model.get("files")
    weight_files = (
        [
            item
            for item in files
            if isinstance(item, Mapping)
            and Path(str(item.get("name", ""))).suffix.lower() in WEIGHT_SUFFIXES
        ]
        if isinstance(files, list)
        else []
    )
    weights_hashed = bool(
        model.get("weights_hashed")
        and weight_files
        and all(bool(item.get("sha256")) for item in weight_files)
    )
    source_support = all(
        len(row.get("supporting_sources", []))
        >= int(summary["minimum_supporting_sources"])
        for row in summary["layers"]
        if bool(row.get("catastrophic"))
    )
    return {
        "all_28_single_layer_conditions_complete": complete,
        "intact_baseline_reference_nll_finite": finite_baseline,
        "automatic_core_depth_in_preregistered_range_1_to_14": within_depth,
        "manual_hypotheses_not_promoted": no_manual_promotion,
        "automatic_members_have_required_cross_source_support": source_support,
        "diagnostic_case_count_meets_minimum": diagnostic_count >= minimum_cases,
        "checkpoint_weight_files_hashed": weights_hashed,
    }


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _render_report(
    *,
    summary: Mapping[str, Any],
    evidence: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    automatic = [int(value) for value in summary["automatic_critical_layers"]]
    g0 = _require_mapping(evidence["g0"], "evidence.g0")
    baseline = _require_mapping(summary["baseline"], "summary.baseline")
    lines = [
        "# CRCC v1 — first unsealed discovery evidence",
        "",
        f"**Program status:** **{evidence['status']}**  ",
        f"**Preregistration head:** `{evidence['source_head']}`  ",
        f"**Execution commit:** `{evidence['execution_commit']}`  ",
        f"**Run identity:** `{summary['run_identity_sha256']}`  ",
        f"**Stable discovery summary:** `{summary['summary_sha256']}`",
        "",
        "## Scope and claim boundary",
        "",
        "This package contains only the intact Qwen baseline and the 28 preregistered "
        "one-layer bypass conditions on deterministic training-corpus diagnostics. It "
        "does not contain recovery training, matched-arm design, donor merges, DEV-gate "
        "selection, the historical 64-row long-context evaluation, or sealed evaluation.",
        "",
        "The historical 64-row evaluator remains outside the canonical registry. Therefore "
        "nothing in this package reproduces, replaces, beats, or is directly comparable to "
        "Halo's historical `30/64` result.",
        "",
        "## Intact baseline",
        "",
        f"- Diagnostic cases: **{int(baseline['case_count'])}**",
        f"- Mean reference NLL: **{_format_float(baseline.get('mean_reference_nll'), 6)}**",
        f"- Collapse-case rate: **{_format_float(baseline.get('collapse_case_rate'))}**",
        f"- Intelligibility rate: **{_format_float(baseline.get('intelligibility_rate'))}**",
        f"- EOS rate: **{_format_float(baseline.get('eos_rate'))}**",
        "",
        "## Frozen automatic membership",
        "",
        (
            f"Automatic CRCC layers: **`{automatic}`** (depth **{len(automatic)}**)."
            if automatic
            else "Automatic CRCC layers: **none**."
        ),
        "",
        "Manual hypothesis layers remained separate and were not used to alter membership.",
        "",
        "## Discovery gate G0",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key, value in g0["checks"].items():
        lines.append(f"| `{key}` | {'PASS' if value else 'FAIL'} |")
    lines.extend(
        [
            f"| **G0 overall** | **{'PASS' if g0['pass'] else 'FAIL'}** |",
            "",
            "## Layer scan",
            "",
            "| Layer | Auto CRCC | Score | NLL ratio | Collapse Δ | Intelligibility Δ | EOS Δ | Supporting sources |",
            "|---:|:---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["layers"]:
        signals = _require_mapping(row["signals"], "layer.signals")
        sources = ", ".join(
            str(value) for value in row.get("supporting_sources", [])
        ) or "—"
        lines.append(
            "| {layer} | {auto} | {score} | {nll} | {collapse} | {intel} | {eos} | {sources} |".format(
                layer=int(row["layer"]),
                auto="YES" if bool(row["catastrophic"]) else "no",
                score=_format_float(row.get("criticality_score")),
                nll=_format_float(signals.get("reference_nll_ratio")),
                collapse=_format_float(signals.get("collapse_rate_delta")),
                intel=_format_float(signals.get("intelligibility_delta")),
                eos=_format_float(signals.get("eos_rate_delta")),
                sources=sources,
            )
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Official checkpoint: `{evidence['model_source']['repo_id']}@{evidence['model_source']['revision']}`",
            f"- Runtime device/precision: `{evidence['execution_runtime']['resolved_device']}` / `{evidence['execution_runtime']['resolved_dtype']}`",
            f"- PyTorch / Transformers: `{evidence['execution_runtime']['torch_version']}` / `{evidence['execution_runtime']['transformers_version']}`",
            f"- Checkpoint inventory: `{manifest['model']['inventory_sha256']}`",
            f"- Config SHA-256: `{manifest['config_sha256']}`",
            f"- Raw workflow artifact: `{evidence['raw_artifact']['name']}` in Actions run `{evidence['raw_artifact']['workflow_run_id']}`",
            f"- Sealed evaluation opened: **{str(evidence['evaluation_boundaries']['sealed_evaluation_opened']).lower()}**",
            f"- Historical 64-row evaluator executed: **{str(evidence['evaluation_boundaries']['historical_long_context_64_executed']).lower()}**",
            f"- Halo-comparable claim permitted: **{str(evidence['evaluation_boundaries']['halo_30_of_64_comparable_claim_permitted']).lower()}**",
            "",
            "## Next permitted action",
            "",
            (
                "G0 passed, so this automatic layer set is frozen for the subsequent matched "
                "recovery-arm design. That later design still cannot support a Halo-comparable "
                "claim until the exact historical 64-row protocol is canonically registered and "
                "the known Qwen/Halo anchors are reproduced."
                if bool(g0["pass"])
                else "G0 failed. The experiment must stop for instrument review; no matched "
                "recovery arms may be designed from a hand-selected substitute core."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--layers-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--raw-artifact-name", required=True)
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, must_exist=True)
    manifest_path = resolve_repo_path(args.run_manifest, must_exist=True)
    baseline_path = resolve_repo_path(args.baseline, must_exist=True)
    layers_dir = resolve_repo_path(args.layers_dir, must_exist=True)
    out_dir = resolve_repo_path(args.out_dir)

    config = load_yaml(config_path)
    manifest = _require_mapping(read_json(manifest_path), str(manifest_path))
    baseline = _require_mapping(read_json(baseline_path), str(baseline_path))

    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        raise ContractError("run manifest schema is not CRCC v1")
    _validate_run_identity(manifest)
    if sha256_file(config_path) != str(manifest.get("config_sha256")):
        raise ContractError("run manifest does not match the committed scan config")
    source_provenance = _validate_sources(config, manifest)

    if baseline.get("condition") != "intact":
        raise ContractError("baseline payload is not marked intact")
    baseline_ids = _case_ids(baseline, "baseline")
    diagnostic_rows = manifest.get("diagnostic_cases")
    if not isinstance(diagnostic_rows, list):
        raise ContractError("run manifest lacks diagnostic_cases")
    manifest_ids = [str(row["id"]) for row in diagnostic_rows]
    if baseline_ids != manifest_ids:
        raise ContractError("baseline case order differs from the run manifest")

    model_cfg = _require_mapping(config.get("model"), "config.model")
    expected_layers = int(model_cfg.get("num_hidden_layers", 28))
    if expected_layers != 28:
        raise ContractError(
            "CRCC v1 evidence assembly requires the preregistered 28-layer coordinate"
        )

    results = _load_layer_results(layers_dir, expected_layers=expected_layers)
    thresholds = _require_mapping(
        _require_mapping(config.get("criticality"), "config.criticality").get(
            "thresholds"
        ),
        "config.criticality.thresholds",
    )
    minimum_sources = int(config["criticality"].get("minimum_supporting_sources", 1))
    baseline_summary = _validated_summary(baseline, "baseline")

    layer_rows: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    for layer in range(expected_layers):
        result_path, result = results[layer]
        if _case_ids(result, f"layer {layer}") != baseline_ids:
            raise ContractError(
                f"layer {layer} case order differs from the intact baseline"
            )
        result_summary = _validated_summary(result, f"layer {layer}")
        overall = classify_layer(result_summary, baseline_summary, thresholds)
        baseline_per_source = _require_mapping(
            baseline_summary.get("per_source"), "baseline.summary.per_source"
        )
        result_per_source = _require_mapping(
            result_summary.get("per_source"), f"layer {layer}.summary.per_source"
        )
        per_source: dict[str, Any] = {}
        for source_label, source_baseline in baseline_per_source.items():
            if source_label not in result_per_source:
                raise ContractError(
                    f"layer {layer} lacks source group {source_label!r}"
                )
            per_source[str(source_label)] = classify_layer(
                _require_mapping(
                    result_per_source[source_label], "result source summary"
                ),
                _require_mapping(source_baseline, "baseline source summary"),
                thresholds,
            )
        supporting = sorted(
            label
            for label, classified in per_source.items()
            if bool(classified["catastrophic"])
        )
        automatic = bool(
            overall["catastrophic"] and len(supporting) >= minimum_sources
        )
        file_hash = sha256_file(result_path)
        artifact_hashes[f"layer_{layer:02d}.json"] = file_hash
        layer_rows.append(
            {
                "layer": layer,
                **deepcopy(dict(result_summary)),
                **overall,
                "overall_catastrophic": bool(overall["catastrophic"]),
                "catastrophic": automatic,
                "minimum_supporting_sources": minimum_sources,
                "supporting_sources": supporting,
                "per_source_classification": per_source,
                "result_path": repo_relative(result_path),
                "result_sha256": file_hash,
                "result_semantic_sha256": _semantic_result_sha(result),
            }
        )

    manual = (
        parse_layer_selection(
            config["criticality"].get("manual_hypothesis_layers", []),
            num_layers=expected_layers,
        )
        if config["criticality"].get("manual_hypothesis_layers")
        else []
    )
    automatic_layers = [
        int(row["layer"]) for row in layer_rows if bool(row["catastrophic"])
    ]
    resolved_device = str(manifest.get("device", "unknown"))
    requested_dtype = str(model_cfg.get("dtype", "unknown"))
    execution_runtime = {
        "requested_device": str(config.get("device", "auto")),
        "resolved_device": resolved_device,
        "requested_dtype": requested_dtype,
        "resolved_dtype": (
            "float32" if resolved_device.lower().startswith("cpu") else requested_dtype
        ),
        "torch_version": str(manifest.get("torch_version", "unknown")),
        "transformers_version": str(manifest.get("transformers_version", "unknown")),
        "cuda_total_mib": manifest.get("cuda_total_mib"),
    }

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "catastrophic-removal critical-core discovery",
        "status": "MEASURED_DISCOVERY_ONLY",
        "run_identity_sha256": str(manifest["run_identity_sha256"]),
        "model": deepcopy(
            dict(_require_mapping(manifest.get("model"), "run_manifest.model"))
        ),
        "execution_runtime": execution_runtime,
        "num_hidden_layers": expected_layers,
        "evaluated_layers": list(range(expected_layers)),
        "baseline": deepcopy(dict(baseline_summary)),
        "layers": layer_rows,
        "automatic_critical_layers": automatic_layers,
        "manual_hypothesis_layers": manual,
        "manual_layers_are_not_automatic_evidence": True,
        "thresholds": deepcopy(dict(thresholds)),
        "minimum_supporting_sources": minimum_sources,
    }
    _join_layer_profile(
        summary,
        _require_mapping(config.get("scan"), "config.scan").get("layer_profile"),
    )

    stable_summary = {
        **{key: deepcopy(value) for key, value in summary.items() if key != "layers"},
        "layers": [_canonical_layer_row(row) for row in summary["layers"]],
    }
    summary["summary_sha256"] = object_sha256(stable_summary)
    summary["completed_at"] = _utc_now()

    checks = _g0_checks(summary=summary, manifest=manifest, config=config)
    g0_pass = all(bool(value) for value in checks.values())
    status = (
        "DISCOVERY_COMPLETE__MEMBERSHIP_FROZEN"
        if g0_pass
        else "DISCOVERY_COMPLETE__G0_STOP"
    )

    model_repo = str(model_cfg.get("source_repo", ""))
    model_revision = str(model_cfg.get("revision", ""))
    if not model_repo or not model_revision:
        raise ContractError(
            "config.model.source_repo and config.model.revision must be pinned"
        )

    artifact_integrity = {
        "run_manifest_sha256": sha256_file(manifest_path),
        "baseline_sha256": sha256_file(baseline_path),
        "layer_files": artifact_hashes,
    }
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(config.get("experiment_id", "crcc-v1")),
        "status": status,
        "source_head": args.source_head,
        "execution_commit": args.execution_commit,
        "assembled_at": _utc_now(),
        "model_source": {"repo_id": model_repo, "revision": model_revision},
        "execution_runtime": execution_runtime,
        "run_identity_sha256": str(summary["run_identity_sha256"]),
        "scan_summary_sha256": str(summary["summary_sha256"]),
        "automatic_critical_layers": automatic_layers,
        "membership_depth": len(automatic_layers),
        "membership_sha256": object_sha256(
            {
                "source_head": args.source_head,
                "config_sha256": str(manifest["config_sha256"]),
                "model_inventory_sha256": str(
                    manifest["model"]["inventory_sha256"]
                ),
                "run_identity_sha256": str(summary["run_identity_sha256"]),
                "scan_summary_sha256": str(summary["summary_sha256"]),
                "automatic_critical_layers": automatic_layers,
            }
        ),
        "diagnostic_sources": source_provenance,
        "g0": {"pass": g0_pass, "checks": checks},
        "artifact_integrity": artifact_integrity,
        "raw_artifact": {
            "name": args.raw_artifact_name,
            "workflow_run_id": str(args.workflow_run_id),
            "contains_generated_case_outputs": True,
            "committed_to_git": False,
        },
        "evaluation_boundaries": {
            "sealed_evaluation_opened": False,
            "development_gates_executed": False,
            "matched_recovery_arms_designed": False,
            "recovery_training_executed": False,
            "donor_merges_executed": False,
            "historical_long_context_64_canonically_registered": False,
            "historical_long_context_64_executed": False,
            "halo_30_of_64_comparable_claim_permitted": False,
        },
    }
    evidence["evidence_sha256"] = object_sha256(
        {key: value for key, value in evidence.items() if key != "assembled_at"}
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "run_manifest.json", manifest)
    atomic_write_json(out_dir / "scan_summary.json", summary)
    atomic_write_json(out_dir / "evidence_manifest.json", evidence)
    atomic_write_json(
        out_dir / "diagnostic_case_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_identity_sha256": manifest["run_identity_sha256"],
            "sources": source_provenance,
            "diagnostic_cases": diagnostic_rows,
        },
    )
    atomic_write_text(
        out_dir / "CRCC_V1_DISCOVERY_EVIDENCE.md",
        _render_report(summary=summary, evidence=evidence, manifest=manifest),
    )

    print(f"status: {status}")
    print(f"automatic CRCC layers: {automatic_layers}")
    print(f"G0: {'PASS' if g0_pass else 'FAIL'}")
    print(f"summary sha256: {summary['summary_sha256']}")
    print(f"wrote {repo_relative(out_dir)}")


if __name__ == "__main__":
    main()
