#!/usr/bin/env python3
"""CPU-only verification for CRCC sharded discovery evidence assembly.

Synthetic fixtures prove exact 28-layer coverage, cross-source membership, stable
summary hashing, and the sealed/64-row claim boundary. They are structural tests,
not model evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import object_sha256, read_json, sha256_file  # noqa: E402
from scripts.critical_core.metrics import summarize_cases  # noqa: E402


class VerificationFailure(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def raw_cases(
    *, nll: float, collapse: float, intelligible: bool, eos: bool
) -> list[dict]:
    rows = []
    for case_id, source in (
        ("rag_train:r:1", "rag_train"),
        ("short_preservation_train:s:1", "short_preservation_train"),
    ):
        rows.append(
            {
                "id": case_id,
                "source": source,
                "category": "fixture",
                "reference_nll": nll,
                "generated_text": "fixture",
                "generated_token_ids": [1, 2, 3],
                "completion_metrics": {
                    "collapse_score": collapse,
                    "intelligible_by_mechanical_screen": intelligible,
                    "empty": not intelligible,
                    "ended_with_eos": eos,
                    "hit_max_new_tokens": not eos,
                    "token_count": 3,
                    "weird_unicode_rate": 0.0,
                    "dominant_token_fraction": collapse,
                    "repeated_bigram_fraction": collapse,
                },
            }
        )
    return rows


def build_fixture(tmp: Path) -> tuple[Path, Path, Path, Path]:
    rag = tmp / "rag.jsonl"
    short = tmp / "short.jsonl"
    rag.write_text(
        '{"messages":[{"role":"user","content":"r"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )
    short.write_text(
        '{"messages":[{"role":"user","content":"s"},{"role":"assistant","content":"b"}]}\n',
        encoding="utf-8",
    )

    config = {
        "experiment_id": "crcc-fixture",
        "seed": 20260831,
        "model": {
            "path": "Qwen3-0.6B",
            "source_repo": "Qwen/Qwen3-0.6B",
            "revision": "c916fa4defd319b7d4e4da17604ca7338f4d99f5",
            "num_hidden_layers": 28,
            "dtype": "bfloat16",
        },
        "data": {
            "sources": [
                {"label": "rag_train", "path": str(rag), "take": 1},
                {"label": "short_preservation_train", "path": str(short), "take": 1},
            ],
            "minimum_cases": 2,
        },
        "scan": {"layer_profile": None},
        "criticality": {
            "minimum_supporting_sources": 2,
            "manual_hypothesis_layers": [],
            "thresholds": {
                "composite_score": 0.60,
                "hard_nll_ratio": 4.0,
                "hard_min_collapse_rate": 0.25,
                "hard_intelligibility_drop": 0.60,
                "hard_nonfinite_nll_rate": 0.0,
            },
        },
    }
    config_path = tmp / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    case_ids = ["rag_train:r:1", "short_preservation_train:s:1"]
    baseline_cases = raw_cases(nll=1.0, collapse=0.0, intelligible=True, eos=True)
    baseline = {
        "condition": "intact",
        "cases": baseline_cases,
        "summary": summarize_cases(baseline_cases),
        "elapsed_s": 1.234,
    }
    baseline_path = tmp / "baseline.json"
    write_json(baseline_path, baseline)

    sources = [
        {
            "label": "rag_train",
            "path": str(rag),
            "sha256": sha256_file(rag),
            "bytes": rag.stat().st_size,
            "total_nonempty_rows": 1,
            "usable_rows": 1,
            "selected_rows": 1,
            "selection_seed": 20260831,
        },
        {
            "label": "short_preservation_train",
            "path": str(short),
            "sha256": sha256_file(short),
            "bytes": short.stat().st_size,
            "total_nonempty_rows": 1,
            "usable_rows": 1,
            "selected_rows": 1,
            "selection_seed": 20260831,
        },
    ]
    model = {
        "path": "Qwen3-0.6B",
        "config_sha256": "1" * 64,
        "files": [{"name": "model.safetensors", "bytes": 1, "sha256": "2" * 64}],
        "weights_hashed": True,
    }
    model["inventory_sha256"] = object_sha256(model)
    manifest = {
        "schema_version": "crcc.v1",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "seed": 20260831,
        "model": model,
        "sources": sources,
        "diagnostic_cases": [
            {
                "id": case_ids[0],
                "source": "rag_train",
                "category": "fixture",
                "prompt_tokens": 1,
                "target_tokens": 1,
            },
            {
                "id": case_ids[1],
                "source": "short_preservation_train",
                "category": "fixture",
                "prompt_tokens": 1,
                "target_tokens": 1,
            },
        ],
        "generation": {
            "max_new_tokens": 48,
            "do_sample": False,
            "enable_thinking": False,
        },
        "criticality": config["criticality"],
        "code": {},
    }
    manifest["run_identity_sha256"] = object_sha256(manifest)
    manifest_path = tmp / "run_manifest.json"
    write_json(manifest_path, manifest)

    layers_dir = tmp / "layers"
    layers_dir.mkdir()
    for layer in range(28):
        catastrophic = layer in {0, 27}
        cases = raw_cases(
            nll=8.0 if catastrophic else 1.05,
            collapse=1.0 if catastrophic else 0.0,
            intelligible=not catastrophic,
            eos=not catastrophic,
        )
        write_json(
            layers_dir / f"layer_{layer:02d}.json",
            {
                "condition": "single_layer_bypass",
                "layer": layer,
                "cases": cases,
                "summary": summarize_cases(cases),
                "elapsed_s": float(layer) + 0.5,
            },
        )
    return config_path, manifest_path, baseline_path, layers_dir


def run_assembler(
    *,
    config: Path,
    manifest: Path,
    baseline: Path,
    layers: Path,
    out: Path,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/critical_core/assemble_scan_evidence.py",
        "--config",
        str(config),
        "--run-manifest",
        str(manifest),
        "--baseline",
        str(baseline),
        "--layers-dir",
        str(layers),
        "--out-dir",
        str(out),
        "--source-head",
        "242613d1b4b5e0a5a4a7154c60337115ca394bd7",
        "--execution-commit",
        "f" * 40,
        "--workflow-run-id",
        "123",
        "--raw-artifact-name",
        "crcc-v1-discovery-raw-123",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if expect_success and completed.returncode != 0:
        raise VerificationFailure(
            f"assembler failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not expect_success and completed.returncode == 0:
        raise VerificationFailure("assembler accepted an incomplete layer set")
    return completed


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="crcc-evidence-verify-") as temporary:
        tmp = Path(temporary)
        config, manifest, baseline, layers = build_fixture(tmp)
        first = tmp / "out-first"
        second = tmp / "out-second"
        run_assembler(
            config=config, manifest=manifest, baseline=baseline, layers=layers, out=first
        )
        run_assembler(
            config=config, manifest=manifest, baseline=baseline, layers=layers, out=second
        )

        evidence = read_json(first / "evidence_manifest.json")
        summary_first = read_json(first / "scan_summary.json")
        summary_second = read_json(second / "scan_summary.json")
        require(
            evidence["status"] == "DISCOVERY_COMPLETE__MEMBERSHIP_FROZEN",
            "fixture did not pass G0",
        )
        require(evidence["automatic_critical_layers"] == [0, 27], "membership changed")
        require(bool(evidence["g0"]["pass"]), "G0 should pass")
        require(
            not evidence["evaluation_boundaries"]["sealed_evaluation_opened"],
            "sealed boundary changed",
        )
        require(
            not evidence["evaluation_boundaries"][
                "halo_30_of_64_comparable_claim_permitted"
            ],
            "64-row claim boundary changed",
        )
        require(
            summary_first["summary_sha256"] == summary_second["summary_sha256"],
            "stable summary hash depends on timestamps/output paths",
        )

        missing = tmp / "layers-missing"
        missing.mkdir()
        for path in layers.glob("*.json"):
            if path.name != "layer_13.json":
                (missing / path.name).write_bytes(path.read_bytes())
        run_assembler(
            config=config,
            manifest=manifest,
            baseline=baseline,
            layers=missing,
            out=tmp / "out-missing",
            expect_success=False,
        )

    print("CRCC discovery evidence verification: PASS")


if __name__ == "__main__":
    main()
