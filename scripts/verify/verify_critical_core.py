#!/usr/bin/env python3
"""CPU-only structural verification for the CRCC pipeline.

This does not claim a model result. It verifies deterministic arm construction,
sealed-path refusal, matched recovery config generation, merge-plan expansion,
and the exact layer-delta arithmetic on tiny tensors.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.apply_layer_deltas import merge_tensor_in_place  # noqa: E402
from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    assert_unsealed_path,
    dump_yaml,
    load_yaml,
    mapping_source_layers,
    object_sha256,
    read_json,
)
from scripts.critical_core.metrics import (  # noqa: E402
    classify_layer,
    completion_metrics,
)
from scripts.critical_core.scan_layer_ablation import _bypass_layer  # noqa: E402


class VerificationFailure(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise VerificationFailure(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def verify_metrics() -> None:
    ordinary = completion_metrics(
        "That makes sense.", [1, 2, 3, 4, 5], ended_with_eos=True, hit_max_new_tokens=False
    )
    collapsed = completion_metrics(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
        [7] * 48,
        ended_with_eos=False,
        hit_max_new_tokens=True,
    )
    require(ordinary["collapse_score"] < 0.1, "ordinary completion falsely collapsed")
    require(collapsed["collapse_score"] > 0.5, "repetitive completion not detected")

    baseline = {
        "mean_reference_nll": 1.0,
        "collapse_case_rate": 0.0,
        "intelligibility_rate": 1.0,
        "eos_rate": 1.0,
        "nonfinite_reference_nll_rate": 0.0,
    }
    broken = {
        "mean_reference_nll": 8.0,
        "collapse_case_rate": 0.75,
        "intelligibility_rate": 0.1,
        "eos_rate": 0.1,
        "nonfinite_reference_nll_rate": 0.0,
    }
    classification = classify_layer(broken, baseline, {})
    require(classification["catastrophic"], "catastrophic fixture not classified")
    require(classification["criticality_score"] >= 0.6, "criticality score too low")


def verify_path_and_mapping_contracts() -> None:
    try:
        assert_unsealed_path(Path("artifacts/eval/rag_test_SEALED.jsonl"), purpose="test")
    except ContractError:
        pass
    else:
        raise VerificationFailure("sealed path guard accepted a marked asset")

    source = mapping_source_layers(
        [
            {"reduced_layer": 0, "source_layer": 2},
            {"reduced_layer": 1, "source_layer": 9},
        ]
    )
    require(source == [2, 9], "valid layer map changed")
    try:
        mapping_source_layers(
            [
                {"reduced_layer": 0, "source_layer": 9},
                {"reduced_layer": 1, "source_layer": 2},
            ]
        )
    except ContractError:
        pass
    else:
        raise VerificationFailure("out-of-order source mapping was accepted")


def verify_delta_arithmetic() -> None:
    try:
        import torch
    except ImportError as exc:
        raise VerificationFailure("torch is required by the repository test environment") from exc

    target = torch.tensor([10.0, 20.0])
    base = torch.tensor([1.0, 2.0])
    recovered = torch.tensor([3.0, 6.0])
    with torch.no_grad():
        stats = merge_tensor_in_place(target, base, recovered, 0.5, torch)
    require(torch.equal(target, torch.tensor([11.0, 22.0])), "delta formula is incorrect")
    require(stats["elements"] == 2, "delta element count is incorrect")
    require(abs(float(stats["delta_sq"]) - 20.0) < 1e-9, "delta norm is incorrect")

    class TensorLayer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 1

    class TupleLayer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 1, "aux"

    tensor_layers = torch.nn.ModuleList([TensorLayer()])
    tuple_layers = torch.nn.ModuleList([TupleLayer()])
    value = torch.tensor([3.0])
    with _bypass_layer(tensor_layers, 0):
        require(torch.equal(tensor_layers[0](value), value), "tensor bypass changed API")
    require(torch.equal(tensor_layers[0](value), value + 1), "tensor hook was not removed")
    with _bypass_layer(tuple_layers, 0):
        output = tuple_layers[0](value)
        require(torch.equal(output[0], value), "tuple bypass did not preserve hidden state")
        require(output[1] == "aux", "tuple bypass did not preserve auxiliary outputs")
    require(
        torch.equal(tuple_layers[0](value)[0], value + 1),
        "tuple hook was not removed",
    )


def verify_committed_configs() -> None:
    scan = load_yaml(ROOT / "configs/critical_core_scan_v1.yaml")
    recovery = load_yaml(ROOT / "configs/critical_core_recovery_v1.yaml")
    merge = load_yaml(ROOT / "configs/critical_core_merge_v1.yaml")

    for source in scan["data"]["sources"]:
        require("SEALED" not in str(source["path"]).upper(), "scan config names a sealed source")
    require(
        int(scan["criticality"]["minimum_supporting_sources"]) == 2,
        "structural CRCC must require both diagnostic domains",
    )
    seeds = [int(seed) for seed in recovery["recovery"]["seeds"]]
    require(
        seeds == [20260831, 20260901, 20260902],
        "screen and confirmatory recovery seeds drifted",
    )
    kept = recovery["historical_reference"]["aegis_19_source_layers"]
    deleted = sorted(set(range(28)) - set(int(layer) for layer in kept))
    require(
        deleted == [6, 7, 8, 9, 10, 14, 18, 23, 24],
        "Aegis 19-layer mapping drifted from the recorded precursor",
    )
    require(
        float(merge["primary_matched_coefficient"]) == 0.5,
        "primary donor coefficient drifted",
    )

    coefficient_sets = merge["coefficient_sets"]
    candidate_count = 0
    for recipe in merge["recipes"]:
        combinations = 1
        for donor in recipe["donors"]:
            values = donor["coefficients"]
            if isinstance(values, str):
                values = coefficient_sets[values]
            combinations *= len(values)
        candidate_count += combinations
    require(candidate_count == 56, "committed merge matrix no longer expands to 56 candidates")


def _write_scan_fixture(path: Path) -> None:
    layers = []
    for layer in range(28):
        catastrophic = layer in {0, 27}
        layers.append(
            {
                "layer": layer,
                "criticality_score": 0.95 if catastrophic else 0.05 + layer / 1000.0,
                "catastrophic": catastrophic,
                "single_layer_angular_distance": (
                    0.50 if layer == 0 else 0.40 if layer == 27 else 0.08 + layer / 200.0
                ),
            }
        )
    payload = {
        "schema_version": "crcc.v1",
        "experiment": "synthetic structural fixture",
        "status": "MEASURED_DISCOVERY_ONLY",
        "run_identity_sha256": "1" * 64,
        "model": {
            "path": "fixture-model",
            "config_sha256": "2" * 64,
            "files": [],
            "weights_hashed": False,
            "inventory_sha256": "3" * 64,
        },
        "num_hidden_layers": 28,
        "evaluated_layers": list(range(28)),
        "layers": layers,
        "automatic_critical_layers": [0, 27],
        "manual_hypothesis_layers": [],
        "summary_sha256": "4" * 64,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_design_and_config_generation(tmp: Path) -> tuple[Path, Path]:
    scan = tmp / "sample_scan.json"
    _write_scan_fixture(scan)
    fixture_config = tmp / "design_config.yaml"
    fixture_payload = {
        "experiment_id": "crcc-fixture",
        "discovery": {"scan_summary": scan.as_posix()},
        "design": {
            "output": (tmp / "unused_arms.json").as_posix(),
            "allow_partial_scan": False,
            "include_manual_hypothesis": False,
            "minimum_core_depth": 1,
            "maximum_core_depth": 14,
            "random_seed": 20260831,
            "controls": ["random", "low_criticality", "angular_high"],
        },
        "historical_reference": {
            "include": True,
            "aegis_19_source_layers": [
                0, 1, 2, 3, 4, 5, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 25, 26, 27
            ],
        },
    }
    fixture_config.write_text(dump_yaml(fixture_payload), encoding="utf-8")
    first = tmp / "arms_first.json"
    second = tmp / "arms_second.json"
    command = [
        sys.executable,
        "scripts/critical_core/design_arms.py",
        "--config",
        str(fixture_config),
        "--scan",
        str(scan),
    ]
    run(command + ["--out", str(first)])
    run(command + ["--out", str(second)])
    a, b = read_json(first), read_json(second)
    require(a["design_sha256"] == b["design_sha256"], "arm design hash is time-dependent")
    require(
        [(arm["name"], arm["source_layers"]) for arm in a["arms"]]
        == [(arm["name"], arm["source_layers"]) for arm in b["arms"]],
        "arm selection is nondeterministic",
    )
    matched = [arm for arm in a["arms"] if arm["match_group"]]
    require({arm["depth"] for arm in matched} == {2}, "matched controls are not depth matched")
    require(len(matched) == 4, "fixture should create one hypothesis and three controls")

    mix = tmp / "mix.jsonl"
    mix.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    template = tmp / "template.yaml"
    template_payload = {
        "seed": 1,
        "device": "cpu",
        "run": {"dir": "unused"},
        "model": {"path": "unused", "dtype": "float32"},
        "data": {"mix": mix.as_posix(), "max_len": 32},
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["q_proj"],
        },
        "train": {
            "epochs": 1,
            "lr": 0.0001,
            "grad_accum": 1,
            "gradient_checkpointing": False,
        },
        "memory": {
            "max_tokens_per_microbatch": 32,
            "max_rows_per_microbatch": 1,
        },
        "determinism": {"strict": True},
        "resume": {"allow_code_drift": True},
    }
    template.write_text(dump_yaml(template_payload), encoding="utf-8")

    recovery_config = tmp / "recovery.yaml"
    recovery_payload = {
        "experiment_id": "fixture",
        "design": {"output": first.as_posix()},
        "build": {"model_root": (tmp / "models").as_posix()},
        "recovery": {
            "template": template.as_posix(),
            "arms": [
                "crcc_monster",
                "control_random",
                "control_low_criticality",
                "control_angular_high",
            ],
            "reference_arms": [],
            "seeds": [11],
            "config_root": (tmp / "configs").as_posix(),
            "run_root": (tmp / "runs").as_posix(),
            "adapter_root": (tmp / "adapters").as_posix(),
            "matrix_output": (tmp / "recovery_matrix.json").as_posix(),
        },
    }
    recovery_config.write_text(dump_yaml(recovery_payload), encoding="utf-8")
    run(
        [
            sys.executable,
            "scripts/critical_core/make_recovery_configs.py",
            "--config",
            str(recovery_config),
            "--design-only",
        ]
    )
    recovery_matrix = read_json(tmp / "recovery_matrix.json")
    require(len(recovery_matrix["entries"]) == 4, "wrong recovery matrix size")
    require(
        len({entry["training_contract_sha256"] for entry in recovery_matrix["entries"]}) == 1,
        "matched recovery configs differ outside allowed fields",
    )
    for entry in recovery_matrix["entries"]:
        text = Path(entry["config"]).read_text(encoding="utf-8")
        require("allow_code_drift: false" in text, "temporary code-drift waiver survived")
    return first, tmp / "recovery_matrix.json"


def verify_merge_planning(tmp: Path, recovery_matrix: Path) -> None:
    plan_config = tmp / "merge.yaml"
    recipe_names = ["monster", "random", "low", "angular"]
    plan_payload = {
        "experiment_id": "fixture",
        "recovery_matrix": recovery_matrix.as_posix(),
        "dtype": "float32",
        "paths": {
            "plan_root": (tmp / "plans").as_posix(),
            "model_root": (tmp / "merged").as_posix(),
            "matrix_output": (tmp / "merge_matrix.json").as_posix(),
        },
        "targets": {
            "qwen": {"path": (tmp / "missing_target").as_posix()}
        },
        "donors": {
            name: {
                "arm": {
                    "monster": "crcc_monster",
                    "random": "control_random",
                    "low": "control_low_criticality",
                    "angular": "control_angular_high",
                }[name],
                "seed": 11,
            }
            for name in recipe_names
        },
        "coefficient_sets": {"sweep": [0.5, 1.0]},
        "primary_matched_coefficient": 0.5,
        "recipes": [
            {
                "id": f"qwen_{name}",
                "stage": "donor_qualification",
                "target": "qwen",
                "donors": [{"id": name, "coefficients": "sweep"}],
            }
            for name in recipe_names
        ],
        "development_evaluation": {
            "datasets": ["chat_dev", "rag_dev"],
            "cap_mib": 1024,
        },
    }
    plan_config.write_text(dump_yaml(plan_payload), encoding="utf-8")
    run(
        [
            sys.executable,
            "scripts/critical_core/plan_merge_grid.py",
            "--config",
            str(plan_config),
        ]
    )
    matrix = read_json(tmp / "merge_matrix.json")
    require(matrix["candidate_count"] == 8, "merge grid cardinality is wrong")
    require(
        sum(row["analysis_role"] == "PRIMARY_MATCHED_CONTROL" for row in matrix["candidates"])
        == 4,
        "primary matched-control set is wrong",
    )
    for candidate in matrix["candidates"]:
        plan = read_json(Path(candidate["plan"]))
        expected = plan.pop("plan_sha256")
        require(expected == object_sha256(plan), "merge plan hash does not reproduce")


def main() -> None:
    checks = [
        ("collapse metrics and classifier", verify_metrics),
        ("sealed path and layer-map contracts", verify_path_and_mapping_contracts),
        ("layer-delta arithmetic", verify_delta_arithmetic),
        ("committed preregistration configs", verify_committed_configs),
    ]
    for name, check in checks:
        check()
        print(f"PASS  {name}")

    with tempfile.TemporaryDirectory(prefix="crcc_verify_") as directory:
        tmp = Path(directory)
        _, recovery_matrix = verify_design_and_config_generation(tmp)
        print("PASS  deterministic arm design and matched recovery configs")
        verify_merge_planning(tmp, recovery_matrix)
        print("PASS  merge-grid expansion and plan hashes")

    print("\n6/6 CRCC structural verification groups passed")
    print("No model weights were loaded and no scientific result is claimed.")


if __name__ == "__main__":
    main()
