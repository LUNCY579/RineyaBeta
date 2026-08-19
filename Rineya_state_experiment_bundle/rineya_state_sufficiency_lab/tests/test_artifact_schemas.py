"""Artifacts produced by real runs must validate against the shipped schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from rineya_lab.analysis.compare import compare_runs  # noqa: E402
from rineya_lab.config import load_config  # noqa: E402
from rineya_lab.evaluation.runner import evaluate  # noqa: E402

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def validator(name: str):
    return jsonschema.Draft202012Validator(json.loads((SCHEMA_DIR / name).read_text()))


def eval_cfg(dataset_root, tmp_path, name, condition, view, protocols):
    return load_config(None, extra={
        "dataset": {"root": str(dataset_root), "verify_hashes": False},
        "experiment": {"name": name, "condition": condition, "seed": 20260818,
                       "output_dir": str(tmp_path / name)},
        "data_view": {"view": view},
        "evaluation": {"split": "dev", "limit": 8, "protocols": protocols, "audit_sample_size": 3,
                       "backend": {"kind": "reference", "reference_mode": "noisy", "reference_noise": 0.6}},
    })


@pytest.fixture(scope="module")
def runs(dataset, dataset_root, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("schema_runs")
    full = eval_cfg(dataset_root, tmp_path, "full", "FULL", "full_context", ["full_direct"])
    split = eval_cfg(dataset_root, tmp_path, "split", "SPLIT_STATEFUL", "split_stateful",
                     ["oracle_transition", "generated_rollout"])
    return {"full": (full, evaluate(full, dataset)), "split": (split, evaluate(split, dataset)),
            "tmp_path": tmp_path}


def test_eval_report_matches_schema(runs):
    check = validator("eval_report.schema.json")
    for key in ("full", "split"):
        _, report = runs[key]
        errors = sorted(check.iter_errors(report), key=lambda e: list(e.path))
        assert not errors, [f"{list(e.path)}: {e.message}" for e in errors[:5]]


def test_every_eval_record_matches_schema(runs):
    check = validator("eval_record.schema.json")
    seen = 0
    for key in ("full", "split"):
        cfg, _ = runs[key]
        for path in (Path(cfg.experiment.output_dir) / "records").glob("*.jsonl"):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                errors = sorted(check.iter_errors(record), key=lambda e: list(e.path))
                assert not errors, [f"{path.name} {list(e.path)}: {e.message}" for e in errors[:3]]
                seen += 1
    assert seen > 0


def test_comparison_report_matches_schema(runs):
    compare_config = load_config(None, extra={"comparison": {
        "bootstrap_iterations": 200, "randomization_iterations": 200,
        "protocol_pairs": ["full_direct:generated_rollout"], "margin_frozen": True}})
    report = compare_runs(runs["full"][0].experiment.output_dir,
                          runs["split"][0].experiment.output_dir,
                          compare_config, output_dir=runs["tmp_path"] / "comparison")
    errors = sorted(validator("comparison_report.schema.json").iter_errors(report),
                    key=lambda e: list(e.path))
    assert not errors, [f"{list(e.path)}: {e.message}" for e in errors[:5]]


@pytest.mark.torch
def test_run_manifest_matches_schema(dataset, dataset_root, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    from rineya_lab.training.loop import train
    from rineya_lab.training.stub_tokenizer import VOCAB_SIZE

    tiny = tmp_path / "tiny"
    LlamaForCausalLM(LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)).save_pretrained(str(tiny))

    cfg = load_config(None, extra={
        "dataset": {"root": str(dataset_root), "verify_hashes": False, "train_split": "dev"},
        "experiment": {"name": "schema_train", "condition": "FULL", "output_dir": str(tmp_path / "run")},
        "model": {"name_or_path": str(tiny), "tokenizer_name_or_path": "builtin:byte",
                  "precision": "fp32", "local_files_only": True},
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        "data_view": {"view": "full_context", "max_records": 8},
        "training": {"per_device_batch_size": 2, "gradient_accumulation_steps": 2,
                     "max_seq_len": 256, "logging_steps": 1, "save_steps": 0},
    })
    manifest = train(cfg, dataset)
    errors = sorted(validator("run_manifest.schema.json").iter_errors(manifest),
                    key=lambda e: list(e.path))
    assert not errors, [f"{list(e.path)}: {e.message}" for e in errors[:5]]
    on_disk = json.loads((Path(cfg.experiment.output_dir) / "run_manifest.json").read_text())
    assert on_disk["config_fingerprint"] == manifest["config_fingerprint"]
