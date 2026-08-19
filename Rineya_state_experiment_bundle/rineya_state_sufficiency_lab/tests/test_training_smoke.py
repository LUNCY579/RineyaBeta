"""Real (tiny) adapter training. Skipped when torch/transformers/peft are absent."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from rineya_lab.config import load_config  # noqa: E402
from rineya_lab.training.chat import IGNORE_INDEX  # noqa: E402
from rineya_lab.training.collate import weighted_causal_lm_loss  # noqa: E402
from rineya_lab.training.loop import train  # noqa: E402
from rineya_lab.training.modeling import select_precision  # noqa: E402
from rineya_lab.training.stub_tokenizer import VOCAB_SIZE  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM

    directory = tmp_path_factory.mktemp("tiny_model")
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    LlamaForCausalLM(config).save_pretrained(str(directory))
    return directory


def train_cfg(dataset_root, tiny_model_dir, tmp_path, *, name, condition, view, **extra):
    data = {
        "dataset": {"root": str(dataset_root), "verify_hashes": False, "train_split": "dev"},
        "experiment": {"name": name, "condition": condition, "seed": 20260818,
                       "output_dir": str(tmp_path / name)},
        "model": {"name_or_path": str(tiny_model_dir), "tokenizer_name_or_path": "builtin:byte",
                  "precision": "fp32", "local_files_only": True},
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        "data_view": {"view": view, "max_records": 16},
        "training": {"epochs": 1, "per_device_batch_size": 2, "gradient_accumulation_steps": 2,
                     "max_seq_len": 128, "learning_rate": 1e-3, "logging_steps": 1, "save_steps": 0},
    }
    for key, value in extra.items():
        data.setdefault(key, {}).update(value)
    return load_config(None, extra=data)


def test_full_arm_produces_an_adapter(dataset, dataset_root, tiny_model_dir, tmp_path):
    cfg = train_cfg(dataset_root, tiny_model_dir, tmp_path, name="full", condition="FULL", view="full_context")
    manifest = train(cfg, dataset)
    adapter = Path(cfg.experiment.output_dir) / "adapter"
    assert any(p.name.startswith("adapter_model") for p in adapter.iterdir())
    assert (adapter / "adapter_config.json").exists()
    exposure = manifest["training"]["exposure"]
    assert exposure["records_seen"] == 16 and exposure["optimizer_updates"] >= 1
    assert exposure["target_tokens"] > 0
    assert manifest["model"]["trainable_parameter_count"] < manifest["model"]["total_parameter_count"]
    assert manifest["training"]["loss_history"]


def test_split_arm_produces_an_adapter(dataset, dataset_root, tiny_model_dir, tmp_path):
    cfg = train_cfg(dataset_root, tiny_model_dir, tmp_path, name="split", condition="SPLIT_STATEFUL",
                    view="split_stateful", training={"loss_weighting": "inverse_transitions"})
    manifest = train(cfg, dataset)
    assert (Path(cfg.experiment.output_dir) / "adapter" / "adapter_config.json").exists()
    assert manifest["training"]["exposure"]["weighted_records"] < manifest["training"]["exposure"]["records_seen"]


def test_run_manifest_records_everything_the_protocol_requires(dataset, dataset_root, tiny_model_dir, tmp_path):
    cfg = train_cfg(dataset_root, tiny_model_dir, tmp_path, name="manifest", condition="FULL", view="full_context")
    manifest = train(cfg, dataset)
    for key in ("resolved_config", "config_fingerprint", "seeding", "environment", "dataset",
                "chat_template", "view_plan", "order_diagnostics", "model", "optimization", "training"):
        assert key in manifest, key
    assert manifest["dataset"]["file_sha256"]
    assert manifest["environment"]["packages"]["torch"] != "not-installed"
    exposure = manifest["training"]["exposure"]
    for key in ("input_tokens", "target_tokens", "padded_tokens", "optimizer_updates",
                "unique_problems_seen", "truncated_prompts", "truncated_targets"):
        assert key in exposure, key


def test_budget_guard_stops_training_early(dataset, dataset_root, tiny_model_dir, tmp_path):
    cfg = train_cfg(dataset_root, tiny_model_dir, tmp_path, name="budget", condition="FULL", view="full_context",
                    data_view={"max_records": None},
                    budget={"mode": "native", "max_records": 8})
    manifest = train(cfg, dataset)
    exposure = manifest["training"]["exposure"]
    assert exposure["records_seen"] <= 10
    assert "max_records" in (manifest["training"]["stop_reason"] or "")


def test_checkpoint_resume_restores_position(dataset, dataset_root, tiny_model_dir, tmp_path):
    cfg = train_cfg(dataset_root, tiny_model_dir, tmp_path, name="ckpt", condition="FULL", view="full_context",
                    training={"save_steps": 1, "save_total_limit": 5})
    train(cfg, dataset)
    checkpoints = sorted((Path(cfg.experiment.output_dir) / "checkpoints").iterdir())
    assert checkpoints
    resumed = train_cfg(dataset_root, tiny_model_dir, tmp_path, name="ckpt_resume", condition="FULL",
                        view="full_context", training={"resume_from": str(checkpoints[0])})
    manifest = train(resumed, dataset)
    # resuming mid-epoch must consume strictly fewer records than a fresh run
    assert manifest["training"]["exposure"]["records_seen"] < 16


def test_weighted_loss_respects_example_weights():
    logits = torch.zeros(2, 5, 8)
    logits[0, :, 3] = 10.0     # example 0 is confident and correct
    labels = torch.full((2, 5), IGNORE_INDEX)
    labels[0, 1:] = 3
    labels[1, 1:] = 4          # example 1 is confidently wrong
    equal, _ = weighted_causal_lm_loss(logits, labels, torch.tensor([1.0, 1.0]))
    favour_good, _ = weighted_causal_lm_loss(logits, labels, torch.tensor([1.0, 0.0]))
    assert favour_good.item() < equal.item()


def test_weighted_loss_ignores_examples_with_no_supervised_tokens():
    logits = torch.zeros(2, 5, 8)
    labels = torch.full((2, 5), IGNORE_INDEX)
    labels[0, 1:] = 3
    loss, meta = weighted_causal_lm_loss(logits, labels, torch.tensor([1.0, 1.0]))
    assert meta["active_examples"] == 1
    assert torch.isfinite(loss)


def test_precision_selection_is_explicit():
    assert select_precision("fp32")[0] == "fp32"
    assert select_precision("bf16")[0] == "bf16"
    assert select_precision("auto")[0] in ("bf16", "fp16", "fp32")
