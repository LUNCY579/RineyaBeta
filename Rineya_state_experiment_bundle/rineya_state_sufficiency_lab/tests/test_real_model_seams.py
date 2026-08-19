"""Seams that only a real training/evaluation run exercises.

These use a *randomly initialised* Qwen3 of toy size together with the **real**
Qwen3 tokenizer and chat template. The weights are meaningless; the point is that
the tokenizer, chat template, left-padding, batched generation, LoRA attachment,
precision promotion and checkpoint-selection code paths are the same ones the GPU
run will take. Every one of these was previously uncovered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from rineya_lab.config import load_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_qwen_dirs() -> list[Path]:
    """Where the official Qwen3-0.6B checkpoint might be, most explicit first.

    These eight tests are the only ones that exercise the real tokenizer, chat
    template, left-padding, LoRA attachment and batched-generation seams -- the
    code paths the GPU run actually takes. Skipping them because the checkpoint
    sits one directory over is the worst outcome available: the run proceeds
    with its highest-risk seams unverified.

    RINEYA_QWEN_DIR mirrors RINEYA_DATASET_ROOT, so a checkpoint anywhere can be
    tested without moving 1.5 GB or hard-coding a machine path into the repo.
    """
    candidates: list[Path] = []
    override = os.environ.get("RINEYA_QWEN_DIR")
    if override:
        candidates.append(Path(override))
    # The layout README_BUNDLE.md documents, then one level further out, where a
    # checkpoint shared across sibling projects naturally lives. Windows and
    # macOS are case-insensitive; Linux is not, so try both spellings.
    for parent in (REPO_ROOT.parent, REPO_ROOT.parent.parent):
        candidates.extend([parent / "qwen3-0.6b", parent / "Qwen3-0.6B"])
    return candidates


@pytest.fixture(scope="module")
def qwen_tokenizer_dir() -> Path:
    for candidate in _candidate_qwen_dirs():
        if (candidate / "tokenizer_config.json").exists():
            return candidate
    searched = "\n  ".join(str(c) for c in _candidate_qwen_dirs())
    pytest.skip(
        "official Qwen3-0.6B tokenizer not found. Set RINEYA_QWEN_DIR, or place the "
        f"checkpoint at one of:\n  {searched}"
    )


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory, qwen_tokenizer_dir) -> Path:
    """A toy Qwen3 whose vocabulary matches the real tokenizer."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    directory = tmp_path_factory.mktemp("tiny_qwen")
    config = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=1024, tie_word_embeddings=True,
        bos_token_id=151643, eos_token_id=151645,
    )
    Qwen3ForCausalLM(config).save_pretrained(str(directory))
    AutoTokenizer.from_pretrained(str(qwen_tokenizer_dir), local_files_only=True).save_pretrained(str(directory))
    return directory


def base_cfg(dataset_root, tiny_qwen, tmp_path, **extra):
    data = {
        "dataset": {"root": str(dataset_root), "verify_hashes": False, "train_split": "dev", "dev_split": "dev"},
        "experiment": {"name": "seams", "condition": "FULL", "seed": 7, "output_dir": str(tmp_path / "run")},
        "model": {"name_or_path": str(tiny_qwen), "local_files_only": True, "precision": "fp32",
                  "chat_template_kwargs": {"enable_thinking": False}},
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        "data_view": {"view": "full_context", "max_records": 8},
        "training": {"per_device_batch_size": 2, "gradient_accumulation_steps": 2, "max_seq_len": 192,
                     "logging_steps": 1, "save_steps": 0},
    }
    for key, value in extra.items():
        data.setdefault(key, {}).update(value)
    return load_config(None, extra=data)


# -- the real chat template, through the real formatter ---------------------------------


def test_real_template_renders_with_thinking_pinned_off(tiny_qwen):
    from rineya_lab.training.chat import ChatFormatter, ensure_pad_token
    from rineya_lab.training.modeling import build_tokenizer

    cfg = load_config(None, extra={"model": {"name_or_path": str(tiny_qwen), "local_files_only": True,
                                             "chat_template_kwargs": {"enable_thinking": False}}})
    tokenizer = build_tokenizer(cfg.model)
    ensure_pad_token(tokenizer)
    formatter = ChatFormatter(tokenizer, cfg.model)
    assert formatter.template_source == "tokenizer_chat_template"

    prompt = formatter.render_prompt([{"role": "user", "content": "Solve 7x + 5 = 47."}])
    assert prompt.endswith("<think>\n\n</think>\n\n"), prompt[-40:]
    assert formatter.template_info()["template_kwargs"] == {"enable_thinking": False}

    # assistant-only masking must still be exact under the real template
    encoded = formatter.encode_example(
        [{"role": "user", "content": "Solve 7x + 5 = 47."},
         {"role": "assistant", "content": "Divide by 7: x = 6."}],
        max_seq_len=256,
    )
    from rineya_lab.training.chat import IGNORE_INDEX

    assert all(label == IGNORE_INDEX for label in encoded.labels[: encoded.prompt_token_count])
    assert encoded.labels[encoded.prompt_token_count :] == encoded.input_ids[encoded.prompt_token_count :]


def test_thinking_on_and_off_render_differently(tiny_qwen):
    from rineya_lab.training.chat import ChatFormatter
    from rineya_lab.training.modeling import build_tokenizer

    messages = [{"role": "user", "content": "hi"}]
    off = load_config(None, extra={"model": {"name_or_path": str(tiny_qwen), "local_files_only": True,
                                             "chat_template_kwargs": {"enable_thinking": False}}})
    on = load_config(None, extra={"model": {"name_or_path": str(tiny_qwen), "local_files_only": True}})
    tokenizer = build_tokenizer(off.model)
    rendered_off = ChatFormatter(tokenizer, off.model).render_prompt(messages)
    rendered_on = ChatFormatter(tokenizer, on.model).render_prompt(messages)
    assert rendered_off != rendered_on
    assert "<think>" in rendered_off and "<think>" not in rendered_on


# -- HF backend: batched generation must equal unbatched --------------------------------


def test_hf_backend_batching_matches_single_generation(dataset, dataset_root, tiny_qwen, tmp_path):
    """Left-padding and attention masks must not change what the model emits.

    This is the classic batched-decoding bug, and the previous equivalence test
    only covered the reference backend, which has no padding at all.
    """
    from rineya_lab.evaluation.backends import HFBackend

    cfg = base_cfg(dataset_root, tiny_qwen, tmp_path, evaluation={
        "split": "dev", "limit": 6, "protocols": ["full_direct"],
        "backend": {"kind": "hf", "adapter_path": None, "device": "cpu"},
        "generation": {"batch_size": 4, "max_new_tokens": 12, "do_sample": False},
    })
    cfg.lora.enabled = False
    backend = HFBackend(cfg)

    prompts = [
        [{"role": "user", "content": text}]
        for text in ["Solve 7x + 5 = 47.", "A short one.",
                     "A considerably longer prompt that pads the others out to a different length entirely.",
                     "Third."]
    ]
    single = [backend.generate(p).text for p in prompts]
    batched = [r.text for r in backend.generate_batch(prompts)]
    assert batched == single, "batched decoding diverged from single-prompt decoding"


def test_hf_backend_passes_every_declared_stop_token(dataset_root, tiny_qwen, tmp_path):
    from rineya_lab.evaluation.backends import HFBackend

    cfg = base_cfg(dataset_root, tiny_qwen, tmp_path, evaluation={
        "backend": {"kind": "hf", "adapter_path": None, "device": "cpu"},
        "generation": {"batch_size": 2, "max_new_tokens": 4, "do_sample": False},
    })
    cfg.lora.enabled = False
    backend = HFBackend(cfg)
    stop_ids = backend._stop_token_ids()
    assert 151645 in stop_ids                      # <|im_end|>
    assert backend.describe()["stop_token_ids"] == stop_ids


# -- precision promotion ----------------------------------------------------------------


def test_low_precision_base_keeps_fp32_adapter_weights(tiny_qwen):
    """End state: frozen base in fp16, trainable adapter in fp32.

    peft already builds adapters in fp32, so `cast_trainable_params_to_fp32` is a
    safety net rather than the mechanism. What matters for the experiment is the
    invariant, which is asserted here regardless of who established it.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from rineya_lab.training.modeling import cast_trainable_params_to_fp32

    model = AutoModelForCausalLM.from_pretrained(str(tiny_qwen), local_files_only=True, dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    cast_trainable_params_to_fp32(model)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert any(p.dtype == torch.float16 for p in model.parameters() if not p.requires_grad)


def test_promotion_helper_repairs_fp16_adapter_weights(tiny_qwen):
    """If anything ever does hand us fp16 trainable weights, the net must catch it."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from rineya_lab.training.modeling import cast_trainable_params_to_fp32

    model = AutoModelForCausalLM.from_pretrained(str(tiny_qwen), local_files_only=True, dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.half()
    assert any(p.requires_grad and p.dtype == torch.float16 for p in model.parameters())

    promoted = cast_trainable_params_to_fp32(model)
    assert promoted > 0
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)


# -- DEV-based checkpoint selection ------------------------------------------------------


def test_best_dev_loss_checkpoint_selection(dataset, dataset_root, tiny_qwen, tmp_path):
    """This path runs on every configured GPU training run and had no coverage."""
    from rineya_lab.training.loop import train

    cfg = base_cfg(dataset_root, tiny_qwen, tmp_path, training={
        "eval_steps": 1, "checkpoint_selection": "best_dev_loss",
        "dev_eval_max_records": 4, "dev_eval_batch_size": 2,
    })
    manifest = train(cfg, dataset)
    output_dir = Path(cfg.experiment.output_dir)

    training = manifest["training"]
    assert training["dev_loss_history"], "no DEV loss was ever measured"
    assert all(entry["dev_loss"] > 0 for entry in training["dev_loss_history"])
    assert training["best_dev_loss"] == min(e["dev_loss"] for e in training["dev_loss_history"])
    assert training["selected_checkpoint"].startswith("best_dev_loss@update_")

    # the evaluated adapter is the best one; the final-step adapter is preserved beside it
    assert (output_dir / "adapter" / "adapter_config.json").exists()
    assert (output_dir / "best_adapter" / "adapter_config.json").exists()
    assert (output_dir / "last_adapter" / "adapter_config.json").exists()
    assert manifest["dev_probe"]["records"] == 4
    assert manifest["checkpoint_selection"]["rule"] == "best_dev_loss"

    on_disk = json.loads((output_dir / "run_manifest.json").read_text())
    assert on_disk["training"]["selected_checkpoint"] == training["selected_checkpoint"]


def test_last_checkpoint_selection_leaves_no_best_adapter(dataset, dataset_root, tiny_qwen, tmp_path):
    from rineya_lab.training.loop import train

    cfg = base_cfg(dataset_root, tiny_qwen, tmp_path, training={"checkpoint_selection": "last", "eval_steps": 0})
    manifest = train(cfg, dataset)
    output_dir = Path(cfg.experiment.output_dir)
    assert manifest["training"]["selected_checkpoint"] == "last"
    assert not (output_dir / "best_adapter").exists()
    assert not (output_dir / "last_adapter").exists()
