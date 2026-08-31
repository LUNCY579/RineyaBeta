"""Offline end-to-end engineering smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch

from dast.artifacts import atomic_write_json
from dast.config import (
    DASTConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    LoraConfig,
    ModelConfig,
    OperatorConfig,
    SemanticConfig,
    TrainingConfig,
)
from dast.data import build_examples, load_records, validate_dataset
from dast.interventions import negate_state
from dast.model import DASTModel
from dast.operators import LowRankOperatorBank
from dast.tokenization import CausalBatchCollator
from dast.toy import HashTokenizer, ToyCausalLM
from dast.training import train


def fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "tiny_reasoning.jsonl"


def build_smoke_config(output_dir: Path, *, arm: str = "operator_ordered") -> DASTConfig:
    uses_operators = arm.startswith("operator_")
    baseline = arm == "cot_baseline"
    config = DASTConfig(
        experiment=ExperimentConfig(
            name=f"smoke-{arm}",
            arm=arm,  # type: ignore[arg-type]
            seed=17,
            output_dir=str(output_dir),
        ),
        model=ModelConfig(
            name_or_path="toy",
            precision="fp32",
            max_length=160,
            # 0.9 * 8 -> 7, deliberately below the top layer: with readout
            # 'final_norm' the last checkpoint must not be the already-normed
            # final hidden state. Mirrors the shipped 28-layer schedule so the
            # offline smoke actually exercises the corrected geometry instead of
            # the pre-repair one.
            checkpoint_fractions=(0.25, 0.5, 0.75, 0.9),
            checkpoint_readout="final_norm",
            pooling="mean",
        ),
        data=DataConfig(
            path=str(fixture_path()),
            train_split="train",
            eval_split="dev",
            max_train_records=None,
            max_eval_records=None,
            system_prompt="Solve carefully and state the final answer.",
        ),
        semantic=SemanticConfig(
            state_dim=24,
            projector_hidden_dim=48,
            target_projection_seed=1701,
            target_max_tokens=32,
            target_centering="within_problem",
            state_weight=0.0 if baseline else 0.5,
            contrastive_weight=0.0 if baseline else 0.05,
        ),
        operators=OperatorConfig(
            enabled=uses_operators,
            num_operators=4,
            rank=4,
            transition_weight=0.25 if uses_operators else 0.0,
            consistency_weight=0.05 if uses_operators else 0.0,
            diversity_weight=0.001 if uses_operators else 0.0,
            balance_weight=0.001 if uses_operators else 0.0,
        ),
        lora=LoraConfig(enabled=False),
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            eval_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-3,
            auxiliary_learning_rate=5e-3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            max_grad_norm=1.0,
            gradient_checkpointing=False,
            freeze_base_without_lora=False,
            log_every_steps=1,
            dataloader_shuffle=True,
        ),
        evaluation=EvaluationConfig(do_generation=False, bootstrap_samples=100),
    )
    config.validate()
    return config


def _make_model(config: DASTConfig, *, feedback: bool = False) -> tuple[DASTModel, HashTokenizer]:
    tokenizer = HashTokenizer(vocab_size=512)
    base = ToyCausalLM(vocab_size=512, hidden_size=48, num_hidden_layers=8)
    # The semantic target geometry must be fixed.  Other toy backbone parameters
    # remain trainable so the smoke path exercises LM and auxiliary gradients.
    base.get_input_embeddings().weight.requires_grad_(False)
    operator = None
    if config.operators.enabled:
        operator = LowRankOperatorBank(
            config.semantic.state_dim,
            config.operators.num_operators,
            config.operators.rank,
            router_temperature=config.operators.router_temperature,
        )
    model = DASTModel(
        base,
        checkpoint_fractions=config.model.checkpoint_fractions,
        state_dim=config.semantic.state_dim,
        projector_hidden_dim=config.semantic.projector_hidden_dim,
        pooling=config.model.pooling,
        operator_bank=operator,
        feedback_enabled=feedback,
        feedback_scale=0.2 if feedback else config.model.feedback_scale,
        checkpoint_readout=config.model.checkpoint_readout,
    )
    return model, tokenizer


def _feedback_qualification(config: DASTConfig) -> dict[str, Any]:
    """Qualify the activation-patching PATH, and report the shipped-init no-op.

    The feedback decoder is zero-initialised by contract (model.py), so at
    initialisation ``delta = decoder(state) * gate`` is identically zero and every
    intervention is a bitwise no-op. This function previously overwrote the decoder
    and the gates and reported only the perturbed number, which made a green smoke
    read as "the configured causal mechanism works". Both numbers are now reported:
    the shipped-init delta (expected exactly 0.0) and the perturbed delta that
    shows the hook, pooling, intervention and residual-injection path execute.
    """

    feedback_config = build_smoke_config(Path(config.experiment.output_dir) / "feedback", arm="state_ordered")
    feedback_config.model.feedback_enabled = True
    model, tokenizer = _make_model(feedback_config, feedback=True)
    assert model.feedback_decoder is not None and model.feedback_gates is not None
    shipped_init_delta = _feedback_delta(model, tokenizer, feedback_config)
    torch.nn.init.normal_(model.feedback_decoder.weight, std=0.02)
    with torch.no_grad():
        model.feedback_gates.fill_(2.0)
    perturbed_delta = _feedback_delta(model, tokenizer, feedback_config)
    return {
        "feedback_logits_changed": perturbed_delta > 0.0,
        "feedback_max_logit_delta": perturbed_delta,
        "feedback_stage_count": model.num_stages,
        "shipped_init_max_logit_delta": shipped_init_delta,
        "shipped_init_is_no_op": shipped_init_delta == 0.0,
        "claim_boundary": (
            "The perturbed delta qualifies the activation-patching PATH only. The "
            "zero-initialised decoder makes interventions a bitwise no-op until the "
            "decoder has trained, so an untrained feedback run cannot produce a "
            "causal result and must not be reported as one."
        ),
    }


def _feedback_delta(model: DASTModel, tokenizer: Any, feedback_config: DASTConfig) -> float:
    records = load_records(fixture_path(), split="dev", limit=2)
    examples = build_examples(records, arm=feedback_config.arm_spec, num_stages=4, seed=17)
    batch = CausalBatchCollator(
        tokenizer,
        system_prompt=feedback_config.data.system_prompt,
        max_length=feedback_config.model.max_length,
    )(examples)
    model.eval()
    with torch.no_grad():
        ordinary = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            prompt_mask=batch["prompt_mask"],
        )
        intervened = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            prompt_mask=batch["prompt_mask"],
            interventions={1: negate_state},
        )
    return float((ordinary.logits - intervened.logits).abs().max())


def run_smoke(output_dir: str | Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="dast-smoke-")
        root = Path(temporary.name)
    else:
        temporary = None
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
    try:
        validation = validate_dataset(fixture_path())
        if not validation.passed:
            raise RuntimeError(f"fixture validation failed: {validation.errors}")
        config = build_smoke_config(root / "run", arm="operator_ordered")
        model, tokenizer = _make_model(config)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.state_projector.named_parameters()
        }
        result = train(
            model,
            tokenizer,
            config,
            model_info={"kind": "offline_toy", "quality_claim_allowed": False},
        )
        # train() moves the model to the accelerator, so the pre-training snapshot
        # and the live parameter can sit on different devices. Compare on one device
        # or this check raises instead of answering, on every CUDA host.
        projector_changed = any(
            not torch.equal(before[name], parameter.detach().to(before[name].device))
            for name, parameter in model.state_projector.named_parameters()
        )
        report = {
            "status": "PASS",
            "dataset": validation.as_dict(),
            "training": {
                "global_steps": result.global_steps,
                "optimizer_steps": result.optimizer_steps,
                "finite_loss": bool(torch.isfinite(torch.tensor(result.final_metrics["loss/total"]))),
                "projector_updated": projector_changed,
                "checkpoint_layers": list(model.checkpoint_layers),
                "state_shape": [2, config.num_stages, config.semantic.state_dim],
            },
            "feedback": _feedback_qualification(config),
            "claim_boundary": "Toy smoke proves software execution only; it is not Rineya model evidence.",
        }
        if not report["training"]["projector_updated"]:
            raise RuntimeError("smoke training did not update the shared state projector")
        if not report["feedback"]["feedback_logits_changed"]:
            raise RuntimeError("semantic feedback intervention did not change downstream logits")
        atomic_write_json(root / "smoke_report.json", report)
        return report
    finally:
        if temporary is not None:
            temporary.cleanup()
