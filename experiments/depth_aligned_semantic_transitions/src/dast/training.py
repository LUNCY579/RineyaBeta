"""Deterministic training and teacher-forced DEV evaluation."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dast.artifacts import append_jsonl, atomic_write_json, build_run_manifest, save_checkpoint
from dast.config import DASTConfig, assert_split_allowed
from dast.data import DASTExample, build_examples, load_records
from dast.evaluation import evaluate_generation
from dast.losses import LossBreakdown, compute_loss, depth_alignment_metrics
from dast.model import DASTModel
from dast.semantic import FrozenTextTargetEncoder
from dast.tokenization import CausalBatchCollator


class ExampleDataset(Dataset[DASTExample]):
    def __init__(self, examples: Sequence[DASTExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DASTExample:
        return self.examples[index]


@dataclass
class TrainingResult:
    output_dir: Path
    global_steps: int
    optimizer_steps: int
    final_metrics: dict[str, float]


def set_global_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(parameter.device for parameter in model.parameters())
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def build_dataloaders(
    config: DASTConfig,
    tokenizer: Any,
) -> tuple[DataLoader[Any], DataLoader[Any], list[Path]]:
    assert_split_allowed(config, config.data.train_split)
    assert_split_allowed(config, config.data.eval_split)
    train_path = config.data_path_for_split(config.data.train_split)
    eval_path = config.data_path_for_split(config.data.eval_split)
    train_records = load_records(
        train_path,
        split=config.data.train_split,
        limit=config.data.max_train_records,
    )
    eval_records = load_records(
        eval_path,
        split=config.data.eval_split,
        limit=config.data.max_eval_records,
    )
    # Every corpus file a run touched, so the manifest can hash each one.
    data_paths = [train_path] if train_path == eval_path else [train_path, eval_path]
    if not train_records:
        raise ValueError(f"no records found for train split {config.data.train_split!r}")
    if not eval_records:
        raise ValueError(f"no records found for eval split {config.data.eval_split!r}")
    train_examples = build_examples(
        train_records,
        arm=config.arm_spec,
        num_stages=config.num_stages,
        seed=config.experiment.seed,
    )
    eval_examples = build_examples(
        eval_records,
        arm=config.arm_spec,
        num_stages=config.num_stages,
        seed=config.experiment.seed,
    )
    collator = CausalBatchCollator(
        tokenizer,
        system_prompt=config.data.system_prompt,
        max_length=config.model.max_length,
        truncation=config.data.truncation,
    )
    generator = torch.Generator()
    generator.manual_seed(config.experiment.seed)
    train_loader = DataLoader(
        ExampleDataset(train_examples),
        batch_size=config.training.batch_size,
        shuffle=config.training.dataloader_shuffle,
        num_workers=config.data.num_workers,
        collate_fn=collator,
        generator=generator,
    )
    eval_loader = DataLoader(
        ExampleDataset(eval_examples),
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )
    return train_loader, eval_loader, data_paths


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _resolve_runtime_precision(model: torch.nn.Module, requested: str) -> str:
    if requested != "auto":
        return requested
    for parameter in model.parameters():
        if not parameter.is_floating_point():
            continue
        if parameter.dtype == torch.bfloat16:
            return "bf16"
        if parameter.dtype == torch.float16:
            return "fp16"
    return "fp32"


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_optimizer(model: DASTModel, config: DASTConfig) -> torch.optim.Optimizer:
    backbone: list[torch.nn.Parameter] = []
    auxiliary: list[torch.nn.Parameter] = []
    auxiliary_ids = {
        id(parameter)
        for module in (model.state_projector, model.operator_bank, model.feedback_decoder)
        if module is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if model.feedback_gates is not None and model.feedback_gates.requires_grad:
        auxiliary_ids.add(id(model.feedback_gates))
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (auxiliary if id(parameter) in auxiliary_ids else backbone).append(parameter)
    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone, "lr": config.training.learning_rate})
    if auxiliary:
        groups.append(
            {
                "params": auxiliary,
                "lr": config.training.auxiliary_learning_rate or config.training.learning_rate,
            }
        )
    if not groups:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=config.training.weight_decay)


def _make_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _run_status(config: DASTConfig, *, finished: bool) -> str:
    """Compose the machine-readable run status.

    Derived from the declared evidence tier AND the split, so an engineering
    rehearsal can never be mistaken for a qualifying DEV result, and a run that
    touches the sealed split says so regardless of tier. Split comparison is
    case-insensitive; a config saying ``eval_split: "TEST"`` previously produced a
    manifest labelled DEV.
    """

    # BOTH splits, not just eval: a run that TRAINS on the sealed split would
    # otherwise report ..__DEV__.. and read as an ordinary DEV result.
    sealed = "test" in {
        config.data.train_split.strip().lower(),
        config.data.eval_split.strip().lower(),
    }
    tier = {
        "engineering": "ENGINEERING",
        "exploratory_dev": "EXPLORATORY_DEV",
        "confirmatory_dev": "CONFIRMATORY_DEV",
    }[config.experiment.evidence_status]
    scope = "SEALED_TEST" if sealed else "DEV"
    phase = "COMPLETE" if finished else "RUNNING"
    return f"{tier}__{scope}__{phase}"


def _encode_targets(
    encoder: FrozenTextTargetEncoder,
    texts: Sequence[Sequence[str]],
    *,
    device: torch.device,
    centering: str = "none",
    reference_texts: Sequence[Sequence[str]] | None = None,
) -> torch.Tensor | None:
    """Encode a [batch, stage] text grid into semantic targets.

    ``centering='within_problem'`` subtracts each problem's own mean over its
    CANONICAL states and renormalises. Phase-0 measurement motivating this: as
    shipped, the targets carry 56.6% of their variance in the problem factor and
    only 9.3% in the stage factor, so a depth permutation rearranges only the
    small component and `state_ordered` sits at 0.86 cosine from
    `state_shuffled`. Centering moves that to -0.02 and flips the factor ratio
    from 0.16 to 6.3, aligning the target side with the hidden side.

    The mean is always taken over the canonical grid, never the arm's own
    training grid. That keeps the subtracted constant identical across arms, so
    the controls stay matched: a permutation leaves the mean unchanged, and
    `state_final_repeated` still receives a constant non-zero target at every
    depth rather than degenerating to zero.
    """

    if not texts or not texts[0]:
        return None
    grid = encoder.encode_grid(texts, device=device)
    if centering == "none":
        return grid
    if centering != "within_problem":
        raise ValueError(f"unsupported target centering mode: {centering}")
    if reference_texts is None:
        # Centering on the arm's OWN grid is the mode this function forbids: it
        # would give each arm a different subtracted mean, and state_final_repeated
        # would collapse to a zero target. Refuse rather than silently break the
        # matched-control contract.
        raise ValueError(
            "target centering requires reference_texts (the canonical grid); "
            "centering on the arm's own targets would unmatch the controls"
        )
    reference = encoder.encode_grid(reference_texts, device=device)
    return torch.nn.functional.normalize(grid - reference.mean(dim=1, keepdim=True), dim=-1)


@torch.no_grad()
def evaluate_teacher_forced(
    model: DASTModel,
    loader: DataLoader[Any],
    target_encoder: FrozenTextTargetEncoder,
    config: DASTConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    weighted: dict[str, float] = {}
    total_examples = 0
    canonical_predictions: list[torch.Tensor] = []
    canonical_targets_all: list[torch.Tensor] = []
    routing_entropies: list[float] = []

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = int(batch["input_ids"].shape[0])
        training_targets = _encode_targets(
            target_encoder,
            batch["training_target_texts"],
            device=device,
            centering=config.semantic.target_centering,
            reference_texts=batch["canonical_target_texts"],
        )
        canonical_targets = _encode_targets(
            target_encoder,
            batch["canonical_target_texts"],
            device=device,
            centering=config.semantic.target_centering,
            reference_texts=batch["canonical_target_texts"],
        )
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            prompt_mask=batch["prompt_mask"],
        )
        breakdown = compute_loss(
            output,
            config=config,
            training_targets=training_targets,
            training_group_ids=batch["training_group_ids"],
            operator_bank=model.operator_bank,
        )
        for key, value in breakdown.detached().items():
            weighted[key] = weighted.get(key, 0.0) + value * batch_size
        if canonical_targets is not None:
            canonical_predictions.append(output.states.detach().cpu())
            canonical_targets_all.append(canonical_targets.detach().cpu())
        if output.router_weights is not None and model.operator_bank is not None:
            routing_entropies.append(float(model.operator_bank.routing_entropy(output.router_weights).cpu()))
        total_examples += batch_size

    metrics = {key: value / max(1, total_examples) for key, value in weighted.items()}
    if canonical_predictions:
        metrics.update(
            depth_alignment_metrics(
                torch.cat(canonical_predictions, dim=0),
                torch.cat(canonical_targets_all, dim=0),
            )
        )
    if routing_entropies:
        metrics["operator/routing_entropy"] = float(np.mean(routing_entropies))
    if "loss/lm" in metrics:
        metrics["lm/perplexity"] = float(math.exp(min(20.0, metrics["loss/lm"])))
    metrics["examples"] = float(total_examples)
    if was_training:
        model.train()
    return metrics


def train(
    model: DASTModel,
    tokenizer: Any,
    config: DASTConfig,
    *,
    model_info: dict[str, Any] | None = None,
) -> TrainingResult:
    config.validate()
    set_global_seed(
        config.experiment.seed,
        deterministic_algorithms=config.training.deterministic_algorithms,
    )
    train_loader, eval_loader, data_paths = build_dataloaders(config, tokenizer)
    device = resolve_device(model)
    if config.model.device_map is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device)
    else:
        # Keep the shared semantic interface on the input device. Hidden states
        # from other shards are copied to this small head explicitly in DASTModel.
        model.state_projector.to(device)
        if model.operator_bank is not None:
            model.operator_bank.to(device)
        if model.feedback_decoder is not None:
            model.feedback_decoder.to(device)
        if model.feedback_gates is not None:
            model.feedback_gates.data = model.feedback_gates.data.to(device)
    embedding = model.base_model.get_input_embeddings()
    target_encoder = FrozenTextTargetEncoder(
        embedding,
        tokenizer,
        state_dim=config.semantic.state_dim,
        projection_seed=config.semantic.target_projection_seed,
        max_tokens=config.semantic.target_max_tokens,
    )
    optimizer = _make_optimizer(model, config)
    runtime_precision = _resolve_runtime_precision(model.base_model, config.model.precision)
    updates_per_epoch = math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.training.epochs
    scheduler = _make_scheduler(
        optimizer,
        total_steps=total_updates,
        warmup_ratio=config.training.warmup_ratio,
    )
    scaler_enabled = device.type == "cuda" and runtime_precision == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # pragma: no cover - torch 2.1 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    output_dir = config.resolved_path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config,
        data_path=data_paths,
        checkpoint_layers=model.checkpoint_layers,
        model_info=model_info or {},
        status=_run_status(config, finished=False),
    )
    manifest["runtime_precision"] = runtime_precision
    manifest["trainable_parameter_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    manifest["total_parameter_count"] = sum(p.numel() for p in model.parameters())
    atomic_write_json(output_dir / "run_manifest.json", manifest)

    global_step = 0
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.perf_counter()
    last_breakdown: LossBreakdown | None = None

    for epoch in range(config.training.epochs):
        for raw_batch in train_loader:
            global_step += 1
            batch = _move_batch(raw_batch, device)
            training_targets = _encode_targets(
                target_encoder,
                batch["training_target_texts"],
                device=device,
                centering=config.semantic.target_centering,
                reference_texts=batch["canonical_target_texts"],
            )
            with _autocast_context(device, runtime_precision):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    prompt_mask=batch["prompt_mask"],
                )
                breakdown = compute_loss(
                    output,
                    config=config,
                    training_targets=training_targets,
                    training_group_ids=batch["training_group_ids"],
                    operator_bank=model.operator_bank,
                )
                scaled_loss = breakdown.total / config.training.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            last_breakdown = breakdown

            boundary = global_step % config.training.gradient_accumulation_steps == 0
            final_batch = global_step == len(train_loader) * (epoch + 1)
            if boundary or final_batch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.training.max_grad_norm,
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if global_step == 1 or global_step % config.training.log_every_steps == 0:
                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.perf_counter() - start,
                    **breakdown.detached(),
                }
                append_jsonl(output_dir / "metrics.jsonl", row)

            if config.training.eval_every_steps and optimizer_step and optimizer_step % config.training.eval_every_steps == 0 and boundary:
                metrics = evaluate_teacher_forced(
                    model,
                    eval_loader,
                    target_encoder,
                    config,
                    device=device,
                )
                append_jsonl(output_dir / "evaluations.jsonl", {"optimizer_step": optimizer_step, **metrics})

            if config.training.save_every_steps and optimizer_step and optimizer_step % config.training.save_every_steps == 0 and boundary:
                save_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=optimizer_step,
                    config_sha256=config.canonical_sha256(),
                )

    final_metrics = evaluate_teacher_forced(
        model,
        eval_loader,
        target_encoder,
        config,
        device=device,
    )
    if config.evaluation.do_generation:
        eval_records = load_records(
            config.data_path_for_split(config.data.eval_split),
            split=config.data.eval_split,
            limit=config.data.max_eval_records,
        )
        generation = evaluate_generation(
            model,
            tokenizer,
            eval_records,
            config,
            output_dir=output_dir,
        )
        final_metrics["generation/accuracy"] = float(generation["accuracy"])
        final_metrics["generation/records"] = float(generation["records"])
    final_metrics["global_steps"] = float(global_step)
    final_metrics["optimizer_steps"] = float(optimizer_step)
    final_metrics["elapsed_seconds"] = time.perf_counter() - start
    if last_breakdown is not None:
        final_metrics.update({f"final_train/{k}": v for k, v in last_breakdown.detached().items()})
    summary_name = "test_summary.json" if config.data.eval_split.lower() == "test" else "dev_summary.json"
    atomic_write_json(output_dir / summary_name, final_metrics)
    torch.save(model.auxiliary_state_dict(), output_dir / "dast_auxiliary.pt")
    if hasattr(model.base_model, "save_pretrained"):
        model.base_model.save_pretrained(output_dir / "adapter")
    manifest["status"] = _run_status(config, finished=True)
    manifest["final_metrics_file"] = summary_name
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return TrainingResult(output_dir=output_dir, global_steps=global_step, optimizer_steps=optimizer_step, final_metrics=final_metrics)
