"""Deterministic controlled sequence-model reference experiment.

This is not a Rineya result. It is a small residual sequence model that executes
all PDRR paths and ablations without external checkpoints, allowing the research
contract to be tested on CPU and in CI.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .arms import arm_registry, get_arm
from .losses import (
    LossBreakdown,
    causal_significance,
    component_recovery_loss,
    feature_distillation_loss,
    token_kl,
)
from .metrics import handoff_completeness, recovery_ratio
from .routing import build_router, router_regularization
from .schedules import RecoverySchedule


@dataclass
class Trace:
    logits: Tensor
    hidden: Tensor
    inputs: dict[int, Tensor]
    residuals: dict[int, Tensor]


class ResidualSequenceBlock(nn.Module):
    """A small causal self-attention + MLP residual block."""

    def __init__(self, hidden_dim: int, ff_dim: int, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        q = self.norm1(x)
        attention, _ = self.attn(q, q, q, attn_mask=causal_mask, need_weights=False)
        x = x + attention
        x = x + self.ff(self.norm2(x))
        return x


class TinyResidualSequenceModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        ff_dim: int,
        layers: int,
        sequence_length: int,
        output_classes: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(sequence_length, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            ResidualSequenceBlock(hidden_dim, ff_dim) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_classes, bias=False)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def forward(
        self,
        x: Tensor,
        *,
        gates: dict[int, float | Tensor] | None = None,
        capture: Iterable[int] = (),
    ) -> Trace:
        if x.ndim != 3 or x.shape[1] > self.sequence_length:
            raise ValueError("expected [batch, sequence, input_dim] within configured length")
        capture_set = set(int(i) for i in capture)
        h = self.input_proj(x) + self.position[:x.shape[1]].unsqueeze(0)
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        inputs: dict[int, Tensor] = {}
        residuals: dict[int, Tensor] = {}
        for index, block in enumerate(self.blocks):
            before = h
            after = block(before, mask)
            delta = after - before
            gate: float | Tensor = 1.0 if gates is None else gates.get(index, 1.0)
            if isinstance(gate, Tensor):
                while gate.ndim < delta.ndim:
                    gate = gate.unsqueeze(-1)
                h = before + gate.to(delta) * delta
            else:
                h = before + float(gate) * delta
            if index in capture_set:
                inputs[index] = before
                residuals[index] = delta
        hidden = self.final_norm(h)
        return Trace(logits=self.head(hidden), hidden=hidden,
                     inputs=inputs, residuals=residuals)

    def block_residual(self, layer: int, hidden: Tensor) -> Tensor:
        seq = hidden.shape[1]
        mask = torch.triu(
            torch.ones(seq, seq, device=hidden.device, dtype=torch.bool), diagonal=1)
        return self.blocks[layer](hidden, mask) - hidden


@dataclass(frozen=True)
class ToyConfig:
    input_dim: int = 24
    hidden_dim: int = 32
    ff_dim: int = 64
    layers: int = 6
    sequence_length: int = 6
    output_classes: int = 19
    target_layer: int = 3
    recipients: tuple[int, ...] = (2, 4, 5)
    train_examples: int = 768
    eval_examples: int = 512
    steps: int = 160
    batch_size: int = 48
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    temperature: float = 2.0
    gradient_clip: float = 1.0
    notice_fraction: float = 0.20
    withdrawal_end_fraction: float = 0.70
    recovery_end_fraction: float = 0.78
    lambda_supervised: float = 0.35
    lambda_kd: float = 0.65
    lambda_feature: float = 0.20
    lambda_recovery: float = 0.80
    lambda_anchor: float = 0.10
    router_top_k: int = 2
    router_temperature: float = 0.8
    router_entropy_weight: float = 0.002
    router_load_balance_weight: float = 0.01


@dataclass
class EvalMetrics:
    accuracy: float
    mean_kl: float
    quality: float


@dataclass
class ArmResult:
    arm: str
    seed: int
    metrics: EvalMetrics
    recovery_ratio: float | None
    handoff_completeness: float | None
    mean_route: list[float] | None
    recipient_importance_before: list[float] | None
    recipient_importance_after: list[float] | None
    recipient_importance_gain: list[float] | None
    route_alignment_spearman: float | None
    top_route_top_gain_match: bool | None
    gate_zero_physical_max_abs: float | None
    final_parameter_count: int
    hard_parameter_count: int
    trainable_parameter_count: int
    last_losses: dict[str, float] | None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = asdict(self.metrics)
        return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_teacher(config: ToyConfig, seed: int) -> TinyResidualSequenceModel:
    seed_everything(seed)
    model = TinyResidualSequenceModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        ff_dim=config.ff_dim,
        layers=config.layers,
        sequence_length=config.sequence_length,
        output_classes=config.output_classes,
    )
    source = model.blocks[config.target_layer]
    with torch.no_grad():
        source.attn.out_proj.weight.mul_(1.8)
        source.ff[-1].weight.mul_(1.8)
        model.head.weight.mul_(1.25)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_data(config: ToyConfig, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = config.train_examples + config.eval_examples
    x = torch.randn(
        total, config.sequence_length, config.input_dim, generator=generator)
    positions = torch.linspace(-1.0, 1.0, config.sequence_length).view(1, -1, 1)
    x[:, :, :1] += positions * 1.5
    regime = torch.sign(x[:, :1, 1:2]).clamp(min=-1, max=1)
    x[:, :, 2:3] += regime * positions
    return x[:config.train_examples], x[config.train_examples:]


def _teacher_targets(teacher: TinyResidualSequenceModel, x: Tensor,
                     target_layer: int, recipients: Sequence[int]) -> tuple[Trace, Trace]:
    capture = {target_layer, *recipients}
    with torch.no_grad():
        full = teacher(x, capture=capture)
        ablated = teacher(x, gates={target_layer: 0.0}, capture=capture)
    return full, ablated


def evaluate(
    model: TinyResidualSequenceModel,
    teacher: TinyResidualSequenceModel,
    x: Tensor,
    *,
    target_layer: int,
    physically_pruned: bool = False,
) -> EvalMetrics:
    with torch.no_grad():
        teacher_logits = teacher(x).logits
        gates = None if physically_pruned else {target_layer: 0.0}
        logits = model(x, gates=gates).logits
        labels = teacher_logits.argmax(dim=-1)
        accuracy = float((logits.argmax(dim=-1) == labels).float().mean())
        mean_kl = float(token_kl(teacher_logits, logits).mean())
        quality = 0.5 * accuracy + 0.5 * math.exp(-mean_kl)
    return EvalMetrics(accuracy=accuracy, mean_kl=mean_kl, quality=quality)


def _freeze_for_recovery(model: TinyResidualSequenceModel,
                         recipients: Sequence[int]) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    for layer in recipients:
        for parameter in model.blocks[layer].parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return 0.0
    x_rank = np.argsort(np.argsort(x_arr)).astype(float)
    y_rank = np.argsort(np.argsort(y_arr)).astype(float)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _recipient_importance(
    model: TinyResidualSequenceModel,
    x: Tensor,
    *,
    target_layer: int,
    recipients: Sequence[int],
) -> list[float]:
    with torch.no_grad():
        base = model(x, gates={target_layer: 0.0}).logits
        values = []
        for recipient in recipients:
            gates = {target_layer: 0.0, recipient: 0.0}
            ablated = model(x, gates=gates).logits
            values.append(float(token_kl(base, ablated).mean()))
    return values


def _mean_route(router: nn.Module | None, teacher: TinyResidualSequenceModel,
                x: Tensor, config: ToyConfig) -> list[float] | None:
    if router is None:
        return None
    with torch.no_grad():
        trace = teacher(x, capture={config.target_layer})
        state = trace.inputs[config.target_layer]
        positions = torch.arange(x.shape[1])
        weights = router(state, positions)
        axes = tuple(range(weights.ndim - 1))
        return weights.float().mean(dim=axes).tolist()


def _physical_delete(model: TinyResidualSequenceModel, target_layer: int) -> TinyResidualSequenceModel:
    compact = copy.deepcopy(model)
    compact.blocks = nn.ModuleList([
        block for index, block in enumerate(compact.blocks) if index != target_layer
    ])
    return compact


def _physical_equivalence(
    model: TinyResidualSequenceModel,
    x: Tensor,
    target_layer: int,
) -> tuple[float, int]:
    compact = _physical_delete(model, target_layer)
    with torch.no_grad():
        gated = model(x, gates={target_layer: 0.0}).logits
        physical = compact(x).logits
    return float((gated - physical).abs().max()), parameter_count(compact)


def run_arm(
    arm_name: str,
    *,
    teacher: TinyResidualSequenceModel,
    train_x: Tensor,
    eval_x: Tensor,
    config: ToyConfig,
    seed: int,
    full_quality: float,
    hard_quality: float,
    hard_divergence: float,
) -> ArmResult:
    arm = get_arm(arm_name)
    seed_everything(seed)
    student = copy.deepcopy(teacher)
    student.train()
    trainable = _freeze_for_recovery(student, config.recipients)
    router = build_router(
        arm.routing,
        state_dim=config.hidden_dim,
        recipient_layers=config.recipients,
        source_layer=config.target_layer,
        seed=seed + 991,
        hidden_dim=48,
        temperature=config.router_temperature,
        top_k=config.router_top_k,
        max_positions=config.sequence_length,
    )
    router_params: list[nn.Parameter] = []
    if router is not None and arm.router_trainable:
        router_params = [p for p in router.parameters() if p.requires_grad]
    params = trainable + router_params
    optimizer = None
    if arm.train:
        optimizer = torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay)
    schedule = RecoverySchedule(
        config.steps,
        notice_fraction=config.notice_fraction,
        withdrawal_end_fraction=config.withdrawal_end_fraction,
        recovery_end_fraction=config.recovery_end_fraction,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 12345)
    last: LossBreakdown | None = None

    if arm.train:
        assert optimizer is not None
        for step in range(config.steps):
            indices = torch.randint(
                0, train_x.shape[0], (config.batch_size,), generator=generator)
            batch = train_x[indices]
            teacher_full, teacher_ablated = _teacher_targets(
                teacher, batch, config.target_layer, config.recipients)
            if arm.deletion == "hard":
                gate = 0.0
            elif arm.deletion == "gradual":
                gate = schedule.doomed_gate(step)
            else:
                gate = 1.0
            capture = {config.target_layer, *config.recipients}
            scheduled = student(batch, gates={config.target_layer: gate}, capture=capture)
            compact = student(batch, gates={config.target_layer: 0.0}, capture=capture)

            zero = scheduled.logits.new_zeros(())
            labels = teacher_full.logits.argmax(dim=-1)
            supervised = (F.cross_entropy(
                scheduled.logits.reshape(-1, scheduled.logits.shape[-1]),
                labels.reshape(-1)) if arm.supervised else zero)
            kd = (token_kl(
                teacher_full.logits.detach(), scheduled.logits,
                config.temperature).mean() if arm.kd else zero)
            feature = (feature_distillation_loss(
                teacher_full.hidden.detach(), scheduled.hidden)
                if arm.feature_kd else zero)

            recovery = zero
            router_reg = zero
            anchor = zero
            if router is not None:
                positions = torch.arange(batch.shape[1])
                route = router(teacher_full.inputs[config.target_layer].detach(), positions)
                reg = router_regularization(
                    route,
                    entropy_weight=config.router_entropy_weight,
                    load_balance_weight=config.router_load_balance_weight)
                router_reg = reg.total
                if arm.recovery_loss and schedule.recovery_weight(step) > 0:
                    recipient_deltas = []
                    anchor_distances = []
                    for recipient in config.recipients:
                        current_input = compact.inputs[recipient]
                        with torch.no_grad():
                            anchor_residual = teacher.block_residual(
                                recipient, current_input.detach())
                        current_residual = compact.residuals[recipient]
                        recipient_deltas.append(current_residual - anchor_residual)
                        anchor_distances.append(
                            ((scheduled.residuals[recipient] - anchor_residual) ** 2).mean())
                    significance = (
                        causal_significance(
                            teacher_full.logits, teacher_ablated.logits)
                        if arm.causal_significance else None)
                    recovery, _ = component_recovery_loss(
                        teacher_full.residuals[config.target_layer].detach(),
                        recipient_deltas, route, significance)
                    anchor = torch.stack(anchor_distances).mean()

            recovery_scale = schedule.recovery_weight(step)
            total = (
                config.lambda_supervised * supervised
                + config.lambda_kd * kd
                + config.lambda_feature * feature
                + config.lambda_recovery * recovery_scale * recovery
                + config.lambda_anchor * recovery_scale * anchor
                + router_reg
            )
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss in {arm.name} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
            optimizer.step()
            last = LossBreakdown(
                total=total, supervised=supervised, kd=kd, feature=feature,
                recovery=recovery, anchor=anchor,
                router_regularization=router_reg)

    student.eval()
    metrics = evaluate(student, teacher, eval_x, target_layer=config.target_layer)
    rr = recovery_ratio(full_quality, hard_quality, metrics.quality)
    hc = handoff_completeness(hard_divergence, metrics.mean_kl)
    before = _recipient_importance(
        teacher, eval_x, target_layer=config.target_layer, recipients=config.recipients)
    after = _recipient_importance(
        student, eval_x, target_layer=config.target_layer, recipients=config.recipients)
    gain = [a - b for a, b in zip(after, before)]
    mean_route = _mean_route(router, teacher, eval_x, config)
    alignment = None if mean_route is None else _spearman(mean_route, gain)
    match = None if mean_route is None else int(np.argmax(mean_route)) == int(np.argmax(gain))
    max_abs, physical_params = _physical_equivalence(student, eval_x[:32], config.target_layer)
    hard_params = parameter_count(_physical_delete(teacher, config.target_layer))
    if physical_params != hard_params:
        raise RuntimeError("final compact parameter count differs from ordinary hard pruning")

    return ArmResult(
        arm=arm.name,
        seed=seed,
        metrics=metrics,
        recovery_ratio=rr,
        handoff_completeness=hc,
        mean_route=mean_route,
        recipient_importance_before=before,
        recipient_importance_after=after,
        recipient_importance_gain=gain,
        route_alignment_spearman=alignment,
        top_route_top_gain_match=match,
        gate_zero_physical_max_abs=max_abs,
        final_parameter_count=physical_params,
        hard_parameter_count=hard_params,
        trainable_parameter_count=sum(p.numel() for p in trainable),
        last_losses=None if last is None else last.detached(),
    )


def run_reference(
    config: ToyConfig,
    seeds: Sequence[int],
    *,
    arms: Sequence[str] | None = None,
) -> list[ArmResult]:
    selected = list(arms or arm_registry().keys())
    results: list[ArmResult] = []
    for seed in seeds:
        teacher = make_teacher(config, seed)
        train_x, eval_x = make_data(config, seed + 7000)
        full_metrics = EvalMetrics(accuracy=1.0, mean_kl=0.0, quality=1.0)
        hard_metrics = evaluate(
            teacher, teacher, eval_x, target_layer=config.target_layer)
        for arm_name in selected:
            if arm_name == "full":
                max_abs, _ = _physical_equivalence(
                    teacher, eval_x[:32], config.target_layer)
                hard_params = parameter_count(_physical_delete(teacher, config.target_layer))
                results.append(ArmResult(
                    arm="full", seed=seed, metrics=full_metrics,
                    recovery_ratio=1.0, handoff_completeness=1.0,
                    mean_route=None, recipient_importance_before=None,
                    recipient_importance_after=None, recipient_importance_gain=None,
                    route_alignment_spearman=None, top_route_top_gain_match=None,
                    gate_zero_physical_max_abs=max_abs,
                    final_parameter_count=parameter_count(teacher),
                    hard_parameter_count=hard_params,
                    trainable_parameter_count=0, last_losses=None))
                continue
            if arm_name == "hard_prune":
                max_abs, physical_params = _physical_equivalence(
                    teacher, eval_x[:32], config.target_layer)
                results.append(ArmResult(
                    arm="hard_prune", seed=seed, metrics=hard_metrics,
                    recovery_ratio=0.0, handoff_completeness=0.0,
                    mean_route=None, recipient_importance_before=None,
                    recipient_importance_after=None, recipient_importance_gain=None,
                    route_alignment_spearman=None, top_route_top_gain_match=None,
                    gate_zero_physical_max_abs=max_abs,
                    final_parameter_count=physical_params,
                    hard_parameter_count=physical_params,
                    trainable_parameter_count=0, last_losses=None))
                continue
            results.append(run_arm(
                arm_name, teacher=teacher, train_x=train_x, eval_x=eval_x,
                config=config, seed=seed, full_quality=full_metrics.quality,
                hard_quality=hard_metrics.quality,
                hard_divergence=hard_metrics.mean_kl))
    return results


def config_from_mapping(payload: dict[str, Any]) -> tuple[ToyConfig, list[int]]:
    model = payload.get("model", {})
    data = payload.get("data", {})
    train = payload.get("train", {})
    routing = payload.get("routing", {})
    config = ToyConfig(
        input_dim=int(model.get("input_dim", 24)),
        hidden_dim=int(model.get("hidden_dim", 32)),
        ff_dim=int(model.get("ff_dim", 64)),
        layers=int(model.get("layers", 6)),
        sequence_length=int(model.get("sequence_length", 6)),
        output_classes=int(model.get("output_classes", 19)),
        target_layer=int(model.get("target_layer", 3)),
        recipients=tuple(int(x) for x in model.get("recipients", [2, 4, 5])),
        train_examples=int(data.get("train_examples", 768)),
        eval_examples=int(data.get("eval_examples", 512)),
        steps=int(train.get("steps", 160)),
        batch_size=int(train.get("batch_size", 48)),
        learning_rate=float(train.get("learning_rate", 0.003)),
        weight_decay=float(train.get("weight_decay", 0.0)),
        temperature=float(train.get("temperature", 2.0)),
        gradient_clip=float(train.get("gradient_clip", 1.0)),
        notice_fraction=float(train.get("notice_fraction", 0.20)),
        withdrawal_end_fraction=float(train.get("withdrawal_end_fraction", 0.70)),
        recovery_end_fraction=float(train.get("recovery_end_fraction", 0.78)),
        lambda_supervised=float(train.get("lambda_supervised", 0.35)),
        lambda_kd=float(train.get("lambda_kd", 0.65)),
        lambda_feature=float(train.get("lambda_feature", 0.20)),
        lambda_recovery=float(train.get("lambda_recovery", 0.80)),
        lambda_anchor=float(train.get("lambda_anchor", 0.10)),
        router_top_k=int(routing.get("top_k", 2)),
        router_temperature=float(routing.get("temperature", 0.8)),
        router_entropy_weight=float(routing.get("entropy_weight", 0.002)),
        router_load_balance_weight=float(routing.get("load_balance_weight", 0.01)),
    )
    seeds = [int(s) for s in payload.get("seeds", [17, 29, 43])]
    return config, seeds
