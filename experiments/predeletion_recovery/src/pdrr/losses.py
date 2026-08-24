"""PDRR losses and interventional significance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def token_kl(teacher_logits: Tensor, student_logits: Tensor,
             temperature: float = 1.0) -> Tensor:
    """KL(teacher || student) at every non-vocabulary position."""
    t = max(float(temperature), 1e-6)
    teacher = torch.softmax(teacher_logits.float() / t, dim=-1)
    student_log = torch.log_softmax(student_logits.float() / t, dim=-1)
    return F.kl_div(student_log, teacher, reduction="none").sum(dim=-1) * (t * t)


def causal_significance(intact_logits: Tensor, ablated_logits: Tensor,
                        *, floor: float = 1e-6, normalize: bool = True) -> Tensor:
    significance = token_kl(intact_logits.detach(), ablated_logits.detach(), 1.0)
    significance = significance.clamp_min(float(floor))
    if normalize:
        axes = tuple(range(1, significance.ndim))
        scale = significance.mean(dim=axes, keepdim=True).clamp_min(float(floor))
        significance = significance / scale
    return significance


def normalized_residual_distance(target: Tensor, prediction: Tensor) -> Tensor:
    target_f = target.float()
    pred_f = prediction.float()
    target_norm = target_f.norm(dim=-1).clamp_min(1e-6)
    magnitude = ((pred_f - target_f).norm(dim=-1) / target_norm) ** 2
    cosine = 1.0 - F.cosine_similarity(pred_f, target_f, dim=-1, eps=1e-6)
    return 0.5 * magnitude + 0.5 * cosine


def component_recovery_loss(
    source_residual: Tensor,
    recipient_deltas: Sequence[Tensor],
    route_weights: Tensor,
    significance: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if not recipient_deltas:
        raise ValueError("recipient_deltas may not be empty")
    if route_weights.shape[-1] != len(recipient_deltas):
        raise ValueError("router recipient axis does not match recipient deltas")
    stacked = torch.stack([delta.to(source_residual.dtype) for delta in recipient_deltas], dim=-2)
    reconstruction = (stacked * route_weights.unsqueeze(-1)).sum(dim=-2)
    per_token = normalized_residual_distance(source_residual, reconstruction)
    weights = torch.ones_like(per_token) if significance is None else significance.to(per_token)
    loss = (per_token * weights).sum() / weights.sum().clamp_min(1e-6)
    return loss, reconstruction


def feature_distillation_loss(teacher_hidden: Tensor, student_hidden: Tensor) -> Tensor:
    teacher = F.layer_norm(teacher_hidden.float(), teacher_hidden.shape[-1:])
    student = F.layer_norm(student_hidden.float(), student_hidden.shape[-1:])
    return F.mse_loss(student, teacher)


def anchor_preservation_loss(
    current: Mapping[int, Tensor],
    anchor: Mapping[int, Tensor],
    *,
    recipient_layers: Sequence[int],
) -> Tensor:
    terms = []
    for layer in recipient_layers:
        if layer in current and layer in anchor:
            terms.append(normalized_residual_distance(anchor[layer].detach(), current[layer]).mean())
    if not terms:
        any_tensor = next(iter(current.values()))
        return any_tensor.new_zeros(())
    return torch.stack(terms).mean()


@dataclass
class LossBreakdown:
    total: Tensor
    supervised: Tensor
    kd: Tensor
    feature: Tensor
    recovery: Tensor
    anchor: Tensor
    router_regularization: Tensor

    def detached(self) -> dict[str, float]:
        return {
            key: float(value.detach().cpu())
            for key, value in self.__dict__.items()
        }
