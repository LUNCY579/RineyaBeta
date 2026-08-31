"""Losses and diagnostics for depth-aligned semantic supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from dast.config import DASTConfig
from dast.model import DASTOutput
from dast.operators import LowRankOperatorBank


@dataclass
class LossBreakdown:
    total: torch.Tensor
    lm: torch.Tensor
    state: torch.Tensor
    contrastive: torch.Tensor
    transition: torch.Tensor
    consistency: torch.Tensor
    operator_diversity: torch.Tensor
    operator_balance: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss/total": float(self.total.detach().cpu()),
            "loss/lm": float(self.lm.detach().cpu()),
            "loss/state": float(self.state.detach().cpu()),
            "loss/contrastive": float(self.contrastive.detach().cpu()),
            "loss/transition": float(self.transition.detach().cpu()),
            "loss/consistency": float(self.consistency.detach().cpu()),
            "loss/operator_diversity": float(self.operator_diversity.detach().cpu()),
            "loss/operator_balance": float(self.operator_balance.detach().cpu()),
        }


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def cosine_alignment_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    stage_weights: tuple[float, ...] = (),
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(f"state/target shapes differ: {tuple(predictions.shape)} vs {tuple(targets.shape)}")
    loss = 1.0 - F.cosine_similarity(predictions.float(), targets.float(), dim=-1)
    if stage_weights:
        weights = loss.new_tensor(stage_weights)
        weights = weights / weights.sum().clamp_min(1e-12)
        return (loss * weights.unsqueeze(0)).sum(dim=1).mean()
    return loss.mean()


def multi_positive_contrastive_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_ids: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """Symmetric stage-wise InfoNCE with same-target rows treated as positives."""

    if predictions.shape != targets.shape:
        raise ValueError("contrastive predictions and targets must have the same shape")
    batch, stages, _ = predictions.shape
    if batch < 2:
        return predictions.new_zeros(())
    total = predictions.new_zeros(())
    for stage in range(stages):
        pred = F.normalize(predictions[:, stage].float(), dim=-1)
        target = F.normalize(targets[:, stage].float(), dim=-1)
        logits = pred @ target.T / temperature
        if group_ids is None:
            positive = torch.eye(batch, device=logits.device, dtype=torch.bool)
        else:
            ids = group_ids[:, stage].to(logits.device)
            positive = ids[:, None].eq(ids[None, :])
        row_log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        col_log_prob = logits - torch.logsumexp(logits, dim=0, keepdim=True)
        row_count = positive.sum(dim=1).clamp_min(1)
        col_count = positive.sum(dim=0).clamp_min(1)
        row_loss = -((row_log_prob * positive).sum(dim=1) / row_count).mean()
        col_loss = -((col_log_prob * positive).sum(dim=0) / col_count).mean()
        total = total + 0.5 * (row_loss + col_loss)
    return total / stages


def compute_loss(
    output: DASTOutput,
    *,
    config: DASTConfig,
    training_targets: torch.Tensor | None,
    training_group_ids: torch.Tensor | None,
    operator_bank: LowRankOperatorBank | None,
) -> LossBreakdown:
    if output.lm_loss is None:
        raise ValueError("base model did not return a language-model loss")
    lm = output.lm_loss.float().to(output.states.device)
    zero = _zero_like(lm)
    state = zero
    contrastive = zero
    transition = zero
    consistency = zero
    diversity = zero
    balance = zero

    if config.arm_spec.target_kind != "none":
        if training_targets is None:
            raise ValueError("semantic arm requires training_targets")
        state = cosine_alignment_loss(
            output.states,
            training_targets,
            stage_weights=config.semantic.stage_weights,
        )
        contrastive = multi_positive_contrastive_loss(
            output.states,
            training_targets,
            group_ids=training_group_ids,
            temperature=config.semantic.contrastive_temperature,
        )

    if config.operators.enabled:
        if output.transition_predictions is None or output.router_weights is None or operator_bank is None:
            raise ValueError("operator arm did not return operator outputs")
        assert training_targets is not None
        transition = cosine_alignment_loss(output.transition_predictions, training_targets[:, 1:])
        # The next depth state is an independently supervised teacher.  Stop its
        # gradient here so consistency cannot pull both sides into a shortcut.
        consistency = cosine_alignment_loss(
            output.transition_predictions,
            output.states[:, 1:].detach(),
        )
        diversity = operator_bank.diversity_loss()
        balance = operator_bank.balance_loss(output.router_weights)

    total = (
        lm
        + config.semantic.state_weight * state
        + config.semantic.contrastive_weight * contrastive
        + config.operators.transition_weight * transition
        + config.operators.consistency_weight * consistency
        + config.operators.diversity_weight * diversity
        + config.operators.balance_weight * balance
    )
    return LossBreakdown(
        total=total,
        lm=lm,
        state=state,
        contrastive=contrastive,
        transition=transition,
        consistency=consistency,
        operator_diversity=diversity,
        operator_balance=balance,
    )


@torch.no_grad()
def depth_alignment_metrics(predictions: torch.Tensor, canonical_targets: torch.Tensor) -> dict[str, float]:
    """Measure whether canonical reasoning progress is aligned with depth."""

    if predictions.shape != canonical_targets.shape:
        raise ValueError("predictions and canonical targets must have the same shape")
    pred = F.normalize(predictions.float(), dim=-1)
    target = F.normalize(canonical_targets.float(), dim=-1)
    similarities = torch.einsum("bkd,bjd->bkj", pred, target)
    stages = similarities.shape[1]
    diagonal = similarities.diagonal(dim1=1, dim2=2)
    if stages > 1:
        identity = torch.eye(stages, dtype=torch.bool, device=similarities.device).unsqueeze(0)
        off_diagonal = similarities.masked_select(~identity).reshape(similarities.shape[0], -1)
        margin = diagonal.mean() - off_diagonal.mean()
    else:
        margin = diagonal.mean()
    top1 = similarities.argmax(dim=-1)
    expected = torch.arange(stages, device=similarities.device).unsqueeze(0)
    stage_accuracy = top1.eq(expected).float().mean()
    monotonic = (top1[:, 1:] >= top1[:, :-1]).all(dim=1).float().mean() if stages > 1 else stage_accuracy
    return {
        "state/canonical_cosine": float(diagonal.mean().cpu()),
        "state/final_cosine": float(diagonal[:, -1].mean().cpu()),
        "state/order_margin": float(margin.cpu()),
        "state/checkpoint_top1": float(stage_accuracy.cpu()),
        "state/decoded_progress_monotonic": float(monotonic.cpu()),
    }
