#!/usr/bin/env python3
"""Causal recovery-utility definitions for SGQR significance routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def teacher_kl(
    teacher_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute KL(teacher || candidate) over the vocabulary dimension."""
    if teacher_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"logit shapes must match, got {tuple(teacher_logits.shape)} and {tuple(candidate_logits.shape)}"
        )
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    candidate_logp = F.log_softmax(candidate_logits.float(), dim=-1)
    teacher_p = teacher_logp.exp()
    values = (teacher_p * (teacher_logp - candidate_logp)).sum(dim=-1)
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"unsupported reduction {reduction!r}")


def marginal_recovery_utility(
    teacher_logits: torch.Tensor,
    deleted_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Return KL(T||deleted) - KL(T||candidate).

    Positive values mean the candidate remnant moved the model closer to the
    dense teacher. Negative values mean it was harmful relative to deletion.
    """
    return teacher_kl(teacher_logits, deleted_logits) - teacher_kl(
        teacher_logits, candidate_logits
    )


def recovery_ratio(
    method_quality: float,
    deleted_quality: float,
    dense_quality: float,
    *,
    minimum_denominator: float = 1e-12,
) -> float:
    denominator = dense_quality - deleted_quality
    if abs(denominator) < minimum_denominator:
        raise ZeroDivisionError(
            "dense and deleted qualities are too close for a stable Recovery Ratio"
        )
    return (method_quality - deleted_quality) / denominator


@dataclass
class SignificanceSummary:
    utility_sum: torch.Tensor
    observation_count: torch.Tensor

    def scores(self, exposure_exponent: float = 1.0) -> torch.Tensor:
        if exposure_exponent < 0:
            raise ValueError("exposure_exponent must be non-negative")
        denominator = self.observation_count.clamp_min(1.0).pow(exposure_exponent)
        return self.utility_sum / denominator

    def as_dict(self, exposure_exponent: float = 1.0) -> dict[str, list[float]]:
        return {
            "utility_sum": self.utility_sum.tolist(),
            "observation_count": self.observation_count.tolist(),
            "scores": self.scores(exposure_exponent).tolist(),
        }


class SignificanceAccumulator:
    """Accumulate group utility and exposure separately."""

    def __init__(self, group_count: int) -> None:
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        self.group_count = int(group_count)
        self.utility_sum = torch.zeros(group_count, dtype=torch.float64)
        self.observation_count = torch.zeros(group_count, dtype=torch.float64)

    def update(
        self,
        utility: torch.Tensor,
        observed: torch.Tensor | None = None,
    ) -> None:
        values = utility.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[-1] != self.group_count:
            raise ValueError(
                f"expected utility last dimension {self.group_count}, got {values.shape[-1]}"
            )
        flat = values.reshape(-1, self.group_count)

        if observed is None:
            mask = torch.isfinite(flat)
        else:
            mask = observed.detach().to(device="cpu", dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            mask = mask.reshape(-1, self.group_count)
            if mask.shape != flat.shape:
                raise ValueError("observed mask shape does not match utility")
            mask = mask & torch.isfinite(flat)

        safe = torch.where(mask, flat, torch.zeros_like(flat))
        self.utility_sum += safe.sum(dim=0)
        self.observation_count += mask.sum(dim=0).to(torch.float64)

    def summary(self) -> SignificanceSummary:
        return SignificanceSummary(
            utility_sum=self.utility_sum.clone(),
            observation_count=self.observation_count.clone(),
        )


def utility_capture(
    utilities: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    """Fraction of positive oracle utility captured by selected group ids."""
    if utilities.ndim < 2:
        raise ValueError("utilities must have a group dimension")
    if selected.shape[:-1] != utilities.shape[:-1]:
        raise ValueError("selected prefix must match utilities prefix")
    if selected.shape[-1] == 0:
        return torch.zeros(utilities.shape[:-1], dtype=utilities.dtype, device=utilities.device)
    gathered = utilities.gather(-1, selected)
    captured = gathered.clamp_min(0).sum(dim=-1)
    available = utilities.clamp_min(0).sum(dim=-1)
    return torch.where(available > 0, captured / available, torch.ones_like(available))
