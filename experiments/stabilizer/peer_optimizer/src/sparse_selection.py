#!/usr/bin/env python3
"""Exact-budget selection, straight-through masks, and support auditing."""

from __future__ import annotations

import torch
from torch import Tensor

try:  # package import
    from .peer_features import SupportAudit
except ImportError:  # direct script/module import
    from peer_features import SupportAudit


def resolve_budget(budget: int | float, numel: int) -> int:
    """Convert an integer count or a (0, 1] fraction into an exact support size."""

    if isinstance(budget, bool):
        raise TypeError("budget must be an integer count or floating-point fraction")
    if isinstance(budget, int):
        k = budget
    elif isinstance(budget, float):
        if not (0.0 < budget <= 1.0):
            raise ValueError(f"fractional budget must be in (0, 1], got {budget}")
        k = max(1, int(round(budget * numel)))
    else:
        raise TypeError(f"unsupported budget type: {type(budget)!r}")
    if not (1 <= k <= numel):
        raise ValueError(f"resolved budget {k} must be between 1 and {numel}")
    return k


def exact_topk_mask(scores: Tensor, budget: int | float) -> Tensor:
    """Return a binary mask containing exactly K ones."""

    k = resolve_budget(budget, scores.numel())
    flat = scores.reshape(-1)
    # ``topk`` is substantially cheaper than sorting the full tensor.  Exact ties
    # have no unique mathematical support; deterministic seeds and fp32 scoring
    # keep experiments reproducible on a fixed backend.
    indices = torch.topk(flat, k=k, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(0, indices, 1.0)
    return mask.reshape_as(scores)


class _ExactForwardStraightThrough(torch.autograd.Function):
    """Return an exact hard tensor while differentiating through ``soft``."""

    @staticmethod
    def forward(ctx, soft: Tensor, hard: Tensor) -> Tensor:  # type: ignore[override]
        return hard.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return grad_output, None


def straight_through_topk(
    scores: Tensor,
    budget: int | float,
    temperature: float = 0.25,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Exact hard top-k forward pass with a smooth surrogate backward pass."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    hard = exact_topk_mask(scores, budget)
    k = int(hard.sum().item())
    threshold = torch.topk(scores.detach().reshape(-1), k=k).values[-1]
    soft = torch.sigmoid((scores - threshold) / temperature)
    soft = soft * (float(k) / soft.sum().clamp_min(eps))
    soft = soft.clamp(0.0, 1.0)
    training = _ExactForwardStraightThrough.apply(soft, hard)
    return hard, training


def audit_sparse_delta(
    initial: Tensor,
    current: Tensor,
    declared_mask: Tensor,
    *,
    atol: float = 0.0,
) -> SupportAudit:
    """Verify that native weights outside a declared support never moved."""

    if initial.shape != current.shape or initial.shape != declared_mask.shape:
        raise ValueError("initial, current, and declared_mask must have equal shapes")
    delta = (current.detach() - initial.detach()).abs()
    changed = delta > atol
    inside = declared_mask.detach().bool()
    outside_changed = changed & ~inside
    max_outside = delta.masked_select(~inside).max().item() if bool((~inside).any()) else 0.0
    return SupportAudit(
        selected=int(inside.sum().item()),
        changed_total=int(changed.sum().item()),
        changed_inside=int((changed & inside).sum().item()),
        changed_outside=int(outside_changed.sum().item()),
        max_abs_outside_delta=float(max_outside),
        support_union_fraction=float(changed.float().mean().item()),
    )
