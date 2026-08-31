"""Leakage-resistant pooling of hidden states over prompt tokens only."""

from __future__ import annotations

import torch


def pool_prompt_hidden(hidden: torch.Tensor, prompt_mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape [batch, sequence, hidden], got {tuple(hidden.shape)}")
    if prompt_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"prompt_mask shape {tuple(prompt_mask.shape)} does not match hidden prefix {tuple(hidden.shape[:2])}"
        )
    mask = prompt_mask.to(dtype=torch.bool, device=hidden.device)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("every example must retain at least one prompt token")

    if mode == "mean":
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / counts.to(hidden.dtype).unsqueeze(-1)
    if mode == "last_prompt":
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand_as(mask)
        last = positions.masked_fill(~mask, -1).max(dim=1).values
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, last]
    raise ValueError(f"unsupported prompt pooling mode: {mode}")
