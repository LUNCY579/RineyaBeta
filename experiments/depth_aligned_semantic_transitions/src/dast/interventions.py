"""Causal semantic-state interventions used only by the feedback-qualified phase."""

from __future__ import annotations

import torch


def zero_state(states: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(states)


def negate_state(states: torch.Tensor) -> torch.Tensor:
    return -states


def swap_batch_states(permutation: torch.Tensor):
    """Return an intervention that swaps semantic states across batch examples."""

    def intervention(states: torch.Tensor) -> torch.Tensor:
        order = permutation.to(device=states.device, dtype=torch.long)
        if order.ndim != 1 or order.numel() != states.shape[0]:
            raise ValueError("batch-state permutation has the wrong shape")
        return states.index_select(0, order)

    return intervention


def replace_with(replacement: torch.Tensor):
    def intervention(states: torch.Tensor) -> torch.Tensor:
        value = replacement.to(device=states.device, dtype=states.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0).expand_as(states)
        return value

    return intervention
