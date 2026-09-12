"""Pure, auditable interventions on persistent KV rows.

The inherited HLGA sinks are ordinary prefix-token KV states retained by local
attention.  This module changes one causal factor at a time without adding
trainable parameters or changing the total local key budget.
"""
from dataclasses import asdict, dataclass
from typing import Literal

import torch

ContentMode = Literal["intact", "zero_k", "zero_v", "permute_v", "mean_v"]
Placement = Literal["prefix", "interior", "recent"]


@dataclass(frozen=True)
class SinkIntervention:
    count: int = 4
    placement: Placement = "prefix"
    content: ContentMode = "intact"
    visible: bool = True
    permutation_seed: int = 17

    def __post_init__(self):
        if type(self.count) is not int or self.count < 0:
            raise ValueError("count must be a nonnegative integer")
        if self.placement not in ("prefix", "interior", "recent"):
            raise ValueError("unknown placement")
        if self.content not in ("intact", "zero_k", "zero_v", "permute_v", "mean_v"):
            raise ValueError("unknown content intervention")
        if type(self.visible) is not bool:
            raise ValueError("visible must be boolean")

    def to_dict(self):
        return asdict(self)


def select_persistent_positions(total_tokens: int, count: int, placement: Placement):
    if type(total_tokens) is not int or total_tokens < 1:
        raise ValueError("total_tokens must be positive")
    if type(count) is not int or not 0 <= count < total_tokens:
        raise ValueError("require 0 <= count < total_tokens")
    if count == 0:
        return tuple()
    if placement == "prefix":
        return tuple(range(count))
    if placement == "recent":
        return tuple(range(total_tokens - count, total_tokens))
    if placement == "interior":
        start = (total_tokens - count) // 2
        return tuple(range(start, start + count))
    raise ValueError("unknown placement")


def persistent_mask(query_positions, key_positions, *, window, persistent_positions=()):
    """Causal local mask whose total per-query capacity never exceeds window."""
    if window < 1 or len(persistent_positions) >= window:
        raise ValueError("persistent rows must fit strictly inside window")
    causal = key_positions[None, :] <= query_positions[:, None]
    persistent = torch.zeros_like(key_positions, dtype=torch.bool)
    for position in persistent_positions:
        persistent |= key_positions == position
    recent_budget = window - len(persistent_positions)
    recent = key_positions[None, :] >= query_positions[:, None] - recent_budget + 1
    return causal & (persistent[None, :] | recent)


def apply_sink_intervention(k, v, key_positions, persistent_positions, intervention):
    """Return cloned K/V with a registered intervention applied only to sinks."""
    if k.ndim != 4 or v.shape != k.shape or key_positions.ndim != 1:
        raise ValueError("expected K/V [B,H,T,D] and positions [T]")
    if k.shape[2] != len(key_positions):
        raise ValueError("position count does not match K/V")
    out_k, out_v = k.clone(), v.clone()
    idx = torch.tensor(
        [i for i, p in enumerate(key_positions.tolist()) if p in set(persistent_positions)],
        device=k.device, dtype=torch.long,
    )
    if len(idx) != intervention.count:
        raise ValueError("registered sink count does not match selected positions")
    if not intervention.visible or intervention.content == "intact" or len(idx) == 0:
        return out_k, out_v
    if intervention.content == "zero_k":
        out_k.index_fill_(2, idx, 0)
    elif intervention.content == "zero_v":
        out_v.index_fill_(2, idx, 0)
    elif intervention.content == "mean_v":
        mean = out_v.index_select(2, idx).mean(dim=2, keepdim=True)
        out_v.index_copy_(2, idx, mean.expand(-1, -1, len(idx), -1))
    elif intervention.content == "permute_v":
        generator = torch.Generator(device="cpu").manual_seed(intervention.permutation_seed)
        order = torch.randperm(len(idx), generator=generator).to(idx.device)
        out_v.index_copy_(2, idx, out_v.index_select(2, idx).index_select(2, order))
    return out_k, out_v


def attention_diagnostics(probabilities, values, sink_columns):
    """Per-head sink mass, entropy and sink-value contribution norm."""
    if probabilities.ndim != 4 or values.ndim != 4:
        raise ValueError("expected [B,H,Q,K] probabilities and [B,H,K,D] values")
    idx = torch.as_tensor(sink_columns, device=probabilities.device, dtype=torch.long)
    eps = torch.finfo(probabilities.dtype).tiny
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(-1)
    if len(idx) == 0:
        shape = probabilities.shape[:-1]
        return {"sink_mass": torch.zeros(shape, device=probabilities.device),
                "attention_entropy": entropy,
                "sink_value_norm": torch.zeros(shape, device=probabilities.device)}
    sink_p = probabilities.index_select(-1, idx)
    sink_v = values.index_select(2, idx)
    contribution = torch.einsum("bhqs,bhsd->bhqd", sink_p, sink_v)
    return {"sink_mass": sink_p.sum(-1), "attention_entropy": entropy,
            "sink_value_norm": contribution.float().norm(dim=-1)}
