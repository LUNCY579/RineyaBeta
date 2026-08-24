"""Temporary responsibility routers.

Routers never modify the model forward path. They only allocate the recovery loss
among surviving recipients and are deleted before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


def _sparsify(weights: Tensor, top_k: int | None) -> Tensor:
    if top_k is None or top_k >= weights.shape[-1]:
        return weights
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    values, indices = torch.topk(weights, k=top_k, dim=-1)
    sparse = torch.zeros_like(weights).scatter(-1, indices, values)
    return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class ResponsibilityRouter(nn.Module):
    mode: str

    def forward(self, state: Tensor, positions: Tensor | None = None) -> Tensor:
        raise NotImplementedError


class UniformRouter(ResponsibilityRouter):
    mode = "uniform"

    def __init__(self, recipients: int, top_k: int | None = None) -> None:
        super().__init__()
        self.recipients = recipients
        self.top_k = top_k

    def forward(self, state: Tensor, positions: Tensor | None = None) -> Tensor:
        shape = (*state.shape[:-1], self.recipients)
        weights = state.new_full(shape, 1.0 / self.recipients)
        return _sparsify(weights, self.top_k)


class FixedRouter(ResponsibilityRouter):
    def __init__(self, weights: Sequence[float], *, mode: str,
                 top_k: int | None = None) -> None:
        super().__init__()
        tensor = torch.as_tensor(weights, dtype=torch.float32)
        if tensor.ndim != 1 or tensor.numel() == 0 or torch.any(tensor < 0):
            raise ValueError("fixed routing weights must be a non-empty non-negative vector")
        if float(tensor.sum()) <= 0:
            raise ValueError("fixed routing weights must have positive mass")
        self.register_buffer("weights", tensor / tensor.sum())
        self.mode = mode
        self.top_k = top_k

    def forward(self, state: Tensor, positions: Tensor | None = None) -> Tensor:
        weights = self.weights.to(device=state.device, dtype=state.dtype)
        weights = weights.view(*([1] * (state.ndim - 1)), -1).expand(
            *state.shape[:-1], weights.numel())
        return _sparsify(weights, self.top_k)


class LearnedRouter(ResponsibilityRouter):
    def __init__(
        self,
        state_dim: int,
        recipients: int,
        *,
        mode: str,
        hidden_dim: int = 64,
        max_positions: int = 4096,
        position_dim: int = 16,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"state", "state_position"}:
            raise ValueError(f"unsupported learned routing mode {mode!r}")
        self.mode = mode
        self.temperature = float(temperature)
        self.top_k = top_k
        self.position = None
        input_dim = state_dim
        if mode == "state_position":
            self.position = nn.Embedding(max_positions, position_dim)
            input_dim += position_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, recipients),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state: Tensor, positions: Tensor | None = None) -> Tensor:
        features = state
        if self.position is not None:
            if positions is None:
                positions = torch.arange(state.shape[-2], device=state.device)
            pos = self.position(positions.long().clamp_max(self.position.num_embeddings - 1))
            while pos.ndim < state.ndim:
                pos = pos.unsqueeze(0)
            pos = pos.expand(*state.shape[:-1], pos.shape[-1])
            features = torch.cat((state, pos.to(dtype=state.dtype)), dim=-1)
        logits = self.network(features.float()) / max(self.temperature, 1e-6)
        weights = torch.softmax(logits, dim=-1).to(dtype=state.dtype)
        return _sparsify(weights, self.top_k)


@dataclass(frozen=True)
class RouterRegularization:
    entropy: Tensor
    load_balance: Tensor

    @property
    def total(self) -> Tensor:
        return self.entropy + self.load_balance


def router_regularization(
    weights: Tensor,
    *,
    entropy_weight: float = 0.0,
    load_balance_weight: float = 0.0,
) -> RouterRegularization:
    probs = weights.float().clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1).mean() * float(entropy_weight)
    axes = tuple(range(probs.ndim - 1))
    mean_load = probs.mean(dim=axes)
    uniform = torch.full_like(mean_load, 1.0 / mean_load.numel())
    load_balance = torch.mean((mean_load - uniform) ** 2) * float(load_balance_weight)
    return RouterRegularization(entropy=entropy, load_balance=load_balance)


def build_router(
    mode: str,
    *,
    state_dim: int,
    recipient_layers: Sequence[int],
    source_layer: int,
    seed: int,
    hidden_dim: int = 64,
    temperature: float = 1.0,
    top_k: int | None = None,
    max_positions: int = 4096,
) -> ResponsibilityRouter | None:
    n = len(recipient_layers)
    if n == 0:
        raise ValueError("at least one recipient is required")
    if mode == "none":
        return None
    if mode == "uniform":
        return UniformRouter(n, top_k=top_k)
    if mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        raw = torch.rand(n, generator=generator).tolist()
        return FixedRouter(raw, mode="random", top_k=top_k)
    if mode == "static":
        raw = [1.0 / (1.0 + abs(int(layer) - int(source_layer)))
               for layer in recipient_layers]
        return FixedRouter(raw, mode="static", top_k=top_k)
    if mode in {"state", "state_position"}:
        return LearnedRouter(
            state_dim, n, mode=mode, hidden_dim=hidden_dim,
            max_positions=max_positions, temperature=temperature, top_k=top_k)
    raise ValueError(f"unknown routing mode {mode!r}")
