"""Low-rank transferable semantic transformation operators."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LowRankOperatorBank(nn.Module):
    """A shared dictionary of low-rank residual transformations.

    For state ``s`` and router weights ``alpha``::

        T(s) = normalize(s + sum_j alpha_j (s A_j) B_j)

    The bank describes semantic change; the Transformer remains responsible for
    the full nonlinear computation that produces each checkpoint state.
    """

    def __init__(
        self,
        state_dim: int,
        num_operators: int,
        rank: int,
        *,
        router_temperature: float = 1.0,
        frozen_random: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= rank <= state_dim:
            raise ValueError("rank must lie in [1, state_dim]")
        self.state_dim = state_dim
        self.num_operators = num_operators
        self.rank = rank
        self.router_temperature = router_temperature
        scale = 1.0 / math.sqrt(state_dim)
        self.left = nn.Parameter(torch.randn(num_operators, state_dim, rank) * scale)
        self.right = nn.Parameter(torch.zeros(num_operators, rank, state_dim))
        self.router = nn.Linear(state_dim, num_operators)
        nn.init.zeros_(self.router.bias)
        if frozen_random:
            nn.init.normal_(self.right, std=scale / math.sqrt(rank))
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.shape[-1] != self.state_dim:
            raise ValueError("operator input has the wrong state dimension")
        logits = self.router(states) / self.router_temperature
        weights = torch.softmax(logits, dim=-1)
        # [*, D] x [M, D, R] -> [*, M, R] -> [*, M, D]
        low = torch.einsum("...d,mdr->...mr", states, self.left)
        deltas = torch.einsum("...mr,mrd->...md", low, self.right)
        delta = torch.einsum("...m,...md->...d", weights, deltas)
        return F.normalize(states + delta, dim=-1), weights

    def diversity_loss(self) -> torch.Tensor:
        matrices = torch.einsum("mdr,mre->mde", self.left, self.right).flatten(1)
        matrices = F.normalize(matrices, dim=-1, eps=1e-8)
        gram = matrices @ matrices.T
        off_diagonal = gram - torch.eye(self.num_operators, device=gram.device, dtype=gram.dtype)
        return off_diagonal.square().sum() / max(1, self.num_operators * (self.num_operators - 1))

    @staticmethod
    def balance_loss(weights: torch.Tensor) -> torch.Tensor:
        if weights.numel() == 0:
            return weights.sum()
        mean_load = weights.reshape(-1, weights.shape[-1]).mean(dim=0)
        target = torch.full_like(mean_load, 1.0 / weights.shape[-1])
        return (mean_load - target).square().mean()

    @staticmethod
    def routing_entropy(weights: torch.Tensor) -> torch.Tensor:
        return -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(dim=-1).mean()
