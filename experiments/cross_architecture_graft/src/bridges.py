"""Representation bridges between a host residual stream and a graft block.

A bridge is intentionally explicit: cross-architecture work is not allowed to
hide a width resize inside an unrelated module.  Every bridge reports its shape
and parameter count in the evidence manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BridgeSpec:
    """Configuration for one directed residual-stream bridge."""

    kind: str = "linear"
    rank: int | None = None
    bias: bool = False
    identity_init: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BridgeSpec":
        raw = dict(value or {})
        allowed = {"kind", "rank", "bias", "identity_init"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown bridge fields: {unknown}")
        spec = cls(**raw)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.kind not in {"identity", "linear", "low_rank"}:
            raise ValueError(f"unsupported bridge kind: {self.kind!r}")
        if self.kind == "low_rank" and (self.rank is None or self.rank <= 0):
            raise ValueError("low_rank bridge requires rank > 0")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("bridge rank must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IdentityBridge(nn.Module):
    """A shape-checked identity mapping."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim != out_dim:
            raise ValueError(
                f"identity bridge requires equal dimensions, got {in_dim} -> {out_dim}"
            )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected last dimension {self.in_dim}, got {hidden_states.shape[-1]}"
            )
        return hidden_states


class LinearBridge(nn.Module):
    """A full linear basis map, identity-initialised when dimensions match."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters(identity_init=identity_init)

    def reset_parameters(self, *, identity_init: bool) -> None:
        if identity_init and self.in_dim == self.out_dim:
            with torch.no_grad():
                self.proj.weight.zero_()
                self.proj.weight.diagonal().fill_(1.0)
                if self.proj.bias is not None:
                    self.proj.bias.zero_()
        else:
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj(hidden_states)


class LowRankBridge(nn.Module):
    """A factorised bridge with an optional identity skip.

    When ``in_dim == out_dim`` and ``identity_init`` is true, the bridge begins
    as ``x + B(A(x))`` with B zero-initialised.  For unequal dimensions there is
    no canonical identity, so both factors use variance-preserving initialisation.
    The outer graft gate still guarantees exact host equivalence at step zero.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        rank: int,
        bias: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.use_identity = bool(identity_init and in_dim == out_dim)
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        if self.use_identity:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.xavier_uniform_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        projected = self.up(self.down(hidden_states))
        return hidden_states + projected if self.use_identity else projected


def build_bridge(
    spec: BridgeSpec,
    in_dim: int,
    out_dim: int,
) -> nn.Module:
    """Construct a bridge from a validated specification."""

    spec.validate()
    if spec.kind == "identity":
        return IdentityBridge(in_dim, out_dim)
    if spec.kind == "linear":
        return LinearBridge(
            in_dim,
            out_dim,
            bias=spec.bias,
            identity_init=spec.identity_init,
        )
    assert spec.rank is not None
    return LowRankBridge(
        in_dim,
        out_dim,
        rank=spec.rank,
        bias=spec.bias,
        identity_init=spec.identity_init,
    )


def bridge_signature(module: nn.Module) -> dict[str, Any]:
    """Return a JSON-safe description used by evidence manifests."""

    return {
        "class": module.__class__.__name__,
        "parameters": sum(p.numel() for p in module.parameters()),
        "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
    }
