"""Function-preserving heterogeneous residual grafts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GateSpec:
    kind: str = "scalar"
    bound: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GateSpec":
        raw = dict(value or {})
        allowed = {"kind", "bound"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown gate fields: {unknown}")
        spec = cls(**raw)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.kind not in {"scalar", "channel", "token"}:
            raise ValueError(f"unsupported gate kind: {self.kind!r}")
        if self.bound <= 0:
            raise ValueError("gate bound must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualGate(nn.Module):
    """A bounded, exactly-zero-initialised gate.

    ``tanh`` permits positive or negative corrections.  The exact zero matters:
    at construction the host's function is preserved, rather than merely close
    under a tiny sigmoid value.
    """

    def __init__(self, d_model: int, spec: GateSpec) -> None:
        super().__init__()
        spec.validate()
        self.d_model = int(d_model)
        self.kind = spec.kind
        self.bound = float(spec.bound)
        if spec.kind == "scalar":
            self.raw = nn.Parameter(torch.zeros(1))
            self.token_gate = None
        elif spec.kind == "channel":
            self.raw = nn.Parameter(torch.zeros(d_model))
            self.token_gate = None
        else:
            self.register_parameter("raw", None)
            self.token_gate = nn.Linear(d_model, 1, bias=True)
            nn.init.zeros_(self.token_gate.weight)
            nn.init.zeros_(self.token_gate.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.kind == "scalar":
            assert self.raw is not None
            return self.bound * torch.tanh(self.raw).view(1, 1, 1)
        if self.kind == "channel":
            assert self.raw is not None
            return self.bound * torch.tanh(self.raw).view(1, 1, self.d_model)
        assert self.token_gate is not None
        return self.bound * torch.tanh(self.token_gate(hidden_states))

    def is_exactly_zero(self) -> bool:
        with torch.no_grad():
            if self.kind in {"scalar", "channel"}:
                assert self.raw is not None
                return bool(torch.count_nonzero(self.raw).item() == 0)
            assert self.token_gate is not None
            return all(torch.count_nonzero(p).item() == 0 for p in self.token_gate.parameters())

    def reset_zero(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()


class GraftBranch(nn.Module):
    """Map host states into a foreign block and return a gated host-space delta."""

    def __init__(
        self,
        *,
        host_dim: int,
        foreign_dim: int,
        input_bridge: nn.Module,
        donor: nn.Module,
        output_bridge: nn.Module,
        gate: ResidualGate,
    ) -> None:
        super().__init__()
        self.host_dim = int(host_dim)
        self.foreign_dim = int(foreign_dim)
        self.input_bridge = input_bridge
        self.donor = donor
        self.output_bridge = output_bridge
        self.gate = gate
        self.last_stats: dict[str, float] = {}

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        if hidden_states.shape[-1] != self.host_dim:
            raise ValueError(
                f"host hidden size mismatch: expected {self.host_dim}, "
                f"got {hidden_states.shape[-1]}"
            )
        foreign_input = self.input_bridge(hidden_states)
        if foreign_input.shape[-1] != self.foreign_dim:
            raise ValueError(
                f"input bridge produced {foreign_input.shape[-1]}, expected {self.foreign_dim}"
            )
        foreign_delta = self.donor(foreign_input, **kwargs)
        if not isinstance(foreign_delta, Tensor) or foreign_delta.shape != foreign_input.shape:
            raise ValueError(
                "donor must return a tensor with the same shape as its foreign-space input"
            )
        host_delta = self.output_bridge(foreign_delta)
        if host_delta.shape != hidden_states.shape:
            raise ValueError(
                f"output bridge produced {tuple(host_delta.shape)}, expected "
                f"{tuple(hidden_states.shape)}"
            )
        gate = self.gate(hidden_states)
        # Sanitize a malformed foreign branch before gating.  Multiplication by
        # an exact-zero gate then preserves the host while retaining a useful
        # gradient with respect to the gate on the first optimisation step.
        safe_delta = torch.nan_to_num(host_delta, nan=0.0, posinf=0.0, neginf=0.0)
        contribution = gate * safe_delta
        with torch.no_grad():
            self.last_stats = {
                "gate_mean_abs": float(gate.detach().abs().mean().cpu()),
                "gate_max_abs": float(gate.detach().abs().max().cpu()),
                "delta_rms": float(host_delta.detach().float().pow(2).mean().sqrt().cpu()),
                "contribution_rms": float(
                    contribution.detach().float().pow(2).mean().sqrt().cpu()
                ),
            }
        return contribution

    def is_exact_bypass(self) -> bool:
        return self.gate.is_exactly_zero()


class GraftedDecoderLayer(nn.Module):
    """Wrap one decoder layer without changing its external return contract."""

    def __init__(
        self,
        host_layer: nn.Module,
        graft_branch: GraftBranch,
        *,
        exact_eval_bypass: bool = True,
    ) -> None:
        super().__init__()
        self.host_layer = host_layer
        self.graft_branch = graft_branch
        self.exact_eval_bypass = bool(exact_eval_bypass)

    @staticmethod
    def _hidden_from_output(output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            return output[0]
        if isinstance(output, Mapping):
            for key in ("hidden_states", "last_hidden_state"):
                value = output.get(key)
                if isinstance(value, Tensor):
                    return value
        raise TypeError(
            "host layer must return a tensor, a tuple/list beginning with one, "
            "or a mapping containing hidden_states"
        )

    @staticmethod
    def _replace_hidden(output: Any, hidden_states: Tensor) -> Any:
        if isinstance(output, Tensor):
            return hidden_states
        if isinstance(output, tuple):
            return (hidden_states, *output[1:])
        if isinstance(output, list):
            return [hidden_states, *output[1:]]
        if isinstance(output, Mapping):
            updated = dict(output)
            key = "hidden_states" if "hidden_states" in updated else "last_hidden_state"
            updated[key] = hidden_states
            try:
                return output.__class__(**updated)
            except TypeError:
                return updated
        raise TypeError("unsupported host output type")

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        host_output = self.host_layer(hidden_states, *args, **kwargs)
        if (
            self.exact_eval_bypass
            and not self.training
            and self.graft_branch.is_exact_bypass()
        ):
            return host_output
        host_hidden = self._hidden_from_output(host_output)
        graft_delta = self.graft_branch(hidden_states, **kwargs)
        return self._replace_hidden(host_output, host_hidden + graft_delta)


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def configure_graft_training(
    layer: GraftedDecoderLayer,
    *,
    freeze_host: bool = True,
    freeze_donor: bool = True,
) -> None:
    """Apply the v1 trainability boundary to one grafted layer."""

    if freeze_host:
        freeze_module(layer.host_layer)
    if freeze_donor:
        freeze_module(layer.graft_branch.donor)
    for parameter in layer.graft_branch.input_bridge.parameters():
        parameter.requires_grad_(True)
    for parameter in layer.graft_branch.output_bridge.parameters():
        parameter.requires_grad_(True)
    for parameter in layer.graft_branch.gate.parameters():
        parameter.requires_grad_(True)


def parameter_counts(module: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
    }
