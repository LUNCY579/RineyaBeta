"""Matched-capacity controls and accounting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from .donors import SwiGLUBranch
from .graft import GraftBranch, GraftedDecoderLayer, ResidualGate, GateSpec, parameter_counts
from .bridges import IdentityBridge


@dataclass(frozen=True)
class MatchResult:
    intermediate_size: int
    candidate_trainable: int
    control_trainable: int
    relative_gap: float


def nearest_swiglu_width(d_model: int, target_parameters: int) -> int:
    """Find the positive SwiGLU width whose parameter count is closest to target.

    A bias-free SwiGLU branch has exactly ``3 * d_model * width`` parameters.
    The gate is accounted for by the caller because scalar/channel/token gates
    have different sizes.
    """

    if d_model <= 0 or target_parameters <= 0:
        raise ValueError("d_model and target_parameters must be positive")
    estimate = max(1, round(target_parameters / (3 * d_model)))
    candidates = range(max(1, estimate - 3), estimate + 4)
    return min(candidates, key=lambda width: abs(3 * d_model * width - target_parameters))


def build_parameter_matched_mlp_control(
    *,
    host_layer: nn.Module,
    d_model: int,
    target_trainable_parameters: int,
    gate_spec: GateSpec,
    exact_eval_bypass: bool = True,
) -> tuple[GraftedDecoderLayer, MatchResult]:
    gate_probe = ResidualGate(d_model, gate_spec)
    gate_parameters = sum(parameter.numel() for parameter in gate_probe.parameters())
    branch_target = max(1, target_trainable_parameters - gate_parameters)
    width = nearest_swiglu_width(d_model, branch_target)
    donor = SwiGLUBranch(d_model, width)
    branch = GraftBranch(
        host_dim=d_model,
        foreign_dim=d_model,
        input_bridge=IdentityBridge(d_model, d_model),
        donor=donor,
        output_bridge=IdentityBridge(d_model, d_model),
        gate=gate_probe,
    )
    control = GraftedDecoderLayer(
        host_layer,
        branch,
        exact_eval_bypass=exact_eval_bypass,
    )
    for parameter in control.host_layer.parameters():
        parameter.requires_grad_(False)
    for parameter in branch.donor.parameters():
        parameter.requires_grad_(True)
    control_counts = parameter_counts(control)
    gap = abs(control_counts["trainable"] - target_trainable_parameters) / max(
        target_trainable_parameters, 1
    )
    return control, MatchResult(
        intermediate_size=width,
        candidate_trainable=target_trainable_parameters,
        control_trainable=control_counts["trainable"],
        relative_gap=gap,
    )


def arm_record(name: str, module: nn.Module, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, **parameter_counts(module)}
    record.update(extra)
    return record
