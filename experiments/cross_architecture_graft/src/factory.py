"""Construct graft branches from validated experiment configurations."""

from __future__ import annotations

from typing import Any

from torch import nn

from .bridges import build_bridge
from .config import ExperimentConfig
from .donors import build_donor
from .graft import GraftBranch, ResidualGate


def build_graft_branch(config: ExperimentConfig, *, host_dim: int | None = None) -> GraftBranch:
    raw = config.raw
    resolved_host_dim = int(host_dim or raw["host"]["hidden_size"])
    foreign_dim = int(raw["graft"]["foreign_dim"])
    donor = build_donor(config.donor_spec, foreign_dim)
    return GraftBranch(
        host_dim=resolved_host_dim,
        foreign_dim=foreign_dim,
        input_bridge=build_bridge(config.input_bridge_spec, resolved_host_dim, foreign_dim),
        donor=donor,
        output_bridge=build_bridge(config.output_bridge_spec, foreign_dim, resolved_host_dim),
        gate=ResidualGate(resolved_host_dim, config.gate_spec),
    )


def component_plan(config: ExperimentConfig) -> dict[str, Any]:
    branch = build_graft_branch(config)
    if bool(config.raw["graft"].get("freeze_donor", True)):
        for parameter in branch.donor.parameters():
            parameter.requires_grad_(False)

    def counts(module: nn.Module) -> dict[str, int]:
        return {
            "parameters": sum(parameter.numel() for parameter in module.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in module.parameters() if parameter.requires_grad
            ),
        }

    return {
        "host_hidden_size": branch.host_dim,
        "foreign_hidden_size": branch.foreign_dim,
        "donor": {"class": branch.donor.__class__.__name__, **counts(branch.donor)},
        "input_bridge": {
            "class": branch.input_bridge.__class__.__name__,
            **counts(branch.input_bridge),
        },
        "output_bridge": {
            "class": branch.output_bridge.__class__.__name__,
            **counts(branch.output_bridge),
        },
        "gate": {"class": branch.gate.__class__.__name__, **counts(branch.gate)},
        "branch_total": counts(branch),
        "initial_gate_exactly_zero": branch.gate.is_exactly_zero(),
    }
