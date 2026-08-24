"""Frozen experiment-arm definitions.

The registry is data, not control flow scattered through the trainer. Tests assert
that the two decisive ablations remain present and semantically distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RoutingMode = Literal["none", "uniform", "random", "static", "state", "state_position"]
DeletionMode = Literal["none", "hard", "gradual"]


@dataclass(frozen=True)
class ArmSpec:
    name: str
    description: str
    deletion: DeletionMode
    train: bool
    supervised: bool
    kd: bool
    feature_kd: bool
    recovery_loss: bool
    routing: RoutingMode
    causal_significance: bool
    router_trainable: bool


_SPECS = (
    ArmSpec("full", "Intact quality anchor", "none", False, False, False, False,
            False, "none", False, False),
    ArmSpec("hard_prune", "Immediate deletion without recovery", "hard", False,
            False, False, False, False, "none", False, False),
    ArmSpec("hard_prune_sft", "Immediate deletion followed by supervised recovery",
            "hard", True, True, False, False, False, "none", False, False),
    ArmSpec("hard_prune_kd", "Immediate deletion followed by output KD", "hard", True,
            True, True, False, False, "none", False, False),
    ArmSpec("gradual_kd", "Graceful withdrawal plus output KD", "gradual", True,
            True, True, False, False, "none", False, False),
    ArmSpec("gradual_feature_kd", "Graceful withdrawal plus output and feature KD",
            "gradual", True, True, True, True, False, "none", False, False),
    ArmSpec("matrix_no_recovery",
            "Learned Recovery Matrix plus KD, but no component-specific recovery loss",
            "gradual", True, True, True, False, False, "state_position", True, True),
    ArmSpec("random_matrix_recovery",
            "Component-specific recovery loss with a fixed random Recovery Matrix",
            "gradual", True, True, True, False, True, "random", True, False),
    ArmSpec("uniform_matrix_recovery",
            "Component-specific recovery loss with uniform responsibility",
            "gradual", True, True, True, False, True, "uniform", True, False),
    ArmSpec("static_matrix_recovery",
            "Component-specific recovery loss with fixed distance-based routing",
            "gradual", True, True, True, False, True, "static", True, False),
    ArmSpec("state_matrix_recovery",
            "Learned state-dependent, position-independent responsibility routing",
            "gradual", True, True, True, False, True, "state", True, True),
    ArmSpec("state_position_no_causal",
            "Learned state-position routing without interventional weighting",
            "gradual", True, True, True, False, True, "state_position", False, True),
    ArmSpec("causal_state_position_recovery",
            "Full state-position responsibility transfer with causal significance",
            "gradual", True, True, True, True, True, "state_position", True, True),
)


def arm_registry() -> dict[str, ArmSpec]:
    return {spec.name: spec for spec in _SPECS}


def get_arm(name: str) -> ArmSpec:
    try:
        return arm_registry()[name]
    except KeyError as exc:
        known = ", ".join(arm_registry())
        raise ValueError(f"unknown PDRR arm {name!r}; expected one of: {known}") from exc


def validate_registry() -> None:
    registry = arm_registry()
    if len(registry) != len(_SPECS):
        raise RuntimeError("duplicate PDRR arm name")
    required = {
        "full", "hard_prune", "hard_prune_kd", "gradual_kd",
        "matrix_no_recovery", "random_matrix_recovery",
        "state_position_no_causal", "causal_state_position_recovery",
    }
    missing = required - registry.keys()
    if missing:
        raise RuntimeError(f"required ablations missing: {sorted(missing)}")
    if registry["matrix_no_recovery"].recovery_loss:
        raise RuntimeError("matrix_no_recovery must disable component recovery loss")
    random_arm = registry["random_matrix_recovery"]
    if not random_arm.recovery_loss or random_arm.routing != "random":
        raise RuntimeError("random_matrix_recovery contract was changed")


validate_registry()
