"""Explicit, auditable optimizer routing for the RGE-PEFT harness."""

from __future__ import annotations

import math
from typing import Any

import torch

from .muon import (
    MUON_IMPLEMENTATION_VERSION,
    MUON_UPSTREAM_COMMIT,
    QUINTIC_COEFFICIENTS,
    VALID_SCALE_MODES,
    Muon,
)

OPTIMIZER_CONTRACT_VERSION = "rge-optim-v2"
ADAMW_IMPLEMENTATION = "torch.optim.AdamW"
MUON_IMPLEMENTATION = "rge.optim.Muon"


def _trainable_named(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _require(cfg: dict, fields: set[str], *, optimizer_name: str) -> None:
    missing = sorted(fields - set(cfg))
    if missing:
        raise ValueError(
            f"{optimizer_name} optimizer config is missing required keys: {missing}"
        )


def _adamw_kwargs(cfg: dict) -> dict[str, Any]:
    _require(
        cfg,
        {"name", "implementation", "lr", "betas", "eps", "weight_decay"},
        optimizer_name="AdamW",
    )
    if cfg["implementation"] != ADAMW_IMPLEMENTATION:
        raise ValueError(f"AdamW implementation must be exactly {ADAMW_IMPLEMENTATION}")
    betas = tuple(float(value) for value in cfg["betas"])
    if len(betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    if not all(0.0 <= value < 1.0 for value in betas):
        raise ValueError("optimizer.betas values must be in [0, 1)")
    lr = float(cfg["lr"])
    eps = float(cfg["eps"])
    weight_decay = float(cfg["weight_decay"])
    if (not all(math.isfinite(value) for value in (*betas, lr, eps, weight_decay))
            or lr < 0.0 or eps <= 0.0 or weight_decay < 0.0):
        raise ValueError("AdamW lr/eps/weight_decay are outside their valid ranges")
    return {
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
    }


def _muon_kwargs(cfg: dict) -> dict[str, Any]:
    _require(
        cfg,
        {
            "name", "implementation", "lr", "momentum", "nesterov",
            "weight_decay", "ns_steps", "ns_eps", "coefficient_type",
            "scale_mode", "extra_scale_factor", "optimizer_math_dtype",
        },
        optimizer_name="Muon",
    )
    if cfg["implementation"] != MUON_IMPLEMENTATION:
        raise ValueError(f"Muon implementation must be exactly {MUON_IMPLEMENTATION}")
    if str(cfg["coefficient_type"]) != "quintic":
        raise ValueError("RGE M2 permits only the pinned quintic coefficient schedule")
    if str(cfg["optimizer_math_dtype"]) != "float32":
        raise ValueError("RGE M2 requires optimizer_math_dtype=float32")
    kwargs = {
        "lr": float(cfg["lr"]),
        "momentum": float(cfg["momentum"]),
        "nesterov": bool(cfg["nesterov"]),
        "weight_decay": float(cfg["weight_decay"]),
        "ns_steps": int(cfg["ns_steps"]),
        "ns_eps": float(cfg["ns_eps"]),
        "scale_mode": str(cfg["scale_mode"]),
        "extra_scale_factor": float(cfg["extra_scale_factor"]),
    }
    numeric = (kwargs["lr"], kwargs["momentum"], kwargs["weight_decay"],
               kwargs["ns_eps"], kwargs["extra_scale_factor"])
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Muon numeric settings must be finite")
    if kwargs["lr"] < 0.0:
        raise ValueError("Muon lr must be non-negative")
    if not 0.0 <= kwargs["momentum"] < 1.0:
        raise ValueError("Muon momentum must be in [0, 1)")
    if kwargs["weight_decay"] < 0.0:
        raise ValueError("Muon weight_decay must be non-negative")
    if kwargs["ns_steps"] < 1 or kwargs["ns_eps"] <= 0.0:
        raise ValueError("Muon ns_steps/ns_eps are outside their valid ranges")
    if kwargs["scale_mode"] not in VALID_SCALE_MODES:
        raise ValueError(
            f"invalid Muon scale_mode={kwargs['scale_mode']!r}; "
            f"expected {VALID_SCALE_MODES}"
        )
    if kwargs["extra_scale_factor"] <= 0.0:
        raise ValueError("Muon extra_scale_factor must be positive")
    return kwargs


def _parameter_payload(named: list[tuple[str, torch.nn.Parameter]]) -> dict:
    return {
        "names": [name for name, _ in named],
        "shapes": {name: list(parameter.shape) for name, parameter in named},
        "dtypes": {name: str(parameter.dtype) for name, parameter in named},
        "parameter_count": int(sum(parameter.numel() for _, parameter in named)),
        "tensor_count": len(named),
    }


def optimizer_contract(model, cfg: dict) -> dict:
    """Return the frozen parameter-routing and numerical optimizer contract."""
    named = _trainable_named(model)
    if not named:
        raise ValueError("model has no trainable parameters")
    name = str(cfg["name"]).lower()
    parameters = _parameter_payload(named)

    if name == "adamw":
        kwargs = _adamw_kwargs(cfg)
        return {
            "contract_version": OPTIMIZER_CONTRACT_VERSION,
            "name": "adamw",
            "implementation": ADAMW_IMPLEMENTATION,
            "version": cfg.get("version"),
            "numerics": {
                "lr": kwargs["lr"],
                "betas": list(kwargs["betas"]),
                "eps": kwargs["eps"],
                "weight_decay": kwargs["weight_decay"],
                "weight_decay_method": "decoupled",
            },
            "parameter_groups": [{"route": "adamw", **parameters}],
        }

    if name == "muon":
        kwargs = _muon_kwargs(cfg)
        non_matrix = [name for name, parameter in named if parameter.ndim != 2]
        if non_matrix:
            raise ValueError(
                "the registered O-M arm has no implicit AdamW fallback; "
                f"non-2-D trainables require a future hybrid contract: {non_matrix[:8]}"
            )
        non_fp32 = [name for name, parameter in named if parameter.dtype != torch.float32]
        if non_fp32:
            raise ValueError(
                "RGE Muon requires FP32 LoRA trainables; unexpected dtypes for "
                f"{non_fp32[:8]}"
            )
        return {
            "contract_version": OPTIMIZER_CONTRACT_VERSION,
            "name": "muon",
            "implementation": MUON_IMPLEMENTATION,
            "version": MUON_IMPLEMENTATION_VERSION,
            "upstream_reference_commit": MUON_UPSTREAM_COMMIT,
            "numerics": {
                **kwargs,
                "coefficient_type": "quintic",
                "coefficient_schedule": [list(values) for values in QUINTIC_COEFFICIENTS],
                "optimizer_math_dtype": "float32",
                "weight_decay_method": "decoupled",
                "tensor_parallel": False,
                "hyperball": False,
            },
            "parameter_groups": [{"route": "muon_matrix", **parameters}],
            "fallback_policy": "reject_non_matrix_trainables",
        }

    raise ValueError(f"unsupported optimizer.name={cfg['name']!r}; expected adamw or muon")


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    """Build the optimizer with no implicit route or hidden default."""
    named = _trainable_named(model)
    if not named:
        raise ValueError("model has no trainable parameters")
    name = str(cfg["name"]).lower()
    parameters = [parameter for _, parameter in named]

    if name == "adamw":
        return torch.optim.AdamW(parameters, **_adamw_kwargs(cfg))

    if name == "muon":
        # Validate through the exact same contract emitted into the run manifest.
        optimizer_contract(model, cfg)
        return Muon(parameters, **_muon_kwargs(cfg))

    raise ValueError(f"unsupported optimizer.name={cfg['name']!r}; expected adamw or muon")
