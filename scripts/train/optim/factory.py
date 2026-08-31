"""Explicit optimizer routing for the RGE-PEFT experimental harness."""

from __future__ import annotations

import torch

OPTIMIZER_CONTRACT_VERSION = "rge-optim-v1"


def _trainable_named(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters()
            if param.requires_grad]


def optimizer_contract(model, cfg: dict) -> dict:
    """Return the frozen, auditable parameter-routing contract."""
    named = _trainable_named(model)
    return {
        "contract_version": OPTIMIZER_CONTRACT_VERSION,
        "name": str(cfg["name"]),
        "implementation": str(cfg["implementation"]),
        "parameter_groups": [{
            "route": "adamw",
            "names": [name for name, _ in named],
            "parameter_count": int(sum(param.numel() for _, param in named)),
            "tensor_count": len(named),
        }],
    }


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    """Build the M1 optimizer without implicit fallbacks or hidden defaults."""
    required = {"name", "implementation", "lr", "betas", "eps", "weight_decay"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"optimizer config is missing required keys: {missing}")
    if str(cfg["name"]).lower() != "adamw":
        raise ValueError("M1 supports only optimizer.name=adamw; Muon is gated by M2")
    if cfg["implementation"] != "torch.optim.AdamW":
        raise ValueError("M1 AdamW implementation must be exactly torch.optim.AdamW")
    named = _trainable_named(model)
    if not named:
        raise ValueError("model has no trainable parameters")
    betas = tuple(float(value) for value in cfg["betas"])
    if len(betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    return torch.optim.AdamW(
        [param for _, param in named],
        lr=float(cfg["lr"]), betas=betas,
        weight_decay=float(cfg["weight_decay"]), eps=float(cfg["eps"]))
