"""LM vs state gradient cosine. Diagnosis only: no optimizer step.

Production training still backprops a single combined loss. This module measures
whether those two terms point in different directions on a matched minibatch.
A negative cosine is not itself a capability result.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch


def vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denom = float(left.norm() * right.norm())
    if denom <= 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denom)


def _group_key(name: str) -> str:
    if "state_projector" in name:
        return "state_projector"
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part.startswith("lora_"):
            return ".".join(parts[:index]) or "lora"
    return "other"


def _flatten(grads: Iterable[torch.Tensor | None], params: Iterable[torch.Tensor]) -> torch.Tensor:
    chunks = []
    for grad, param in zip(grads, params, strict=True):
        piece = param.new_zeros(param.numel()) if grad is None else grad.detach().reshape(-1)
        chunks.append(piece.float())
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


def conflict_from_losses(
    lm_loss: torch.Tensor,
    state_loss: torch.Tensor,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict:
    """Cosine between ∇L_LM and ∇L_state. Does not call optimizer.step."""

    named = [(name, param) for name, param in named_parameters if param.requires_grad]
    if not named:
        raise ValueError("no trainable parameters")
    names = [name for name, _ in named]
    params = [param for _, param in named]
    grad_lm = torch.autograd.grad(lm_loss, params, retain_graph=True, allow_unused=True)
    grad_state = torch.autograd.grad(state_loss, params, retain_graph=True, allow_unused=True)
    flat_lm = _flatten(grad_lm, params)
    flat_state = _flatten(grad_state, params)
    per_module: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"lm": [], "state": []})
    for name, glm, gst, param in zip(names, grad_lm, grad_state, params, strict=True):
        key = _group_key(name)
        per_module[key]["lm"].append(param.new_zeros(param.numel()) if glm is None else glm.detach().reshape(-1))
        per_module[key]["state"].append(param.new_zeros(param.numel()) if gst is None else gst.detach().reshape(-1))
    module_rows = []
    negative = 0
    usable = 0
    for key, buckets in sorted(per_module.items()):
        left = torch.cat([item.float() for item in buckets["lm"]])
        right = torch.cat([item.float() for item in buckets["state"]])
        cosine = vector_cosine(left, right)
        lm_norm = float(left.norm())
        state_norm = float(right.norm())
        if lm_norm > 0 and state_norm > 0:
            usable += 1
            if cosine < 0:
                negative += 1
        module_rows.append(
            {
                "module": key,
                "cosine": cosine,
                "lm_norm": lm_norm,
                "state_norm": state_norm,
            }
        )
    lm_norm = float(flat_lm.norm())
    state_norm = float(flat_state.norm())
    return {
        "global_cosine": vector_cosine(flat_lm, flat_state),
        "lm_grad_norm": lm_norm,
        "state_grad_norm": state_norm,
        "grad_norm_ratio_state_over_lm": (state_norm / lm_norm) if lm_norm > 0 else float("nan"),
        "modules": module_rows,
        "n_modules_with_both_grads": usable,
        "fraction_modules_negative_cosine": (negative / usable) if usable else float("nan"),
        "lm_loss": float(lm_loss.detach().cpu()),
        "state_loss": float(state_loss.detach().cpu()),
    }
