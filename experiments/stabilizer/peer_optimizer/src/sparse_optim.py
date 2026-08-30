#!/usr/bin/env python3
"""Exact-support AdamW for sparse native-weight fine-tuning.

Masking gradients alone is insufficient: ordinary AdamW can still move a frozen
coordinate through decoupled weight decay or stale optimizer momentum.  This
optimizer masks the gradient, first moment, second moment, update, and weight
decay.  Assuming no external in-place mutation, coordinates where M=0 remain
bitwise equal to their initial checkpoint values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class SparseParameterSpec:
    parameter: nn.Parameter
    mask: Tensor
    step_scale: Tensor | float = 1.0
    name: str = "parameter"

    def validate(self) -> None:
        if self.parameter.shape != self.mask.shape:
            raise ValueError(
                f"{self.name}: parameter shape {tuple(self.parameter.shape)} != "
                f"mask shape {tuple(self.mask.shape)}"
            )
        if isinstance(self.step_scale, Tensor):
            if self.step_scale.shape != self.parameter.shape:
                raise ValueError(
                    f"{self.name}: scale shape {tuple(self.step_scale.shape)} != "
                    f"parameter shape {tuple(self.parameter.shape)}"
                )
            scale_values = self.step_scale.detach()
            if not bool(torch.isfinite(scale_values).all()):
                raise ValueError(f"{self.name}: step scale must be finite")
            if bool((scale_values < 0).any()):
                raise ValueError(f"{self.name}: step scale must be non-negative")
        else:
            scale_value = float(self.step_scale)
            if not math.isfinite(scale_value) or scale_value < 0:
                raise ValueError(f"{self.name}: scalar step scale must be finite and non-negative")
        mask_values = self.mask.detach()
        if not bool(((mask_values == 0) | (mask_values == 1)).all()):
            raise ValueError(f"{self.name}: mask must be exactly binary")


class SparseNativeAdamW(Optimizer):
    """AdamW whose complete optimizer dynamics are restricted to a fixed support."""

    def __init__(
        self,
        specs: Iterable[SparseParameterSpec],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("betas must lie in [0, 1)")

        groups = []
        seen: set[int] = set()
        for spec in specs:
            spec.validate()
            if id(spec.parameter) in seen:
                raise ValueError(f"parameter {spec.name!r} was supplied more than once")
            seen.add(id(spec.parameter))
            groups.append(
                {
                    "params": [spec.parameter],
                    "mask": spec.mask.detach().clone(),
                    "step_scale": (
                        spec.step_scale.detach().clone()
                        if isinstance(spec.step_scale, Tensor)
                        else float(spec.step_scale)
                    ),
                    "name": spec.name,
                }
            )
        if not groups:
            raise ValueError("at least one sparse parameter specification is required")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            mask_source = group["mask"]
            scale_source = group["step_scale"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SparseNativeAdamW does not accept sparse-gradient tensors")

                mask = mask_source.to(device=parameter.device, dtype=parameter.dtype)
                grad = parameter.grad
                if maximize:
                    grad = -grad
                # Mask the task gradient before moment accumulation, but do not put
                # the peer-generated scale inside Adam's normalization.  A positive
                # constant applied to the gradient is largely cancelled by
                # exp_avg / sqrt(exp_avg_sq), so it would not reliably control the
                # realized step.  The scale is applied to the final update below.
                effective_grad = grad * mask
                state_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                effective_grad_state = effective_grad.to(state_dtype)
                state_mask = mask.to(state_dtype)
                if isinstance(scale_source, Tensor):
                    state_scale: Tensor | float = scale_source.to(
                        device=parameter.device, dtype=state_dtype
                    )
                else:
                    state_scale = float(scale_source)

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    # Keep optimizer statistics in fp32 for fp16/bf16 model weights.
                    # This matches the stability expectation of ordinary AdamW.
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=state_dtype)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(effective_grad_state, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    effective_grad_state, effective_grad_state, value=1.0 - beta2
                )
                # State masking matters when checkpoints are resumed or a caller
                # accidentally reuses optimizer state from a denser experiment.
                exp_avg.mul_(state_mask)
                exp_avg_sq.mul_(state_mask)

                step = state["step"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(eps)

                if weight_decay != 0:
                    decay = parameter.to(state_dtype) * state_mask * state_scale
                    parameter.add_(decay.to(parameter.dtype), alpha=-lr * weight_decay)
                update = ((exp_avg * state_mask) / denom) * state_scale
                parameter.add_(update.to(parameter.dtype), alpha=-step_size)

        return loss

    def selected_parameter_count(self) -> int:
        return sum(int(group["mask"].sum().item()) for group in self.param_groups)

    def support_fraction(self) -> float:
        selected = self.selected_parameter_count()
        total = sum(group["mask"].numel() for group in self.param_groups)
        return selected / total

    @torch.no_grad()
    def clear_state_outside_support(self) -> None:
        """Reassert the fixed-support invariant after loading an external state dict."""

        for group in self.param_groups:
            mask_source = group["mask"]
            for parameter in group["params"]:
                state = self.state.get(parameter, {})
                if not state:
                    continue
                if "exp_avg" in state:
                    state["exp_avg"].mul_(
                        mask_source.to(device=state["exp_avg"].device, dtype=state["exp_avg"].dtype)
                    )
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].mul_(
                        mask_source.to(
                            device=state["exp_avg_sq"].device, dtype=state["exp_avg_sq"].dtype
                        )
                    )
