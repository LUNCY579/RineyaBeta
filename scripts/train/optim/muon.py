# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Rineya contributors.
# SPDX-License-Identifier: Apache-2.0
"""Small, deterministic Muon subset for the RGE-PEFT optimizer pilot.

This module adapts the matrix-update semantics used by NVIDIA NeMo Emerging
Optimizers, but deliberately keeps only the part required by the registered
RGE-PEFT O-M arm:

* two-dimensional FP32 LoRA parameters;
* EMA-style momentum with optional Nesterov interpolation;
* five-step optimized quintic Newton--Schulz orthogonalization;
* explicit update scaling; and
* decoupled weight decay.

It does not implement MuonHyperball, tensor parallelism, fused logical QKV/MLP
partitioning, Triton kernels, or reduced-precision optimizer math.  Those are
outside the M2 optimizer-only ablation and must not be silently enabled.

Upstream reference (Apache-2.0):
https://github.com/NVIDIA-NeMo/Emerging-Optimizers/tree/02b6ec843b392475a9dd29fc68dbdb27c48bea4c
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer

MUON_IMPLEMENTATION_VERSION = "rge-muon-v1"
MUON_UPSTREAM_COMMIT = "02b6ec843b392475a9dd29fc68dbdb27c48bea4c"

# Optimized quintic schedule used by the pinned upstream implementation.  Five
# steps consume each tuple once; longer exploratory schedules cycle explicitly.
QUINTIC_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
VALID_SCALE_MODES = ("shape_scaling", "spectral", "unit_rms_norm")


def get_muon_scale_factor(size_out: int, size_in: int, *, mode: str) -> float:
    """Return the explicit shape-dependent Muon update multiplier."""
    if size_out < 1 or size_in < 1:
        raise ValueError("Muon matrix dimensions must be positive")
    if mode == "shape_scaling":
        return math.sqrt(max(1.0, size_out / size_in))
    if mode == "spectral":
        return math.sqrt(max(size_out, size_in))
    if mode == "unit_rms_norm":
        return math.sqrt(size_out / size_in)
    raise ValueError(f"invalid Muon scale_mode={mode!r}; expected {VALID_SCALE_MODES}")


@torch.no_grad()
def newton_schulz_quintic(
    matrix: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1.0e-15,
) -> torch.Tensor:
    """Orthogonalize a 2-D matrix with the pinned quintic NS schedule.

    The computation is always FP32.  The larger first dimension is transposed so
    the Gram matrix is formed on the smaller axis, matching the upstream
    non-tensor-parallel path.  The input is never mutated.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Muon requires a 2-D matrix, got {matrix.ndim}-D")
    if steps < 1:
        raise ValueError("Muon ns_steps must be at least 1")
    if eps <= 0.0:
        raise ValueError("Muon ns_eps must be positive")
    if not matrix.is_floating_point():
        raise TypeError("Muon matrix must use a floating dtype")

    value = matrix.detach().to(dtype=torch.float32)
    transpose = value.shape[0] > value.shape[1]
    if transpose:
        value = value.mT

    # F.normalize has a well-defined zero-input result and avoids a Python-side
    # device synchronization.  Frobenius normalization bounds the spectral norm.
    value = torch.nn.functional.normalize(value, p=2, dim=(-2, -1), eps=eps)
    for index in range(steps):
        a, b, c = QUINTIC_COEFFICIENTS[index % len(QUINTIC_COEFFICIENTS)]
        gram = value @ value.mT
        polynomial = b * gram + c * (gram @ gram)
        value = a * value + polynomial @ value

    if transpose:
        value = value.mT
    return value.to(dtype=torch.float32)


class Muon(Optimizer):
    """Muon optimizer restricted to ordinary two-dimensional LoRA tensors.

    All configuration is explicit so the optimizer contract can be compared
    across checkpoints and experiment arms.  Parameters with any other rank are
    rejected rather than silently routed to a different optimizer; the current
    registered LoRA arm contains only 2-D A/B matrices.  A future DoRA or bias
    experiment must add an explicit hybrid route and a new contract version.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float,
        momentum: float,
        nesterov: bool,
        weight_decay: float,
        ns_steps: int,
        ns_eps: float,
        scale_mode: str,
        extra_scale_factor: float,
    ) -> None:
        if lr < 0.0:
            raise ValueError("Muon lr must be non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Muon momentum must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("Muon weight_decay must be non-negative")
        if ns_steps < 1:
            raise ValueError("Muon ns_steps must be at least 1")
        if ns_eps <= 0.0:
            raise ValueError("Muon ns_eps must be positive")
        if scale_mode not in VALID_SCALE_MODES:
            raise ValueError(
                f"invalid Muon scale_mode={scale_mode!r}; expected {VALID_SCALE_MODES}"
            )
        if not math.isfinite(extra_scale_factor) or extra_scale_factor <= 0.0:
            raise ValueError("Muon extra_scale_factor must be finite and positive")

        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "weight_decay": float(weight_decay),
            "ns_steps": int(ns_steps),
            "ns_eps": float(ns_eps),
            "scale_mode": str(scale_mode),
            "extra_scale_factor": float(extra_scale_factor),
        }
        super().__init__(params, defaults)
        self._validate_parameter_groups()

    def _validate_parameter_groups(self) -> None:
        count = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                count += 1
                if parameter.ndim != 2:
                    raise ValueError(
                        "RGE Muon route supports only 2-D tensors; "
                        f"received shape {tuple(parameter.shape)}"
                    )
                if not parameter.is_floating_point():
                    raise TypeError("RGE Muon parameters must use a floating dtype")
        if count == 0:
            raise ValueError("Muon received no parameters")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("RGE Muon does not support closures")

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            ns_eps = float(group["ns_eps"])
            scale_mode = str(group["scale_mode"])
            extra = float(group["extra_scale_factor"])
            nesterov = bool(group["nesterov"])

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("RGE Muon does not support sparse gradients")
                if gradient.ndim != 2:
                    raise RuntimeError("RGE Muon encountered a non-matrix gradient")

                # Decoupled weight decay is applied to the parameter, not folded
                # into the gradient/momentum stream.
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)

                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1.0 - momentum)
                update = gradient.lerp(buffer, momentum) if nesterov else buffer

                orthogonal = newton_schulz_quintic(
                    update, steps=ns_steps, eps=ns_eps
                )
                scale = get_muon_scale_factor(
                    parameter.shape[-2], parameter.shape[-1], mode=scale_mode
                )
                parameter.add_(
                    orthogonal.to(dtype=parameter.dtype),
                    alpha=-lr * scale * extra,
                )
        return None
