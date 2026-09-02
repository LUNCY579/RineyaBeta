"""LM vs state gradient cosine. Diagnosis only: no optimizer step.

Production training still backprops a single combined loss. This module measures
whether those two terms point in different directions on a matched minibatch.
A negative cosine is not itself a capability result.
"""

from __future__ import annotations

import math
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


class TrajectoryProbe:
    """Record cosine(grad L_LM, grad L_state) at scheduled steps DURING training.

    The step-250 snapshot in `artifacts/b2_verdict_s29.json` found all arms within 2
    sigma of orthogonal. That refutes H2 at convergence, not as a mechanism: LM loss
    there is ~7e-4, and two objectives cannot conflict over a loss that is already
    zero. A conflict that appears early and dissipates would look exactly like that
    snapshot. This probe measures the trajectory instead of its endpoint.

    Read-only by construction. `conflict_from_losses` uses `torch.autograd.grad`,
    which returns gradients without accumulating into `.grad`, and backward consumes
    no RNG, so the optimizer trajectory is unchanged. `retain_graph=True` leaves the
    graph intact for the production backward that follows.

    Cost. Every measured step pays two extra backward passes, and with gradient
    checkpointing on each one re-runs the checkpointed forward. So the schedule is
    dense only where the hypothesis lives; see PREREGISTRATION_B2_TRAJECTORY.md.
    """

    def __init__(
        self,
        sink: Any,
        *,
        schedule: Iterable[int] | None = None,
        dense_until: int = 120,
        mid_until: int = 400,
        mid_every: int = 4,
        late_every: int = 20,
    ) -> None:
        self.sink = sink
        self.explicit = set(int(step) for step in schedule) if schedule is not None else None
        self.dense_until = dense_until
        self.mid_until = mid_until
        self.mid_every = mid_every
        self.late_every = late_every
        self.rows: list[dict] = []

    def should_measure(self, global_step: int) -> bool:
        if self.explicit is not None:
            return global_step in self.explicit
        if global_step <= self.dense_until:
            return True
        if global_step <= self.mid_until:
            return global_step % self.mid_every == 0
        return global_step % self.late_every == 0

    def __call__(
        self,
        *,
        global_step: int,
        optimizer_step: int,
        model: Any,
        breakdown: Any,
        learning_rate: float | None = None,
    ) -> None:
        if not self.should_measure(global_step):
            return
        # An arm with no state loss has a constant-zero state term; its gradient is
        # zero and the cosine is undefined. Record the step and skip the grads rather
        # than writing a NaN that later reads as a measurement.
        if not breakdown.state.requires_grad:
            row = {
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "learning_rate": learning_rate,
                "measured": False,
                "reason": "state loss is not differentiable in this arm",
            }
        else:
            report = conflict_from_losses(
                breakdown.lm, breakdown.state, model.named_parameters()
            )
            report.update(
                global_step=global_step,
                optimizer_step=optimizer_step,
                learning_rate=learning_rate,
                measured=True,
                lora_only_cosine=lora_only_cosine(report["modules"]),
            )
            row = report
        self.rows.append(row)
        if self.sink is not None:
            self.sink(row)


def lora_only_cosine(modules: Iterable[dict]) -> float:
    """Cosine over the parameters BOTH losses touch, i.e. excluding the projector.

    `L_LM` does not depend on `state_projector` at all, so grad_lm is identically zero
    on that whole block. Keeping it in the global cosine inflates ||g_state|| without
    contributing to the dot product, shrinking |cosine| toward zero for every arm
    alike and hiding exactly the between-arm difference H2 predicts. Reconstructed
    exactly from per-module norms and cosines:

        cos = sum_m cos_m |lm_m| |st_m| / sqrt(sum_m |lm_m|^2 * sum_m |st_m|^2)
    """

    numerator = left = right = 0.0
    for module in modules:
        if module["module"] == "state_projector":
            continue
        lm_norm, state_norm = module["lm_norm"], module["state_norm"]
        if lm_norm <= 0 or state_norm <= 0:
            continue
        numerator += module["cosine"] * lm_norm * state_norm
        left += lm_norm ** 2
        right += state_norm ** 2
    if left <= 0 or right <= 0:
        return float("nan")
    return numerator / math.sqrt(left * right)
