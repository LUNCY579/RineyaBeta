"""Gauge-invariant geometry of the effective LoRA update.

The canonical object is ``delta = scale * B @ A``.  Factor norms are not
intrinsic because ``B @ A == (B @ Q) @ (Q^-1 @ A)``.  This module therefore
computes every reported metric from the induced product, using rank-r or rank-2r
algebra for the norms and materialising no dense ``B @ A`` in the training path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

GEOMETRY_IMPLEMENTATION_VERSION = "rge-delta-geometry-v1"
DELTABALL_IMPLEMENTATION_VERSION = "rge-deltaball-v1"


def _work(tensor: torch.Tensor) -> torch.Tensor:
    """Detached FP32 view used by runtime metrics without mutating the source."""
    return tensor.detach().to(dtype=torch.float32)


@torch.no_grad()
def low_rank_frobenius_sq(
    left: torch.Tensor, right: torch.Tensor, *, scale: float = 1.0,
) -> torch.Tensor:
    """Return ``||scale * left @ right||_F^2`` without forming the product."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("left/right must be compatible 2-D low-rank factors")
    l, r = _work(left), _work(right)
    gram_l = l.transpose(0, 1) @ l
    gram_r = r @ r.transpose(0, 1)
    # trace(gram_l @ gram_r); both Gram matrices are symmetric, but spell out
    # the transpose so the identity remains clear under finite precision.
    value = (gram_l * gram_r.transpose(0, 1)).sum()
    return value.clamp_min(0.0) * (float(scale) ** 2)


@torch.no_grad()
def low_rank_frobenius_norm(
    left: torch.Tensor, right: torch.Tensor, *, scale: float = 1.0,
) -> torch.Tensor:
    return low_rank_frobenius_sq(left, right, scale=scale).sqrt()


@torch.no_grad()
def successive_delta_norm(
    next_b: torch.Tensor,
    next_a: torch.Tensor,
    prev_b: torch.Tensor,
    prev_a: torch.Tensor,
    *,
    next_scale: float,
    prev_scale: float,
) -> torch.Tensor:
    """Norm of two induced deltas' difference via a rank-at-most-2r product."""
    device = next_b.device
    nb, na = _work(next_b), _work(next_a)
    pb = _work(prev_b).to(device=device)
    pa = _work(prev_a).to(device=device)
    if nb.shape != pb.shape or na.shape != pa.shape:
        raise ValueError("successive LoRA factors must have identical shapes")
    b_cat = torch.cat((nb, pb), dim=1)
    a_cat = torch.cat((float(next_scale) * na,
                       -float(prev_scale) * pa), dim=0)
    return low_rank_frobenius_norm(b_cat, a_cat)


@torch.no_grad()
def base_delta_inner(
    base: torch.Tensor, b: torch.Tensor, a: torch.Tensor, *, scale: float,
) -> torch.Tensor:
    """Return ``<base, scale * B @ A>`` without materialising ``B @ A``."""
    w, bb, aa = _work(base), _work(b), _work(a)
    if w.shape != (bb.shape[0], aa.shape[1]):
        raise ValueError("base matrix shape is incompatible with LoRA factors")
    # <W, BA> = <W A^T, B>.  This is the only dense-base interaction needed for
    # both radius and angle; it is sampled on a deterministic cadence.
    return (bb * (w @ aa.transpose(0, 1))).sum() * float(scale)


@torch.no_grad()
def effective_weight_metrics(
    base: torch.Tensor, b: torch.Tensor, a: torch.Tensor, *, scale: float,
    base_norm_sq: float | None = None,
) -> dict[str, float]:
    """Dense-reference-equivalent radius and angle using low-rank identities."""
    base_sq = (_work(base).square().sum() if base_norm_sq is None else
               torch.as_tensor(base_norm_sq, device=base.device, dtype=torch.float32))
    delta_sq = low_rank_frobenius_sq(b, a, scale=scale)
    cross = base_delta_inner(base, b, a, scale=scale)
    effective_sq = (base_sq + 2.0 * cross + delta_sq).clamp_min(0.0)
    base_norm = base_sq.sqrt()
    effective_norm = effective_sq.sqrt()
    denominator = base_norm * effective_norm
    cosine = ((base_sq + cross) / denominator.clamp_min(torch.finfo(torch.float32).tiny))
    cosine = cosine.clamp(-1.0, 1.0)
    return {
        "base_norm": float(base_norm.item()),
        "delta_norm": float(delta_sq.sqrt().item()),
        "relative_delta": float((delta_sq.sqrt() / base_norm).item()),
        "radius_drift": float((effective_norm / base_norm).item()),
        "angle_radians": float(torch.acos(cosine).item()),
        "base_delta_inner": float(cross.item()),
    }


@dataclass(frozen=True)
class LoRATarget:
    name: str
    base: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    scale: float


def iter_lora_targets(model, adapter: str = "default") -> Iterator[LoRATarget]:
    """Discover PEFT LoRA linear targets by their public factor containers."""
    for name, module in sorted(model.named_modules(), key=lambda item: item[0]):
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        base_layer = getattr(module, "base_layer", None)
        scaling = getattr(module, "scaling", None)
        if lora_a is None or lora_b is None or base_layer is None or scaling is None:
            continue
        if adapter not in lora_a or adapter not in lora_b or adapter not in scaling:
            continue
        base = getattr(base_layer, "weight", None)
        if base is None:
            continue
        yield LoRATarget(
            name=name,
            base=base,
            a=lora_a[adapter].weight,
            b=lora_b[adapter].weight,
            scale=float(scaling[adapter]),
        )


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


class DeltaGeometryTracker:
    """Deterministic, detached effective-delta measurements for a PEFT model."""

    def __init__(self, model, *, cadence_steps: int, include_radius_angle: bool,
                 run_id: str, enabled: bool = True) -> None:
        if cadence_steps < 1:
            raise ValueError("geometry cadence_steps must be positive")
        self.enabled = bool(enabled)
        self.cadence_steps = int(cadence_steps)
        self.include_radius_angle = bool(include_radius_angle)
        self.run_id = str(run_id)
        self.targets = list(iter_lora_targets(model)) if self.enabled else []
        if self.enabled and not self.targets:
            raise ValueError("geometry enabled but no PEFT LoRA targets were found")
        self.previous: dict[str, dict[str, Any]] = {}
        self.base_norms: dict[str, float] = {}
        self.base_norm_sqs: dict[str, float] = {}
        for target in self.targets:
            base_sq = float(_work(target.base).square().sum().item())
            self.base_norm_sqs[target.name] = base_sq
            self.base_norms[target.name] = math.sqrt(base_sq)
            self.previous[target.name] = {
                "step": 0,
                "a": _cpu_clone(target.a),
                "b": _cpu_clone(target.b),
                "scale": target.scale,
            }

    @property
    def contract(self) -> dict:
        return {
            "implementation": GEOMETRY_IMPLEMENTATION_VERSION,
            "enabled": self.enabled,
            "cadence_steps": self.cadence_steps,
            "include_radius_angle": self.include_radius_angle,
            "target_count": len(self.targets),
            "targets": [target.name for target in self.targets],
            "successive_step_semantics": "difference_since_previous_measurement",
            "base_norm_cache": "detached_fp32_frobenius_at_run_start",
        }

    def due(self, step: int) -> bool:
        return self.enabled and step > 0 and step % self.cadence_steps == 0

    @torch.no_grad()
    def measure(self, step: int) -> list[dict]:
        if not self.due(step):
            return []
        rows = []
        for target in self.targets:
            prev = self.previous[target.name]
            base_norm = self.base_norms[target.name]
            delta_norm = low_rank_frobenius_norm(target.b, target.a,
                                                   scale=target.scale)
            step_norm = successive_delta_norm(
                target.b, target.a, prev["b"], prev["a"],
                next_scale=target.scale, prev_scale=float(prev["scale"]))
            row = {
                "schema_version": 1,
                "run_id": self.run_id,
                "step": int(step),
                "previous_measurement_step": int(prev["step"]),
                "step_span": int(step - int(prev["step"])),
                "module": target.name,
                "base_norm": float(base_norm),
                "delta_norm": float(delta_norm.item()),
                "relative_delta": float((delta_norm / base_norm).item()),
                "effective_step_norm": float(step_norm.item()),
                "base_relative_effective_step": float((step_norm / base_norm).item()),
                "lora_scale": float(target.scale),
                "implementation": GEOMETRY_IMPLEMENTATION_VERSION,
                "cadence_steps": self.cadence_steps,
            }
            if self.include_radius_angle:
                dense_interaction = effective_weight_metrics(
                    target.base, target.b, target.a, scale=target.scale,
                    base_norm_sq=self.base_norm_sqs[target.name])
                row["radius_drift"] = dense_interaction["radius_drift"]
                row["angle_radians"] = dense_interaction["angle_radians"]
                row["base_delta_inner"] = dense_interaction["base_delta_inner"]
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite geometry metric {key} for {target.name}")
            rows.append(row)
            self.previous[target.name] = {
                "step": int(step),
                "a": _cpu_clone(target.a),
                "b": _cpu_clone(target.b),
                "scale": target.scale,
            }
        return rows

    def state_dict(self) -> dict:
        return {
            "implementation": GEOMETRY_IMPLEMENTATION_VERSION,
            "cadence_steps": self.cadence_steps,
            "include_radius_angle": self.include_radius_angle,
            "base_norms": self.base_norms,
            "base_norm_sqs": self.base_norm_sqs,
            "previous": self.previous,
        }

    def load_state_dict(self, state: dict) -> None:
        if not self.enabled:
            return
        if state.get("implementation") != GEOMETRY_IMPLEMENTATION_VERSION:
            raise ValueError("geometry implementation/version changed across resume")
        if int(state.get("cadence_steps", -1)) != self.cadence_steps:
            raise ValueError("geometry cadence changed across resume")
        if bool(state.get("include_radius_angle")) != self.include_radius_angle:
            raise ValueError("geometry radius/angle policy changed across resume")
        if state.get("base_norms") != self.base_norms:
            raise ValueError("cached base norms changed across resume")
        if state.get("base_norm_sqs") != self.base_norm_sqs:
            raise ValueError("cached squared base norms changed across resume")
        previous = state.get("previous", {})
        expected = {target.name for target in self.targets}
        if set(previous) != expected:
            raise ValueError("geometry target set changed across resume")
        self.previous = {
            name: {
                "step": int(payload["step"]),
                "a": _cpu_clone(payload["a"]),
                "b": _cpu_clone(payload["b"]),
                "scale": float(payload["scale"]),
            }
            for name, payload in previous.items()
        }


class DeltaBallProjector:
    """Base-relative trust-region projection on each induced LoRA delta."""

    def __init__(self, model, *, enabled: bool, epsilon_max: float,
                 warmin_steps: int, schedule: str, run_id: str) -> None:
        if epsilon_max < 0.0 or not math.isfinite(epsilon_max):
            raise ValueError("DeltaBall epsilon_max must be finite and non-negative")
        if warmin_steps < 0:
            raise ValueError("DeltaBall warmin_steps must be non-negative")
        if schedule != "linear":
            raise ValueError("DeltaBall v1 supports only schedule=linear")
        self.enabled = bool(enabled)
        self.epsilon_max = float(epsilon_max)
        self.warmin_steps = int(warmin_steps)
        self.schedule = str(schedule)
        self.run_id = str(run_id)
        self.targets = list(iter_lora_targets(model)) if self.enabled else []
        if self.enabled and not self.targets:
            raise ValueError("DeltaBall enabled but no PEFT LoRA targets were found")
        self.base_norms = {
            target.name: math.sqrt(float(_work(target.base).square().sum().item()))
            for target in self.targets
        }
        self.steps_projected = 0
        self.matrix_checks = 0
        self.boundary_hits = 0

    @property
    def contract(self) -> dict:
        return {
            "implementation": DELTABALL_IMPLEMENTATION_VERSION,
            "enabled": self.enabled,
            "epsilon_max": self.epsilon_max,
            "warmin_steps": self.warmin_steps,
            "schedule": self.schedule,
            "target_count": len(self.targets),
            "optimizer_state_projection": "none",
            "factor_projection": "symmetric_sqrt_c",
        }

    def epsilon_at(self, step: int) -> float:
        if step < 1:
            return 0.0
        fraction = (1.0 if self.warmin_steps == 0 else
                    min(float(step) / float(self.warmin_steps), 1.0))
        return self.epsilon_max * fraction

    @torch.no_grad()
    def project(self, step: int) -> list[dict]:
        if not self.enabled:
            return []
        epsilon = self.epsilon_at(step)
        rows = []
        step_hit = False
        for target in self.targets:
            base_norm = self.base_norms[target.name]
            delta_norm = float(low_rank_frobenius_norm(
                target.b, target.a, scale=target.scale).item())
            relative_before = delta_norm / base_norm
            hit = relative_before > epsilon
            projection_c = 1.0
            if hit:
                if delta_norm <= 0.0 or epsilon <= 0.0:
                    projection_c = 0.0
                else:
                    projection_c = epsilon * base_norm / delta_norm
                factor_scale = math.sqrt(max(projection_c, 0.0))
                target.a.mul_(factor_scale)
                target.b.mul_(factor_scale)
                step_hit = True
                self.boundary_hits += 1
            relative_after = relative_before * projection_c
            if hit and not math.isclose(relative_after, epsilon,
                                        rel_tol=2e-5, abs_tol=2e-8):
                raise FloatingPointError(
                    f"DeltaBall projection missed boundary for {target.name}: "
                    f"{relative_after} != {epsilon}")
            self.matrix_checks += 1
            rows.append({
                "schema_version": 1,
                "record_type": "deltaball_projection",
                "run_id": self.run_id,
                "step": int(step),
                "module": target.name,
                "epsilon": float(epsilon),
                "base_norm": float(base_norm),
                "relative_delta_before": float(relative_before),
                "relative_delta_after": float(relative_after),
                "boundary_hit": bool(hit),
                "projection_c": float(projection_c),
                "factor_scale": float(math.sqrt(max(projection_c, 0.0))),
                "implementation": DELTABALL_IMPLEMENTATION_VERSION,
            })
        self.steps_projected += 1
        for row in rows:
            row["cumulative_matrix_hit_rate"] = (
                self.boundary_hits / max(self.matrix_checks, 1))
            row["step_had_boundary_hit"] = step_hit
        return rows

    def state_dict(self) -> dict:
        return {
            "implementation": DELTABALL_IMPLEMENTATION_VERSION,
            "contract": self.contract,
            "steps_projected": self.steps_projected,
            "matrix_checks": self.matrix_checks,
            "boundary_hits": self.boundary_hits,
            "base_norms": self.base_norms,
        }

    def load_state_dict(self, state: dict) -> None:
        if not self.enabled:
            return
        if state.get("implementation") != DELTABALL_IMPLEMENTATION_VERSION:
            raise ValueError("DeltaBall implementation/version changed across resume")
        if state.get("contract") != self.contract:
            raise ValueError("DeltaBall contract changed across resume")
        if state.get("base_norms") != self.base_norms:
            raise ValueError("DeltaBall base norms changed across resume")
        self.steps_projected = int(state["steps_projected"])
        self.matrix_checks = int(state["matrix_checks"])
        self.boundary_hits = int(state["boundary_hits"])


def append_geometry(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False) + "\n")


def reconcile_geometry_log(path: Path, *, through_step: int) -> None:
    """Drop uncheckpointed tail records before an exact resume."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["step"]) <= through_step:
            kept.append(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False))
    text = "\n".join(kept) + ("\n" if kept else "")
    path.write_text(text, encoding="utf-8", newline="\n")
