#!/usr/bin/env python3
"""Hugging Face/Qwen-style integration for fixed peer-generated sparse plans.

The integration intentionally separates three phases:

1. run an ordinary calibration backward pass to measure task gradients;
2. generate and persist a fixed support plan from compatible layer families;
3. restart/freeze the model and train through ``SparseNativeAdamW``.

This prevents a dynamic mask from eventually touching a large union of weights and
makes the native delta exactly auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn

try:  # package import
    from .peer_stabilizer import LayerMetadata, LayerView, PeerStabilizerMatrix
    from .sparse_optim import SparseNativeAdamW, SparseParameterSpec
except ImportError:  # direct script/module import
    from peer_stabilizer import LayerMetadata, LayerView, PeerStabilizerMatrix
    from sparse_optim import SparseNativeAdamW, SparseParameterSpec

PLAN_VERSION = 1


@dataclass
class SparsePlanEntry:
    parameter_name: str
    module_type: str
    depth: int
    mask: Tensor
    step_scale: Tensor
    attention: Tensor
    feature_weights: Tensor

    @property
    def selected(self) -> int:
        return int(self.mask.sum().item())

    @property
    def total(self) -> int:
        return self.mask.numel()

    def mask_sha256(self) -> str:
        raw = self.mask.detach().to(torch.uint8).cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()


@dataclass
class SparsePlan:
    entries: list[SparsePlanEntry]
    budget: int | float
    module_paths: tuple[str, ...]
    source: str = "peer_stabilizer_matrix_v1"

    @property
    def selected(self) -> int:
        return sum(entry.selected for entry in self.entries)

    @property
    def total(self) -> int:
        return sum(entry.total for entry in self.entries)

    @property
    def support_fraction(self) -> float:
        return self.selected / self.total if self.total else 0.0

    def manifest(self) -> dict[str, Any]:
        return {
            "version": PLAN_VERSION,
            "source": self.source,
            "budget": self.budget,
            "module_paths": list(self.module_paths),
            "selected": self.selected,
            "total": self.total,
            "support_fraction": self.support_fraction,
            "entries": [
                {
                    "parameter_name": e.parameter_name,
                    "module_type": e.module_type,
                    "depth": e.depth,
                    "shape": list(e.mask.shape),
                    "selected": e.selected,
                    "total": e.total,
                    "support_fraction": e.selected / e.total,
                    "mask_sha256": e.mask_sha256(),
                    "attention": e.attention.tolist(),
                    "feature_weights": e.feature_weights.tolist(),
                }
                for e in self.entries
            ],
        }


def transformer_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Resolve the decoder-layer list used by Qwen-style Hugging Face models."""

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("could not resolve transformer layers at model.model.layers or model.layers")


def resolve_submodule(root: nn.Module, dotted_path: str) -> nn.Module:
    current: Any = root
    for component in dotted_path.split("."):
        if not hasattr(current, component):
            raise AttributeError(
                f"{type(current).__name__} has no component {component!r} "
                f"while resolving {dotted_path!r}"
            )
        current = getattr(current, component)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{dotted_path!r} did not resolve to an nn.Module")
    return current


def _parameter_name_map(model: nn.Module) -> dict[int, str]:
    return {id(parameter): name for name, parameter in model.named_parameters()}


def build_sparse_plan(
    model: nn.Module,
    generator: PeerStabilizerMatrix,
    *,
    module_paths: Iterable[str],
    budget: int | float,
) -> SparsePlan:
    """Generate exact fixed supports from calibration gradients without mutation.

    Every candidate matrix must have a populated ``weight.grad`` from the same
    calibration objective.  A separate peer family is built for each module path,
    so q_proj matrices advise q_proj matrices and down_proj matrices advise
    down_proj matrices by default.
    """

    paths = tuple(module_paths)
    if not paths:
        raise ValueError("module_paths cannot be empty")
    layers = transformer_layers(model)
    names = _parameter_name_map(model)
    entries: list[SparsePlanEntry] = []
    generator.eval()

    generator_device = next(generator.parameters()).device
    for module_path in paths:
        views: list[LayerView] = []
        parameters: list[nn.Parameter] = []
        for depth, layer in enumerate(layers):
            module = resolve_submodule(layer, module_path)
            if not hasattr(module, "weight") or not isinstance(module.weight, nn.Parameter):
                raise TypeError(f"layer {depth} {module_path!r} has no trainable weight parameter")
            parameter = module.weight
            if parameter.ndim != 2:
                raise ValueError(
                    f"{module_path} layer {depth}: expected 2-D weight, got {tuple(parameter.shape)}"
                )
            if parameter.grad is None:
                raise RuntimeError(
                    f"{names.get(id(parameter), module_path)} has no calibration gradient; "
                    "run loss.backward() before building the sparse plan"
                )
            if parameter.device != generator_device:
                raise RuntimeError(
                    f"generator is on {generator_device}, but {module_path} layer {depth} "
                    f"is on {parameter.device}; move the generator to the model device"
                )
            parameters.append(parameter)
            views.append(
                LayerView(
                    weight=parameter.detach(),
                    gradient=parameter.grad.detach(),
                    metadata=LayerMetadata(
                        name=names.get(id(parameter), f"layers.{depth}.{module_path}.weight"),
                        depth=depth,
                        module_type=module_path,
                    ),
                )
            )

        with torch.no_grad():
            for target_index, (view, parameter) in enumerate(zip(views, parameters)):
                output = generator(
                    views,
                    target_index,
                    budget,
                    task_gradient=view.gradient,
                    straight_through=False,
                )
                entries.append(
                    SparsePlanEntry(
                        parameter_name=names[id(parameter)],
                        module_type=module_path,
                        depth=target_index,
                        mask=output.hard_mask.detach().cpu(),
                        step_scale=output.step_scale.detach().cpu(),
                        attention=output.attention.detach().cpu(),
                        feature_weights=output.feature_weights.detach().cpu(),
                    )
                )

    return SparsePlan(entries=entries, budget=budget, module_paths=paths)


def apply_sparse_plan(
    model: nn.Module,
    plan: SparsePlan,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    freeze_unplanned: bool = True,
) -> SparseNativeAdamW:
    """Freeze unplanned parameters and construct the exact-support optimizer."""

    named = dict(model.named_parameters())
    if freeze_unplanned:
        for parameter in named.values():
            parameter.requires_grad_(False)

    specs: list[SparseParameterSpec] = []
    seen: set[str] = set()
    for entry in plan.entries:
        if entry.parameter_name in seen:
            raise ValueError(f"duplicate plan entry for {entry.parameter_name}")
        seen.add(entry.parameter_name)
        if entry.parameter_name not in named:
            raise KeyError(f"plan parameter not found in model: {entry.parameter_name}")
        parameter = named[entry.parameter_name]
        if tuple(parameter.shape) != tuple(entry.mask.shape):
            raise ValueError(
                f"plan shape {tuple(entry.mask.shape)} != model shape {tuple(parameter.shape)} "
                f"for {entry.parameter_name}"
            )
        parameter.requires_grad_(True)
        specs.append(
            SparseParameterSpec(
                parameter=parameter,
                mask=entry.mask.to(parameter.device),
                step_scale=entry.step_scale.to(parameter.device),
                name=entry.parameter_name,
            )
        )

    model.zero_grad(set_to_none=True)
    return SparseNativeAdamW(
        specs,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def save_sparse_plan(plan: SparsePlan, path: str | Path) -> None:
    """Persist exact masks/scales and all audit metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PLAN_VERSION,
        "budget": plan.budget,
        "module_paths": plan.module_paths,
        "source": plan.source,
        "entries": [
            {
                "parameter_name": e.parameter_name,
                "module_type": e.module_type,
                "depth": e.depth,
                "mask": e.mask.to(torch.uint8).cpu(),
                "step_scale": e.step_scale.float().cpu(),
                "attention": e.attention.float().cpu(),
                "feature_weights": e.feature_weights.float().cpu(),
            }
            for e in plan.entries
        ],
    }
    torch.save(payload, destination)


def load_sparse_plan(path: str | Path) -> SparsePlan:
    """Load a plan created by :func:`save_sparse_plan` on CPU."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("version") != PLAN_VERSION:
        raise ValueError(
            f"unsupported plan version {payload.get('version')}; expected {PLAN_VERSION}"
        )
    entries = [
        SparsePlanEntry(
            parameter_name=row["parameter_name"],
            module_type=row["module_type"],
            depth=int(row["depth"]),
            mask=row["mask"].float(),
            step_scale=row["step_scale"].float(),
            attention=row["attention"].float(),
            feature_weights=row["feature_weights"].float(),
        )
        for row in payload["entries"]
    ]
    return SparsePlan(
        entries=entries,
        budget=payload["budget"],
        module_paths=tuple(payload["module_paths"]),
        source=payload.get("source", "peer_stabilizer_matrix_v1"),
    )
