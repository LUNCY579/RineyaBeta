#!/usr/bin/env python3
"""Causal controls and compact one-step sparse-update helpers for PCSM Phase C."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

try:
    from .peer_features import LayerMetadata, LayerView, matrix_property_map
    from .sparse_selection import exact_topk_mask, resolve_budget
except ImportError:
    from peer_features import LayerMetadata, LayerView, matrix_property_map
    from sparse_selection import exact_topk_mask, resolve_budget


def stable_u64(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    prefix = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(prefix + b"\0" + raw).hexdigest()


def _permutation(length: int, *, seed: int, label: str, device: torch.device) -> Tensor:
    if length < 2:
        return torch.arange(length, device=device)
    generator = torch.Generator(device="cpu").manual_seed(stable_u64(seed, label) % (2**63 - 1))
    order = torch.randperm(length, generator=generator)
    if torch.equal(order, torch.arange(length)):
        order = torch.roll(order, shifts=1)
    return order.to(device)


def coordinate_shuffle_view(view: LayerView, *, seed: int, label: str) -> LayerView:
    """Shuffle row/column coordinate identities while preserving structural marginals.

    The same row and column permutations are applied to a peer's weight and optional
    calibration gradient. This preserves the exact value multiset and paired
    weight/gradient relationship, plus the row/column-energy distributions, while
    destroying alignment to the target coordinate system.
    """
    rows, cols = view.weight.shape
    row_perm = _permutation(rows, seed=seed, label=f"{label}:rows", device=view.weight.device)
    col_perm = _permutation(cols, seed=seed, label=f"{label}:cols", device=view.weight.device)

    def permute(tensor: Tensor | None) -> Tensor | None:
        if tensor is None:
            return None
        return tensor.index_select(0, row_perm).index_select(1, col_perm)

    return LayerView(
        weight=permute(view.weight),
        gradient=permute(view.gradient),
        activation=view.activation,
        metadata=view.metadata,
    )


def coordinate_shuffled_views(
    views: Sequence[LayerView], *, target_index: int, seed: int
) -> tuple[list[LayerView], dict[str, Any]]:
    transformed: list[LayerView] = []
    ledger: list[dict[str, Any]] = []
    for index, view in enumerate(views):
        if index == target_index:
            transformed.append(view)
            continue
        shuffled = coordinate_shuffle_view(
            view, seed=seed, label=f"peer:{index}:{view.metadata.name}"
        )
        transformed.append(shuffled)
        ledger.append(
            {
                "peer_index": index,
                "peer_name": view.metadata.name,
                # Row/column index_select operations are bijections, so value
                # multisets and paired weight/gradient coordinates are preserved
                # by construction without an O(N log N) full-tensor sort.
                "weight_multiset_preserved_by_bijection": True,
                "weight_gradient_pairing_preserved": view.gradient is None or shuffled.gradient is not None,
            }
        )
    if not ledger:
        raise RuntimeError("coordinate-shuffle control has no eligible peers")
    return transformed, {"kind": "coordinate_shuffle", "seed": seed, "peers": ledger}


def layer_shuffled_views(
    views: Sequence[LayerView], *, target_index: int, seed: int
) -> tuple[list[LayerView], dict[str, Any]]:
    """Move intact peer contents to incorrect depth identities via a derangement."""
    peers = [index for index in range(len(views)) if index != target_index]
    if len(peers) < 2:
        raise ValueError("layer shuffle requires at least two peers")
    shift = 1 + (stable_u64("layer-shuffle", seed, target_index) % (len(peers) - 1))
    sources = peers[shift:] + peers[:shift]
    mapping = dict(zip(peers, sources))
    if any(dst == src for dst, src in mapping.items()):
        raise RuntimeError("layer shuffle unexpectedly produced a fixed point")

    transformed: list[LayerView] = []
    for destination, destination_view in enumerate(views):
        if destination == target_index:
            transformed.append(destination_view)
            continue
        source_view = views[mapping[destination]]
        transformed.append(
            LayerView(
                weight=source_view.weight,
                gradient=source_view.gradient,
                activation=source_view.activation,
                # Preserve destination identity/depth while changing intact content.
                metadata=destination_view.metadata,
            )
        )
    return transformed, {
        "kind": "layer_identity_shuffle",
        "seed": seed,
        "mapping": {str(k): int(v) for k, v in mapping.items()},
    }


def _standardize(tensor: Tensor, eps: float = 1e-6) -> Tensor:
    value = tensor.float()
    return (value - value.mean()) / value.std(unbiased=False).clamp_min(eps)


def target_only_scores(weight: Tensor, gradient: Tensor) -> Tensor:
    """Simple matched-capacity target-only diagnostic, with no peer features."""
    grad = gradient.float()
    grad_mag = grad.abs() / grad.square().mean().add(1e-8).sqrt()
    target_elem = matrix_property_map(weight)[0].to(grad_mag.device)
    return 0.85 * _standardize(grad_mag) + 0.15 * _standardize(target_elem)


@dataclass
class CompactSupport:
    parameter_name: str
    shape: tuple[int, int]
    depth: int
    indices: Tensor
    scales: Tensor
    mask_sha256: str
    attention: list[float]
    feature_weights: list[float]

    @property
    def selected(self) -> int:
        return int(self.indices.numel())

    def manifest(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "shape": list(self.shape),
            "depth": self.depth,
            "selected": self.selected,
            "mask_sha256": self.mask_sha256,
            "indices_sha256": tensor_sha256(self.indices.to(torch.int64)),
            "scale_sha256": tensor_sha256(self.scales.float()),
            "scale_min": float(self.scales.min().item()),
            "scale_max": float(self.scales.max().item()),
            "scale_mean": float(self.scales.float().mean().item()),
            "attention": self.attention,
            "feature_weights": self.feature_weights,
        }


@dataclass
class CompactPlan:
    arm: str
    budget: int | float
    module_path: str
    supports: list[CompactSupport]
    transform: dict[str, Any]
    generator_sha256: str | None = None

    @property
    def selected(self) -> int:
        return sum(s.selected for s in self.supports)

    @property
    def count_signature(self) -> str:
        payload = [
            (support.parameter_name, support.shape, support.selected)
            for support in self.supports
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def manifest(self) -> dict[str, Any]:
        payload = {
            "arm": self.arm,
            "budget": self.budget,
            "module_path": self.module_path,
            "selected": self.selected,
            "count_signature": self.count_signature,
            "transform": self.transform,
            "generator_sha256": self.generator_sha256,
            "supports": [support.manifest() for support in self.supports],
        }
        payload["plan_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload


def mask_sha256(shape: tuple[int, int], indices: Tensor) -> str:
    mask = torch.zeros(math.prod(shape), dtype=torch.uint8)
    cpu_indices = indices.detach().cpu().to(torch.int64)
    mask.scatter_(0, cpu_indices, 1)
    return hashlib.sha256(mask.numpy().tobytes()).hexdigest()


def support_from_scores(
    *,
    parameter_name: str,
    depth: int,
    scores: Tensor,
    budget: int | float,
    scales: Tensor | float,
    attention: Tensor | None = None,
    feature_weights: Tensor | None = None,
) -> CompactSupport:
    hard_mask = exact_topk_mask(scores, budget)
    indices = hard_mask.reshape(-1).nonzero(as_tuple=False).flatten().to(torch.int64)
    if isinstance(scales, Tensor):
        selected_scales = scales.reshape(-1).index_select(0, indices).detach().float().cpu()
    else:
        selected_scales = torch.full((indices.numel(),), float(scales), dtype=torch.float32)
    indices_cpu = indices.detach().cpu()
    shape = (int(scores.shape[0]), int(scores.shape[1]))
    return CompactSupport(
        parameter_name=parameter_name,
        shape=shape,
        depth=depth,
        indices=indices_cpu,
        scales=selected_scales,
        mask_sha256=mask_sha256(shape, indices_cpu),
        attention=[] if attention is None else attention.detach().float().cpu().tolist(),
        feature_weights=[]
        if feature_weights is None
        else feature_weights.detach().float().cpu().tolist(),
    )


def random_support(
    parameter: nn.Parameter,
    *,
    parameter_name: str,
    depth: int,
    budget: int | float,
    seed: int,
    scale: float = 1.0,
) -> CompactSupport:
    generator = torch.Generator(device="cpu").manual_seed(stable_u64(seed, parameter_name) % (2**63 - 1))
    scores = torch.rand(tuple(parameter.shape), generator=generator, dtype=torch.float32)
    return support_from_scores(
        parameter_name=parameter_name,
        depth=depth,
        scores=scores,
        budget=budget,
        scales=scale,
    )


def gradient_topk_support(
    gradient: Tensor,
    *,
    parameter_name: str,
    depth: int,
    budget: int | float,
    scale: float = 1.0,
) -> CompactSupport:
    return support_from_scores(
        parameter_name=parameter_name,
        depth=depth,
        scores=gradient.abs().float(),
        budget=budget,
        scales=scale,
    )


@torch.no_grad()
def apply_first_adamw_step(
    model: nn.Module,
    plan: CompactPlan,
    gradients: dict[str, Tensor],
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Apply exactly the empty-state first SparseNativeAdamW step at selected indices."""
    if lr <= 0 or eps <= 0 or weight_decay < 0:
        raise ValueError("invalid AdamW hyperparameters")
    beta1, beta2 = betas
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("invalid AdamW betas")

    named = dict(model.named_parameters())
    audits: list[dict[str, Any]] = []
    for support in plan.supports:
        parameter = named[support.parameter_name]
        gradient = gradients[support.parameter_name].to(parameter.device, dtype=torch.float32)
        indices = support.indices.to(parameter.device)
        scale = support.scales.to(parameter.device, dtype=torch.float32)
        flat_parameter = parameter.reshape(-1)
        flat_gradient = gradient.reshape(-1)
        before = flat_parameter.index_select(0, indices).detach().clone()

        g = flat_gradient.index_select(0, indices)
        # Empty-state AdamW: m_hat=g, v_hat=g^2. PCSM scale is applied to the
        # realized update, matching SparseNativeAdamW rather than being normalized away.
        update = g / (g.abs() + eps)
        current = before.float()
        if weight_decay:
            current = current - lr * weight_decay * current * scale
        current = current - lr * update * scale
        flat_parameter.index_copy_(0, indices, current.to(parameter.dtype))
        after = flat_parameter.index_select(0, indices).detach()

        changed = int((after != before).sum().item())
        audits.append(
            {
                "parameter_name": support.parameter_name,
                "selected": support.selected,
                "changed_inside": changed,
                "changed_outside": 0,
                "max_abs_selected_delta": float(
                    (after.float() - before.float()).abs().max().item()
                )
                if support.selected
                else 0.0,
            }
        )
    return {
        "arm": plan.arm,
        "selected": plan.selected,
        "changed_outside": 0,
        "entries": audits,
    }


def support_jaccard(left: CompactPlan, right: CompactPlan) -> float:
    right_map = {support.parameter_name: support for support in right.supports}
    intersection = 0
    union = 0
    for support in left.supports:
        other = right_map[support.parameter_name]
        a = set(int(x) for x in support.indices.tolist())
        b = set(int(x) for x in other.indices.tolist())
        intersection += len(a & b)
        union += len(a | b)
    return intersection / union if union else 1.0
