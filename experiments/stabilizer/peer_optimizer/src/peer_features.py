#!/usr/bin/env python3
"""Layer descriptors and structural property maps used by PCSM."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

DESCRIPTOR_DIM = 20
PROPERTY_CHANNELS = 4
LOCAL_FEATURES = 8


@dataclass(frozen=True)
class LayerMetadata:
    """Identity used to decide which layers are semantically compatible peers."""

    name: str
    depth: int
    module_type: str = "linear"


@dataclass
class LayerView:
    """A matrix plus optional calibration behavior.

    ``activation`` and ``gradient`` are summaries from a calibration batch, not
    required persistent tensors.  ``weight`` must be a two-dimensional native
    parameter matrix.  The caller should pass detached peer tensors during normal
    sparse fine-tuning; ``PeerStabilizerMatrix(detach_peers=True)`` enforces this
    by default.
    """

    weight: Tensor
    metadata: LayerMetadata
    activation: Tensor | None = None
    gradient: Tensor | None = None

    def validate(self) -> None:
        if self.weight.ndim != 2:
            raise ValueError(
                f"{self.metadata.name}: expected a 2-D weight matrix, "
                f"got shape {tuple(self.weight.shape)}"
            )
        if self.gradient is not None and self.gradient.shape != self.weight.shape:
            raise ValueError(
                f"{self.metadata.name}: gradient shape {tuple(self.gradient.shape)} "
                f"does not match weight shape {tuple(self.weight.shape)}"
            )


@dataclass
class StabilizerOutput:
    """All generated objects needed for training and scientific auditing."""

    scores: Tensor
    hard_mask: Tensor
    training_mask: Tensor
    step_scale: Tensor
    attention: Tensor
    peer_properties: Tensor
    peer_template: Tensor
    feature_weights: Tensor
    eligible_peers: Tensor
    target_descriptor: Tensor
    context: Tensor

    @property
    def selected_count(self) -> int:
        return int(self.hard_mask.detach().sum().item())


@dataclass(frozen=True)
class SupportAudit:
    """Audit of a realized native-weight delta against its declared support."""

    selected: int
    changed_total: int
    changed_inside: int
    changed_outside: int
    max_abs_outside_delta: float
    support_union_fraction: float

    @property
    def exact(self) -> bool:
        return self.changed_outside == 0


def _as_float_tensor(x: Tensor) -> Tensor:
    # Descriptor and scoring arithmetic is intentionally fp32 even when the model
    # is bf16/fp16.  Generated masks/scales are cast back by the caller/optimizer.
    return x if x.dtype == torch.float32 else x.float()


def _rms(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x.square().mean().add(eps).sqrt()


def normalize_matrix(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Scale-normalize a matrix without assuming equal magnitudes across layers."""

    x = _as_float_tensor(x)
    centered = x - x.mean()
    return centered / _rms(centered, eps)


def _normalized_summary(x: Tensor | None, eps: float = 1e-8) -> tuple[Tensor, Tensor, Tensor]:
    """Return signed mean, standard deviation, and absolute mean in RMS units."""

    if x is None:
        # Device is corrected by ``matrix_descriptor`` before concatenation.
        z = torch.tensor(0.0)
        return z, z, z
    xf = _as_float_tensor(x)
    scale = _rms(xf, eps)
    return xf.mean() / scale, xf.std(unbiased=False) / scale, xf.abs().mean() / scale


def matrix_descriptor(view: LayerView, total_layers: int, eps: float = 1e-8) -> Tensor:
    """Build a compact, differentiable 20-value description of one layer.

    The descriptor records scale-free weight geometry plus optional activation and
    gradient behavior.  It is deliberately a summary: the peer-attention network
    should reason over layer properties rather than ingest every raw parameter.
    Coordinate-level peer information is supplied separately through bounded
    property maps.
    """

    view.validate()
    w = _as_float_tensor(view.weight)
    scale = _rms(w, eps)
    wn = w / scale

    row_rms = w.square().mean(dim=1).add(eps).sqrt()
    col_rms = w.square().mean(dim=0).add(eps).sqrt()
    row_mean = row_rms.mean().clamp_min(eps)
    col_mean = col_rms.mean().clamp_min(eps)

    act_mean, act_std, act_abs = _normalized_summary(view.activation, eps)
    grad_mean, grad_std, grad_abs = _normalized_summary(view.gradient, eps)
    act_mean = act_mean.to(w.device)
    act_std = act_std.to(w.device)
    act_abs = act_abs.to(w.device)
    grad_mean = grad_mean.to(w.device)
    grad_std = grad_std.to(w.device)
    grad_abs = grad_abs.to(w.device)

    depth_den = max(1, total_layers - 1)
    depth = torch.tensor(float(view.metadata.depth) / depth_den, device=w.device)
    rows = torch.tensor(math.log1p(w.shape[0]) / 12.0, device=w.device)
    cols = torch.tensor(math.log1p(w.shape[1]) / 12.0, device=w.device)

    descriptor = torch.stack(
        [
            w.mean() / scale,
            w.std(unbiased=False) / scale,
            w.abs().mean() / scale,
            w.abs().amax() / scale,
            row_rms.mean() / scale,
            row_rms.std(unbiased=False) / row_mean,
            col_rms.mean() / scale,
            col_rms.std(unbiased=False) / col_mean,
            (wn > 0).float().mean(),
            (wn.abs() > 1.0).float().mean(),
            (wn.abs() > 2.0).float().mean(),
            rows,
            cols,
            act_mean,
            act_std,
            act_abs,
            grad_mean,
            grad_std,
            grad_abs,
            depth,
        ]
    )
    if descriptor.numel() != DESCRIPTOR_DIM:
        raise RuntimeError(f"descriptor contract changed: {descriptor.numel()} != {DESCRIPTOR_DIM}")
    return descriptor


def matrix_property_map(weight: Tensor, eps: float = 1e-8) -> Tensor:
    """Return four non-negative coordinate maps describing matrix structure.

    Channels are:

    0. element magnitude in centered RMS units;
    1. normalized row energy broadcast over columns;
    2. normalized column energy broadcast over rows;
    3. row-column joint leverage proxy.

    These are properties, not copied signed weights.  That makes peer advice less
    sensitive to arbitrary scale and reduces the risk of forcing different layers
    to become numerically identical.
    """

    w = _as_float_tensor(weight)
    if w.ndim != 2:
        raise ValueError(f"property map expects a 2-D matrix, got {tuple(w.shape)}")
    m, n = w.shape
    elem = normalize_matrix(w, eps).abs()

    row = w.square().mean(dim=1).add(eps).sqrt()
    row = row / row.mean().clamp_min(eps)
    row_map = row[:, None].expand(m, n)

    col = w.square().mean(dim=0).add(eps).sqrt()
    col = col / col.mean().clamp_min(eps)
    col_map = col[None, :].expand(m, n)

    joint = row_map * col_map
    joint = joint / joint.mean().clamp_min(eps)
    return torch.stack([elem, row_map, col_map, joint], dim=0)


def project_property_map(properties: Tensor, target_shape: tuple[int, int]) -> Tensor:
    """Project peer property channels to a target matrix shape.

    Equal shapes are passed through exactly.  Different shapes use bilinear
    interpolation over normalized structural maps.  This is a coarse structural
    bridge, not a claim that unlike modules share one-to-one neuron identities.
    """

    if properties.ndim != 3:
        raise ValueError(
            f"properties must have [channels, rows, cols], got {tuple(properties.shape)}"
        )
    if properties.shape[-2:] == target_shape:
        return properties
    return F.interpolate(
        properties.unsqueeze(0),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
