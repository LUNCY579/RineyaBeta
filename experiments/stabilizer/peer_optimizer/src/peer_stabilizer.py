#!/usr/bin/env python3
"""Peer-conditioned stabilizer matrices for sparse native-weight fine-tuning.

Other transformer layers generate an exact support mask and a bounded step-scale
matrix for a target native weight matrix.  The task gradient retains control of
update direction:

    W_t' = W_t - eta * M_t * A_t * grad(L, W_t)

The implementation is split into ``peer_features`` (descriptors/properties),
``sparse_selection`` (exact masks/audits), and this peer-conditioning module.
Public objects are re-exported here for a stable research API.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:  # package import
    from .peer_features import (
        DESCRIPTOR_DIM,
        LOCAL_FEATURES,
        PROPERTY_CHANNELS,
        LayerMetadata,
        LayerView,
        StabilizerOutput,
        SupportAudit,
        _as_float_tensor,
        _rms,
        matrix_descriptor,
        matrix_property_map,
        normalize_matrix,
        project_property_map,
    )
    from .sparse_selection import (
        audit_sparse_delta,
        exact_topk_mask,
        resolve_budget,
        straight_through_topk,
    )
except ImportError:  # direct script/module import
    from peer_features import (
        DESCRIPTOR_DIM,
        LOCAL_FEATURES,
        PROPERTY_CHANNELS,
        LayerMetadata,
        LayerView,
        StabilizerOutput,
        SupportAudit,
        _as_float_tensor,
        _rms,
        matrix_descriptor,
        matrix_property_map,
        normalize_matrix,
        project_property_map,
    )
    from sparse_selection import (
        audit_sparse_delta,
        exact_topk_mask,
        resolve_budget,
        straight_through_topk,
    )


def _standardize_map(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(eps)


class PeerStabilizerMatrix(nn.Module):
    """Generate peer-conditioned sparse-plasticity matrices for one target layer."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        same_module_only: bool = True,
        allow_peer_fallback: bool = False,
        detach_peers: bool = True,
        cosine_temperature: float = 0.75,
        min_step_scale: float = 0.20,
        max_step_scale: float = 1.20,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if cosine_temperature <= 0:
            raise ValueError("cosine_temperature must be positive")
        if not (0.0 < min_step_scale <= max_step_scale):
            raise ValueError("step-scale bounds must satisfy 0 < min <= max")

        self.same_module_only = same_module_only
        self.allow_peer_fallback = allow_peer_fallback
        self.detach_peers = detach_peers
        self.cosine_temperature = cosine_temperature
        self.min_step_scale = min_step_scale
        self.max_step_scale = max_step_scale

        self.descriptor_encoder = nn.Sequential(
            nn.LayerNorm(DESCRIPTOR_DIM),
            nn.Linear(DESCRIPTOR_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Learned pairwise attention begins as a small residual around transparent
        # descriptor-cosine attention, rather than a random routing policy.
        self.learned_mix_logit = nn.Parameter(torch.tensor(-4.0))
        self.raw_depth_penalty = nn.Parameter(torch.tensor(-2.0))

        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, LOCAL_FEATURES),
        )
        self.step_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # A scientifically useful bootstrap before meta-calibration: task gradient
        # and peer structural evidence dominate; target magnitude is weak.  The
        # final layers are zero-initialized so these biases are the exact initial
        # policy and runs do not begin from arbitrary support masks.
        feature_bias = torch.tensor([1.25, 1.10, 0.55, 0.55, 0.85, 0.15, 0.65, 0.90])
        final_feature = self.feature_head[-1]
        nn.init.zeros_(final_feature.weight)
        with torch.no_grad():
            final_feature.bias.copy_(feature_bias)

        final_step = self.step_head[-1]
        nn.init.zeros_(final_step.weight)
        with torch.no_grad():
            final_step.bias.copy_(torch.tensor([0.10, 0.55, 0.30]))

    def _eligible_mask(self, layers: Sequence[LayerView], target_index: int) -> Tensor:
        target = layers[target_index]
        device = target.weight.device
        eligible = torch.zeros(len(layers), dtype=torch.bool, device=device)
        for i, peer in enumerate(layers):
            if i == target_index:
                continue
            if not self.same_module_only or peer.metadata.module_type == target.metadata.module_type:
                eligible[i] = True
        if not bool(eligible.any()) and self.allow_peer_fallback:
            eligible[:] = True
            eligible[target_index] = False
        if not bool(eligible.any()):
            raise ValueError(
                f"target {target.metadata.name!r} has no eligible peer; "
                "supply another layer with the same module_type, disable "
                "same_module_only, or explicitly allow fallback"
            )
        return eligible

    def _peer_attention(
        self,
        descriptors: Tensor,
        encoded: Tensor,
        layers: Sequence[LayerView],
        target_index: int,
        eligible: Tensor,
    ) -> tuple[Tensor, Tensor]:
        target_z = encoded[target_index]
        base = F.cosine_similarity(
            descriptors[target_index].unsqueeze(0), descriptors, dim=-1
        ) / self.cosine_temperature
        learned = (self.query(target_z) * self.key(encoded)).sum(dim=-1) / math.sqrt(
            encoded.shape[-1]
        )
        learned_mix = torch.sigmoid(self.learned_mix_logit)

        depth = torch.tensor(
            [float(v.metadata.depth) for v in layers],
            device=encoded.device,
            dtype=encoded.dtype,
        )
        depth_den = max(1.0, float(depth.max().item() - depth.min().item()))
        distance = (depth - depth[target_index]).abs() / depth_den
        depth_penalty = F.softplus(self.raw_depth_penalty) * distance

        logits = base + learned_mix * learned - depth_penalty
        logits = logits.masked_fill(~eligible, -torch.inf)
        attention = torch.softmax(logits, dim=0)
        # Explicitly force excluded entries to exact zero and renormalize.  This
        # avoids tiny numerical values undermining support/ablation audits.
        attention = attention * eligible.to(attention.dtype)
        attention = attention / attention.sum().clamp_min(1e-12)
        context = (attention[:, None] * self.value(encoded)).sum(dim=0)
        return attention, context

    def forward(
        self,
        layers: Sequence[LayerView],
        target_index: int,
        budget: int | float,
        *,
        task_gradient: Tensor | None = None,
        straight_through: bool = True,
        mask_temperature: float = 0.25,
    ) -> StabilizerOutput:
        if len(layers) < 2:
            raise ValueError("at least two layers are required for peer conditioning")
        if not (0 <= target_index < len(layers)):
            raise IndexError(f"target_index {target_index} outside [0, {len(layers)})")
        for layer in layers:
            layer.validate()

        target = layers[target_index]
        gradient = task_gradient if task_gradient is not None else target.gradient
        if gradient is None:
            raise ValueError("task_gradient is required, either explicitly or on the target LayerView")
        if gradient.shape != target.weight.shape:
            raise ValueError(
                f"task gradient shape {tuple(gradient.shape)} does not match target "
                f"shape {tuple(target.weight.shape)}"
            )

        device = target.weight.device
        dtype = torch.float32
        descriptors = torch.stack(
            [matrix_descriptor(v, len(layers)).to(device=device, dtype=dtype) for v in layers]
        )
        encoded = self.descriptor_encoder(descriptors)
        eligible = self._eligible_mask(layers, target_index)
        attention, context = self._peer_attention(
            descriptors, encoded, layers, target_index, eligible
        )

        target_shape = tuple(target.weight.shape)
        peer_properties = torch.zeros(
            PROPERTY_CHANNELS, *target_shape, device=device, dtype=dtype
        )
        for i, peer in enumerate(layers):
            if not bool(eligible[i]):
                continue
            pw = peer.weight.detach() if self.detach_peers else peer.weight
            props = matrix_property_map(pw).to(device=device, dtype=dtype)
            peer_properties = peer_properties + attention[i] * project_property_map(
                props, target_shape
            )

        target_properties = matrix_property_map(target.weight).to(device=device, dtype=dtype)
        peer_template = peer_properties.mean(dim=0)
        target_template = target_properties.mean(dim=0)
        grad = _as_float_tensor(gradient).to(device)
        grad_mag = grad.abs() / _rms(grad)

        # Each feature map is standardized before mixing, so learned coefficients
        # express evidential preference rather than compensating arbitrary units.
        local_features = torch.stack(
            [
                grad_mag,
                peer_properties[0],
                peer_properties[1],
                peer_properties[2],
                peer_properties[3],
                target_properties[0],
                (peer_template - target_template).abs(),
                grad_mag * peer_template,
            ],
            dim=0,
        )
        local_features = torch.stack([_standardize_map(f) for f in local_features], dim=0)

        target_z = encoded[target_index]
        joint_context = torch.cat([target_z, context], dim=0)
        feature_weights = torch.softmax(self.feature_head(joint_context), dim=0)
        scores = (feature_weights[:, None, None] * local_features).sum(dim=0)

        if straight_through:
            hard_mask, training_mask = straight_through_topk(
                scores, budget, temperature=mask_temperature
            )
        else:
            hard_mask = exact_topk_mask(scores, budget)
            training_mask = hard_mask

        step_parameters = self.step_head(joint_context)
        global_bias = step_parameters[0]
        score_slope = F.softplus(step_parameters[1])
        peer_slope = F.softplus(step_parameters[2])
        scale_logits = (
            global_bias
            + score_slope * _standardize_map(scores)
            + peer_slope * _standardize_map(peer_template)
        )
        step_scale = self.min_step_scale + (
            self.max_step_scale - self.min_step_scale
        ) * torch.sigmoid(scale_logits)

        return StabilizerOutput(
            scores=scores,
            hard_mask=hard_mask,
            training_mask=training_mask,
            step_scale=step_scale,
            attention=attention,
            peer_properties=peer_properties,
            peer_template=peer_template,
            feature_weights=feature_weights,
            eligible_peers=eligible,
            target_descriptor=descriptors[target_index],
            context=context,
        )


def effective_gradient(gradient: Tensor, output: StabilizerOutput, *, training: bool = True) -> Tensor:
    """Apply generated support and step scales without changing gradient direction."""

    if gradient.shape != output.hard_mask.shape:
        raise ValueError(
            f"gradient shape {tuple(gradient.shape)} != mask shape {tuple(output.hard_mask.shape)}"
        )
    mask = output.training_mask if training else output.hard_mask
    return gradient * mask.to(gradient.dtype) * output.step_scale.to(gradient.dtype)


def one_step_weight(
    weight: Tensor,
    gradient: Tensor,
    output: StabilizerOutput,
    learning_rate: float,
    *,
    training: bool = True,
) -> Tensor:
    """Functional one-step update used by outer-loop calibration and tests."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return weight - learning_rate * effective_gradient(gradient, output, training=training)


def structural_profile_loss(
    updated_weight: Tensor,
    peer_properties: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Regularize row/column energy profiles without matching raw peer weights.

    This optional term is safer than ``||W_t - W_j||``: it asks a target layer to
    remain structurally compatible with peers while preserving its distinct signed
    computation.
    """

    if updated_weight.ndim != 2:
        raise ValueError("updated_weight must be two-dimensional")
    if peer_properties.shape[0] != PROPERTY_CHANNELS:
        raise ValueError(
            f"peer_properties needs {PROPERTY_CHANNELS} channels, got {peer_properties.shape[0]}"
        )
    if peer_properties.shape[-2:] != updated_weight.shape:
        raise ValueError("peer property shape does not match updated weight")

    w = _as_float_tensor(updated_weight)
    row = w.square().mean(dim=1).add(eps).sqrt()
    row = row / row.mean().clamp_min(eps)
    col = w.square().mean(dim=0).add(eps).sqrt()
    col = col / col.mean().clamp_min(eps)

    peer_row = peer_properties[1].mean(dim=1)
    peer_row = peer_row / peer_row.mean().clamp_min(eps)
    peer_col = peer_properties[2].mean(dim=0)
    peer_col = peer_col / peer_col.mean().clamp_min(eps)
    return F.mse_loss(row, peer_row) + F.mse_loss(col, peer_col)


def detach_layer_views(layers: Iterable[LayerView]) -> list[LayerView]:
    """Create an explicitly detached snapshot for mask generation/calibration."""

    detached: list[LayerView] = []
    for layer in layers:
        detached.append(
            LayerView(
                weight=layer.weight.detach(),
                metadata=layer.metadata,
                activation=None if layer.activation is None else layer.activation.detach(),
                gradient=None if layer.gradient is None else layer.gradient.detach(),
            )
        )
    return detached
