"""Peer-conditioned sparse native-weight fine-tuning prototype."""

from .peer_stabilizer import (
    LayerMetadata,
    LayerView,
    PeerStabilizerMatrix,
    StabilizerOutput,
    SupportAudit,
    audit_sparse_delta,
    detach_layer_views,
    effective_gradient,
    exact_topk_mask,
    matrix_descriptor,
    matrix_property_map,
    one_step_weight,
    project_property_map,
    resolve_budget,
    straight_through_topk,
    structural_profile_loss,
)
from .sparse_optim import SparseNativeAdamW, SparseParameterSpec

__all__ = [
    "LayerMetadata",
    "LayerView",
    "PeerStabilizerMatrix",
    "StabilizerOutput",
    "SupportAudit",
    "SparseNativeAdamW",
    "SparseParameterSpec",
    "audit_sparse_delta",
    "detach_layer_views",
    "effective_gradient",
    "exact_topk_mask",
    "matrix_descriptor",
    "matrix_property_map",
    "one_step_weight",
    "project_property_map",
    "resolve_budget",
    "straight_through_topk",
    "structural_profile_loss",
]
