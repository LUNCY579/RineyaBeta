#!/usr/bin/env python3
"""Install SGQR MLP interventions into Hugging Face causal language models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .remnant import (
    DeletedMLP,
    DynamicUtilityRouter,
    FrozenQuantizedMLPRemnant,
    NormalizedUtilityRouter,
    SGQRMLPWrapper,
)


@dataclass
class InstalledSGQR:
    layer_idx: int
    wrapper: SGQRMLPWrapper | None
    original_mlp: nn.Module | None
    mode: str

    def storage_manifest(self) -> dict[str, object]:
        if self.wrapper is None:
            return {
                "layer_idx": self.layer_idx,
                "mode": self.mode,
                "remnant_present": False,
                "packed_storage_bytes": 0,
            }
        payload = self.wrapper.remnant.storage_manifest()
        payload.update(
            {
                "layer_idx": self.layer_idx,
                "mode": self.mode,
                "remnant_present": True,
                "dense_reference_retained": self.wrapper.dense_mlp is not None,
            }
        )
        return payload


def resolve_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Resolve common Hugging Face decoder-layer locations without guessing silently."""
    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    found: list[tuple[str, nn.ModuleList]] = []
    for path in candidates:
        current: object = model
        valid = True
        for part in path:
            if not hasattr(current, part):
                valid = False
                break
            current = getattr(current, part)
        if valid and isinstance(current, nn.ModuleList):
            found.append((".".join(path), current))
    if len(found) != 1:
        labels = [name for name, _ in found]
        raise RuntimeError(
            "expected exactly one supported decoder-layer path; "
            f"found {labels or 'none'}. Add an explicit resolver for this architecture."
        )
    return found[0][1]


def _validate_layer_idx(layers: Sequence[nn.Module], layer_idx: int) -> None:
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx={layer_idx} outside [0, {len(layers)})")


def load_static_scores(path: str | Path, *, exposure_exponent: float = 1.0) -> torch.Tensor:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "significance" in payload:
        payload = payload["significance"]
    if "scores" in payload:
        values = payload["scores"]
    elif "utility_sum" in payload and "observation_count" in payload:
        utility_sum = torch.tensor(payload["utility_sum"], dtype=torch.float32)
        count = torch.tensor(payload["observation_count"], dtype=torch.float32)
        values = (utility_sum / count.clamp_min(1.0).pow(exposure_exponent)).tolist()
    else:
        raise KeyError("score file must contain scores or utility_sum + observation_count")
    return torch.tensor(values, dtype=torch.float32)


def load_router(
    path: str | Path,
    *,
    hidden_size: int,
    group_count: int,
    rank: int | None = None,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    resolved_rank = int(rank or metadata.get("rank", 32))
    router = DynamicUtilityRouter(hidden_size, group_count, resolved_rank)
    router.load_state_dict(state, strict=True)
    if isinstance(payload, dict) and all(
        key in payload
        for key in ("feature_mean", "feature_scale", "utility_mean", "utility_scale")
    ):
        return NormalizedUtilityRouter(
            router,
            feature_mean=payload["feature_mean"],
            feature_scale=payload["feature_scale"],
            utility_mean=payload["utility_mean"],
            utility_scale=payload["utility_scale"],
        )
    return router


def install_sgqr_mlp(
    model: nn.Module,
    *,
    layer_idx: int,
    mode: str,
    bits: int = 4,
    group_size: int = 128,
    budget_groups: int | None = None,
    static_scores: torch.Tensor | None = None,
    gains: Sequence[float] | None = None,
    dynamic_router: nn.Module | None = None,
    retain_dense_reference: bool = False,
    random_seed: int = 20260824,
) -> InstalledSGQR:
    """Replace one decoder MLP with an SGQR intervention.

    ``dense`` and ``fp_frozen`` leave the original module in place. ``deleted``
    installs a zero branch and stores no quantized remnant. Other modes construct
    a frozen quantized remnant from the original weights. By default the original
    dense MLP is not retained, preventing a hidden deployment copy.
    """
    layers = resolve_decoder_layers(model)
    _validate_layer_idx(layers, layer_idx)
    layer = layers[layer_idx]
    if not hasattr(layer, "mlp"):
        raise TypeError(f"decoder layer {layer_idx} has no mlp attribute")
    original_mlp = layer.mlp

    if mode == "dense":
        return InstalledSGQR(layer_idx, None, original_mlp, mode)

    if mode == "fp_frozen":
        for parameter in original_mlp.parameters():
            parameter.requires_grad_(False)
        return InstalledSGQR(layer_idx, None, original_mlp, mode)

    if mode == "deleted":
        layer.mlp = DeletedMLP()
        return InstalledSGQR(layer_idx, None, original_mlp, mode)

    remnant = FrozenQuantizedMLPRemnant.from_dense_mlp(
        original_mlp,
        bits=bits,
        group_size=group_size,
        gains=gains,
    )
    dense_reference = original_mlp if retain_dense_reference else None
    wrapper = SGQRMLPWrapper(
        remnant=remnant,
        dense_mlp=dense_reference,
        mode=mode,
        budget_groups=budget_groups,
        static_scores=static_scores,
        router=dynamic_router,
        random_seed=random_seed,
    )
    layer.mlp = wrapper
    return InstalledSGQR(layer_idx, wrapper, original_mlp, mode)


def restore_original_mlp(model: nn.Module, installation: InstalledSGQR) -> None:
    if installation.original_mlp is None:
        raise RuntimeError("installation does not carry the original MLP")
    layers = resolve_decoder_layers(model)
    _validate_layer_idx(layers, installation.layer_idx)
    layers[installation.layer_idx].mlp = installation.original_mlp


def save_deployment_artifact(
    installation: InstalledSGQR,
    out_dir: str | Path,
    *,
    model_identifier: str,
    config: dict[str, object],
) -> Path:
    """Serialize only SGQR-specific state and metadata.

    The base model remains separately identified by ``model_identifier``. A
    deployment loader reconstructs the base model and replaces the target MLP;
    the dense donor is never serialized in this artifact.
    """
    if installation.wrapper is None:
        raise ValueError("only quantized SGQR installations have a deployment artifact")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "sgqr_state.pt"
    torch.save(installation.wrapper.state_dict(), state_path)
    manifest = {
        "format": "sgqr-reference-v1",
        "model_identifier": model_identifier,
        "layer_idx": installation.layer_idx,
        "mode": installation.wrapper.mode,
        "budget_groups": installation.wrapper.budget_groups,
        "storage": installation.storage_manifest(),
        "config": config,
        "state_file": state_path.name,
    }
    (out / "sgqr_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out
