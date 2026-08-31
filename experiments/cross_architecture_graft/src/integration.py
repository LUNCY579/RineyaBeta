"""Framework-neutral and Hugging Face-oriented graft attachment helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

from .graft import GraftBranch, GraftedDecoderLayer, configure_graft_training


@dataclass(frozen=True)
class Attachment:
    layer_index: int
    original_class: str
    graft_class: str


COMMON_LAYER_PATHS = (
    "model.layers",          # Qwen/Llama/DeepSeek-style causal LMs
    "model.model.layers",    # wrappers around a causal-LM body
    "transformer.h",         # GPT-style
    "layers",                # minimal/custom models
)


def resolve_module(root: nn.Module, dotted_path: str) -> Any:
    current: Any = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            raise AttributeError(f"module path {dotted_path!r} failed at {part!r}")
        current = getattr(current, part)
    return current


def find_layer_container(
    model: nn.Module,
    explicit_path: str | None = None,
) -> tuple[str, nn.ModuleList]:
    candidates = (explicit_path,) if explicit_path else COMMON_LAYER_PATHS
    failures: list[str] = []
    for path in candidates:
        if path is None:
            continue
        try:
            value = resolve_module(model, path)
        except AttributeError as exc:
            failures.append(str(exc))
            continue
        if isinstance(value, nn.ModuleList):
            return path, value
        failures.append(f"{path!r} resolved to {type(value).__name__}, not ModuleList")
    joined = "; ".join(failures)
    raise ValueError(f"could not locate decoder layer ModuleList: {joined}")


def attach_grafts(
    model: nn.Module,
    layer_indices: Iterable[int],
    branch_factory: Callable[[int, nn.Module], GraftBranch],
    *,
    layer_path: str | None = None,
    freeze_host: bool = True,
    freeze_donor: bool = True,
    exact_eval_bypass: bool = True,
) -> list[Attachment]:
    """Replace selected decoder layers with function-preserving wrappers."""

    _, layers = find_layer_container(model, layer_path)
    indices = [int(index) for index in layer_indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("layer_indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= len(layers):
        raise IndexError(
            f"layer indices {indices} outside decoder range [0, {len(layers) - 1}]"
        )

    attachments: list[Attachment] = []
    for index in indices:
        original = layers[index]
        if isinstance(original, GraftedDecoderLayer):
            raise ValueError(f"layer {index} is already grafted")
        branch = branch_factory(index, original)
        wrapper = GraftedDecoderLayer(
            original,
            branch,
            exact_eval_bypass=exact_eval_bypass,
        )
        configure_graft_training(
            wrapper,
            freeze_host=freeze_host,
            freeze_donor=freeze_donor,
        )
        layers[index] = wrapper
        attachments.append(
            Attachment(
                layer_index=index,
                original_class=original.__class__.__name__,
                graft_class=branch.donor.__class__.__name__,
            )
        )
    return attachments


def iter_grafted_layers(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, GraftedDecoderLayer):
            yield name, module


def graft_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, layer in iter_grafted_layers(model):
        for key, value in layer.graft_branch.state_dict().items():
            state[f"{name}.graft_branch.{key}"] = value.detach().cpu()
    if not state:
        raise ValueError("model contains no grafted layers")
    return state


def save_graft_pack(
    model: nn.Module,
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save only graft parameters plus provenance; never duplicate host weights."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "graft_state.pt"
    metadata_path = destination / "graft_manifest.json"
    tmp_weights = destination / ".graft_state.pt.tmp"
    tmp_metadata = destination / ".graft_manifest.json.tmp"
    torch.save(graft_state_dict(model), tmp_weights)
    tmp_weights.replace(weights_path)
    tmp_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_metadata.replace(metadata_path)
    return weights_path, metadata_path


def load_graft_pack(
    model: nn.Module,
    weights_path: str | Path,
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a graft pack into an already-attached architecture."""

    raw = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("graft state must be a dictionary")
    current = model.state_dict()
    unexpected = sorted(set(raw) - set(current))
    missing = sorted(
        key
        for name, _ in iter_grafted_layers(model)
        for key in current
        if key.startswith(f"{name}.graft_branch.") and key not in raw
    )
    if strict and (unexpected or missing):
        raise RuntimeError(
            f"graft state mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    compatible = {key: value for key, value in raw.items() if key in current}
    model.load_state_dict(compatible, strict=False)
    return missing, unexpected


def model_signature(model: nn.Module, layer_path: str | None = None) -> dict[str, Any]:
    path, layers = find_layer_container(model, layer_path)
    config = getattr(model, "config", None)
    return {
        "model_class": model.__class__.__name__,
        "config_class": config.__class__.__name__ if config is not None else None,
        "layer_path": path,
        "num_layers": len(layers),
        "layer_classes": [layer.__class__.__name__ for layer in layers],
        "hidden_size": getattr(config, "hidden_size", None),
        "model_type": getattr(config, "model_type", None),
    }
