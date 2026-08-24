"""Optional Hugging Face/Qwen integration.

This module has no import-time dependency on Transformers, so CPU contract tests
remain runnable in a minimal environment.
"""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .surgery import GatedModule, compare_logits, count_parameters, deployment_state_is_clean


_LAYER_PATHS = (
    "model.layers",
    "model.model.layers",
    "transformer.h",
    "gpt_neox.layers",
)


def resolve_attr(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(path)
        current = getattr(current, part)
    return current


def assign_attr(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def resolve_decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    for path in _LAYER_PATHS:
        try:
            value = resolve_attr(model, path)
        except AttributeError:
            continue
        if isinstance(value, nn.ModuleList):
            return path, value
    candidates = [name for name, module in model.named_modules()
                  if isinstance(module, nn.ModuleList)]
    raise ValueError(
        "could not locate decoder ModuleList; checked " + ", ".join(_LAYER_PATHS)
        + f"; available ModuleLists: {candidates[:12]}")


def unwrap_gate(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, GatedModule) else module


def install_layer_gate(model: nn.Module, layer_index: int, gate: float = 1.0) -> GatedModule:
    _, layers = resolve_decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise IndexError(layer_index)
    existing = layers[layer_index]
    if isinstance(existing, GatedModule):
        existing.set_gate(gate)
        return existing
    wrapped = GatedModule(existing, gate)
    layers[layer_index] = wrapped
    return wrapped


def set_layer_gate(model: nn.Module, layer_index: int, gate: float) -> None:
    _, layers = resolve_decoder_layers(model)
    module = layers[layer_index]
    if not isinstance(module, GatedModule):
        raise ValueError("target layer is not gated; call install_layer_gate first")
    module.set_gate(gate)


@dataclass
class LayerTrace:
    inputs: dict[int, Tensor]
    residuals: dict[int, Tensor]


class TraceCollector(AbstractContextManager["TraceCollector"]):
    """Capture block inputs and residual deltas without retaining unrelated layers."""

    def __init__(self, model: nn.Module, layers: Iterable[int], *, detach: bool = False) -> None:
        self.model = model
        self.layer_ids = tuple(sorted(set(int(i) for i in layers)))
        self.detach = detach
        self.inputs: dict[int, Tensor] = {}
        self.residuals: dict[int, Tensor] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "TraceCollector":
        _, modules = resolve_decoder_layers(self.model)
        for index in self.layer_ids:
            if not 0 <= index < len(modules):
                raise IndexError(index)
            module = modules[index]

            def hook(_module: nn.Module, args: tuple[Any, ...], output: Any,
                     *, layer: int = index) -> None:
                if not args or not isinstance(args[0], Tensor):
                    raise RuntimeError(f"layer {layer} did not receive hidden states positionally")
                before = args[0]
                after = output[0] if isinstance(output, tuple) else output
                if not isinstance(after, Tensor):
                    raise RuntimeError(f"layer {layer} returned unsupported output")
                residual = after - before
                self.inputs[layer] = before.detach() if self.detach else before
                self.residuals[layer] = residual.detach() if self.detach else residual

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return None

    def trace(self) -> LayerTrace:
        missing = set(self.layer_ids) - self.residuals.keys()
        if missing:
            raise RuntimeError(f"requested layers were not executed: {sorted(missing)}")
        return LayerTrace(inputs=dict(self.inputs), residuals=dict(self.residuals))


def physical_delete_layer(model: nn.Module, target_index: int, *, in_place: bool = False) -> tuple[nn.Module, int]:
    compact = model if in_place else copy.deepcopy(model)
    path, layers = resolve_decoder_layers(compact)
    if not 0 <= target_index < len(layers):
        raise IndexError(target_index)
    removed_module = unwrap_gate(layers[target_index])
    removed_params = count_parameters(removed_module)
    new_layers = nn.ModuleList([
        unwrap_gate(layer) for index, layer in enumerate(layers) if index != target_index
    ])
    assign_attr(compact, path, new_layers)
    config = getattr(compact, "config", None)
    if config is not None:
        for field in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(config, field):
                setattr(config, field, len(new_layers))
        if hasattr(config, "max_window_layers"):
            config.max_window_layers = min(int(config.max_window_layers), len(new_layers))
    return compact, removed_params


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if name in names}


def load_partial_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    current = model.state_dict()
    unknown = set(state) - current.keys()
    if unknown:
        raise ValueError(f"checkpoint contains unknown tensors: {sorted(unknown)[:8]}")
    with torch.no_grad():
        for name, tensor in state.items():
            if current[name].shape != tensor.shape:
                raise ValueError(f"shape mismatch for {name}: {tensor.shape} vs {current[name].shape}")
            current[name].copy_(tensor.to(current[name]))


def recipient_layers(source: int, layer_count: int, *, radius: int = 2,
                     max_count: int = 3) -> list[int]:
    candidates = [index for index in range(layer_count)
                  if index != source and abs(index - source) <= radius]
    candidates.sort(key=lambda index: (abs(index - source), index))
    return sorted(candidates[:max_count])


def freeze_except_recipients(model: nn.Module, recipients: Sequence[int]) -> list[nn.Parameter]:
    _, layers = resolve_decoder_layers(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params: list[nn.Parameter] = []
    for index in recipients:
        module = unwrap_gate(layers[index])
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            params.append(parameter)
    if not params:
        raise ValueError("recipient selection produced no trainable parameters")
    return params


def scar_free_check(
    gated_model: nn.Module,
    sample: Mapping[str, Tensor],
    target_layer: int,
    *,
    atol: float = 1e-5,
) -> tuple[nn.Module, dict[str, Any]]:
    install_layer_gate(gated_model, target_layer, 0.0)
    gated_model.eval()
    with torch.inference_mode():
        gated_logits = gated_model(**sample).logits
    compact, removed = physical_delete_layer(gated_model, target_layer)
    compact.eval()
    with torch.inference_mode():
        physical_logits = compact(**sample).logits
    max_abs, greedy_equal = compare_logits(gated_logits, physical_logits, atol=atol)
    source_count = count_parameters(gated_model)
    compact_count = count_parameters(compact)
    report = {
        "max_abs_logit_difference": max_abs,
        "greedy_equal": greedy_equal,
        "source_parameter_count": source_count,
        "compact_parameter_count": compact_count,
        "expected_removed_parameters": removed,
        "actual_removed_parameters": source_count - compact_count,
        "deployment_state_clean": deployment_state_is_clean(compact),
        "passed": (
            max_abs <= atol and greedy_equal
            and source_count - compact_count == removed
            and deployment_state_is_clean(compact)
        ),
    }
    return compact, report


class RecipientProbe(nn.Module):
    """Measure a recipient's acquired residual on the exact current input.

    The trainable module and a frozen anchor copy receive identical hidden states
    and decoder kwargs. Their output difference is therefore attributable to the
    recipient's parameter change rather than upstream state drift. Recovery runs
    must use ``use_cache=False`` so the local anchor call cannot mutate a cache.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.anchor = copy.deepcopy(module).eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.capture_anchor = True
        self.last_input: Tensor | None = None
        self.last_output: Tensor | None = None
        self.last_anchor_output: Tensor | None = None
        self.last_delta: Tensor | None = None

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("use_cache"):
            raise RuntimeError("PDRR recipient probes require use_cache=False")
        output = self.module(hidden_states, *args, **kwargs)
        current = output[0] if isinstance(output, tuple) else output
        if not isinstance(current, Tensor):
            raise TypeError("recipient layer returned unsupported output")
        self.last_input = hidden_states
        self.last_output = current
        if self.capture_anchor:
            detached_args = tuple(value.detach() if isinstance(value, Tensor) else value
                                  for value in args)
            detached_kwargs = {
                key: (value.detach() if isinstance(value, Tensor) else value)
                for key, value in kwargs.items()
            }
            with torch.no_grad():
                anchor_output = self.anchor(hidden_states.detach(), *detached_args,
                                            **detached_kwargs)
            anchor_hidden = anchor_output[0] if isinstance(anchor_output, tuple) else anchor_output
            if not isinstance(anchor_hidden, Tensor):
                raise TypeError("recipient anchor returned unsupported output")
            self.last_anchor_output = anchor_hidden
            self.last_delta = current - anchor_hidden.detach()
        else:
            self.last_anchor_output = None
            self.last_delta = None
        return output


def install_recipient_probes(model: nn.Module, recipients: Sequence[int]) -> dict[int, RecipientProbe]:
    _, layers = resolve_decoder_layers(model)
    probes: dict[int, RecipientProbe] = {}
    for index in sorted(set(int(value) for value in recipients)):
        if not 0 <= index < len(layers):
            raise IndexError(index)
        if isinstance(layers[index], RecipientProbe):
            probes[index] = layers[index]
            continue
        probe = RecipientProbe(layers[index])
        layers[index] = probe
        probes[index] = probe
    return probes


def recipient_deltas(probes: Mapping[int, RecipientProbe]) -> list[Tensor]:
    missing = [index for index, probe in probes.items() if probe.last_delta is None]
    if missing:
        raise RuntimeError(f"recipient deltas unavailable for layers {missing}")
    return [probes[index].last_delta for index in sorted(probes)]  # type: ignore[list-item]


def set_probe_capture(probes: Mapping[int, RecipientProbe], enabled: bool) -> None:
    for probe in probes.values():
        probe.capture_anchor = bool(enabled)


def remove_recipient_probes(model: nn.Module, probes: Mapping[int, RecipientProbe]) -> None:
    _, layers = resolve_decoder_layers(model)
    for index, probe in probes.items():
        if layers[index] is not probe:
            raise RuntimeError(f"recipient probe at layer {index} was replaced unexpectedly")
        layers[index] = probe.module
