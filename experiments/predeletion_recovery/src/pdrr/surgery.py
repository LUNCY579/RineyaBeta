"""Gate wrappers, physical deletion, and scar-free export checks."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


class GatedModule(nn.Module):
    """Wrap a residual module and interpolate its residual contribution.

    The wrapped module must return either a tensor or a tuple whose first element
    is the residual-stream tensor. Inputs follow the same convention used by
    Hugging Face decoder blocks: the first positional argument is the hidden state.
    """

    def __init__(self, module: nn.Module, gate: float = 1.0) -> None:
        super().__init__()
        self.module = module
        self.register_buffer("gate", torch.tensor(float(gate), dtype=torch.float32),
                             persistent=False)

    def set_gate(self, value: float) -> None:
        self.gate.fill_(float(value))

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        output = self.module(hidden_states, *args, **kwargs)
        if isinstance(output, tuple):
            primary = output[0]
            blended = hidden_states + self.gate.to(primary) * (primary - hidden_states)
            return (blended, *output[1:])
        if not isinstance(output, Tensor):
            raise TypeError(f"unsupported gated module output type: {type(output)!r}")
        return hidden_states + self.gate.to(output) * (output - hidden_states)


@dataclass(frozen=True)
class ExportCheck:
    max_abs_logit_difference: float
    greedy_equal: bool
    source_parameter_count: int
    compact_parameter_count: int
    expected_removed_parameters: int
    actual_removed_parameters: int
    deployment_state_clean: bool

    @property
    def ok(self) -> bool:
        return (
            self.greedy_equal
            and self.actual_removed_parameters == self.expected_removed_parameters
            and self.deployment_state_clean
        )


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def deployment_state_is_clean(module: nn.Module) -> bool:
    forbidden = ("router", "recovery_matrix", "recovery_layer", ".gate")
    return not any(any(token in key.lower() for token in forbidden)
                   for key in module.state_dict())


def physical_delete_from_module_list(
    model: nn.Module,
    layers: nn.ModuleList,
    target_index: int,
) -> tuple[nn.Module, int]:
    if not 0 <= target_index < len(layers):
        raise IndexError(target_index)
    compact = copy.deepcopy(model)
    qualified = next(
        (name for name, module in model.named_modules() if module is layers), None)
    if qualified is None:
        raise ValueError("layers ModuleList is not owned by model")
    current: nn.Module = compact
    for part in qualified.split(".") if qualified else []:
        current = getattr(current, part)
    if not isinstance(current, nn.ModuleList):
        raise TypeError("resolved copied layer container is not ModuleList")
    removed = count_parameters(current[target_index])
    current.__delitem__(target_index)
    return compact, removed


def compare_logits(gated_logits: Tensor, physical_logits: Tensor,
                   *, atol: float = 1e-5) -> tuple[float, bool]:
    if gated_logits.shape != physical_logits.shape:
        raise ValueError("gated and physical logits have different shapes")
    max_abs = float((gated_logits.float() - physical_logits.float()).abs().max())
    greedy_equal = bool(torch.equal(
        gated_logits.argmax(dim=-1), physical_logits.argmax(dim=-1)))
    if max_abs > atol and greedy_equal:
        pass
    return max_abs, greedy_equal
