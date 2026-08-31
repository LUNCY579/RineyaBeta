"""DAST model wrapper: hidden-depth readout, optional operators, and causal feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch import nn

from dast.alignment import resolve_checkpoint_layers
from dast.operators import LowRankOperatorBank
from dast.pooling import pool_prompt_hidden
from dast.semantic import SharedStateProjector

StateIntervention = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class DASTOutput:
    lm_loss: torch.Tensor | None
    logits: torch.Tensor
    states: torch.Tensor
    effective_states: torch.Tensor
    transition_predictions: torch.Tensor | None
    router_weights: torch.Tensor | None
    checkpoint_layers: tuple[int, ...]


def _config_value(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def infer_num_hidden_layers(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        value = _config_value(config, "num_hidden_layers", "n_layer", "num_layers")
        if value is not None:
            return value
    return len(locate_decoder_layers(model))


def infer_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        value = _config_value(config, "hidden_size", "n_embd", "d_model")
        if value is not None:
            return value
    embedding = model.get_input_embeddings()
    return int(embedding.weight.shape[1])


def _unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:  # pragma: no cover - defensive for third-party PEFT wrappers
            pass
    return model


def locate_decoder_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    """Locate common decoder-layer containers without trusting remote model code."""

    root = _unwrap_model(model)
    candidates = (
        getattr(root, "layers", None),
        getattr(getattr(root, "model", None), "layers", None),
        getattr(getattr(root, "transformer", None), "h", None),
        getattr(getattr(root, "gpt_neox", None), "layers", None),
        getattr(getattr(getattr(root, "model", None), "decoder", None), "layers", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "__len__") and len(candidate) > 0:
            return candidate
    raise ValueError("could not locate decoder layers for feedback hooks")


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported decoder-layer output type for feedback: {type(output)!r}")


def _extract_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"unsupported decoder-layer output type: {type(output)!r}")


class DASTModel(nn.Module):
    """Wrap a causal LM with one shared semantic interface across depth.

    Default V1 is auxiliary-only and obtains hidden states from the ordinary
    end-to-end forward pass.  ``feedback_enabled`` is an explicitly gated causal
    phase: a small decoded semantic residual is inserted at the final prompt token
    after selected layers.  It is intentionally incompatible with gradient
    checkpointing and cached generation until separately qualified.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        checkpoint_fractions: tuple[float, ...],
        state_dim: int,
        projector_hidden_dim: int,
        pooling: str,
        operator_bank: LowRankOperatorBank | None = None,
        feedback_enabled: bool = False,
        feedback_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_hidden_layers = infer_num_hidden_layers(base_model)
        self.hidden_size = infer_hidden_size(base_model)
        self.checkpoint_layers = resolve_checkpoint_layers(self.num_hidden_layers, checkpoint_fractions)
        self.pooling = pooling
        self.state_projector = SharedStateProjector(self.hidden_size, state_dim, projector_hidden_dim)
        self.operator_bank = operator_bank
        self.feedback_enabled = feedback_enabled
        self.feedback_scale = feedback_scale
        self.feedback_decoder: nn.Linear | None = None
        self.feedback_gates: nn.Parameter | None = None
        self._feedback_handles: list[Any] = []
        self._active_prompt_mask: torch.Tensor | None = None
        self._active_interventions: Mapping[int, StateIntervention | torch.Tensor] = {}
        self._hook_raw_states: dict[int, torch.Tensor] = {}
        self._hook_effective_states: dict[int, torch.Tensor] = {}

        if feedback_enabled:
            self.feedback_decoder = nn.Linear(state_dim, self.hidden_size, bias=False)
            nn.init.zeros_(self.feedback_decoder.weight)
            # tanh(-3) starts near zero but not exactly zero; the causal path can
            # learn without destabilising the pretrained residual stream.
            self.feedback_gates = nn.Parameter(torch.full((len(self.checkpoint_layers),), -3.0))
            self._install_feedback_hooks()

    @property
    def num_stages(self) -> int:
        return len(self.checkpoint_layers)

    def _install_feedback_hooks(self) -> None:
        layers = locate_decoder_layers(self.base_model)
        for stage_index, layer_number in enumerate(self.checkpoint_layers):
            layer = layers[layer_number - 1]

            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, stage: int = stage_index) -> Any:
                if self._active_prompt_mask is None:
                    return output
                hidden = _extract_hidden(output)
                pooled = pool_prompt_hidden(hidden, self._active_prompt_mask, self.pooling)
                projector_device = next(self.state_projector.parameters()).device
                raw_state = self.state_projector(pooled.to(projector_device))
                effective_state = self._apply_intervention(stage, raw_state)
                self._hook_raw_states[stage] = raw_state
                self._hook_effective_states[stage] = effective_state
                assert self.feedback_decoder is not None
                assert self.feedback_gates is not None
                decoded = self.feedback_decoder(effective_state)
                scale = self.feedback_scale * torch.tanh(self.feedback_gates[stage])
                delta = decoded.to(hidden.dtype) * scale.to(hidden.dtype)
                modified = hidden.clone()
                mask = self._active_prompt_mask.to(device=hidden.device, dtype=torch.bool)
                positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand_as(mask)
                last_prompt = positions.masked_fill(~mask, -1).max(dim=1).values
                if torch.any(last_prompt < 0):
                    raise ValueError("feedback requires at least one prompt token per example")
                batch = torch.arange(hidden.shape[0], device=hidden.device)
                modified[batch, last_prompt] = modified[batch, last_prompt] + delta
                return _replace_hidden(output, modified)

            self._feedback_handles.append(layer.register_forward_hook(hook))

    def _apply_intervention(self, stage: int, state: torch.Tensor) -> torch.Tensor:
        intervention = self._active_interventions.get(stage)
        if intervention is None:
            return state
        if callable(intervention):
            result = intervention(state)
        else:
            result = intervention.to(device=state.device, dtype=state.dtype)
            if result.ndim == 1:
                result = result.unsqueeze(0).expand_as(state)
        if result.shape != state.shape:
            raise ValueError(
                f"intervention at stage {stage} returned {tuple(result.shape)}; expected {tuple(state.shape)}"
            )
        return torch.nn.functional.normalize(result, dim=-1)

    def remove_feedback_hooks(self) -> None:
        for handle in self._feedback_handles:
            handle.remove()
        self._feedback_handles.clear()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        interventions: Mapping[int, StateIntervention | torch.Tensor] | None = None,
    ) -> DASTOutput:
        if prompt_mask.shape != input_ids.shape:
            raise ValueError("prompt_mask must match input_ids")
        if torch.any(prompt_mask & ~attention_mask.bool()):
            raise ValueError("prompt_mask cannot include padding")

        if self.feedback_enabled:
            self._active_prompt_mask = prompt_mask
            self._active_interventions = interventions or {}
            self._hook_raw_states = {}
            self._hook_effective_states = {}
            try:
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
            finally:
                self._active_prompt_mask = None
                self._active_interventions = {}
            missing = [stage for stage in range(self.num_stages) if stage not in self._hook_raw_states]
            if missing:
                raise RuntimeError(f"feedback hooks did not execute for stages: {missing}")
            states = torch.stack([self._hook_raw_states[stage] for stage in range(self.num_stages)], dim=1)
            effective_states = torch.stack(
                [self._hook_effective_states[stage] for stage in range(self.num_stages)], dim=1
            )
        else:
            if interventions:
                raise ValueError("interventions require model.feedback_enabled=true")
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None or len(hidden_states) <= self.checkpoint_layers[-1]:
                raise RuntimeError(
                    f"base model returned {0 if hidden_states is None else len(hidden_states)} hidden states; "
                    f"need index {self.checkpoint_layers[-1]}"
                )
            projected = []
            for layer_number in self.checkpoint_layers:
                pooled = pool_prompt_hidden(hidden_states[layer_number], prompt_mask, self.pooling)
                projector_device = next(self.state_projector.parameters()).device
                projected.append(self.state_projector(pooled.to(projector_device)))
            states = torch.stack(projected, dim=1)
            effective_states = states

        transition_predictions: torch.Tensor | None = None
        router_weights: torch.Tensor | None = None
        if self.operator_bank is not None and self.num_stages > 1:
            transition_predictions, router_weights = self.operator_bank(effective_states[:, :-1])

        return DASTOutput(
            lm_loss=getattr(outputs, "loss", None),
            logits=outputs.logits,
            states=states,
            effective_states=effective_states,
            transition_predictions=transition_predictions,
            router_weights=router_weights,
            checkpoint_layers=self.checkpoint_layers,
        )

    def auxiliary_state_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "state_projector": self.state_projector.state_dict(),
            "checkpoint_layers": self.checkpoint_layers,
            "pooling": self.pooling,
            "feedback_enabled": self.feedback_enabled,
            "feedback_scale": self.feedback_scale,
        }
        if self.operator_bank is not None:
            result["operator_bank"] = self.operator_bank.state_dict()
        if self.feedback_decoder is not None:
            result["feedback_decoder"] = self.feedback_decoder.state_dict()
        if self.feedback_gates is not None:
            result["feedback_gates"] = self.feedback_gates.detach().cpu()
        return result
