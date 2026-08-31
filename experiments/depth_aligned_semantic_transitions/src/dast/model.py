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


def locate_final_norm(model: nn.Module) -> nn.Module | None:
    """Locate the model's final normalisation layer, without remote code.

    Needed because ``hidden_states[-1]`` is already post-final-norm on HF decoder
    models while every earlier entry is a raw residual state. Reading both through
    one projector puts two representation classes into an interface the SSOT calls
    shared; applying this module to the un-normed checkpoints removes that split.
    """

    root = _unwrap_model(model)
    inner = getattr(root, "model", None)
    candidates = (
        getattr(inner, "norm", None),
        getattr(root, "norm", None),
        getattr(root, "final_norm", None),
        getattr(getattr(root, "transformer", None), "ln_f", None),
        getattr(getattr(root, "gpt_neox", None), "final_layer_norm", None),
        getattr(getattr(inner, "decoder", None), "final_layer_norm", None),
    )
    for candidate in candidates:
        if isinstance(candidate, nn.Module):
            return candidate
    return None


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
        checkpoint_readout: str = "raw",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_hidden_layers = infer_num_hidden_layers(base_model)
        self.hidden_size = infer_hidden_size(base_model)
        self.checkpoint_layers = resolve_checkpoint_layers(self.num_hidden_layers, checkpoint_fractions)
        self.pooling = pooling
        if checkpoint_readout not in ("raw", "final_norm"):
            raise ValueError(f"unsupported checkpoint readout: {checkpoint_readout}")
        self.checkpoint_readout = checkpoint_readout
        self._final_norm = None
        if checkpoint_readout == "final_norm":
            self._final_norm = locate_final_norm(base_model)
            if self._final_norm is None:
                raise ValueError(
                    "checkpoint_readout='final_norm' requires a locatable final norm layer"
                )
            # The config-level guard can only see the FRACTION, and
            # int(round(L * f)) maps the whole interval [1 - 0.5/L, 1.0) onto layer
            # L. So 0.983 on a 28-layer model, or the shipped 0.964 on any model
            # with L <= 13, resolves to the top layer while passing config
            # validation. Assert on the resolved depth, where the truth is known.
            if self.checkpoint_layers[-1] >= self.num_hidden_layers:
                raise ValueError(
                    "checkpoint_readout='final_norm' resolved its last checkpoint to layer "
                    f"{self.checkpoint_layers[-1]} of {self.num_hidden_layers}, i.e. the top of "
                    "the stack, whose hidden state is already post-final-norm and measured as a "
                    "near-orthogonal outlier against the other checkpoints. Lower the last "
                    f"checkpoint fraction below {1.0 - 0.5 / self.num_hidden_layers:.4f} for this "
                    "model, or use checkpoint_readout='raw'."
                )
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
            # Zero-init keeps the pretrained residual stream untouched at step 0.
            # Consequence to be explicit about: `delta = decoder(state) * gate` is
            # then identically zero, so EVERY intervention is a bitwise no-op until
            # the decoder has trained. An untrained feedback run therefore cannot
            # produce a causal result. dW is non-zero (d(Wx)/dW = x), so it can
            # learn; it just has not yet.
            nn.init.zeros_(self.feedback_decoder.weight)
            # NOTE: tanh(-3) = -0.995, i.e. near SATURATION, not near zero. The
            # near-zero initial delta comes from the decoder above, not from this
            # gate, and saturation attenuates the gate's own gradient by ~100x
            # (1 - tanh^2(-3) ~ 0.0099).
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
                # A forward hook sees the PRE-norm layer output, while the auxiliary
                # path reads hidden_states[...] which is post-final-norm at the last
                # index. Applying the same readout here keeps the two paths sampling
                # the same tensor, so a projector trained under the auxiliary config
                # is not silently fed a different vector during the causal probe.
                readout = self._final_norm(hidden) if self._final_norm is not None else hidden
                pooled = pool_prompt_hidden(readout, self._active_prompt_mask, self.pooling)
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
                state = hidden_states[layer_number]
                # hidden_states[num_hidden_layers] is ALREADY post-final-norm on HF
                # decoder models, so it is skipped here to avoid double-norming.
                if self._final_norm is not None and layer_number < self.num_hidden_layers:
                    state = self._final_norm(state)
                pooled = pool_prompt_hidden(state, prompt_mask, self.pooling)
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
