"""Architecture-distinct graft blocks.

The modules in this file transfer *mechanisms*, not claims about donor-model
knowledge.  ``latent_kv_attention`` is an intentionally small latent-KV design
inspired by the same compression principle as MLA; it is not represented as a
bit-compatible reimplementation of any DeepSeek checkpoint.  ``nope_attention``
provides a causal attention branch with no rotary position transform.  A generic
``ForeignModuleAdapter`` supports later functional-transplant experiments with
an already-loaded frozen decoder block.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DonorSpec:
    kind: str
    d_model: int | None = None
    num_heads: int = 4
    num_kv_heads: int | None = None
    head_dim: int | None = None
    intermediate_size: int | None = None
    latent_rank: int | None = None
    q_rank: int | None = None
    num_experts: int = 4
    top_k: int = 2
    shared_expert: bool = True
    dropout: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DonorSpec":
        raw = dict(value)
        allowed = {
            "kind",
            "d_model",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "intermediate_size",
            "latent_rank",
            "q_rank",
            "num_experts",
            "top_k",
            "shared_expert",
            "dropout",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown donor fields: {unknown}")
        if "kind" not in raw:
            raise ValueError("donor.kind is required")
        spec = cls(**raw)
        spec.validate()
        return spec

    def validate(self) -> None:
        supported = {
            "swiglu_mlp",
            "nope_attention",
            "latent_kv_attention",
            "routed_moe",
        }
        if self.kind not in supported:
            raise ValueError(f"unsupported donor kind: {self.kind!r}")
        if self.d_model is not None and self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_kv_heads is not None and self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.kind in {"nope_attention", "latent_kv_attention"}:
            kv_heads = self.num_kv_heads or self.num_heads
            if self.num_heads % kv_heads != 0:
                raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.kind == "latent_kv_attention":
            if self.latent_rank is None or self.latent_rank <= 0:
                raise ValueError("latent_kv_attention requires latent_rank > 0")
        if self.kind == "routed_moe":
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive")
            if not 1 <= self.top_k <= self.num_experts:
                raise ValueError("top_k must be in [1, num_experts]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = float(eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype) * self.weight


def _split_heads(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
    batch, seq, _ = x.shape
    return x.view(batch, seq, num_heads, head_dim).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    batch, heads, seq, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq, heads * head_dim)


def _repeat_kv(x: Tensor, repeat: int) -> Tensor:
    return x if repeat == 1 else x.repeat_interleave(repeat, dim=1)


def _apply_attention_mask(scores: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Apply common decoder-mask forms to ``[B,H,Q,K]`` scores."""

    q_len, k_len = scores.shape[-2:]
    causal = torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device).triu(1)
    scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)

    if attention_mask is None:
        return scores

    mask = attention_mask.to(device=scores.device)
    if mask.ndim == 2:
        if mask.shape[-1] != k_len:
            raise ValueError(
                f"2D attention mask length {mask.shape[-1]} != key length {k_len}"
            )
        invalid = mask == 0
        return scores.masked_fill(invalid[:, None, None, :], torch.finfo(scores.dtype).min)

    if mask.ndim == 3:
        mask = mask[:, None, :, :]
    if mask.ndim != 4:
        raise ValueError(f"unsupported attention mask rank: {mask.ndim}")
    mask = mask[..., :q_len, :k_len]
    if mask.dtype == torch.bool:
        # HF boolean masks conventionally mark allowed entries true.
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return scores + mask.to(dtype=scores.dtype)


def _causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attention_mask: Tensor | None,
    dropout_p: float,
    training: bool,
) -> Tensor:
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    scores = scores / math.sqrt(query.shape[-1])
    scores = _apply_attention_mask(scores, attention_mask)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0)
    probabilities = F.dropout(probabilities, p=dropout_p, training=training)
    return torch.matmul(probabilities.to(value.dtype), value)


class SwiGLUBranch(nn.Module):
    """A Llama/Qwen-family dense gated MLP branch used as a control."""

    def __init__(self, d_model: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        self.d_model = int(d_model)
        self.intermediate_size = int(intermediate_size)
        self.norm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, **_: Any) -> Tensor:
        x = self.norm(hidden_states)
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class NoPEAttentionBranch(nn.Module):
    """Causal grouped-query self-attention with no positional rotation."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads or num_heads)
        self.head_dim = int(head_dim or (d_model // num_heads))
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_heads * self.head_dim != d_model:
            raise ValueError("num_heads * head_dim must equal d_model")
        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        x = self.norm(hidden_states)
        q = _split_heads(self.q_proj(x), self.num_heads, self.head_dim)
        k = _split_heads(self.k_proj(x), self.num_kv_heads, self.head_dim)
        v = _split_heads(self.v_proj(x), self.num_kv_heads, self.head_dim)
        repeat = self.num_heads // self.num_kv_heads
        context = _causal_attention(
            q,
            _repeat_kv(k, repeat),
            _repeat_kv(v, repeat),
            attention_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.o_proj(_merge_heads(context))


class LatentKVAttentionBranch(nn.Module):
    """A compact latent-KV causal attention sidecar.

    Keys and values are reconstructed from one shared low-dimensional latent.
    Queries may also be factorised through ``q_rank``.  This captures the
    architectural hypothesis under test without asserting compatibility with a
    particular donor checkpoint's exact MLA parameterisation.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        num_kv_heads: int | None,
        latent_rank: int,
        q_rank: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads or num_heads)
        self.head_dim = int(head_dim or (d_model // num_heads))
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_heads * self.head_dim != d_model:
            raise ValueError("num_heads * head_dim must equal d_model")
        self.norm = RMSNorm(d_model)
        self.kv_down = nn.Linear(d_model, latent_rank, bias=False)
        self.k_up = nn.Linear(latent_rank, self.num_kv_heads * self.head_dim, bias=False)
        self.v_up = nn.Linear(latent_rank, self.num_kv_heads * self.head_dim, bias=False)
        if q_rank is None:
            self.q_down: nn.Module = nn.Identity()
            self.q_up = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        else:
            if q_rank <= 0:
                raise ValueError("q_rank must be positive")
            self.q_down = nn.Linear(d_model, q_rank, bias=False)
            self.q_up = nn.Linear(q_rank, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)
        self.dropout = float(dropout)
        self.latent_rank = int(latent_rank)
        self.q_rank = q_rank

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        x = self.norm(hidden_states)
        q = _split_heads(self.q_up(self.q_down(x)), self.num_heads, self.head_dim)
        latent = self.kv_down(x)
        k = _split_heads(self.k_up(latent), self.num_kv_heads, self.head_dim)
        v = _split_heads(self.v_up(latent), self.num_kv_heads, self.head_dim)
        repeat = self.num_heads // self.num_kv_heads
        context = _causal_attention(
            q,
            _repeat_kv(k, repeat),
            _repeat_kv(v, repeat),
            attention_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.o_proj(_merge_heads(context))


class _SwiGLUExpert(nn.Module):
    """One norm-free expert; the MoE branch owns the shared pre-norm."""

    def __init__(self, d_model: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        )


class RoutedMoEBranch(nn.Module):
    """Token-wise top-k routed SwiGLU experts plus an optional shared expert.

    The reference path evaluates all compact experts and then gathers the selected
    outputs.  This is intentionally less kernel-efficient than dispatch/index-add,
    but it is deterministic on the CPU/CUDA research path and avoids attributing
    nondeterministic accumulation noise to the architecture.
    """

    def __init__(
        self,
        d_model: int,
        *,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        shared_expert: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.norm = RMSNorm(d_model)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_SwiGLUExpert(d_model, intermediate_size, dropout) for _ in range(num_experts)]
        )
        self.shared = (
            _SwiGLUExpert(d_model, intermediate_size, dropout) if shared_expert else None
        )
        self.last_routing_stats: dict[str, Any] = {}

    def forward(self, hidden_states: Tensor, **_: Any) -> Tensor:
        normalized = self.norm(hidden_states)
        probabilities = torch.softmax(self.router(normalized).float(), dim=-1)
        weights, indices = torch.topk(probabilities, k=self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(normalized.dtype)

        # [B, S, E, D], followed by deterministic gather of the top-k experts.
        expert_outputs = torch.stack(
            [expert(normalized) for expert in self.experts], dim=-2
        )
        gather_index = indices.unsqueeze(-1).expand(*indices.shape, self.d_model)
        selected = torch.gather(expert_outputs, dim=-2, index=gather_index)
        output = (selected * weights.unsqueeze(-1)).sum(dim=-2)
        if self.shared is not None:
            output = output + self.shared(normalized)

        with torch.no_grad():
            load = torch.bincount(
                indices.detach().reshape(-1).cpu(), minlength=self.num_experts
            )
            self.last_routing_stats = {
                "selected_tokens_per_expert": load.tolist(),
                "mean_topk_probability": float(weights.detach().float().mean().cpu()),
            }
        return output


class ForeignModuleAdapter(nn.Module):
    """Adapt an already-loaded decoder block to the graft-delta contract.

    The wrapped module may return a tensor or a tuple whose first element is a
    tensor.  If it returns a full residual state, ``returns_delta=False`` turns
    it into a delta by subtracting its input.  Unsupported keyword arguments are
    filtered through the wrapped module's signature instead of being guessed.
    """

    def __init__(self, module: nn.Module, *, returns_delta: bool = False) -> None:
        super().__init__()
        self.module = module
        self.returns_delta = bool(returns_delta)
        try:
            signature = inspect.signature(module.forward)
            self._accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
            self._accepted = set(signature.parameters)
        except (TypeError, ValueError):
            self._accepts_kwargs = True
            self._accepted = set()

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        call_kwargs = kwargs if self._accepts_kwargs else {
            key: value for key, value in kwargs.items() if key in self._accepted
        }
        output = self.module(hidden_states, **call_kwargs)
        if isinstance(output, Tensor):
            transformed = output
        elif isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            transformed = output[0]
        else:
            raise TypeError(
                "foreign module must return a tensor or a tuple/list beginning with one"
            )
        return transformed if self.returns_delta else transformed - hidden_states


def build_donor(spec: DonorSpec, d_model: int | None = None) -> nn.Module:
    spec.validate()
    width = int(spec.d_model or d_model or 0)
    if width <= 0:
        raise ValueError("donor d_model must be supplied in the spec or builder")
    intermediate = int(spec.intermediate_size or 4 * width)
    if spec.kind == "swiglu_mlp":
        return SwiGLUBranch(width, intermediate, spec.dropout)
    if spec.kind == "nope_attention":
        return NoPEAttentionBranch(
            width,
            num_heads=spec.num_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "latent_kv_attention":
        assert spec.latent_rank is not None
        return LatentKVAttentionBranch(
            width,
            num_heads=spec.num_heads,
            num_kv_heads=spec.num_kv_heads,
            latent_rank=spec.latent_rank,
            q_rank=spec.q_rank,
            head_dim=spec.head_dim,
            dropout=spec.dropout,
        )
    return RoutedMoEBranch(
        width,
        intermediate_size=intermediate,
        num_experts=spec.num_experts,
        top_k=spec.top_k,
        shared_expert=spec.shared_expert,
        dropout=spec.dropout,
    )


def donor_signature(module: nn.Module) -> dict[str, Any]:
    return {
        "class": module.__class__.__name__,
        "parameters": sum(p.numel() for p in module.parameters()),
        "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
    }
