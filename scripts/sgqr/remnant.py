#!/usr/bin/env python3
"""Frozen quantized MLP remnants and significance-only routing for SGQR.

SGQR intentionally omits confidence gating. A fixed deployment budget chooses
how many remnant groups may execute; significance decides *which* groups receive
that budget. The module supports static global scores and a state-conditioned
utility probe without any next-token confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .quantization import PackedSymmetricMatrix


@dataclass
class RoutingSnapshot:
    forward_calls: int
    token_count: int
    group_token_invocations: int
    available_group_tokens: int

    @property
    def active_fraction(self) -> float:
        if self.available_group_tokens <= 0:
            return 0.0
        return self.group_token_invocations / self.available_group_tokens

    def as_dict(self) -> dict[str, float | int]:
        return {
            "forward_calls": self.forward_calls,
            "token_count": self.token_count,
            "group_token_invocations": self.group_token_invocations,
            "available_group_tokens": self.available_group_tokens,
            "active_group_fraction": self.active_fraction,
        }


class QuantizedMLPGroup(nn.Module):
    """One additive group of a Qwen-style gated MLP."""

    def __init__(
        self,
        *,
        group_id: int,
        start: int,
        end: int,
        gate: PackedSymmetricMatrix,
        up: PackedSymmetricMatrix,
        down: PackedSymmetricMatrix,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        if end <= start:
            raise ValueError("group end must be greater than start")
        width = end - start
        if gate.rows != width or up.rows != width:
            raise ValueError("gate/up row counts must equal group width")
        if down.cols != width:
            raise ValueError("down column count must equal group width")
        if gate.cols != up.cols or down.rows != gate.cols:
            raise ValueError("incompatible hidden dimensions across group matrices")

        self.group_id = int(group_id)
        self.start = int(start)
        self.end = int(end)
        self.gate = gate
        self.up = up
        self.down = down
        self.register_buffer("gain", torch.tensor(float(gain), dtype=torch.float32), persistent=True)

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def hidden_size(self) -> int:
        return self.gate.cols

    @property
    def bits(self) -> int:
        return self.gate.bits

    @property
    def storage_bytes(self) -> int:
        return self.gate.storage_bytes + self.up.storage_bytes + self.down.storage_bytes + self.gain.numel() * self.gain.element_size()

    @property
    def logical_parameter_count(self) -> int:
        return (
            self.gate.logical_parameter_count
            + self.up.logical_parameter_count
            + self.down.logical_parameter_count
        )

    @property
    def approximate_multiply_adds_per_token(self) -> int:
        # gate + up + down projections. A multiply-add is counted as one unit
        # here because this is a relative routing-cost estimate, not a FLOP claim.
        return self.hidden_size * self.width * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        device = x.device
        gate_w = self.gate.dequantize(dtype=dtype, device=device)
        up_w = self.up.dequantize(dtype=dtype, device=device)
        down_w = self.down.dequantize(dtype=dtype, device=device)
        intermediate = F.silu(F.linear(x, gate_w)) * F.linear(x, up_w)
        out = F.linear(intermediate, down_w)
        return out * self.gain.to(device=device, dtype=out.dtype)

    def storage_manifest(self) -> dict[str, int | float]:
        return {
            "group_id": self.group_id,
            "start": self.start,
            "end": self.end,
            "width": self.width,
            "bits": self.bits,
            "logical_parameters": self.logical_parameter_count,
            "packed_storage_bytes": self.storage_bytes,
            "approximate_multiply_adds_per_token": self.approximate_multiply_adds_per_token,
            "gain": float(self.gain.item()),
        }


class FrozenQuantizedMLPRemnant(nn.Module):
    """A complete MLP represented as independently executable low-bit groups."""

    def __init__(
        self,
        groups: Sequence[QuantizedMLPGroup],
        *,
        magnitude_scores: torch.Tensor,
    ) -> None:
        super().__init__()
        if not groups:
            raise ValueError("an SGQR remnant must contain at least one group")
        hidden_sizes = {group.hidden_size for group in groups}
        bits = {group.bits for group in groups}
        if len(hidden_sizes) != 1 or len(bits) != 1:
            raise ValueError("all remnant groups must share hidden size and bitwidth")
        expected_ids = list(range(len(groups)))
        actual_ids = [group.group_id for group in groups]
        if actual_ids != expected_ids:
            raise ValueError(f"group ids must be contiguous {expected_ids}, got {actual_ids}")
        if magnitude_scores.shape != (len(groups),):
            raise ValueError("magnitude_scores must have one value per group")

        self.groups = nn.ModuleList(groups)
        self.hidden_size = next(iter(hidden_sizes))
        self.bits = next(iter(bits))
        self.register_buffer("magnitude_scores", magnitude_scores.detach().float().contiguous(), persistent=True)

    @classmethod
    def from_dense_mlp(
        cls,
        dense_mlp: nn.Module,
        *,
        bits: int,
        group_size: int,
        gains: Sequence[float] | None = None,
        scale_dtype: torch.dtype = torch.float16,
    ) -> "FrozenQuantizedMLPRemnant":
        required = ("gate_proj", "up_proj", "down_proj")
        if not all(hasattr(dense_mlp, name) for name in required):
            raise TypeError("dense_mlp must expose gate_proj, up_proj and down_proj")
        if group_size <= 0:
            raise ValueError("group_size must be positive")

        gate_w = dense_mlp.gate_proj.weight.detach()
        up_w = dense_mlp.up_proj.weight.detach()
        down_w = dense_mlp.down_proj.weight.detach()
        if gate_w.shape != up_w.shape:
            raise ValueError("gate and up projections must have identical shapes")
        intermediate_size, hidden_size = gate_w.shape
        if tuple(down_w.shape) != (hidden_size, intermediate_size):
            raise ValueError("down projection shape is incompatible with gate/up projections")

        group_count = (intermediate_size + group_size - 1) // group_size
        if gains is not None and len(gains) != group_count:
            raise ValueError("gains must contain one value per group")

        groups: list[QuantizedMLPGroup] = []
        magnitude: list[float] = []
        for group_id, start in enumerate(range(0, intermediate_size, group_size)):
            end = min(start + group_size, intermediate_size)
            gate_slice = gate_w[start:end, :]
            up_slice = up_w[start:end, :]
            down_slice = down_w[:, start:end]
            group = QuantizedMLPGroup(
                group_id=group_id,
                start=start,
                end=end,
                gate=PackedSymmetricMatrix.from_float(gate_slice, bits=bits, scale_dtype=scale_dtype),
                up=PackedSymmetricMatrix.from_float(up_slice, bits=bits, scale_dtype=scale_dtype),
                down=PackedSymmetricMatrix.from_float(down_slice, bits=bits, scale_dtype=scale_dtype),
                gain=1.0 if gains is None else float(gains[group_id]),
            )
            groups.append(group)
            # The magnitude baseline uses all three coupled matrices.
            magnitude.append(
                float(
                    gate_slice.float().pow(2).sum()
                    + up_slice.float().pow(2).sum()
                    + down_slice.float().pow(2).sum()
                )
            )

        remnant = cls(groups, magnitude_scores=torch.tensor(magnitude, dtype=torch.float32))
        # Quantized donor tensors are buffers only. This is an explicit invariant.
        if any(parameter.requires_grad for parameter in remnant.parameters()):
            raise RuntimeError("frozen quantized remnant unexpectedly contains trainable parameters")
        return remnant

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def logical_parameter_count(self) -> int:
        return sum(group.logical_parameter_count for group in self.groups)

    @property
    def packed_storage_bytes(self) -> int:
        return sum(group.storage_bytes for group in self.groups) + self.magnitude_scores.numel() * self.magnitude_scores.element_size()

    @property
    def group_costs(self) -> torch.Tensor:
        return torch.tensor(
            [group.approximate_multiply_adds_per_token for group in self.groups],
            dtype=torch.float32,
            device=self.magnitude_scores.device,
        )

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_groups(x, range(self.group_count))

    def forward_groups(self, x: torch.Tensor, group_ids: Iterable[int]) -> torch.Tensor:
        ids = sorted({int(group_id) for group_id in group_ids})
        if any(group_id < 0 or group_id >= self.group_count for group_id in ids):
            raise IndexError(f"group ids outside [0, {self.group_count})")
        if not ids:
            return torch.zeros_like(x)
        result = torch.zeros_like(x)
        for group_id in ids:
            result = result + self.groups[group_id](x)
        return result

    def forward_masked(self, x: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """Execute token-specific selected groups without evaluating inactive pairs.

        ``selected`` has shape ``x.shape[:-1] + (k,)`` and stores group ids.
        Each group is evaluated only on the flattened token rows that selected it.
        This reference implementation is mathematically sparse; its gather/scatter
        overhead is measured separately and is not assumed to be fast.
        """
        expected_prefix = tuple(x.shape[:-1])
        if selected.ndim != x.ndim or tuple(selected.shape[:-1]) != expected_prefix:
            raise ValueError(
                f"selected shape must be x prefix + k; got x={tuple(x.shape)}, selected={tuple(selected.shape)}"
            )
        if selected.shape[-1] == 0:
            return torch.zeros_like(x)
        if selected.dtype not in (torch.int32, torch.int64):
            raise TypeError("selected group ids must be integer")
        if bool(((selected < 0) | (selected >= self.group_count)).any()):
            raise IndexError("selected contains an invalid group id")

        flat_x = x.reshape(-1, x.shape[-1])
        flat_selected = selected.reshape(-1, selected.shape[-1])
        flat_out = torch.zeros_like(flat_x)

        for group_id in range(self.group_count):
            mask = (flat_selected == group_id).any(dim=-1)
            if not bool(mask.any()):
                continue
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
            contribution = self.groups[group_id](flat_x.index_select(0, indices))
            flat_out = flat_out.index_add(0, indices, contribution)
        return flat_out.reshape_as(x)

    def storage_manifest(self) -> dict[str, object]:
        return {
            "bits": self.bits,
            "group_count": self.group_count,
            "hidden_size": self.hidden_size,
            "logical_parameters": self.logical_parameter_count,
            "packed_storage_bytes": self.packed_storage_bytes,
            "groups": [group.storage_manifest() for group in self.groups],
        }


class DynamicUtilityRouter(nn.Module):
    """Small state-conditioned predictor of per-group marginal utility."""

    def __init__(self, hidden_size: int, group_count: int, rank: int = 32) -> None:
        super().__init__()
        if hidden_size <= 0 or group_count <= 0 or rank <= 0:
            raise ValueError("hidden_size, group_count and rank must be positive")
        self.hidden_size = int(hidden_size)
        self.group_count = int(group_count)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.down = nn.Linear(hidden_size, rank, bias=True)
        self.up = nn.Linear(rank, group_count, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(self.norm(x))))


class NormalizedUtilityRouter(nn.Module):
    """Apply trace normalization around a trained utility probe."""

    def __init__(
        self,
        router: DynamicUtilityRouter,
        *,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        utility_mean: torch.Tensor,
        utility_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (router.hidden_size,) or feature_scale.shape != (router.hidden_size,):
            raise ValueError("feature normalization has the wrong hidden dimension")
        if utility_mean.shape != (router.group_count,) or utility_scale.shape != (router.group_count,):
            raise ValueError("utility normalization has the wrong group dimension")
        self.router = router
        self.register_buffer("feature_mean", feature_mean.detach().float().clone(), persistent=True)
        self.register_buffer("feature_scale", feature_scale.detach().float().clamp_min(1e-6).clone(), persistent=True)
        self.register_buffer("utility_mean", utility_mean.detach().float().clone(), persistent=True)
        self.register_buffer("utility_scale", utility_scale.detach().float().clamp_min(1e-6).clone(), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.feature_mean.to(device=x.device, dtype=x.dtype)
        scale = self.feature_scale.to(device=x.device, dtype=x.dtype)
        normalized = (x - mean) / scale
        prediction = self.router(normalized)
        return prediction * self.utility_scale.to(prediction) + self.utility_mean.to(prediction)


class DeletedMLP(nn.Module):
    """Exact structural deletion at the MLP residual branch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class SGQRMLPWrapper(nn.Module):
    """MLP replacement implementing significance-only quantized routing.

    No confidence, probability lens or uncertainty threshold exists in this
    module. A fixed group budget is supplied by configuration. Significance may
    be static (global calibration) or dynamic (a utility probe over the current
    MLP input).
    """

    MODES = {
        "dense",
        "fp_frozen",
        "deleted",
        "quantized_all",
        "fixed_groups",
        "static_topk",
        "magnitude_topk",
        "random_topk",
        "dynamic_topk",
    }

    def __init__(
        self,
        *,
        remnant: FrozenQuantizedMLPRemnant,
        dense_mlp: nn.Module | None = None,
        mode: str = "quantized_all",
        budget_groups: int | None = None,
        static_scores: torch.Tensor | None = None,
        router: nn.Module | None = None,
        random_seed: int = 20260824,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown SGQR mode {mode!r}")
        self.remnant = remnant
        self.dense_mlp = dense_mlp
        if self.dense_mlp is not None:
            for parameter in self.dense_mlp.parameters():
                parameter.requires_grad_(False)
        self.mode = mode
        self.budget_groups = remnant.group_count if budget_groups is None else int(budget_groups)
        self.router = router
        self.register_buffer(
            "static_scores",
            (
                torch.zeros(remnant.group_count, dtype=torch.float32)
                if static_scores is None
                else static_scores.detach().float().clone()
            ),
            persistent=True,
        )
        if self.static_scores.shape != (remnant.group_count,):
            raise ValueError("static_scores must have one value per group")
        generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
        self.register_buffer("random_scores", torch.rand(remnant.group_count, generator=generator), persistent=True)
        self.fixed_group_ids: list[int] = []
        self.capture_last_input = False
        self.last_input: torch.Tensor | None = None
        self.reset_stats()

    def reset_stats(self) -> None:
        self._forward_calls = 0
        self._token_count = 0
        self._group_token_invocations = 0
        self._available_group_tokens = 0

    def stats(self) -> RoutingSnapshot:
        return RoutingSnapshot(
            forward_calls=self._forward_calls,
            token_count=self._token_count,
            group_token_invocations=self._group_token_invocations,
            available_group_tokens=self._available_group_tokens,
        )

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown SGQR mode {mode!r}")
        if mode in {"dense", "fp_frozen"} and self.dense_mlp is None:
            raise RuntimeError("dense reference was not retained in this wrapper")
        if mode == "dynamic_topk" and self.router is None:
            raise RuntimeError("dynamic_topk requires a DynamicUtilityRouter")
        self.mode = mode

    def set_budget(self, budget_groups: int) -> None:
        budget_groups = int(budget_groups)
        if budget_groups < 0 or budget_groups > self.remnant.group_count:
            raise ValueError(f"budget must be within [0, {self.remnant.group_count}]")
        self.budget_groups = budget_groups

    def set_fixed_groups(self, group_ids: Sequence[int]) -> None:
        ids = sorted({int(group_id) for group_id in group_ids})
        if any(group_id < 0 or group_id >= self.remnant.group_count for group_id in ids):
            raise IndexError("fixed group id out of range")
        self.fixed_group_ids = ids

    def set_static_scores(self, scores: torch.Tensor) -> None:
        if scores.shape != self.static_scores.shape:
            raise ValueError("new static scores have incompatible shape")
        self.static_scores.copy_(scores.detach().to(self.static_scores))

    def drop_dense_reference(self) -> None:
        self.dense_mlp = None

    def _topk(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        if k == 0:
            return torch.empty(*scores.shape[:-1], 0, dtype=torch.long, device=scores.device)
        # stable=True gives deterministic tie handling by original group order.
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return order[..., :k]

    def _static_ids(self, scores: torch.Tensor) -> list[int]:
        k = min(max(self.budget_groups, 0), self.remnant.group_count)
        if k == 0:
            return []
        costs = self.remnant.group_costs.to(device=scores.device).clamp_min(1.0)
        ranked = self._topk(scores / costs, k)
        return [int(value) for value in ranked.tolist()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.remnant.hidden_size:
            raise ValueError(
                f"expected hidden size {self.remnant.hidden_size}, got {x.shape[-1]}"
            )
        if self.capture_last_input:
            self.last_input = x.detach()

        tokens = int(x.numel() // x.shape[-1])
        self._forward_calls += 1
        self._token_count += tokens
        self._available_group_tokens += tokens * self.remnant.group_count

        if self.mode in {"dense", "fp_frozen"}:
            if self.dense_mlp is None:
                raise RuntimeError("dense reference is unavailable")
            self._group_token_invocations += tokens * self.remnant.group_count
            return self.dense_mlp(x)

        if self.mode == "deleted":
            return torch.zeros_like(x)

        if self.mode == "quantized_all":
            self._group_token_invocations += tokens * self.remnant.group_count
            return self.remnant.forward_all(x)

        if self.mode == "fixed_groups":
            ids = self.fixed_group_ids
            self._group_token_invocations += tokens * len(ids)
            return self.remnant.forward_groups(x, ids)

        if self.mode == "static_topk":
            ids = self._static_ids(self.static_scores)
            self._group_token_invocations += tokens * len(ids)
            return self.remnant.forward_groups(x, ids)

        if self.mode == "magnitude_topk":
            ids = self._static_ids(self.remnant.magnitude_scores)
            self._group_token_invocations += tokens * len(ids)
            return self.remnant.forward_groups(x, ids)

        if self.mode == "random_topk":
            ids = self._static_ids(self.random_scores)
            self._group_token_invocations += tokens * len(ids)
            return self.remnant.forward_groups(x, ids)

        if self.mode == "dynamic_topk":
            if self.router is None:
                raise RuntimeError("dynamic_topk requires a router")
            k = min(max(self.budget_groups, 0), self.remnant.group_count)
            utility = self.router(x)
            costs = self.remnant.group_costs.to(device=x.device).clamp_min(1.0)
            selected = self._topk(utility / costs, k)
            self._group_token_invocations += tokens * k
            return self.remnant.forward_masked(x, selected)

        raise AssertionError(f"unreachable mode {self.mode}")
