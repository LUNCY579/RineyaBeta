#!/usr/bin/env python3
"""Self-contained correctness checks for the SGQR reference implementation."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.sgqr.remnant import (  # noqa: E402
    DynamicUtilityRouter,
    FrozenQuantizedMLPRemnant,
    SGQRMLPWrapper,
)
from scripts.sgqr.quantization import (  # noqa: E402
    PackedSymmetricMatrix,
    pack_int4,
    unpack_int4,
)
from scripts.sgqr.utility import (  # noqa: E402
    SignificanceAccumulator,
    marginal_recovery_utility,
    recovery_ratio,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 10) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def assert_close(actual, expected, *, atol=1e-6, rtol=1e-6, label="values") -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        error = float((actual - expected).abs().max())
        raise AssertionError(f"{label} differ; max absolute error={error}")


def run() -> int:
    checks = 0

    values = torch.tensor([-7, -3, 0, 4, 7], dtype=torch.int8)
    packed, original_numel = pack_int4(values)
    assert packed.numel() == 3
    assert torch.equal(unpack_int4(packed, original_numel), values)
    checks += 1

    weight = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.25, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [2.0, -2.0, 1.0, -1.0, 0.1]],
        dtype=torch.float32,
    )
    matrix = PackedSymmetricMatrix.from_float(weight, bits=4)
    assert matrix.packed_weight_bytes == (weight.numel() + 1) // 2
    reconstruction = matrix.dequantize(dtype=torch.float32)
    assert torch.equal(reconstruction[1], torch.zeros_like(reconstruction[1]))
    row_tolerance = matrix.scales.float().expand_as(weight) * 0.51 + 1e-6
    if bool(((reconstruction - weight).abs() > row_tolerance).any()):
        raise AssertionError("INT4 reconstruction exceeded half-step tolerance")
    checks += 1

    matrix8 = PackedSymmetricMatrix.from_float(weight, bits=8)
    assert matrix8.packed_weight_bytes == weight.numel()
    checks += 1

    torch.manual_seed(17)
    dense = TinyMLP()
    x = torch.randn(2, 3, 8)
    remnant = FrozenQuantizedMLPRemnant.from_dense_mlp(
        dense, bits=4, group_size=4
    )
    assert remnant.group_count == 3
    assert not list(remnant.parameters())
    checks += 1

    full = remnant.forward_all(x)
    summed = torch.zeros_like(full)
    for group_id in range(remnant.group_count):
        summed = summed + remnant.forward_groups(x, [group_id])
    assert_close(full, summed, label="quantized group sum")
    checks += 1

    gate_parts = []
    up_parts = []
    down_parts = []
    for group in remnant.groups:
        gate_parts.append(group.gate.dequantize(dtype=x.dtype))
        up_parts.append(group.up.dequantize(dtype=x.dtype))
        down_parts.append(group.down.dequantize(dtype=x.dtype))
    gate_q = torch.cat(gate_parts, dim=0)
    up_q = torch.cat(up_parts, dim=0)
    down_q = torch.cat(down_parts, dim=1)
    monolithic = F.linear(
        F.silu(F.linear(x, gate_q)) * F.linear(x, up_q),
        down_q,
    )
    assert_close(full, monolithic, label="monolithic quantized MLP")
    checks += 1

    assert_close(remnant.forward_groups(x, []), torch.zeros_like(x), label="empty route")
    selected = remnant.forward_groups(x, [0, 2])
    manual = remnant.groups[0](x) + remnant.groups[2](x)
    assert_close(selected, manual, label="selected groups")
    checks += 1

    token_selection = torch.tensor(
        [[[0, 1], [1, 2], [0, 2]], [[2, 1], [0, 1], [2, 0]]],
        dtype=torch.long,
    )
    masked = remnant.forward_masked(x, token_selection)
    manual_masked = torch.zeros_like(x)
    for batch in range(x.shape[0]):
        for token in range(x.shape[1]):
            ids = token_selection[batch, token].tolist()
            manual_masked[batch, token] = remnant.forward_groups(
                x[batch, token].unsqueeze(0), ids
            ).squeeze(0)
    assert_close(masked, manual_masked, label="token-specific groups")
    checks += 1

    wrapper = SGQRMLPWrapper(
        remnant=remnant,
        dense_mlp=copy.deepcopy(dense),
        mode="deleted",
        budget_groups=2,
        static_scores=torch.tensor([1.0, 3.0, 2.0]),
    )
    assert_close(wrapper(x), torch.zeros_like(x), label="wrapper deletion")
    wrapper.set_mode("quantized_all")
    assert_close(wrapper(x), full, label="wrapper all-on")
    wrapper.set_mode("static_topk")
    assert_close(
        wrapper(x),
        remnant.forward_groups(x, [1, 2]),
        label="static significance",
    )
    checks += 1

    router = DynamicUtilityRouter(hidden_size=8, group_count=3, rank=4)
    for parameter in router.parameters():
        nn.init.zeros_(parameter)
    router.up.bias.data.copy_(torch.tensor([1.0, 4.0, 3.0]))
    dynamic_wrapper = SGQRMLPWrapper(
        remnant=remnant,
        mode="dynamic_topk",
        budget_groups=2,
        router=router,
    )
    assert_close(
        dynamic_wrapper(x),
        remnant.forward_groups(x, [1, 2]),
        label="dynamic significance",
    )
    assert abs(dynamic_wrapper.stats().active_fraction - (2.0 / 3.0)) < 1e-12
    checks += 1

    clone = FrozenQuantizedMLPRemnant.from_dense_mlp(
        dense, bits=4, group_size=4
    )
    clone.load_state_dict(remnant.state_dict(), strict=True)
    assert_close(clone.forward_all(x), full, label="state reload")
    checks += 1

    teacher = torch.tensor([[5.0, 0.0]])
    deleted = torch.tensor([[0.0, 0.0]])
    helpful = torch.tensor([[4.0, 0.0]])
    harmful = torch.tensor([[-2.0, 2.0]])
    assert float(marginal_recovery_utility(teacher, deleted, helpful)) > 0
    assert float(marginal_recovery_utility(teacher, deleted, harmful)) < 0
    checks += 1

    accumulator = SignificanceAccumulator(3)
    accumulator.update(torch.tensor([[1.0, 2.0, float("nan")], [3.0, 4.0, 9.0]]))
    summary = accumulator.summary()
    assert torch.equal(summary.observation_count, torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64))
    assert_close(
        summary.scores(1.0),
        torch.tensor([2.0, 3.0, 9.0], dtype=torch.float64),
        label="exposure-normalized significance",
    )
    checks += 1

    assert abs(recovery_ratio(0.8, 0.4, 1.0) - (2.0 / 3.0)) < 1e-12
    checks += 1

    print(f"SGQR verification passed: {checks}/{checks}")
    return checks


if __name__ == "__main__":
    run()
