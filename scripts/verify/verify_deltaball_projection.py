#!/usr/bin/env python3
"""Synthetic proof obligations for base-relative DeltaBall projection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from optim.delta_geometry import DeltaBallProjector  # noqa: E402


class FakeLora(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, rank: int, scale: float,
                 *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.base_layer = torch.nn.Linear(n_in, n_out, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict({"default":
            torch.nn.Linear(n_in, rank, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default":
            torch.nn.Linear(rank, n_out, bias=False)})
        with torch.no_grad():
            self.base_layer.weight.copy_(torch.randn(
                n_out, n_in, generator=generator))
            self.lora_A["default"].weight.copy_(torch.randn(
                rank, n_in, generator=generator) * 0.1)
            self.lora_B["default"].weight.copy_(torch.randn(
                n_out, rank, generator=generator) * 0.1)
        self.scaling = {"default": scale}


class FakeModel(torch.nn.Module):
    def __init__(self, *, seed: int = 1) -> None:
        super().__init__()
        self.target = FakeLora(11, 7, 3, 2.0, seed=seed)


def dense_delta(model: FakeModel) -> torch.Tensor:
    module = model.target
    return (module.scaling["default"] * module.lora_B["default"].weight
            @ module.lora_A["default"].weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()
    checks = []
    model = FakeModel(seed=10)
    base_before = model.target.base_layer.weight.detach().clone()
    a_before = model.target.lora_A["default"].weight.detach().clone()
    b_before = model.target.lora_B["default"].weight.detach().clone()
    inside = DeltaBallProjector(
        model, enabled=True, epsilon_max=100.0, warmin_steps=1,
        schedule="linear", run_id="inside")
    row = inside.project(1)[0]
    assert not row["boundary_hit"]
    assert torch.equal(model.target.lora_A["default"].weight, a_before)
    assert torch.equal(model.target.lora_B["default"].weight, b_before)
    checks.append("inside-ball factors unchanged")

    model = FakeModel(seed=11)
    base_before = model.target.base_layer.weight.detach().clone()
    delta_before = dense_delta(model).detach().clone()
    base_norm = float(base_before.float().norm())
    relative_before = float(delta_before.float().norm()) / base_norm
    epsilon = relative_before * 0.4
    outside_epsilon = epsilon
    outside = DeltaBallProjector(
        model, enabled=True, epsilon_max=epsilon, warmin_steps=1,
        schedule="linear", run_id="outside")
    a_before = model.target.lora_A["default"].weight.detach().clone()
    b_before = model.target.lora_B["default"].weight.detach().clone()
    row = outside.project(1)[0]
    expected_c = epsilon / relative_before
    expected_factor = expected_c ** 0.5
    assert row["boundary_hit"]
    assert abs(row["projection_c"] - expected_c) < 2e-6
    assert torch.allclose(model.target.lora_A["default"].weight,
                          a_before * expected_factor, atol=2e-6, rtol=2e-6)
    assert torch.allclose(model.target.lora_B["default"].weight,
                          b_before * expected_factor, atol=2e-6, rtol=2e-6)
    assert torch.allclose(dense_delta(model), delta_before * expected_c,
                          atol=3e-6, rtol=3e-6)
    assert abs(float(dense_delta(model).detach().float().norm()) / base_norm - epsilon) < 2e-6
    assert torch.equal(model.target.base_layer.weight, base_before)
    checks += ["outside delta projected to boundary", "symmetric sqrt(c) scaling",
               "dense product scaled by c", "frozen base untouched"]

    schedule = DeltaBallProjector(
        FakeModel(seed=12), enabled=True, epsilon_max=0.02, warmin_steps=4,
        schedule="linear", run_id="schedule")
    assert [schedule.epsilon_at(step) for step in range(1, 6)] == [
        0.005, 0.01, 0.015, 0.02, 0.02]
    checks.append("linear warm-in schedule")

    original = FakeModel(seed=13)
    transformed = FakeModel(seed=13)
    q = torch.tensor([[1.2, 0.1, 0.0], [0.0, 0.9, 0.2],
                      [0.1, 0.0, 1.1]], dtype=torch.float32)
    with torch.no_grad():
        b = transformed.target.lora_B["default"].weight
        a = transformed.target.lora_A["default"].weight
        b.copy_(b @ q)
        a.copy_(torch.linalg.solve(q, a))
    epsilon = 1.0e-4
    left = DeltaBallProjector(original, enabled=True, epsilon_max=epsilon,
                              warmin_steps=1, schedule="linear", run_id="left")
    right = DeltaBallProjector(transformed, enabled=True, epsilon_max=epsilon,
                               warmin_steps=1, schedule="linear", run_id="right")
    left.project(1); right.project(1)
    assert torch.allclose(dense_delta(original), dense_delta(transformed),
                          atol=5e-6, rtol=5e-6)
    checks.append("gauge-equivalent products project identically")

    state = outside.state_dict()
    restored_model = FakeModel(seed=11)
    restored = DeltaBallProjector(
        restored_model, enabled=True, epsilon_max=outside_epsilon,
        warmin_steps=1, schedule="linear", run_id="outside")
    restored.load_state_dict(state)
    assert restored.steps_projected == outside.steps_projected
    assert restored.matrix_checks == outside.matrix_checks
    assert restored.boundary_hits == outside.boundary_hits
    checks.append("projection counters restore exactly")

    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    rng_model = FakeModel(seed=14)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    DeltaBallProjector(rng_model, enabled=True, epsilon_max=0.01,
                       warmin_steps=1, schedule="linear", run_id="rng").project(1)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    checks.append("projection consumes no RNG")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "checks": len(checks),
        "check_names": checks,
        "sealed_data_accessed": False,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.evidence:
        path = Path(args.evidence)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
