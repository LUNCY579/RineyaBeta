#!/usr/bin/env python3
"""Deterministic mathematical and side-effect verification for RGE geometry."""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import determinism  # noqa: E402
from optim.delta_geometry import (  # noqa: E402
    DeltaGeometryTracker,
    append_geometry,
    effective_weight_metrics,
    low_rank_frobenius_norm,
    successive_delta_norm,
)


def assert_close(actual, expected, *, atol=2e-5, rtol=2e-5, label="value") -> None:
    a = torch.as_tensor(actual, dtype=torch.float64)
    e = torch.as_tensor(expected, dtype=torch.float64)
    if not torch.allclose(a, e, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: {a} != {e}")


def factors(m: int, r: int, n: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(m, r, generator=generator, dtype=torch.float64),
            torch.randn(r, n, generator=generator, dtype=torch.float64))


def verify_norms_and_scale() -> int:
    checks = 0
    for index, (m, r, n) in enumerate(((7, 3, 5), (64, 4, 11), (9, 4, 48))):
        b, a = factors(m, r, n, 100 + index)
        scale = 2.0  # alpha/r for the canonical alpha=2r contract
        dense = scale * (b @ a)
        assert_close(low_rank_frobenius_norm(b, a, scale=scale), dense.norm(),
                     label=f"delta norm {m}x{n}")
        b2, a2 = factors(m, r, n, 200 + index)
        dense_diff = scale * (b2 @ a2) - scale * (b @ a)
        assert_close(successive_delta_norm(
            b2, a2, b, a, next_scale=scale, prev_scale=scale), dense_diff.norm(),
            label=f"successive norm {m}x{n}")
        checks += 2
    return checks


def verify_effective_weight_metrics() -> int:
    b, a = factors(13, 4, 17, 301)
    base = torch.randn(13, 17, generator=torch.Generator().manual_seed(302),
                       dtype=torch.float64)
    scale = 1.75
    delta, effective = scale * (b @ a), base + scale * (b @ a)
    result = effective_weight_metrics(base, b, a, scale=scale)
    cosine = torch.sum(base * effective) / (base.norm() * effective.norm())
    assert_close(result["base_norm"], base.norm(), label="base norm")
    assert_close(result["delta_norm"], delta.norm(), label="delta norm")
    assert_close(result["relative_delta"], delta.norm() / base.norm(),
                 label="relative delta")
    assert_close(result["radius_drift"], effective.norm() / base.norm(),
                 label="radius drift")
    assert_close(result["angle_radians"], torch.acos(cosine.clamp(-1, 1)),
                 atol=5e-5, rtol=5e-5, label="angle")
    return 5


def verify_gauge_invariance() -> int:
    checks = 0
    for index, (m, r, n) in enumerate(((7, 3, 5), (64, 4, 11), (9, 4, 48))):
        b, a = factors(m, r, n, 400 + index)
        base = torch.randn(m, n, generator=torch.Generator().manual_seed(500 + index),
                           dtype=torch.float64)
        transforms = [
            torch.diag(torch.linspace(0.4, 2.0, r, dtype=torch.float64)),
            torch.eye(r, dtype=torch.float64)
            + 0.12 * torch.triu(torch.ones(r, r, dtype=torch.float64), diagonal=1),
        ]
        for q in transforms:
            bq, aq = b @ q, torch.linalg.solve(q, a)
            if not torch.allclose(b @ a, bq @ aq, atol=1e-10, rtol=1e-10):
                raise AssertionError("gauge transform changed dense delta")
            original = effective_weight_metrics(base, b, a, scale=2.0)
            changed = effective_weight_metrics(base, bq, aq, scale=2.0)
            for key in ("delta_norm", "relative_delta", "radius_drift",
                        "angle_radians", "base_delta_inner"):
                assert_close(changed[key], original[key], atol=8e-5, rtol=8e-5,
                             label=f"gauge {key}")
            if torch.allclose(b.norm(), bq.norm(), atol=1e-10, rtol=1e-10) and \
                    torch.allclose(a.norm(), aq.norm(), atol=1e-10, rtol=1e-10):
                raise AssertionError("raw factor norms unexpectedly appeared gauge invariant")
            checks += 6
    return checks


class FakeLora(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, rank: int, scale: float):
        super().__init__()
        self.base_layer = torch.nn.Linear(n_in, n_out, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict({"default":
            torch.nn.Linear(n_in, rank, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default":
            torch.nn.Linear(rank, n_out, bias=False)})
        self.scaling = {"default": scale}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.targets = torch.nn.ModuleList([
            FakeLora(11, 7, 3, 2.0), FakeLora(9, 13, 4, 1.5)])


def _numpy_state_equal(a, b) -> bool:
    return (a[0] == b[0] and np.array_equal(a[1], b[1]) and
            a[2:] == b[2:])


def verify_tracker_side_effects_and_schema() -> int:
    random.seed(601)
    np.random.seed(601)
    torch.manual_seed(601)
    model = FakeModel()
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)
    tracker = DeltaGeometryTracker(model, cadence_steps=1,
                                   include_radius_angle=True,
                                   run_id="unit", enabled=True)
    params = {name: value.detach().clone() for name, value in model.named_parameters()}
    grads = {name: (None if value.grad is None else value.grad.detach().clone())
             for name, value in model.named_parameters()}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    rows = tracker.measure(1)
    if random.getstate() != python_state:
        raise AssertionError("geometry consumed Python RNG")
    if not _numpy_state_equal(np.random.get_state(), numpy_state):
        raise AssertionError("geometry consumed NumPy RNG")
    if not torch.equal(torch.get_rng_state(), torch_state):
        raise AssertionError("geometry consumed torch RNG")
    for name, value in model.named_parameters():
        if not torch.equal(value, params[name]):
            raise AssertionError(f"geometry mutated parameter {name}")
        before_grad = grads[name]
        if before_grad is None:
            if value.grad is not None:
                raise AssertionError(f"geometry created gradient {name}")
        elif not torch.equal(value.grad, before_grad):
            raise AssertionError(f"geometry mutated gradient {name}")
    required = {"run_id", "step", "module", "base_norm", "delta_norm",
                "relative_delta", "effective_step_norm",
                "base_relative_effective_step", "radius_drift", "angle_radians",
                "implementation", "cadence_steps"}
    if len(rows) != 2 or any(required - set(row) for row in rows):
        raise AssertionError("geometry artifact rows do not satisfy schema")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "geometry.jsonl"
        append_geometry(path, rows)
        decoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if decoded != rows:
            raise AssertionError("geometry JSONL round trip changed records")
    return 4 + len(params) * 2 + len(rows)


def verify_recursive_fingerprint() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "optim").mkdir()
        (root / "top.py").write_text("VALUE = 1\n", encoding="utf-8")
        nested = root / "optim" / "delta.py"
        nested.write_text("VALUE = 1\n", encoding="utf-8")
        first = determinism.code_fingerprint(root)
        if set(first["files"]) != {"top.py", "optim/delta.py"}:
            raise AssertionError("recursive fingerprint did not use stable relative paths")
        nested.write_text("VALUE = 2\n", encoding="utf-8")
        second = determinism.code_fingerprint(root)
        drift = determinism.compare_code_fingerprint(first, second)
        if drift != ["optim/delta.py: modified"]:
            raise AssertionError(f"nested edit was not detected exactly: {drift}")
        (root / "optim" / "new.py").write_text("VALUE = 3\n", encoding="utf-8")
        third = determinism.code_fingerprint(root)
        if "optim/new.py: added" not in determinism.compare_code_fingerprint(second, third):
            raise AssertionError("nested addition was not detected")
        nested.unlink()
        fourth = determinism.code_fingerprint(root)
        if "optim/delta.py: removed" not in determinism.compare_code_fingerprint(third, fourth):
            raise AssertionError("nested removal was not detected")
    return 3


def main() -> None:
    checks = 0
    checks += verify_norms_and_scale()
    checks += verify_effective_weight_metrics()
    checks += verify_gauge_invariance()
    checks += verify_tracker_side_effects_and_schema()
    checks += verify_recursive_fingerprint()
    print(json.dumps({"status": "PASS", "checks": checks,
                      "dense_products_in_runtime_norms": False,
                      "gauge_aspect_ratios": [[7, 3, 5], [64, 4, 11], [9, 4, 48]],
                      "rng_side_effect": False,
                      "parameter_or_gradient_side_effect": False}, indent=2))


if __name__ == "__main__":
    main()
