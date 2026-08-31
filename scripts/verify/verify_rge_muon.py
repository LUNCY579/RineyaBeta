#!/usr/bin/env python3
"""CPU proof obligations for the registered RGE-PEFT Muon optimizer route."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts" / "train"
sys.path.insert(0, str(TRAIN))

from optim.factory import build_optimizer, optimizer_contract  # noqa: E402
from optim.muon import (  # noqa: E402
    MUON_IMPLEMENTATION_VERSION,
    Muon,
    get_muon_scale_factor,
    newton_schulz_quintic,
)


class TinyLoRA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).view(4, 6) / 50)
        self.lora_b = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).view(5, 4) / 40)


class TinyLoRAWithFrozenBase(TinyLoRA):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.arange(30, dtype=torch.float32).view(5, 6),
                                              requires_grad=False)


ADAMW = {
    "name": "adamw", "implementation": "torch.optim.AdamW", "version": "test",
    "lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01,
}
MUON = {
    "name": "muon", "implementation": "rge.optim.Muon", "version": MUON_IMPLEMENTATION_VERSION,
    "lr": 1.0e-3, "momentum": 0.95, "nesterov": True, "weight_decay": 0.0,
    "ns_steps": 5, "ns_eps": 1.0e-15, "coefficient_type": "quintic",
    "scale_mode": "spectral", "extra_scale_factor": 0.2,
    "optimizer_math_dtype": "float32",
}


def reference_newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Independent literal reference for the pinned five-step update."""
    coefficients = (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    )
    value = matrix.detach().float().clone()
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.mT
    norm = torch.linalg.vector_norm(value)
    value = value / norm.clamp_min(1.0e-15)
    for index in range(steps):
        a, b, c = coefficients[index % len(coefficients)]
        gram = value @ value.mT
        value = a * value + (b * gram + c * (gram @ gram)) @ value
    return value.mT.contiguous() if transposed else value


def reference_muon_update(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    buffer: torch.Tensor,
    *,
    lr: float,
    momentum: float,
    nesterov: bool,
    weight_decay: float,
    scale_mode: str,
    extra_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_parameter = parameter.clone()
    if weight_decay:
        next_parameter.mul_(1.0 - lr * weight_decay)
    next_buffer = buffer.lerp(gradient, 1.0 - momentum)
    update = gradient.lerp(next_buffer, momentum) if nesterov else next_buffer
    orthogonal = reference_newton_schulz(update)
    scale = get_muon_scale_factor(
        parameter.shape[-2], parameter.shape[-1], mode=scale_mode
    )
    next_parameter.add_(orthogonal, alpha=-lr * scale * extra_scale_factor)
    return next_parameter, next_buffer


def check(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()
    torch.manual_seed(20260830)
    checks: list[str] = []

    check(get_muon_scale_factor(8, 2, mode="shape_scaling") == 2.0,
          "shape scaling formula", checks)
    check(get_muon_scale_factor(8, 2, mode="spectral") == 8 ** 0.5,
          "spectral scaling formula", checks)
    check(get_muon_scale_factor(8, 2, mode="unit_rms_norm") == 2.0,
          "unit-RMS scaling formula", checks)
    try:
        get_muon_scale_factor(8, 2, mode="unknown")
    except ValueError:
        checks.append("invalid scale mode rejected")
    else:
        raise AssertionError("invalid scale mode was accepted")

    matrix = torch.randn(9, 4, dtype=torch.float32)
    original = matrix.clone()
    orth = newton_schulz_quintic(matrix, steps=5)
    check(torch.equal(matrix, original), "Newton-Schulz does not mutate input", checks)
    check(orth.shape == matrix.shape and orth.dtype == torch.float32,
          "Newton-Schulz preserves shape and FP32 contract", checks)
    check(bool(torch.isfinite(orth).all()), "Newton-Schulz output is finite", checks)
    transposed = newton_schulz_quintic(matrix.mT.contiguous(), steps=5).mT
    check(torch.allclose(orth, transposed, atol=2e-6, rtol=2e-6),
          "Newton-Schulz is transpose-consistent", checks)
    normalized = torch.nn.functional.normalize(matrix, p=2, dim=(-2, -1))
    check(torch.linalg.svdvals(orth).std() < torch.linalg.svdvals(normalized).std(),
          "Newton-Schulz flattens the singular-value spectrum", checks)
    zero = newton_schulz_quintic(torch.zeros(3, 7), steps=5)
    check(torch.equal(zero, torch.zeros_like(zero)), "zero update remains zero", checks)
    try:
        newton_schulz_quintic(torch.ones(2, 3, 4))
    except ValueError:
        checks.append("non-matrix Newton-Schulz input rejected")
    else:
        raise AssertionError("non-matrix input was accepted")
    reference = reference_newton_schulz(matrix)
    check(torch.equal(orth, reference),
          "Newton-Schulz matches independent pinned reference", checks)
    rank_one = newton_schulz_quintic(torch.linspace(-1.0, 1.0, 64).view(1, 64))
    check(rank_one.shape == (1, 64) and bool(torch.isfinite(rank_one).all()),
          "rank-1 Newton-Schulz input remains finite", checks)
    for shape in ((16, 1024), (1024, 16), (16, 3072), (3072, 16)):
        rectangular = newton_schulz_quintic(torch.linspace(-1.0, 1.0,
                                                            math.prod(shape)).view(shape))
        check(rectangular.shape == shape and bool(torch.isfinite(rectangular).all()),
              f"rectangular LoRA aspect {shape} remains finite", checks)
    for nonfinite in (float("nan"), float("inf")):
        invalid = torch.ones(4, 4)
        invalid[0, 0] = nonfinite
        try:
            newton_schulz_quintic(invalid)
        except FloatingPointError:
            checks.append(f"Newton-Schulz rejects {nonfinite!r}")
        else:
            raise AssertionError("non-finite Newton-Schulz input was accepted")

    # Compare two stateful optimizer steps against a literal reference path.
    parameter = torch.nn.Parameter(torch.linspace(-0.4, 0.6, 24).view(4, 6))
    optimizer_under_test = Muon(
        [parameter], lr=1.0e-3, momentum=0.95, nesterov=True,
        weight_decay=0.01, ns_steps=5, ns_eps=1.0e-15,
        scale_mode="spectral", extra_scale_factor=0.2,
    )
    expected_parameter = parameter.detach().clone()
    expected_buffer = torch.zeros_like(expected_parameter)
    for offset in (0.0, 0.125):
        gradient = torch.linspace(-0.3 + offset, 0.5 + offset, 24).view(4, 6)
        expected_parameter, expected_buffer = reference_muon_update(
            expected_parameter, gradient, expected_buffer, lr=1.0e-3,
            momentum=0.95, nesterov=True, weight_decay=0.01,
            scale_mode="spectral", extra_scale_factor=0.2,
        )
        parameter.grad = gradient.clone()
        optimizer_under_test.step()
    check(torch.equal(parameter.detach(), expected_parameter),
          "Muon two-step trajectory matches independent reference", checks)
    check(torch.equal(optimizer_under_test.state[parameter]["momentum_buffer"],
                      expected_buffer),
          "Muon momentum state matches independent reference", checks)
    check(optimizer_under_test.state[parameter]["momentum_buffer"].dtype == torch.float32,
          "Muon momentum state remains FP32", checks)

    model = TinyLoRA()
    contract = optimizer_contract(model, MUON)
    check(contract["name"] == "muon", "factory emits Muon contract", checks)
    check(contract["fallback_policy"] == "reject_non_matrix_trainables",
          "Muon contract has no implicit fallback", checks)
    check(contract["parameter_groups"][0]["tensor_count"] == 2,
          "Muon contract accounts for every trainable tensor", checks)
    check(contract["numerics"]["optimizer_math_dtype"] == "float32",
          "Muon contract freezes FP32 optimizer math", checks)
    optimizer = build_optimizer(model, MUON)
    check(isinstance(optimizer, Muon), "factory builds the registered Muon class", checks)

    frozen_model = TinyLoRAWithFrozenBase()
    frozen_before = frozen_model.base_weight.detach().clone()
    frozen_optimizer = build_optimizer(frozen_model, MUON)
    for trainable in (frozen_model.lora_a, frozen_model.lora_b):
        trainable.grad = torch.full_like(trainable, 0.25)
    frozen_optimizer.step()
    check(torch.equal(frozen_model.base_weight, frozen_before),
          "Muon routing never mutates frozen base weights", checks)

    candidate_path = ROOT / "configs" / "rge_peft" / "m2_o_m_muon.yaml"
    control_path = ROOT / "configs" / "rge_peft" / "m2_o_a_adamw.yaml"
    candidate_cfg = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    control_cfg = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    registered = optimizer_contract(TinyLoRA(), candidate_cfg["optimizer"])
    check(registered["version"] == MUON_IMPLEMENTATION_VERSION,
          "registered candidate config resolves to pinned Muon implementation", checks)
    check(optimizer_contract(TinyLoRA(), control_cfg["optimizer"])["name"] == "adamw",
          "registered control config resolves to AdamW", checks)
    invalid_cfg = copy.deepcopy(candidate_cfg["optimizer"])
    invalid_cfg["coefficient_type"] = "unregistered"
    try:
        optimizer_contract(TinyLoRA(), invalid_cfg)
    except ValueError:
        checks.append("unregistered Muon coefficient schedule rejected")
    else:
        raise AssertionError("unregistered coefficient schedule was accepted")

    bad = TinyLoRA()
    bad.extra = torch.nn.Parameter(torch.ones(3))
    try:
        optimizer_contract(bad, MUON)
    except ValueError:
        checks.append("non-matrix trainable rejected by routing contract")
    else:
        raise AssertionError("non-matrix trainable silently entered Muon")
    try:
        Muon([torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float16))],
             lr=1e-3, momentum=0.95, nesterov=True, weight_decay=0.0,
             ns_steps=5, ns_eps=1e-15, scale_mode="spectral",
             extra_scale_factor=0.2)
    except TypeError:
        checks.append("non-FP32 Muon parameter rejected")
    else:
        raise AssertionError("non-FP32 Muon parameter was accepted")
    nonfinite_parameter = torch.nn.Parameter(torch.ones(2, 2))
    nonfinite_optimizer = Muon(
        [nonfinite_parameter], lr=1e-3, momentum=0.95, nesterov=True,
        weight_decay=0.0, ns_steps=5, ns_eps=1e-15,
        scale_mode="spectral", extra_scale_factor=0.2)
    nonfinite_parameter.grad = torch.full_like(nonfinite_parameter, float("nan"))
    try:
        nonfinite_optimizer.step()
    except FloatingPointError:
        checks.append("non-finite Muon gradient rejected")
    else:
        raise AssertionError("non-finite Muon gradient was accepted")

    # State-dict continuity is the prerequisite for exact interrupted resume.
    first = TinyLoRA()
    opt_first = build_optimizer(first, MUON)
    for parameter in first.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    opt_first.step()
    state = copy.deepcopy(opt_first.state_dict())

    resumed = TinyLoRA()
    with torch.no_grad():
        for target, source in zip(resumed.parameters(), first.parameters()):
            target.copy_(source)
    opt_resumed = build_optimizer(resumed, MUON)
    opt_resumed.load_state_dict(state)
    for left, right in zip(first.parameters(), resumed.parameters()):
        left.grad = torch.linspace(-0.2, 0.3, left.numel()).view_as(left)
        right.grad = left.grad.clone()
    opt_first.step()
    opt_resumed.step()
    check(all(torch.equal(a, b) for a, b in zip(first.parameters(), resumed.parameters())),
          "Muon state_dict resumes the next update bit-exactly on CPU", checks)

    # Identical trajectories must stay identical without consuming RNG.
    left, right = TinyLoRA(), TinyLoRA()
    right.load_state_dict(copy.deepcopy(left.state_dict()))
    left_opt, right_opt = build_optimizer(left, MUON), build_optimizer(right, MUON)
    rng_before = torch.get_rng_state().clone()
    for step in range(3):
        for index, (a, b) in enumerate(zip(left.parameters(), right.parameters())):
            gradient = (
                torch.arange(a.numel(), dtype=torch.float32).view_as(a)
                / (100 + step + index)
            )
            a.grad, b.grad = gradient, gradient.clone()
        left_opt.step(); right_opt.step()
    check(torch.equal(rng_before, torch.get_rng_state()), "Muon consumes no RNG", checks)
    check(all(torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters())),
          "Muon repeated trajectory is deterministic", checks)

    # Decoupled decay remains independent of the gradient stream.
    decay_parameter = torch.nn.Parameter(torch.full((2, 2), 2.0))
    decay_optimizer = Muon(
        [decay_parameter], lr=0.1, momentum=0.0, nesterov=False,
        weight_decay=0.25, ns_steps=1, ns_eps=1e-15,
        scale_mode="shape_scaling", extra_scale_factor=1.0,
    )
    decay_parameter.grad = torch.zeros_like(decay_parameter)
    decay_optimizer.step()
    check(torch.equal(decay_parameter, torch.full((2, 2), 1.95)),
          "Muon applies decoupled weight decay", checks)

    # The AdamW branch must remain numerically identical to the prior factory.
    via_factory, direct = TinyLoRA(), TinyLoRA()
    direct.load_state_dict(copy.deepcopy(via_factory.state_dict()))
    factory_opt = build_optimizer(via_factory, ADAMW)
    direct_opt = torch.optim.AdamW(
        list(direct.parameters()), lr=ADAMW["lr"], betas=tuple(ADAMW["betas"]),
        eps=ADAMW["eps"], weight_decay=ADAMW["weight_decay"])
    for a, b in zip(via_factory.parameters(), direct.parameters()):
        gradient = torch.linspace(-1.0, 1.0, a.numel()).view_as(a)
        a.grad, b.grad = gradient, gradient.clone()
    factory_opt.step(); direct_opt.step()
    check(all(torch.equal(a, b) for a, b in zip(via_factory.parameters(), direct.parameters())),
          "AdamW factory path remains bit-identical to torch.optim.AdamW", checks)
    check(optimizer_contract(via_factory, ADAMW)["name"] == "adamw",
          "factory still emits AdamW control contract", checks)

    result = {
        "schema_version": 1,
        "status": "PASS",
        "checks": len(checks),
        "check_names": checks,
        "torch": torch.__version__,
        "device": "cpu",
        "muon_implementation": MUON_IMPLEMENTATION_VERSION,
        "sealed_data_accessed": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.evidence:
        path = Path(args.evidence)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
