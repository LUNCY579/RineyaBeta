"""B2 gradient-cosine helper: diagnosis only, no optimizer step."""

from __future__ import annotations

import math

import pytest

import torch

from dast.grad_conflict import conflict_from_losses, vector_cosine


def test_opposed_losses_have_cosine_minus_one():
    weight = torch.nn.Linear(4, 4, bias=False)
    data = torch.randn(3, 4)
    predicted = weight(data)
    lm = predicted.pow(2).mean()
    state = -predicted.pow(2).mean()
    report = conflict_from_losses(lm, state, list(weight.named_parameters()))
    assert abs(report["global_cosine"] + 1.0) < 1e-5
    assert report["fraction_modules_negative_cosine"] == 1.0


def test_identical_losses_have_cosine_plus_one():
    weight = torch.nn.Linear(3, 3, bias=False)
    data = torch.randn(2, 3)
    loss = weight(data).pow(2).mean()
    report = conflict_from_losses(loss, loss, list(weight.named_parameters()))
    assert abs(report["global_cosine"] - 1.0) < 1e-5


def test_zero_grad_cosine_is_nan():
    assert math.isnan(vector_cosine(torch.zeros(3), torch.ones(3)))


def test_lora_only_cosine_reconstructs_the_concatenated_cosine() -> None:
    """The per-module recombination must equal the true cosine of the concatenation.

    B2's headline `global_cosine` includes the state_projector, where grad_lm is
    identically zero because L_LM does not depend on it. That inflates ||g_state||
    without touching the dot product and shrinks |cosine| toward 0, making every arm
    look equally conflict-free. The corrected figure is rebuilt from per-module
    norms and cosines, so that reconstruction has to be exact.
    """

    import importlib.util
    import math
    import sys
    from pathlib import Path

    import torch

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "gradient_conflict_compare", scripts / "gradient_conflict_compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gradient_conflict_compare"] = module
    spec.loader.exec_module(module)

    torch.manual_seed(5)
    blocks = {"lora.a": (7, 1.0), "lora.b": (5, 1.0), "state_projector": (11, 0.0)}
    left_parts, right_parts, rows = [], [], []
    for name, (size, lm_scale) in blocks.items():
        left = torch.randn(size, dtype=torch.float64) * lm_scale
        right = torch.randn(size, dtype=torch.float64)
        left_norm, right_norm = float(left.norm()), float(right.norm())
        cosine = (
            float(torch.dot(left, right) / (left_norm * right_norm))
            if left_norm > 0 and right_norm > 0
            else float("nan")
        )
        rows.append(
            {"module": name, "cosine": cosine, "lm_norm": left_norm, "state_norm": right_norm}
        )
        if name != "state_projector":
            left_parts.append(left)
            right_parts.append(right)

    expected_left = torch.cat(left_parts)
    expected_right = torch.cat(right_parts)
    expected = float(
        torch.dot(expected_left, expected_right)
        / (expected_left.norm() * expected_right.norm())
    )
    assert module.lora_only_cosine({"modules": rows}) == pytest.approx(expected, abs=1e-12)

    # And it must genuinely differ from the diluted global figure.
    full_left = torch.cat([expected_left, torch.zeros(11, dtype=torch.float64)])
    full_right = torch.cat([expected_right, torch.randn(11, dtype=torch.float64)])
    diluted = float(
        torch.dot(full_left, full_right) / (full_left.norm() * full_right.norm())
    )
    assert abs(diluted) < abs(expected), "projector block must shrink |cosine|"
    assert math.copysign(1, diluted) == math.copysign(1, expected)
