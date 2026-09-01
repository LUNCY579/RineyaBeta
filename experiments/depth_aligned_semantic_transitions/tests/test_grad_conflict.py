"""B2 gradient-cosine helper: diagnosis only, no optimizer step."""

from __future__ import annotations

import math

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
