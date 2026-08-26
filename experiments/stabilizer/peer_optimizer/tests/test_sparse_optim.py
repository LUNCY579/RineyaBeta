from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from peer_stabilizer import audit_sparse_delta  # noqa: E402
from sparse_optim import SparseNativeAdamW, SparseParameterSpec  # noqa: E402


class SparseNativeAdamWTests(unittest.TestCase):
    def test_only_selected_coordinates_move_with_momentum_and_decay(self) -> None:
        initial = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        parameter = nn.Parameter(initial.clone())
        mask = torch.zeros_like(parameter)
        mask[0, 1] = 1
        mask[1, 3] = 1
        mask[2, 0] = 1
        scale = torch.ones_like(parameter) * 0.75
        optimizer = SparseNativeAdamW(
            [SparseParameterSpec(parameter, mask, scale, name="target")],
            lr=0.05,
            weight_decay=0.1,
        )

        for step in range(8):
            parameter.grad = torch.full_like(parameter, 0.2 + 0.1 * step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        audit = audit_sparse_delta(initial, parameter, mask)
        self.assertTrue(audit.exact)
        self.assertEqual(audit.changed_outside, 0)
        self.assertEqual(audit.changed_inside, 3)
        self.assertTrue(torch.equal(parameter.detach()[mask == 0], initial[mask == 0]))
        self.assertEqual(optimizer.selected_parameter_count(), 3)
        self.assertAlmostEqual(optimizer.support_fraction(), 0.25)

        state = optimizer.state[parameter]
        self.assertTrue(torch.equal(state["exp_avg"][mask == 0], torch.zeros(9)))
        self.assertTrue(torch.equal(state["exp_avg_sq"][mask == 0], torch.zeros(9)))

    def test_zero_gradient_does_not_apply_decay_outside_mask(self) -> None:
        parameter = nn.Parameter(torch.ones(2, 3))
        initial = parameter.detach().clone()
        mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        optimizer = SparseNativeAdamW(
            [SparseParameterSpec(parameter, mask)], lr=0.1, weight_decay=0.5
        )
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        self.assertLess(float(parameter[0, 0].item()), 1.0)
        self.assertTrue(torch.equal(parameter.detach()[mask == 0], initial[mask == 0]))

    def test_nonbinary_mask_rejected(self) -> None:
        parameter = nn.Parameter(torch.ones(2, 2))
        with self.assertRaises(ValueError):
            SparseNativeAdamW(
                [SparseParameterSpec(parameter, torch.full((2, 2), 0.5))]
            )

    def test_shape_mismatch_rejected(self) -> None:
        parameter = nn.Parameter(torch.ones(2, 2))
        with self.assertRaises(ValueError):
            SparseNativeAdamW(
                [SparseParameterSpec(parameter, torch.ones(2, 3))]
            )

    def test_step_scale_multiplies_realized_adam_update(self) -> None:
        parameter = nn.Parameter(torch.zeros(1, 2))
        mask = torch.ones_like(parameter)
        scale = torch.tensor([[0.25, 1.0]])
        optimizer = SparseNativeAdamW(
            [SparseParameterSpec(parameter, mask, scale)],
            lr=0.1,
            weight_decay=0.0,
        )
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        delta = -parameter.detach().reshape(-1)
        self.assertGreater(float(delta[1].item()), 0.0)
        self.assertAlmostEqual(
            float((delta[0] / delta[1]).item()), 0.25, places=5
        )

    def test_negative_or_nonfinite_step_scale_rejected(self) -> None:
        parameter = nn.Parameter(torch.ones(2, 2))
        with self.assertRaises(ValueError):
            SparseNativeAdamW(
                [SparseParameterSpec(parameter, torch.ones(2, 2), -0.1)]
            )
        with self.assertRaises(ValueError):
            SparseNativeAdamW(
                [SparseParameterSpec(
                    parameter, torch.ones(2, 2), torch.full((2, 2), float("nan"))
                )]
            )


class LowPrecisionStateTests(unittest.TestCase):
    def test_bfloat16_parameter_uses_float32_state_and_keeps_frozen_bits(self) -> None:
        parameter = nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))
        initial = parameter.detach().clone()
        mask = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.bfloat16
        )
        optimizer = SparseNativeAdamW(
            [SparseParameterSpec(parameter, mask)], lr=0.05, weight_decay=0.01
        )
        parameter.grad = torch.full_like(parameter, 0.25)
        optimizer.step()
        state = optimizer.state[parameter]
        self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
        self.assertTrue(torch.equal(parameter.detach()[mask == 0], initial[mask == 0]))


if __name__ == "__main__":
    unittest.main()
