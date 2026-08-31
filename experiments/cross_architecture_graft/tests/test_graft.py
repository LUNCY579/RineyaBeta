from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from experiments.cross_architecture_graft.src.bridges import IdentityBridge, LinearBridge
from experiments.cross_architecture_graft.src.graft import (
    GateSpec,
    GraftBranch,
    GraftedDecoderLayer,
    ResidualGate,
    configure_graft_training,
)


class TupleHost(nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states), torch.tensor(123)


class Delta(nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)


class GraftTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def _layer(self, gate_kind: str = "scalar") -> GraftedDecoderLayer:
        d_model = 8
        branch = GraftBranch(
            host_dim=d_model,
            foreign_dim=d_model,
            input_bridge=IdentityBridge(d_model, d_model),
            donor=Delta(d_model),
            output_bridge=IdentityBridge(d_model, d_model),
            gate=ResidualGate(d_model, GateSpec(kind=gate_kind)),
        )
        return GraftedDecoderLayer(TupleHost(d_model), branch)

    def test_zero_gate_preserves_tensor_and_aux_bit_exactly(self) -> None:
        layer = self._layer().eval()
        x = torch.randn(2, 4, 8)
        with torch.no_grad():
            reference = layer.host_layer(x)
            output = layer(x)
        self.assertTrue(torch.equal(reference[0], output[0]))
        self.assertTrue(torch.equal(reference[1], output[1]))

    def test_gate_kinds_start_at_exact_zero(self) -> None:
        x = torch.randn(2, 4, 8)
        for kind in ("scalar", "channel", "token"):
            with self.subTest(kind=kind):
                gate = ResidualGate(8, GateSpec(kind=kind))
                self.assertTrue(gate.is_exactly_zero())
                self.assertEqual(torch.count_nonzero(gate(x)).item(), 0)

    def test_nonzero_gate_changes_only_first_tuple_element(self) -> None:
        layer = self._layer().eval()
        assert layer.graft_branch.gate.raw is not None
        layer.graft_branch.gate.raw.data.fill_(0.5)
        x = torch.randn(2, 4, 8)
        with torch.no_grad():
            reference = layer.host_layer(x)
            output = layer(x)
        self.assertFalse(torch.equal(reference[0], output[0]))
        self.assertTrue(torch.equal(reference[1], output[1]))

    def test_trainability_boundary_freezes_host_and_donor(self) -> None:
        layer = self._layer()
        configure_graft_training(layer, freeze_host=True, freeze_donor=True)
        self.assertFalse(any(p.requires_grad for p in layer.host_layer.parameters()))
        self.assertFalse(any(p.requires_grad for p in layer.graft_branch.donor.parameters()))
        self.assertTrue(any(p.requires_grad for p in layer.graft_branch.gate.parameters()))

    def test_nan_donor_cannot_contaminate_exact_zero_gate(self) -> None:
        class NaNDonor(nn.Module):
            def forward(self, hidden_states, **kwargs):
                return torch.full_like(hidden_states, float("nan"))

        branch = GraftBranch(
            host_dim=8,
            foreign_dim=8,
            input_bridge=LinearBridge(8, 8),
            donor=NaNDonor(),
            output_bridge=LinearBridge(8, 8),
            gate=ResidualGate(8, GateSpec()),
        )
        layer = GraftedDecoderLayer(copy.deepcopy(TupleHost()), branch, exact_eval_bypass=False)
        output = layer(torch.randn(1, 3, 8))[0]
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
