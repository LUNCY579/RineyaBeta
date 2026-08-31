from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from experiments.cross_architecture_graft.src.bridges import IdentityBridge
from experiments.cross_architecture_graft.src.graft import GateSpec, GraftBranch, ResidualGate
from experiments.cross_architecture_graft.src.integration import (
    attach_grafts,
    find_layer_container,
    graft_state_dict,
    load_graft_pack,
    save_graft_pack,
)


class Layer(nn.Module):
    def __init__(self, d_model=8):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, **kwargs):
        return (self.linear(hidden_states),)


class Delta(nn.Module):
    def __init__(self, d_model=8):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(3)])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()


class IntegrationTests(unittest.TestCase):
    def _factory(self, index, layer):
        return GraftBranch(
            host_dim=8,
            foreign_dim=8,
            input_bridge=IdentityBridge(8, 8),
            donor=Delta(),
            output_bridge=IdentityBridge(8, 8),
            gate=ResidualGate(8, GateSpec()),
        )

    def test_common_qwen_layer_path_is_found(self) -> None:
        path, layers = find_layer_container(Model())
        self.assertEqual(path, "model.layers")
        self.assertEqual(len(layers), 3)

    def test_attach_is_additive_and_indices_are_checked(self) -> None:
        model = Model()
        attachments = attach_grafts(model, [1], self._factory)
        self.assertEqual(attachments[0].layer_index, 1)
        self.assertEqual(model.model.layers[0].__class__.__name__, "Layer")
        self.assertEqual(model.model.layers[1].__class__.__name__, "GraftedDecoderLayer")
        with self.assertRaises(IndexError):
            attach_grafts(Model(), [99], self._factory)

    def test_graft_pack_round_trip(self) -> None:
        model = Model()
        attach_grafts(model, [1], self._factory, freeze_donor=False)
        gate = model.model.layers[1].graft_branch.gate
        assert gate.raw is not None
        gate.raw.data.fill_(0.25)
        with tempfile.TemporaryDirectory() as tmp:
            weights, _ = save_graft_pack(model, tmp, {"test": True})
            state = graft_state_dict(model)
            fresh = Model()
            attach_grafts(fresh, [1], self._factory, freeze_donor=False)
            missing, unexpected = load_graft_pack(fresh, weights)
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
            fresh_state = graft_state_dict(fresh)
            self.assertEqual(set(state), set(fresh_state))
            for key in state:
                torch.testing.assert_close(state[key], fresh_state[key])


if __name__ == "__main__":
    unittest.main()
