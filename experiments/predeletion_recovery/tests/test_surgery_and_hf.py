from types import SimpleNamespace

import torch
from torch import nn

from pdrr.hf import (
    install_layer_gate,
    install_recipient_probes,
    recipient_deltas,
    remove_recipient_probes,
    resolve_decoder_layers,
    scar_free_check,
)
from pdrr.surgery import count_parameters, deployment_state_is_clean


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, **kwargs):
        return (hidden_states + torch.tanh(self.linear(hidden_states)),)


class Backbone(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(width) for _ in range(layers)])


class TinyCausal(nn.Module):
    def __init__(self, width=8, layers=5):
        super().__init__()
        self.model = Backbone(width, layers)
        self.head = nn.Linear(width, 11, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=layers, max_window_layers=layers)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = input_ids
        for layer in self.model.layers:
            hidden = layer(hidden, use_cache=False)[0]
        return SimpleNamespace(logits=self.head(hidden))


def test_decoder_layer_resolution_and_gate_zero_physical_equivalence():
    torch.manual_seed(1)
    model = TinyCausal()
    _, layers = resolve_decoder_layers(model)
    assert len(layers) == 5
    install_layer_gate(model, 2, 0.0)
    source_params = count_parameters(model)
    sample = {"input_ids": torch.randn(2, 4, 8),
              "attention_mask": torch.ones(2, 4, dtype=torch.long)}
    compact, report = scar_free_check(model, sample, 2, atol=1e-6)
    assert report["passed"]
    assert report["actual_removed_parameters"] > 0
    assert count_parameters(compact) < source_params
    assert deployment_state_is_clean(compact)


def test_recipient_probe_measures_same_input_parameter_delta():
    torch.manual_seed(2)
    model = TinyCausal()
    probes = install_recipient_probes(model, [1, 3])
    sample = torch.randn(2, 4, 8)
    model(input_ids=sample)
    initial = recipient_deltas(probes)
    assert all(torch.allclose(delta, torch.zeros_like(delta), atol=1e-7) for delta in initial)
    with torch.no_grad():
        probes[1].module.linear.weight.add_(0.1)
    model(input_ids=sample)
    changed = recipient_deltas(probes)
    assert changed[0].abs().max() > 0
    remove_recipient_probes(model, probes)
    assert deployment_state_is_clean(model)
