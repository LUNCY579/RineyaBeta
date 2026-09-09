from types import SimpleNamespace
import unittest
import torch
from torch import nn
from experiments.composition_qrr_v1.identity import ContractError
from experiments.composition_qrr_v1.native import encode, greedy_trace


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [3] if text == 'UNKNOWN' else [1] * len(text)

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(map(str, ids))


class ToyCacheModel(nn.Module):
    def __init__(self, broken=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.broken = broken

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        cumulative = input_ids.cumsum(-1)
        if past_key_values is not None:
            cumulative = cumulative + past_key_values + int(self.broken)
        logits = torch.zeros(*input_ids.shape, 5)
        logits.scatter_(-1, (cumulative % 5).unsqueeze(-1), 5.)
        return SimpleNamespace(logits=logits,
            past_key_values=cumulative[:, -1:] if use_cache else None)


class NativeFixtureTests(unittest.TestCase):
    def test_no_truncation(self):
        with self.assertRaises(ContractError): encode(Tokenizer(), 'abc', 2)

    def test_empty_input_rejected(self):
        with self.assertRaises(ContractError): encode(Tokenizer(), '', 2)

    def test_cached_greedy_eos_and_full_trace(self):
        out = greedy_trace(ToyCacheModel(), Tokenizer(), [1, 1], 4, [4], True)
        self.assertEqual(out['token_ids'], [2, 4])
        self.assertEqual(out['termination'], 'eos')
        self.assertEqual(out['output_tokens_including_eos'], 2)
        self.assertEqual(len(out['steps']), 2)
        self.assertTrue(all(r['parity_max_abs'] == 0 for r in out['steps']))

    def test_forced_four_steps_for_mechanical_qualification(self):
        out = greedy_trace(ToyCacheModel(), Tokenizer(), [1, 1], 4, [], True)
        self.assertEqual(len(out['steps']), 4)
        self.assertEqual(out['termination'], 'budget')

    def test_wrong_cache_rejected(self):
        with self.assertRaises(ContractError):
            greedy_trace(ToyCacheModel(True), Tokenizer(), [1, 1], 4, [], True)

    def test_zero_token_budget_rejected(self):
        with self.assertRaises(ContractError):
            greedy_trace(ToyCacheModel(), Tokenizer(), [1, 1], 0, [4])
