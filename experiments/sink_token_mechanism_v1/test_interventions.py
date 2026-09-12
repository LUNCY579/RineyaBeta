import unittest
import torch

from sink_mechanism_v1.interventions import *


class TestSinkMechanism(unittest.TestCase):
    def test_locations(self):
        self.assertEqual(select_persistent_positions(20, 4, "prefix"), (0, 1, 2, 3))
        self.assertEqual(select_persistent_positions(20, 4, "interior"), (8, 9, 10, 11))
        self.assertEqual(select_persistent_positions(20, 4, "recent"), (16, 17, 18, 19))

    def test_exact_budget_and_causality(self):
        p = torch.arange(30)
        m = persistent_mask(p, p, window=12, persistent_positions=(0, 1, 2, 3))
        self.assertEqual(int(m[-1].sum()), 12)
        self.assertFalse(bool((m & (p[None] > p[:, None])).any()))

    def test_only_sink_rows_change(self):
        k = torch.randn(1, 2, 12, 4); v = torch.randn_like(k); p = torch.arange(12)
        ko, vo = apply_sink_intervention(k, v, p, (0, 1), SinkIntervention(2, content="zero_v"))
        self.assertTrue(torch.equal(ko, k)); self.assertTrue(torch.equal(vo[:, :, 2:], v[:, :, 2:]))

    def test_hidden_is_identity(self):
        k = torch.randn(1, 1, 8, 2); v = torch.randn_like(k); p = torch.arange(8)
        ko, vo = apply_sink_intervention(k, v, p, (0, 1), SinkIntervention(2, visible=False))
        self.assertTrue(torch.equal(ko, k)); self.assertTrue(torch.equal(vo, v))


if __name__ == "__main__": unittest.main()
