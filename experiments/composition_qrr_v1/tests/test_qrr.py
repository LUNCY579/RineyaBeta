import io
import unittest
import torch
from torch import nn
from experiments.composition_qrr_v1.identity import ContractError
from experiments.composition_qrr_v1.qrr import (QueryReadRevise, MatchedMLP, add_logit_residual,
                                              arm_table, initialized_controls)

torch.set_num_threads(1)


class QRRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.m = QueryReadRevise(16, 8, 2)
        self.q = torch.randn(2, 16)
        self.h = torch.randn(2, 6, 16)
        self.v = torch.ones(2, 6, dtype=torch.bool)
        self.qp = torch.tensor([2, 3])
        self.kp = torch.arange(6).expand(2, -1)

    def run_module(self, evidence=None, trace=True):
        return self.m(self.q, self.h if evidence is None else evidence,
                      self.v, self.qp, self.kp, trace)

    def open_output(self):
        with torch.no_grad(): self.m.output.weight.normal_(std=.03)

    def test_zero_output_identity(self):
        r = self.run_module().residual
        self.assertEqual(torch.count_nonzero(r).item(), 0)
        self.assertTrue(torch.equal(self.q + r, self.q))

    def test_zero_logit_bridge_identity(self):
        head = nn.Linear(16, 23, bias=False)
        logits = torch.randn(2, 23)
        self.assertTrue(torch.equal(add_logit_residual(logits, self.run_module().residual, head), logits))

    def test_bridge_rejects_biased_head(self):
        with self.assertRaises(ContractError):
            add_logit_residual(torch.randn(2, 23), self.run_module().residual, nn.Linear(16, 23))

    def test_bridge_rejects_wrong_vocab(self):
        with self.assertRaises(ContractError):
            add_logit_residual(torch.randn(2, 24), self.run_module().residual, nn.Linear(16, 23, bias=False))

    def test_future_evidence_cannot_leak(self):
        self.open_output()
        old = self.run_module()
        changed = self.h.clone()
        changed[self.kp > self.qp[:, None]] = 10000
        new = self.run_module(changed)
        self.assertTrue(torch.equal(old.residual, new.residual))
        for a in old.attention:
            self.assertEqual(a[self.kp > self.qp[:, None]].sum().item(), 0)
            self.assertTrue(torch.allclose(a.sum(-1), torch.ones(2)))

    def test_invisible_nan_does_not_leak(self):
        self.open_output()
        old = self.run_module().residual
        changed = self.h.clone(); changed[self.kp > self.qp[:, None]] = float('nan')
        self.assertTrue(torch.equal(old, self.run_module(changed).residual))

    def test_visible_nan_rejected(self):
        self.h[0, 0, 0] = float('nan')
        with self.assertRaises(ContractError): self.run_module()

    def test_empty_visibility_rejected(self):
        self.v[0] = False
        with self.assertRaises(ContractError): self.run_module()

    def test_float_mask_rejected(self):
        self.v = self.v.float()
        with self.assertRaises(ContractError): self.run_module()

    def test_float_positions_rejected(self):
        self.qp = self.qp.float()
        with self.assertRaises(ContractError): self.run_module()

    def test_batch_isolation(self):
        self.open_output()
        full = self.run_module().residual
        for i in range(2):
            one = self.m(self.q[i:i+1], self.h[i:i+1], self.v[i:i+1], self.qp[i:i+1], self.kp[i:i+1]).residual
            self.assertTrue(torch.allclose(full[i:i+1], one, atol=1e-6))

    def test_no_persistent_state(self):
        self.open_output()
        old = self.run_module().residual.clone()
        self.m(self.q * 3, self.h * 2, self.v, self.qp, self.kp)
        self.assertTrue(torch.equal(old, self.run_module().residual))

    def test_live_first_backward_output(self):
        self.run_module().residual.sum().backward()
        self.assertGreater(torch.count_nonzero(self.m.output.weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(self.m.key.weight.grad).item(), 0)

    def test_controller_gradient_after_opening(self):
        self.open_output()
        self.run_module().residual.square().sum().backward()
        for name, p in self.m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(torch.count_nonzero(p.grad).item(), 0, name)

    def test_read_attention_actually_changes(self):
        r = self.run_module()
        self.assertFalse(torch.equal(r.attention[0], r.attention[1]))

    def test_equal_parameter_counts_across_tied_reads(self):
        counts = [QueryReadRevise(16, 8, r).parameter_count() for r in (1, 2, 4)]
        self.assertEqual(len(set(counts)), 1)
        self.assertEqual(counts[0], 832)

    def test_registered_mlp_exact_match(self):
        table = arm_table()
        self.assertEqual(table['arms'][0]['parameters'], 282624)
        self.assertEqual(table['arms'][-1]['parameters'], 282624)
        self.assertTrue(table['arms'][-1]['exact_parameter_match'])
        self.assertFalse(table['compute_matched'])

    def test_untied_is_explicitly_larger(self):
        self.assertGreater(QueryReadRevise(16, 8, 2, False).parameter_count(), self.m.parameter_count())

    def test_initialization_matched_and_rng_restored(self):
        state = torch.random.get_rng_state().clone()
        controls = initialized_controls(16, 8)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for k, v in controls['one'].state_dict().items():
            self.assertTrue(torch.equal(v, controls['four'].state_dict()[k]))
        for a, b in zip(controls['untied_two'].steps[0].parameters(), controls['untied_two'].steps[1].parameters()):
            self.assertTrue(torch.equal(a, b))
            self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_save_load_roundtrip(self):
        self.open_output()
        buffer = io.BytesIO(); torch.save(self.m.state_dict(), buffer); buffer.seek(0)
        clone = QueryReadRevise(16, 8, 2)
        clone.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(self.run_module().residual,
            clone(self.q, self.h, self.v, self.qp, self.kp).residual))

    def test_mlp_has_live_output_gradient(self):
        m = MatchedMLP(16, 832)
        m(self.q).sum().backward()
        self.assertGreater(torch.count_nonzero(m.output.weight.grad).item(), 0)

    def test_invalid_read_count(self):
        for reads in (0, 3, True):
            with self.assertRaises(ContractError): QueryReadRevise(16, 8, reads)
