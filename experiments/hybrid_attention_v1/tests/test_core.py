import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ..attention import CompactKV, allowed_mask, blocked_attention, dense_oracle, storage_bytes
from ..config import ARMS, Spec, arm_spec
from ..diagnostics import (PRIMARY_METRIC, make_cases, paired_result, score,
                           summarize)
from ..provenance import (checkpoint_inventory, code_identity, digest, environment,
                          new_run, require_receipt, write_json)
from ..runtime import HeadGate, HybridRuntime
from ..run import main, qualify
from .fixture import CharacterTokenizer, TinyQwen

torch.set_num_threads(1)


class ConfigTests(unittest.TestCase):
    def test_28_layer_schedule(self):
        self.assertEqual(arm_spec("A2_local512", 28).global_layers, (3, 7, 11, 15, 19, 23, 27))

    def test_pruned_final_layer_is_explicit(self):
        self.assertEqual(arm_spec("A2_local512", 19).global_layers, (3, 7, 11, 15, 18))
        self.assertEqual(len(arm_spec("A2_local512", 12).global_layers), 3)

    def test_all_arms_construct(self):
        for name in ARMS:
            for layers in (1, 12, 19, 28, 36):
                with self.subTest(arm=name, layers=layers):
                    self.assertEqual(arm_spec(name, layers).layers, layers)

    def test_reject_invalid_specs(self):
        for args in ((0, ()), (4, (4,)), (4, (2, 1)), (4, (1, 1))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Spec(*args)
        for kw in ({"sinks": 512}, {"window": 0}, {"block_size": 0},
                   {"gate": 1}, {"backend": "flash"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                Spec(4, (3,), **kw)

    def test_budget_is_matched_with_sinks(self):
        a, b = arm_spec("A2_local512", 28), arm_spec("A3_sink512", 28)
        self.assertEqual(a.cache_bytes(32768, 8, 128, 2), b.cache_bytes(32768, 8, 128, 2))
        self.assertEqual(a.cache_bytes(4096, 8, 128, 2), (7*4096+21*512)*4096)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.q, self.k, self.v = torch.randn(1, 4, 29, 8), torch.randn(1, 2, 29, 8), torch.randn(1, 2, 29, 8)
        self.p = torch.arange(29)

    def test_boundary_semantics_and_no_future_sinks(self):
        p = torch.arange(12)
        mask = allowed_mask(p, p, 8, 2)
        self.assertEqual(mask[11].nonzero().flatten().tolist(), [0, 1, 6, 7, 8, 9, 10, 11])
        self.assertEqual(mask[0].nonzero().flatten().tolist(), [0])
        self.assertEqual(allowed_mask(p, p, 1, 0)[11].nonzero().flatten().tolist(), [11])

    def test_blocked_dense_equivalence(self):
        for backend in ("math", "sdpa"):
            for window, sinks in ((None, 0), (1, 0), (8, 0), (8, 2), (64, 4)):
                for block in (1, 7, 31):
                    with self.subTest(backend=backend, window=window, sinks=sinks, block=block):
                        args = (self.q, self.k, self.v, self.p, self.p)
                        a = dense_oracle(*args, window=window, sinks=sinks)
                        b = blocked_attention(*args, window=window, sinks=sinks,
                                              block_size=block, backend=backend)
                        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_future_values_cannot_change_past(self):
        v = self.v.clone()
        v[:, :, 14:] = 1e5
        args = (self.q, self.k)
        a = blocked_attention(*args, self.v, self.p, self.p, window=8, sinks=2)
        b = blocked_attention(*args, v, self.p, self.p, window=8, sinks=2)
        torch.testing.assert_close(a[:, :, :14], b[:, :, :14], atol=0, rtol=0)

    def test_bounded_local_mask(self):
        stats = {}
        blocked_attention(self.q, self.k, self.v, self.p, self.p,
                          window=8, sinks=2, block_size=3, stats=stats)
        self.assertLessEqual(stats["max_mask_elements"], 3*(8+3-1))
        self.assertLessEqual(stats["max_key_tokens"], 8+3-1)

    def test_absolute_gapped_positions(self):
        kp = torch.tensor([0, 1, 17, 18, 19, 20])
        qp = torch.tensor([19, 20])
        args = (self.q[:, :, :2], self.k[:, :, :6], self.v[:, :, :6], qp, kp)
        a = dense_oracle(*args, window=5, sinks=2)
        b = blocked_attention(*args, window=5, sinks=2, block_size=1)
        torch.testing.assert_close(a, b)

    def test_gradients_match_oracle(self):
        args = [t.clone().requires_grad_() for t in (self.q, self.k, self.v)]
        args2 = [t.detach().clone().requires_grad_() for t in args]
        dense_oracle(*args, self.p, self.p, window=8, sinks=2).square().sum().backward()
        blocked_attention(*args2, self.p, self.p, window=8, sinks=2, block_size=7).square().sum().backward()
        for a, b in zip(args, args2):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-5)

    def test_invalid_positions_and_shapes(self):
        with self.assertRaises(ValueError):
            blocked_attention(self.q, self.k, self.v, self.p.flip(0), self.p)
        with self.assertRaises(ValueError):
            blocked_attention(self.q[:, :3], self.k, self.v, self.p, self.p)
        with self.assertRaises(ValueError):
            blocked_attention(self.q[:, :, :1], self.k[:, :, :1], self.v[:, :, :1],
                              torch.tensor([0]), torch.tensor([2]))


class CacheTests(unittest.TestCase):
    def test_compact_storage_and_positions(self):
        full = torch.randn(1, 2, 100, 8)
        sliced = full[:, :, -8:]
        self.assertGreater(storage_bytes((sliced,)), sliced.numel()*sliced.element_size())
        cache = CompactKV(8, 2)
        with torch.inference_mode():
            kv = cache.merge(full, full+1, torch.arange(100))
            cache.retain(*kv)
        self.assertEqual(cache.positions.tolist(), [0, 1, 94, 95, 96, 97, 98, 99])
        m = cache.metrics()
        self.assertEqual(m["resident_kv_bytes"], m["logical_kv_bytes"])
        self.assertNotEqual(cache.k.untyped_storage().data_ptr(), full.untyped_storage().data_ptr())

    def test_streaming_attention_matches_dense(self):
        torch.manual_seed(4)
        q, k, v = torch.randn(1, 4, 37, 8), torch.randn(1, 2, 37, 8), torch.randn(1, 2, 37, 8)
        p = torch.arange(37)
        for window, sinks in ((None, 0), (8, 0), (8, 2)):
            for chunk in (1, 7, 16, 37):
                with self.subTest(window=window, sinks=sinks, chunk=chunk), torch.inference_mode():
                    cache, outputs = CompactKV(window, sinks), []
                    for start in range(0, 37, chunk):
                        sl = slice(start, start+chunk)
                        kk, vv, pp = cache.merge(k[:, :, sl], v[:, :, sl], p[sl])
                        outputs.append(blocked_attention(q[:, :, sl], kk, vv, p[sl], pp,
                                                         window=window, sinks=sinks, block_size=3))
                        cache.retain(kk, vv, pp)
                    torch.testing.assert_close(torch.cat(outputs, 2), dense_oracle(q, k, v, p, p,
                                               window=window, sinks=sinks), atol=2e-6, rtol=2e-6)

    def test_cache_refuses_grad_and_replayed_positions(self):
        k = torch.ones(1, 2, 3, 8)
        c = CompactKV(8)
        with self.assertRaises(RuntimeError):
            c.merge(k, k, torch.arange(3))
        with torch.inference_mode():
            c.retain(*c.merge(k, k, torch.arange(3)))
            with self.assertRaises(ValueError):
                c.merge(k, k, torch.arange(3))

    def test_resident_deduplicates_aliases(self):
        t = torch.zeros(32)
        self.assertEqual(storage_bytes((t, t[1:], t)), 128)


class GateRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TinyQwen()
        self.ids = torch.randint(0, 97, (1, 23))

    def test_gate_identity_and_live_gradient(self):
        gate = HeadGate(32, 4)
        x, out = torch.randn(1, 7, 32), torch.randn(1, 4, 7, 8)
        y = gate(x, out)
        self.assertTrue(torch.equal(y, out))
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(gate.proj.weight.grad).all())
        self.assertGreater(float(gate.proj.weight.grad.abs().sum()), 0)

    def test_gate_parameter_count_and_serialization(self):
        gates = [HeadGate(1024, 16) for _ in range(28)]
        self.assertEqual(sum(p.numel() for g in gates for p in g.parameters()), 458752)
        stream = io.BytesIO()
        torch.save(gates[0].state_dict(), stream)
        stream.seek(0)
        restored = HeadGate(1024, 16)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(torch.equal(restored.proj.weight, gates[0].proj.weight))

    def test_native_fixture_equivalence_and_no_weight_mutation(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        rt = HybridRuntime(self.model, arm_spec("A0_global", 4, block_size=3))
        torch.testing.assert_close(rt(self.ids), self.model(self.ids).logits, atol=2e-6, rtol=2e-6)
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[k], v))

    def test_layer_runtime_cache_equivalence(self):
        for sinks in (0, 2):
            for gate in (False, True):
                with self.subTest(sinks=sinks, gate=gate), torch.inference_mode():
                    rt = HybridRuntime(self.model, Spec(4, (3,), 8, sinks, gate, 3))
                    full = rt(self.ids)
                    stream = torch.cat([rt(self.ids[:, i:i+5], use_cache=True)
                                        for i in range(0, 23, 5)], dim=1)
                    torch.testing.assert_close(full, stream, atol=3e-6, rtol=3e-6)
                    self.assertEqual(rt.position, 23)
                    self.assertEqual([x.metrics()["retained_tokens"] for x in rt.cache], [8, 8, 8, 23])

    def test_gate_only_toy_optimizer_step_keeps_backbone(self):
        self.model.requires_grad_(False)
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        rt = HybridRuntime(self.model, Spec(4, (3,), 8, 2, True))
        opt = torch.optim.SGD(rt.gates.parameters(), lr=0.01)
        loss = rt(self.ids).square().mean()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in rt.gates.parameters()), 0)
        opt.step()
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in self.model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_generation_resets_between_requests(self):
        rt = HybridRuntime(self.model, Spec(4, (3,), 8, 2))
        a = rt.generate(self.ids, max_new_tokens=4, chunk_size=5)
        b = rt.generate(self.ids, max_new_tokens=4, chunk_size=5)
        self.assertEqual(a["token_ids"], b["token_ids"])
        self.assertTrue(a["budget_exhausted"])
        self.assertEqual(rt.position, 0)
        self.assertTrue(all(x.k is None for x in rt.cache))
        eos = rt.generate(self.ids, max_new_tokens=4, eos_token_ids=(a["token_ids"][0],))
        self.assertTrue(eos["stopped_eos"])
        self.assertEqual(len(eos["token_ids"]), 1)

    def test_runtime_refuses_unsupported_inputs_and_hooks(self):
        rt = HybridRuntime(self.model, arm_spec("A0_global", 4))
        with self.assertRaises(ValueError):
            rt(self.ids.expand(2, -1))
        with self.assertRaises(ValueError):
            rt.generate(torch.empty(1, 0, dtype=torch.long))
        with self.assertRaises(RuntimeError):
            rt(self.ids, use_cache=True)
        with self.assertRaises(ValueError):
            rt(torch.zeros(1, 2049, dtype=torch.long))
        hook = self.model.lm_head.register_forward_hook(lambda m, i, o: o)
        try:
            with self.assertRaises(ValueError):
                HybridRuntime(self.model, arm_spec("A0_global", 4))
        finally:
            hook.remove()

    def test_qualification_function_on_fixture_not_real_evidence(self):
        for backend in ("math", "sdpa"):
            with self.subTest(backend=backend):
                checks = qualify(self.model, backend, 3)
                self.assertTrue(all(x["passed"] for x in checks.values()), checks)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Transformers absent: native HF integration NOT run")
    def test_optional_native_hf_random_qwen3(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=2048, attention_dropout=0.0)
        cfg._attn_implementation = "eager"
        model = Qwen3ForCausalLM(cfg).eval()
        checks = qualify(model, "math", 3)
        self.assertTrue(all(x["passed"] for x in checks.values()), checks)


class EvidenceTests(unittest.TestCase):
    def test_diagnostics_are_exact_length_unique_and_deterministic(self):
        a, b = make_cases(CharacterTokenizer()), make_cases(CharacterTokenizer())
        self.assertEqual(a, b)
        self.assertEqual(len(a), 72)
        self.assertEqual(len({c["prompt_sha256"] for c in a}), 72)
        self.assertTrue(all(len(c["input_ids"]) == c["tokens"] for c in a))
        self.assertTrue(all(c["split"] == "open_synthetic_dev" for c in a))

    def test_scoring_separates_format_abstention_and_loops(self):
        c = {"id": "x", "family": "retrieval", "depth": 0.5, "tokens": 512,
             "answerable": True, "prompt_sha256": "x", "answer": "12345",
             "evidence_span": [100, 120], "question_span": [500, 512],
             "evidence_to_question_gap": 380, "evidence_kind": "support_or_conflict"}
        g = {"token_ids": [7]*24, "stopped_eos": False, "budget_exhausted": True}
        # A case that predates the attribution schema must fail closed, not
        # silently produce an unstratifiable row.
        for dropped in ("evidence_span", "question_span", "evidence_kind",
                        "evidence_to_question_gap"):
            stale = {k: v for k, v in c.items() if k != dropped}
            with self.subTest(dropped=dropped), self.assertRaises(ValueError):
                score(stale, " 12345\n", g)
        row = score(c, " 12345\n", g)
        self.assertFalse(row["strict_exact"])
        self.assertTrue(row["trimmed_exact"])
        self.assertFalse(score(c, "code 12345", g)["trimmed_exact"])
        loop = score(c, "UNKNOWN UNKNOWN", g)
        self.assertFalse(loop["abstained"])
        self.assertEqual(loop["max_token_repeat"], 24)
        summary = summarize([score(c, "UNKNOWN", g)])["overall"]
        self.assertEqual(summary["false_abstention_rate"], 1)
        self.assertEqual(summary["budget_exhaustion_rate"], 1)

    def test_paired_comparison_enforces_membership_and_content(self):
        base = [{"id": str(i), "prompt_sha256": str(i), "answer": "1",
                 "first_content_exact": False, "trimmed_exact": False} for i in range(6)]
        other = [dict(r, first_content_exact=True, trimmed_exact=True) for r in reversed(base)]
        r = paired_result(base, other)
        self.assertEqual(r["wins"], 6)
        self.assertEqual(r["mcnemar_exact_two_sided"], 0.03125)
        self.assertEqual(r["metric"], PRIMARY_METRIC)
        with self.assertRaises(ValueError):
            paired_result(base, other[:-1])
        other[0]["prompt_sha256"] = "different"
        with self.assertRaises(ValueError):
            paired_result(base, other)

    def test_paired_comparison_reports_the_metric_it_actually_used(self):
        # It used to hardcode trimmed_exact while callers believed they were
        # comparing the registered primary; the two agreed only by accident.
        rows = [{"id": str(i), "prompt_sha256": str(i), "answer": "1",
                 "first_content_exact": False, "trimmed_exact": False} for i in range(4)]
        # An arm that answers correctly but overruns: primary true, trimmed false.
        other = [dict(r, first_content_exact=True) for r in rows]
        self.assertEqual(paired_result(rows, other)["wins"], 4)
        self.assertEqual(paired_result(rows, other, "trimmed_exact")["wins"], 0)
        self.assertEqual(paired_result(rows, other, "trimmed_exact")["metric"], "trimmed_exact")
        with self.assertRaises(ValueError):
            paired_result(rows, other, "no_such_metric")

    def test_missing_real_weights_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "config.json").write_text('{"model_type":"qwen3"}')
            with self.assertRaises(FileNotFoundError):
                checkpoint_inventory(p)
            (p / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaises(ValueError):
                checkpoint_inventory(p)

    def test_receipt_cannot_be_fixture_or_stale(self):
        env = environment()
        r = {"stage": "real_model_qualification", "passed": True,
             "checkpoint_id": "abc", "code_identity": code_identity(), "environment": env,
             "backend": "math", "block_size": 128,
             "licenses_training": False, "licenses_sealed_evaluation": False,
             "checks": {k: {"passed": True} for k in ("native_global", "global_stream", "gate_identity", "local_stream", "sink_stream")}}
        r["receipt_sha256"] = digest(r)
        require_receipt(r, {"id": "abc"}, env, "math", 128)
        for key, value in (("stage", "cpu_fixture"), ("passed", False),
                           ("checkpoint_id", "def"), ("code_identity", "stale"),
                           ("licenses_training", True), ("backend", "sdpa")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_receipt(dict(r, **{key: value}), {"id": "abc"}, env, "math", 128)

    def test_run_paths_and_receipts_are_non_overwriting(self):
        with tempfile.TemporaryDirectory() as d, patch("experiments.hybrid_attention_v1.provenance.ROOT", Path(d)):
            out = new_run("test-1")
            with self.assertRaises(FileExistsError):
                new_run("test-1")
            with self.assertRaises(ValueError):
                new_run("../sealed")
            write_json(out / "a.json", {"a": 1})
            with self.assertRaises(FileExistsError):
                write_json(out / "a.json", {"a": 2})

    def test_training_and_sealed_cli_stages_refuse(self):
        for stage in ("train", "sealed"):
            with self.subTest(stage=stage), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main([stage])


if __name__ == "__main__":
    unittest.main()
