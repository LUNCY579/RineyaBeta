"""Regression/negative-path fixtures only. No Rineya weights or evaluation data."""
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from ..config import ARMS, Spec, arm_spec
from ..diagnostics import make_cases
from ..provenance import (artifact_inventory, require_artifacts, code_identity,
                          digest, require_receipt, require_unchanged_code)
from ..run import main as run_main, qualify, compare_logits
from ..runtime import HybridRuntime
from ..study import validate_execution
from ..verify import qualification_passed
from .fixture import TinyQwen, CharacterTokenizer

torch.set_num_threads(1)


class RuntimeHardeningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TinyQwen(layers=8)
        self.model.config.intermediate_size = 64
        self.rt = HybridRuntime(self.model, Spec(8, (3, 7), 8, 2, block_size=7))
        self.ids = torch.randint(0, 97, (1, 17))

    def assert_reset(self):
        self.assertEqual(self.rt.position, 0)
        self.assertEqual(self.rt.stats, {})
        self.assertTrue(all(c.k is None and c.v is None for c in self.rt.cache))

    def test_validation_and_embedding_errors_reset_existing_cache(self):
        invalid = [torch.empty(1, 0, dtype=torch.long), self.ids.float(),
                   torch.zeros(2, 1, dtype=torch.long), torch.tensor([[97]]),
                   torch.zeros(1, 2048, dtype=torch.long)]
        for ids in invalid:
            with self.subTest(shape=tuple(ids.shape)), torch.inference_mode():
                self.rt(self.ids, use_cache=True)
                self.assertEqual(self.rt.position, 17)
                with self.assertRaises((ValueError, IndexError)):
                    self.rt(ids, use_cache=True)
                self.assert_reset()

    def test_rotary_error_resets_cache(self):
        with torch.inference_mode():
            self.rt(self.ids, use_cache=True)
            with patch.object(self.model.model.rotary_emb, "forward", side_effect=RuntimeError("rotary failure")):
                with self.assertRaises(RuntimeError):
                    self.rt(self.ids[:, :1], use_cache=True)
            self.assert_reset()

    def test_nonfinite_prefill_never_becomes_tokens(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), torch.inference_mode():
                original = self.model.lm_head.weight.clone()
                self.model.lm_head.weight.fill_(value)
                try:
                    with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                        self.rt.generate(self.ids, max_new_tokens=3, chunk_size=7)
                    self.assert_reset()
                finally:
                    self.model.lm_head.weight.copy_(original)

    def test_nonfinite_decode_aborts_and_resets(self):
        forward = self.model.lm_head.forward
        count = 0
        def corrupt(x):
            nonlocal count
            count += 1
            y = forward(x)
            if count == 4:  # Three prefill chunks, then first decode forward.
                y.fill_(float("nan"))
            return y
        with patch.object(self.model.lm_head, "forward", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                self.rt.generate(self.ids, max_new_tokens=3, chunk_size=7)
        self.assertEqual(count, 4)
        self.assert_reset()

    def test_invalid_generate_clears_abandoned_request(self):
        with torch.inference_mode():
            self.rt(self.ids, use_cache=True)
            with self.assertRaises(ValueError):
                self.rt.generate(None)
        self.assert_reset()

    def test_noninteger_budgets_and_invalid_eos_rejected(self):
        for kwargs in ({"max_new_tokens": True}, {"chunk_size": 1.5},
                       {"eos_token_ids": [True]}, {"eos_token_ids": [97]},
                       {"eos_token_ids": [-1]}, {"eos_token_ids": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.rt.generate(self.ids, **kwargs)
            self.assert_reset()

    def test_all_topologies_still_qualify_on_tiny_fixture(self):
        for backend in ("math", "sdpa"):
            with self.subTest(backend=backend):
                self.assertTrue(all(c["passed"] for c in qualify(self.model, backend, 7).values()))

    def test_identity_and_planner_across_all_arms(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        for backend in ("math", "sdpa"):
            rows = {}
            for name in ARMS:
                spec = replace(arm_spec(name, 8, backend=backend, block_size=7),
                               window=8, sinks=2 if ARMS[name][1] else 0)
                rt = HybridRuntime(self.model, spec)
                result = rt.generate(self.ids, max_new_tokens=4, chunk_size=7)
                validate_execution(spec, self.model.config, {"tokens": 17}, result)
                rows[name] = result["token_ids"]
            for a, b in (("A0_global", "A4_global_gate"), ("A2_local512", "A5_local512_gate"),
                         ("A3_sink512", "A6_sink512_gate")):
                self.assertEqual(rows[a], rows[b])
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.model.state_dict().items()))

    def test_real_window_boundary_shapes_on_random_tiny_model(self):
        # Native window sizes, not pretrained quality and not generated DEV prompts.
        for n in (513, 1025):
            ids = torch.randint(0, 97, (1, n))
            for name in ARMS:
                with self.subTest(n=n, arm=name):
                    spec = arm_spec(name, 8, block_size=128)
                    rt = HybridRuntime(self.model, spec)
                    gen = rt.generate(ids, max_new_tokens=1, chunk_size=128)
                    validate_execution(spec, self.model.config, {"tokens": n}, gen)


class ProvenanceHardeningTests(unittest.TestCase):
    def test_markdown_contract_is_fingerprinted_but_reports_are_not(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            (root / "protocol.json").write_text("{}")
            doc = root / "COMPATIBILITY_CONTRACT.md"
            doc.write_text("original")
            with patch("experiments.hybrid_attention_v1.provenance.ROOT", root):
                a = code_identity()
                doc.write_text("changed")
                b = code_identity()
                self.assertNotEqual(a, b)
                with self.assertRaises(RuntimeError):
                    require_unchanged_code(a)
                (root / "reports").mkdir()
                (root / "reports/result.json").write_text("{}")
                self.assertEqual(code_identity(), b)
                require_unchanged_code(b)

    def test_artifacts_bind_content_membership_and_exclude_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "checkpoint_inventory.json"
            p.write_text('{"id":"fixture"}')
            r = {"artifacts": artifact_inventory(root)}
            (root / "receipt.json").write_text("{}")
            require_artifacts(r, root)
            p.write_text('{"id":"edited"}')
            with self.assertRaises(ValueError):
                require_artifacts(r, root)
            p.write_text('{"id":"fixture"}')
            (root / "unexpected.json").write_text("{}")
            with self.assertRaises(ValueError):
                require_artifacts(r, root)
            with self.assertRaises(ValueError):
                require_artifacts({}, root)

    def test_symlink_artifact_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "source.txt"
            target.write_text("{}")
            (root / "artifact.json").symlink_to(target)
            with self.assertRaises(ValueError):
                artifact_inventory(root)

    def test_bool_receipt_flags_cannot_be_replaced_by_integers(self):
        r = {"stage": "real_model_qualification", "passed": True, "checkpoint_id": "id",
             "code_identity": "code", "environment": {}, "backend": "math", "block_size": 7,
             "licenses_training": False, "licenses_sealed_evaluation": False,
             "checks": {k: {"passed": True} for k in
                        ("native_global", "global_stream", "gate_identity", "local_stream", "sink_stream")}}
        with patch("experiments.hybrid_attention_v1.provenance.code_identity", return_value="code"):
            r["receipt_sha256"] = digest(r)
            require_receipt(r, {"id": "id"}, {}, "math", 7)
            for field, value in (("passed", 1), ("licenses_training", 0), ("block_size", 7.0)):
                bad = copy.deepcopy(r)
                bad[field] = value
                bad["receipt_sha256"] = digest({k: v for k, v in bad.items() if k != "receipt_sha256"})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    require_receipt(bad, {"id": "id"}, {}, "math", 7)

    def test_finite_opposite_extreme_logits_keep_failure_serializable(self):
        maximum = torch.finfo(torch.float32).max
        result = compare_logits(torch.tensor([maximum]), torch.tensor([-maximum]))
        self.assertFalse(result["passed"])
        self.assertGreater(result["max_abs_error"], maximum)
        json.dumps(result, allow_nan=False)

    def test_nonfinite_receipt_digest_rejected(self):
        with self.assertRaises(ValueError):
            digest({"value": float("nan")})

    def test_runner_preserves_failed_receipt_for_source_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inventory = {"id": "fixture", "config": {}}
            with patch("experiments.hybrid_attention_v1.run.new_run", return_value=root), \
                 patch("experiments.hybrid_attention_v1.run.checkpoint_inventory", return_value=inventory), \
                 patch("experiments.hybrid_attention_v1.run.code_identity", return_value="before"), \
                 patch("experiments.hybrid_attention_v1.run.require_unchanged_code", side_effect=RuntimeError("changed")), \
                 patch("sys.stdout", io.StringIO()):
                result = run_main(["audit", "--run-id", "fixture"])
            r = json.loads((root / "receipt.json").read_text())
            self.assertEqual(result, 1)
            self.assertFalse(r["passed"])
            self.assertEqual(r["error_type"], "RuntimeError")
            self.assertFalse(r["licenses_training"])
            self.assertFalse(r["quality_improvement_established"])

    def test_runner_binds_artifacts_on_successful_mock_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inventory = {"id": "fixture", "config": {}}
            with patch("experiments.hybrid_attention_v1.run.new_run", return_value=root), \
                 patch("experiments.hybrid_attention_v1.run.checkpoint_inventory", return_value=inventory), \
                 patch("experiments.hybrid_attention_v1.run.code_identity", return_value="unchanged"), \
                 patch("experiments.hybrid_attention_v1.run.require_unchanged_code"), \
                 patch("sys.stdout", io.StringIO()):
                result = run_main(["audit", "--run-id", "fixture"])
            r = json.loads((root / "receipt.json").read_text())
            self.assertEqual(result, 0)
            require_artifacts(r, root)
            self.assertTrue(r["source_unchanged"])
            self.assertFalse(r["licenses_sealed_evaluation"])

    def test_native_requirement_rejects_skips_empty_and_failed_suites(self):
        result = SimpleNamespace(testsRun=1, skipped=[], wasSuccessful=lambda: True)
        self.assertTrue(qualification_passed(result, True, True))
        self.assertFalse(qualification_passed(result, False, True))
        result.skipped = [("fixture", "reason")]
        self.assertFalse(qualification_passed(result, True, True))
        result.skipped = []
        result.testsRun = 0
        self.assertFalse(qualification_passed(result, True, False))
        result.testsRun = 1
        result.wasSuccessful = lambda: False
        self.assertFalse(qualification_passed(result, True, True))


class DiagnosticMetadataTests(unittest.TestCase):
    def test_spans_and_gap_are_actual_positions_not_requested_depth(self):
        for case in make_cases(CharacterTokenizer()):
            start, end = case["evidence_span"]
            query, limit = case["question_span"]
            self.assertTrue(0 <= start < end <= query < limit)
            self.assertEqual(case["tokens"], limit)
            self.assertEqual(case["evidence_to_question_gap"], query-end)
            text = bytes(case["input_ids"][start:end]).decode("ascii")
            self.assertIn("value", text)
            self.assertEqual(case["prompt_sha256"], digest(case["input_ids"]))
            if case["depth"] == 0:
                self.assertGreater(start, 0)  # Instructions precede evidence.
            if case["family"] == "absent":
                self.assertEqual(case["evidence_kind"], "distractor")


if __name__ == "__main__":
    unittest.main()
