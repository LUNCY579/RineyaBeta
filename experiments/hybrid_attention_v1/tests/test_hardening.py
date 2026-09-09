"""Regression/negative-path fixtures only. No Rineya weights or evaluation data."""
from collections import Counter
import copy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from ..config import ARMS, Spec, arm_spec
from ..diagnostics import DIAGNOSTIC_LENGTHS, make_cases
from ..provenance import (artifact_inventory, require_artifacts, code_identity,
                          digest, environment, require_receipt, require_unchanged_code)
from ..run import (DRIFT_MULTIPLE, apply_criterion, compare_logits, declared_eos_ids,
                   main as run_main, qualify, require_deterministic_cublas,
                   stratified_cases)
from ..runtime import (HybridRuntime, model_fingerprint, rope_is_unscaled,
                       rope_theta as rope_theta_of)
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
                self.rt.reset()  # explicit start of a new request
                self.rt(self.ids, use_cache=True)
                self.assertEqual(self.rt.position, 17)
                with self.assertRaises((ValueError, IndexError)):
                    self.rt(ids, use_cache=True)
                self.assert_reset()
                # The cache is clear, but a caller that catches the error and
                # retries the chunk must say so explicitly: an implicit retry
                # would silently restart absolute positions at zero and return
                # a plausible, wrong continuation.
                with self.assertRaisesRegex(RuntimeError, "aborted"):
                    self.rt(self.ids, use_cache=True)
                self.rt(self.ids)  # an uncached forward stays available

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
        # Windows refuses symlink creation without SeCreateSymbolicLinkPrivilege
        # (WinError 1314). Skipping would silently drop the guard from the receipt,
        # so fall back to making exactly this path report as a link instead.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "source.txt"
            target.write_text("{}")
            link = root / "artifact.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                link.write_text("{}")
                real = Path.is_symlink
                with patch.object(Path, "is_symlink",
                                  lambda self: self == link or real(self)):
                    with self.assertRaises(ValueError):
                        artifact_inventory(root)
                return
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


class QualificationCriterionTests(unittest.TestCase):
    """protocol.json qualification.criterion_version 2. A criterion that cannot
    withhold a pass is not a criterion, so every condition is tested failing."""

    def _checks(self, **overrides):
        base = {
            "native_global": {"passed": True, "max_abs_error": 0.0, "argmax_identical": True},
            "gate_identity": {"passed": True, "max_abs_error": 0.0, "argmax_identical": True},
            "global_stream": {"passed": True, "max_abs_error": 2.0e-05, "argmax_identical": True,
                              "attended_positions_identical": True},
            # passed here is compare_logits' v1 verdict: 9.0e-05 misses 2e-5, and
            # that is exactly the real-R2 case the amendment exists to re-judge.
            "local_stream": {"passed": False, "max_abs_error": 9.0e-05, "argmax_identical": True,
                             "attended_positions_identical": True, "compact_storage": True},
        }
        for name, patch_fields in overrides.items():
            base[name] = {**base[name], **patch_fields}
        return base

    def test_real_r2_shaped_numbers_pass(self):
        checks = apply_criterion(self._checks(), "math", single_block=True)
        self.assertTrue(all(c["passed"] for c in checks.values()), checks)
        self.assertAlmostEqual(checks["local_stream"]["drift_ratio_to_all_global"], 4.5)
        # The superseded verdict is still recorded, not silently dropped.
        self.assertFalse(checks["local_stream"]["registered_v1_tolerance_passed"])
        self.assertTrue(checks["gate_identity"]["registered_v1_tolerance_passed"])

    def test_drift_beyond_the_control_multiple_fails(self):
        control = 2.0e-05
        for multiple, expected in ((DRIFT_MULTIPLE, True), (DRIFT_MULTIPLE + 1, False)):
            with self.subTest(multiple=multiple):
                checks = apply_criterion(
                    self._checks(local_stream={"max_abs_error": control * multiple}),
                    "math", single_block=True)
                self.assertEqual(checks["local_stream"]["passed"], expected)

    def test_exact_conditions_outrank_a_tiny_drift(self):
        # The fault that matters most: positions wrong, logits barely moved.
        for field in ("attended_positions_identical", "compact_storage"):
            with self.subTest(field=field):
                checks = apply_criterion(
                    self._checks(local_stream={field: False, "max_abs_error": 2.0e-05}),
                    "math", single_block=True)
                self.assertFalse(checks["local_stream"]["passed"])
                self.assertFalse(checks["local_stream"]["criterion_conditions"][field])

    def test_argmax_and_nonfinite_failures_are_refused(self):
        checks = apply_criterion(self._checks(local_stream={"argmax_identical": False}),
                                 "math", single_block=True)
        self.assertFalse(checks["local_stream"]["passed"])
        checks = apply_criterion(self._checks(local_stream={"max_abs_error": None}),
                                 "math", single_block=True)
        self.assertFalse(checks["local_stream"]["passed"])
        self.assertIsNone(checks["local_stream"]["drift_ratio_to_all_global"])

    def test_gate_identity_must_be_bit_exact(self):
        checks = apply_criterion(self._checks(gate_identity={"max_abs_error": 1e-12}),
                                 "math", single_block=True)
        self.assertFalse(checks["gate_identity"]["passed"])

    def test_native_parity_is_bit_exact_only_where_that_is_attainable(self):
        near = self._checks(native_global={"max_abs_error": 4.8e-07})
        # math + one query block: both sides multiply the same shapes, so demand bits.
        self.assertFalse(apply_criterion(near, "math", single_block=True)["native_global"]["passed"])
        # Split into blocks, or on a different kernel, bit equality is unattainable.
        self.assertTrue(apply_criterion(self._checks(native_global={"max_abs_error": 4.8e-07}),
                                        "math", single_block=False)["native_global"]["passed"])
        self.assertTrue(apply_criterion(self._checks(native_global={"max_abs_error": 4.8e-07}),
                                        "sdpa", single_block=True)["native_global"]["passed"])
        # ...but it is still bounded by the same control multiple.
        blown = self._checks(native_global={"max_abs_error": 2.0e-05 * (DRIFT_MULTIPLE + 1)})
        self.assertFalse(apply_criterion(blown, "sdpa", single_block=True)["native_global"]["passed"])

    def test_a_zero_control_admits_only_zero_drift(self):
        checks = apply_criterion(self._checks(global_stream={"max_abs_error": 0.0}),
                                 "math", single_block=True)
        self.assertFalse(checks["local_stream"]["passed"])


class DiagnosticDecodingTests(unittest.TestCase):
    """The R2 checkpoint declares eos_token_id 32744 in config.json but
    [32744, 32742] in generation_config.json, and 32742 is <|endoftext|> --
    the natural terminator for the raw-completion prompts P3 uses."""

    def _checkpoint(self, root, generation_eos):
        if generation_eos is not None:
            (root / "generation_config.json").write_text(
                json.dumps({"eos_token_id": generation_eos}), encoding="utf-8")

    def test_declared_set_unions_config_and_generation_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkpoint(root, [32744, 32742])
            cfg = SimpleNamespace(eos_token_id=32744)
            self.assertEqual(declared_eos_ids(root, cfg, 32768), (32742, 32744))

    def test_config_only_checkpoint_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(
                declared_eos_ids(root, SimpleNamespace(eos_token_id=[7, 3]), 97), (3, 7))

    def test_out_of_vocabulary_or_nonintegral_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for generation_eos, vocab in (([97], 97), ([-1], 97), ([True], 97), (["3"], 97)):
                with self.subTest(eos=generation_eos):
                    self._checkpoint(root, generation_eos)
                    with self.assertRaises(ValueError):
                        declared_eos_ids(root, SimpleNamespace(eos_token_id=3), vocab)

    def test_a_checkpoint_declaring_no_eos_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                declared_eos_ids(Path(d), SimpleNamespace(eos_token_id=None), 97)


class CaseTruncationTests(unittest.TestCase):
    def setUp(self):
        self.cases = make_cases(CharacterTokenizer())

    def test_full_budget_keeps_the_registered_order(self):
        for limit in (len(self.cases), len(self.cases) + 10):
            self.assertEqual(stratified_cases(self.cases, limit), self.cases)

    def test_a_bare_prefix_would_have_kept_only_short_prompts(self):
        # The defect this guards: make_cases is length-major, so a bare prefix
        # keeps only the shortest lengths -- the ones where the window evicts
        # least, i.e. the subset least able to show window damage.
        half = {c["tokens"] for c in self.cases[:36]}
        self.assertEqual(half, set(DIAGNOSTIC_LENGTHS[:3]))
        self.assertLess(max(half), min(DIAGNOSTIC_LENGTHS[3:]))

    def test_truncation_stays_balanced_across_every_length(self):
        every = {c["tokens"] for c in self.cases}
        for limit in (6, 12, 18, 24, 36, 48, 54, 71):
            with self.subTest(limit=limit):
                kept = stratified_cases(self.cases, limit)
                self.assertEqual(len(kept), limit)
                self.assertEqual(len({c["id"] for c in kept}), limit)
                self.assertEqual({c["tokens"] for c in kept}, every)
                counts = Counter(c["tokens"] for c in kept)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_truncation_also_spans_depths_once_each_length_can_afford_it(self):
        # Length cycles outermost, so a limit below one case per (length, depth)
        # cell cannot cover depths; from 18 up it can and must.
        for limit in (18, 36, 54):
            with self.subTest(limit=limit):
                counts = Counter(c["depth"] for c in stratified_cases(self.cases, limit))
                self.assertEqual(set(counts), {0.0, 0.5, 1.0})
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


class DeterministicCublasTests(unittest.TestCase):
    """Without this the entire --device cuda path raises before loading a model,
    which is why no CUDA receipt has ever been produced for this experiment."""

    def test_unset_config_is_populated(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            self.assertEqual(require_deterministic_cublas(), ":4096:8")
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_both_deterministic_configs_are_preserved(self):
        for value in (":4096:8", ":16:8"):
            with self.subTest(value=value), patch.dict(
                    os.environ, {"CUBLAS_WORKSPACE_CONFIG": value}):
                self.assertEqual(require_deterministic_cublas(), value)

    def test_nondeterministic_config_is_refused(self):
        for value in ("", ":0:0", "garbage"):
            with self.subTest(value=value), patch.dict(
                    os.environ, {"CUBLAS_WORKSPACE_CONFIG": value}):
                with self.assertRaises(RuntimeError):
                    require_deterministic_cublas()

    def test_environment_records_the_setting(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self.assertEqual(environment()["cublas_workspace_config"], ":16:8")


class RopeCompatibilityTests(unittest.TestCase):
    """Transformers 4.x recorded no scaling as None; >=5 records an explicit
    {"rope_type": "default"} descriptor and moves rope_theta off the config
    root. Both denote the SAME unscaled RoPE, and a bare None-test rejected an
    unmodified R2 checkpoint outright."""

    def _model(self, **overrides):
        model = TinyQwen(4)
        for name, value in overrides.items():
            setattr(model.config, name, value)
        return model

    def test_both_unscaled_encodings_are_accepted(self):
        for scaling, theta in ((None, 10000),
                               ({"rope_type": "default", "rope_theta": 10000}, None),
                               ({"rope_type": "default"}, 10000),
                               ({"type": "default", "rope_theta": 10000}, 10000)):
            with self.subTest(rope_scaling=scaling):
                model = self._model(rope_scaling=scaling, rope_theta=theta)
                self.assertTrue(rope_is_unscaled(model.config))
                self.assertEqual(rope_theta_of(model.config), 10000)
                HybridRuntime(model, arm_spec("A0_global", 4))

    def test_scaled_rope_is_still_refused(self):
        for scaling in ({"rope_type": "linear", "factor": 2.0},
                        {"rope_type": "dynamic", "factor": 4.0},
                        {"rope_type": "yarn", "factor": 2.0, "beta_fast": 32},
                        {"rope_type": "llama3", "factor": 8.0},
                        {"type": "longrope"},
                        {"rope_type": "default", "factor": 2.0},
                        {"rope_type": "default", "original_max_position_embeddings": 4096},
                        {}, "default", 1.0):
            with self.subTest(rope_scaling=scaling):
                self.assertFalse(rope_is_unscaled(self._model(rope_scaling=scaling).config))
                with self.assertRaises(ValueError):
                    HybridRuntime(self._model(rope_scaling=scaling, rope_theta=10000),
                                  arm_spec("A0_global", 4))

    def test_absent_or_conflicting_theta_fails_closed(self):
        for scaling, theta in ((None, None),
                               ({"rope_type": "default"}, None),
                               ({"rope_type": "default", "rope_theta": 500000}, 10000),
                               (None, "10000"),
                               (None, True)):
            with self.subTest(rope_scaling=scaling, rope_theta=theta):
                model = self._model(rope_scaling=scaling, rope_theta=theta)
                self.assertIsNone(rope_theta_of(model.config))
                with self.assertRaises(ValueError):
                    HybridRuntime(model, arm_spec("A0_global", 4))

    def test_fingerprint_binds_loaded_module_not_only_config(self):
        model = TinyQwen(4)
        fingerprint = model_fingerprint(model)
        self.assertEqual(fingerprint["layers"], 4)
        self.assertEqual(fingerprint["rope_theta_effective"], 10000)
        self.assertTrue(fingerprint["rope_unscaled"])
        self.assertEqual(fingerprint["parameter_dtypes"], ["torch.float32"])
        self.assertEqual(fingerprint["attention_scaling"], [8**-0.5])
        self.assertFalse(fingerprint["tied_output_embedding"])
        json.dumps(fingerprint, allow_nan=False)  # must survive receipt writing
        # A fingerprint that ignored the live module would not move here.
        model.model.layers = model.model.layers[:2]
        self.assertEqual(model_fingerprint(model)["layers"], 2)


if __name__ == "__main__":
    unittest.main()
