"""Infrastructure tests only: no pretrained weights, training data or sealed assets."""
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ..attention import allowed_mask, blocked_attention
from ..config import ARMS, Spec, arm_spec
from ..provenance import digest
from ..run import compare_logits, main, qualify
from ..runtime import HybridRuntime
from ..study import (CONTRASTS, QUALIFICATION_CRITERION_VERSION, TARGET, VERSION,
                     assert_target_config, causal_pairs, contrast_report,
                     inference_workload, matched_update_budget, plan,
                     require_study_receipt, validate_execution)
from .fixture import TinyQwen

torch.set_num_threads(1)


class ControlTests(unittest.TestCase):
    def test_topologies_and_stable_original_arms(self):
        self.assertEqual(len(ARMS), 10)
        self.assertEqual(arm_spec("A2_local512", 28).global_layers, (3, 7, 11, 15, 19, 23, 27))
        self.assertEqual(arm_spec("A8_all_local512", 28).global_layers, ())
        self.assertEqual(arm_spec("A9_frontloaded512", 28).global_layers, (0, 1, 2, 3, 4, 5, 27))
        self.assertEqual(ARMS["A3_sink512"], (512, 4, False))

    def test_frontloaded_same_global_count_at_all_fixture_depths(self):
        for depth in (1, 4, 8, 12, 19, 28, 36):
            a, b = arm_spec("A2_local512", depth), arm_spec("A9_frontloaded512", depth)
            self.assertEqual(len(a.global_layers), len(b.global_layers))
            self.assertEqual(b.global_layers[-1], depth-1)

    def test_equal_cache_and_modeled_work_for_placement_and_sinks(self):
        for n in (511, 512, 513, 1023, 1024, 1025, 4096):
            a = inference_workload(arm_spec("A2_local512", 28), TARGET, n, 24)
            for name in ("A3_sink512", "A9_frontloaded512"):
                self.assertEqual(a, inference_workload(arm_spec(name, 28), TARGET, n, 24))

    def test_all_local_is_explicitly_not_compute_matched(self):
        a = inference_workload(arm_spec("A2_local512", 28), TARGET, 1025, 24)
        b = inference_workload(arm_spec("A8_all_local512", 28), TARGET, 1025, 24)
        self.assertLess(b["attention_matmul_flops"], a["attention_matmul_flops"])
        self.assertEqual(b["decoder_linear_matmul_flops"], a["decoder_linear_matmul_flops"])

    def test_gate_projection_overhead_is_not_hidden(self):
        a = inference_workload(arm_spec("A0_global", 28), TARGET, 513, 24)
        b = inference_workload(arm_spec("A4_global_gate", 28), TARGET, 513, 24)
        self.assertGreater(b["modeled_matmul_flops"], a["modeled_matmul_flops"])
        self.assertEqual(b["modeled_matmul_flops"]-a["modeled_matmul_flops"], 2*536*28*1024*16)
        self.assertEqual(a["issued_score_elements"], b["issued_score_elements"])

    def test_shape_lock_is_not_family_wide(self):
        assert_target_config(TARGET)
        for key in ("num_hidden_layers", "num_key_value_heads", "head_dim", "vocab_size", "rope_theta"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_target_config(dict(TARGET, **{key: TARGET[key]+1}))


class AccountingTests(unittest.TestCase):
    def test_visible_pairs_against_independent_dense_masks(self):
        for n in (1, 7, 8, 9, 17, 37):
            p = torch.arange(n)
            for window, sinks in ((None, 0), (1, 0), (8, 0), (8, 2), (8, 7), (64, 4)):
                self.assertEqual(causal_pairs(n, window), int(allowed_mask(p, p, window, sinks).sum()))

    def test_rectangles_against_executed_attention(self):
        torch.manual_seed(17)
        cfg = dict(TARGET, num_hidden_layers=1, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=8, hidden_size=32, intermediate_size=64)
        for n in (1, 8, 9, 37):
            p = torch.arange(n)
            q, k, v = torch.randn(1, 4, n, 8), torch.randn(1, 2, n, 8), torch.randn(1, 2, n, 8)
            for window, sinks in ((None, 0), (1, 0), (8, 0), (8, 2), (8, 7), (64, 4)):
                for block in (1, 3, 16):
                    spec = Spec(1, (0,) if window is None else (), window or 512, sinks, False, block)
                    stats = {}
                    blocked_attention(q, k, v, p, p, window=window, sinks=sinks,
                                      block_size=block, stats=stats)
                    expected = inference_workload(spec, cfg, n, 1)
                    self.assertEqual(stats["issued_score_elements"], expected["issued_score_elements"])
                    self.assertEqual(stats["attention_matmul_flops"], expected["attention_matmul_flops"])
                    self.assertGreaterEqual(expected["issued_score_elements"], expected["visible_score_elements"])

    def test_actual_q_width_not_hidden_size_shortcut(self):
        # R2 has Q width 2048 while hidden width is 1024; K/V widths are 1024.
        work = inference_workload(arm_spec("A0_global", 28), TARGET, 17, 1)
        explicit_matrix_parameters = 1024*2048 + 2*1024*1024 + 2048*1024 + 3*1024*3072
        self.assertEqual(work["decoder_linear_matmul_flops"], 2*17*28*explicit_matrix_parameters)

    def test_last_predicted_token_is_not_forwarded(self):
        spec = arm_spec("A0_global", 28, block_size=8)
        a, b = [inference_workload(spec, TARGET, 17, g) for g in (1, 4)]
        self.assertEqual(a["decoder_tokens"], 17)
        self.assertEqual(b["decoder_tokens"], 20)
        self.assertEqual(a["head_projection_calls"], 3)
        self.assertEqual(b["head_projection_calls"], 6)

    def test_invalid_workloads_fail_closed(self):
        spec = arm_spec("A0_global", 28)
        for n, g in ((0, 1), (1, 0), (True, 1), (40960, 1), (17, -1)):
            with self.subTest(n=n, g=g), self.assertRaises(ValueError):
                inference_workload(spec, TARGET, n, g)
        with self.assertRaises(ValueError):
            inference_workload(spec, dict(TARGET, num_hidden_layers=12), 17, 1)

    def test_runtime_generations_and_resident_storage_match_planner(self):
        torch.manual_seed(17)
        model = TinyQwen(layers=8)
        model.config.intermediate_size = 64  # Actual fixture MLP width.
        ids = torch.randint(0, 97, (1, 29))
        for name in ARMS:
            for block in (3, 7):
                spec = replace(arm_spec(name, 8, block_size=block), window=8,
                               sinks=2 if ARMS[name][1] else 0)
                rt = HybridRuntime(model, spec)
                for g in (1, 4):
                    with self.subTest(arm=name, block=block, generated=g):
                        result = rt.generate(ids, max_new_tokens=g, chunk_size=block)
                        validate_execution(spec, model.config, {"tokens": 29}, result)

    def test_early_eos_and_counter_corruption(self):
        model = TinyQwen()
        model.config.intermediate_size = 64
        spec = replace(arm_spec("A3_sink512", 4, block_size=7), window=8, sinks=2)
        rt = HybridRuntime(model, spec)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        first = rt.generate(ids, max_new_tokens=1, chunk_size=7)["token_ids"][0]
        gen = rt.generate(ids, max_new_tokens=4, eos_token_ids=(first,), chunk_size=7)
        self.assertEqual(len(gen["token_ids"]), 1)
        validate_execution(spec, model.config, {"tokens": 9}, gen)
        bad = copy.deepcopy(gen)
        bad["attention_stats"]["issued_score_elements"] += 1
        with self.assertRaises(RuntimeError):
            validate_execution(spec, model.config, {"tokens": 9}, bad)
        bad = copy.deepcopy(gen)
        bad["cache"][0]["retained_tokens"] += 1
        with self.assertRaises(RuntimeError):
            validate_execution(spec, model.config, {"tokens": 9}, bad)


class BudgetAndGuardTests(unittest.TestCase):
    def test_plan_is_bounded_and_does_not_authorize(self):
        p = plan()
        self.assertEqual(p["max_generations"], 720)
        self.assertFalse(p["licenses_training"])
        self.assertFalse(p["licenses_sealed_evaluation"])
        self.assertFalse(p["established_quality_improvement"])
        self.assertEqual(p["future_training_seeds"], (17, 29, 43))

    def test_compute_match_uses_whole_updates_and_reports_remainder(self):
        kw = {"cost_unit": "profiled_update_flops"}
        a = matched_update_budget(64, 10, 8, **kw)
        self.assertEqual(a["candidate_steps"], 80)
        self.assertTrue(a["within_two_percent"])
        self.assertEqual(a["unspent"], 0)
        b = matched_update_budget(64, 10, 9, **kw)
        self.assertEqual(b["candidate_steps"], 71)
        self.assertEqual(b["unspent"], 1)
        self.assertFalse(b["same_token_budget"])
        self.assertFalse(b["licenses_training"])

    def test_cap_hit_is_not_labeled_compute_matched(self):
        a = matched_update_budget(64, 10, 1, cost_unit="isolated_update_seconds")
        self.assertEqual(a["candidate_steps"], 128)
        self.assertTrue(a["cap_hit"])
        self.assertFalse(a["within_two_percent"])
        b = matched_update_budget(64, 10, 1000, cost_unit="profiled_update_flops")
        self.assertEqual(b["candidate_steps"], 0)
        self.assertFalse(b["within_two_percent"])

    def test_invalid_costs_and_attention_only_unit_rejected(self):
        for cost in (0, -1, float("nan"), float("inf"), True, "1"):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                matched_update_budget(64, cost, 1, cost_unit="profiled_update_flops")
        with self.assertRaises(ValueError):
            matched_update_budget(64, 10, 8, cost_unit="attention_only_flops")

    def test_new_controls_and_unchanged_checkpoint_are_required(self):
        r = {"study_version": VERSION, "checkpoint_unchanged": True,
             "criterion_version": QUALIFICATION_CRITERION_VERSION,
             "checks": {k: {"passed": True} for k in ("all_local_stream", "frontloaded_stream")}}
        require_study_receipt(r)
        # A qualification judged under the superseded absolute-tolerance
        # criterion must not be able to open the diagnostic.
        for change in ({"study_version": 1}, {"study_version": 2.0},
                       {"criterion_version": 1}, {"criterion_version": None},
                       {"checkpoint_unchanged": False}, {"checkpoint_unchanged": 1}, {"checks": {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                require_study_receipt(dict(r, **change))

    def test_current_qualification_includes_distinct_topologies(self):
        torch.manual_seed(17)
        model = TinyQwen(layers=8)
        for backend in ("math", "sdpa"):
            checks = qualify(model, backend, 3)
            self.assertEqual(len(checks), 7)
            self.assertTrue(all(c["passed"] for c in checks.values()), checks)

    def test_nonfinite_qualification_failures_remain_serializable(self):
        for b in (torch.tensor([[[float("nan")]]]), torch.tensor([[[float("inf")]]]), torch.zeros(2)):
            result = compare_logits(torch.zeros(1, 1, 1), b)
            self.assertFalse(result["passed"])
            json.dumps(result, allow_nan=False)

    def test_invalid_cli_budgets_and_closed_stages(self):
        cases = [["train"], ["sealed"], ["qualify", "--run-id", "no-write", "--block-size", "129"],
                 ["diagnose", "--run-id", "no-write", "--max-cases", "73"],
                 ["audit", "--run-id", "no-write", "--cap-mib", "6145"]]
        for args in cases:
            with self.subTest(args=args), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_missing_weights_produce_failed_non_authorizing_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ckpt = root / "checkpoint"
            ckpt.mkdir()
            (ckpt / "config.json").write_text('{"model_type":"qwen3"}')
            with patch("experiments.hybrid_attention_v1.provenance.ROOT", root), patch("sys.stdout", io.StringIO()):
                result = main(["audit", "--run-id", "missing-weights", "--model", str(ckpt)])
            receipt = json.loads((root / "runs/missing-weights/receipt.json").read_text())
            self.assertEqual(result, 1)
            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["licenses_training"])
            self.assertFalse(receipt["licenses_sealed_evaluation"])
            sha = receipt.pop("receipt_sha256")
            self.assertEqual(sha, digest(receipt))


class ContrastTests(unittest.TestCase):
    def rows(self):
        return {name: [dict(id=str(i), prompt_sha256=str(i), answer="1" if i == 0 else "UNKNOWN",
                    answerable=i == 0, strict_exact=False, trimmed_exact=False) for i in range(2)] for name in ARMS}

    def test_strict_and_trimmed_endpoints_stay_separate(self):
        rows = self.rows()
        rows["A3_sink512"][0]["trimmed_exact"] = True
        report = contrast_report(rows)
        self.assertEqual(len(report), len(CONTRASTS))
        metrics = report["primary_frozen_sink_retrofit"]["metrics"]
        self.assertEqual(metrics["strict_exact/all"]["accuracy_delta"], 0)
        self.assertEqual(metrics["trimmed_exact/all"]["accuracy_delta"], 0.5)
        self.assertEqual(metrics["trimmed_exact/answerable"]["accuracy_delta"], 1)
        self.assertEqual(metrics["trimmed_exact/unanswerable"]["accuracy_delta"], 0)

    def test_contrast_direction_and_membership(self):
        rows = self.rows()
        rows["A2_local512"][0]["strict_exact"] = True
        report = contrast_report(rows)
        self.assertEqual(report["sink_at_equal_budget"]["metrics"]["strict_exact/all"]["accuracy_delta"], -0.5)
        rows["A3_sink512"][0]["prompt_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            contrast_report(rows)

    def test_empty_subset_is_unmeasured_not_zero(self):
        rows = {k: v[1:] for k, v in self.rows().items()}
        report = contrast_report(rows)
        self.assertIsNone(report["primary_frozen_sink_retrofit"]["metrics"]["strict_exact/answerable"])


if __name__ == "__main__":
    unittest.main()
