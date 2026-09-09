"""Declaration fixtures are invented metadata, never evidence of real model runs."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ..config import ARMS, arm_spec
from ..matching import (COMMON_IDS, COMMON_COUNTS, TRAIN_IDS, CONTRASTS, CLOSED,
                        audit_pair, main)
from ..study import TARGET, CONTRASTS as STUDY_CONTRASTS, plan


def record(arm, trained=False):
    r = {key: "fixture-identity-not-a-real-receipt" for key in COMMON_IDS}
    r.update(arm=arm, backend="math", dtype="float32", quantization="none",
             context_length=2048, batch_size=1, block_size=128, prefill_chunk_size=128,
             evaluation_cases=72, max_new_tokens=24, packing=False, unpadded=True,
             target_config=dict(TARGET), spec=arm_spec(arm, 28).to_dict(),
             source_receipt_id="fixture-receipt-not-authenticated",
             trained=trained, trainable_parameters=458752 if trained else 0,
             identity_gate_verified=True)
    if trained:
        r.update({key: "same-reviewed-registration-fixture" for key in TRAIN_IDS})
        r.update(seed=17, backbone_frozen=True, optimizer_steps=64,
                 input_tokens=64*8*1024, supervised_tokens=1000,
                 microbatch=1, accumulation=8, sequence_length=1024,
                 training_order_id="same-order", teacher_used=False,
                 compute_unit="profiled_total_training_flops", compute_accounting_id="full-update",
                 training_stream_id="same-stream", compute_spent=100,
                 compute_receipt_id="fixture-profiler", accounting_complete=True,
                 compute_scope="student_forward_backward_optimizer_and_teacher",
                 hardware_isolated=True, order_randomized=True)
    return r


class MatchingTests(unittest.TestCase):
    def pair(self, trained=False):
        return [record(a, trained) for a in
                (("A4_global_gate", "A6_sink512_gate") if trained else ("A2_local512", "A3_sink512"))]

    def audit(self, records=None, trained=False, mode=None, **kwargs):
        return audit_pair(*(records or self.pair(trained)),
                          contrast="trained_sink_vs_global" if trained else "sink_at_equal_budget",
                          mode=mode or ("equal_data_training" if trained else "frozen_workload"), **kwargs)

    def test_preserves_live_contrasts_and_directions(self):
        for name, a, b in STUDY_CONTRASTS:
            self.assertEqual(CONTRASTS[name], (a, b))
            result = audit_pair(record(a), record(b), contrast=name, mode="frozen_workload")
            self.assertEqual(result["status"], "matching_declarations_passed_not_efficacy_evidence")
            for key, value in CLOSED.items():
                self.assertIs(result[key], value)

    def test_every_common_identity_and_count_must_match(self):
        for field in COMMON_IDS + COMMON_COUNTS:
            pair = self.pair()
            pair[1][field] = pair[1][field]+1 if field in COMMON_COUNTS else "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(pair)

    def test_missing_fields_and_bool_counts_refused(self):
        for field in COMMON_IDS + COMMON_COUNTS:
            pair = self.pair()
            del pair[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(pair)
        pair = self.pair()
        for rec in pair:
            rec["batch_size"] = True
        with self.assertRaises(ValueError):
            self.audit(pair)

    def test_unregistered_shared_budgets_refused(self):
        for key, val in (("block_size",129), ("context_length",4096), ("evaluation_cases",73),
                         ("max_new_tokens",25), ("dtype","bfloat16"), ("quantization","nf4")):
            pair=self.pair()
            for rec in pair:
                rec[key]=val
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(pair)

    def test_mislabeled_spec_and_chunk_size_refused(self):
        pair=self.pair()
        pair[1]["spec"] = arm_spec("A0_global",28).to_dict()
        with self.assertRaises(ValueError):
            self.audit(pair)
        pair=self.pair()
        pair[1]["prefill_chunk_size"]=64
        with self.assertRaises(ValueError):
            self.audit(pair)

    def test_geometry_types_and_changed_heads_refused(self):
        for field, value in (("num_hidden_layers", 28.0), ("num_key_value_heads",4)):
            pair=self.pair()
            pair[1]["target_config"][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(pair)

    def test_frozen_gate_identity_and_actual_freeze_required(self):
        a,b=record("A0_global"),record("A4_global_gate")
        b["identity_gate_verified"]=False
        with self.assertRaises(ValueError):
            audit_pair(a,b,contrast="global_gate_identity",mode="frozen_workload")
        for key, val in (("trained",True),("trainable_parameters",1),("packing",True),("unpadded",False)):
            pair=self.pair()
            pair[1][key]=val
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(pair)

    def test_equal_data_declarations_pass_without_unlocking(self):
        result=self.audit(trained=True)
        self.assertFalse(result["licenses_training"])
        self.assertTrue(result["requires_independent_receipt_review"])

    def test_equal_data_matching_includes_loss_masks_and_selection(self):
        for key in TRAIN_IDS+("training_order_id","supervised_tokens","seed"):
            pair=self.pair(True)
            pair[1][key]=pair[1][key]+1 if isinstance(pair[1][key],int) else "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(pair,trained=True)

    def test_training_geometry_updates_and_exposure_are_bounded(self):
        for key,val in (("microbatch",2),("accumulation",4),("sequence_length",512),
                        ("optimizer_steps",63),("input_tokens",1),("supervised_tokens",9999999),
                        ("seed",5),("backbone_frozen",False)):
            pair=self.pair(True)
            for r in pair:r[key]=val
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(pair,trained=True)

    def test_equal_compute_allows_more_exposure_but_not_more_than_cap(self):
        pair=self.pair(True)
        pair[1].update(optimizer_steps=80,input_tokens=80*8*1024,supervised_tokens=1250,
                       training_order_id="longer-prefix-of-same-stream")
        result=self.audit(pair,trained=True,mode="equal_compute_training")
        self.assertFalse(result["quality_improvement_established"])
        pair[1].update(optimizer_steps=129,input_tokens=129*8*1024)
        with self.assertRaises(ValueError):
            self.audit(pair,trained=True,mode="equal_compute_training")

    def test_attention_only_or_incomplete_compute_rejected(self):
        for key,val in (("compute_unit","forward_matmul_flops"),("compute_scope","attention_only"),
                        ("accounting_complete",False)):
            pair=self.pair(True)
            for r in pair:r[key]=val
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(pair,trained=True,mode="equal_compute_training")

    def test_two_percent_compute_tolerance_is_not_relaxable(self):
        pair=self.pair(True)
        pair[1]["compute_spent"]=98
        self.audit(pair,trained=True,mode="equal_compute_training")
        pair[1]["compute_spent"]=97.9
        with self.assertRaises(ValueError):
            self.audit(pair,trained=True,mode="equal_compute_training")
        for val in (0.03,float("nan"),True,-1):
            with self.subTest(val=val), self.assertRaises(ValueError):
                self.audit(trained=True,mode="equal_compute_training",compute_tolerance=val)

    def test_nonfinite_nonpositive_and_bool_compute_refused(self):
        for val in (float("nan"),float("inf"),0,-1,True):
            pair=self.pair(True)
            pair[1]["compute_spent"]=val
            with self.subTest(val=val), self.assertRaises(ValueError):
                self.audit(pair,trained=True,mode="equal_compute_training")

    def test_shared_or_fixed_order_seconds_not_compute_matching(self):
        for key in ("hardware_isolated","order_randomized"):
            pair=self.pair(True)
            for rec in pair:rec["compute_unit"]="measured_training_device_seconds"
            pair[1][key]=False
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(pair,trained=True,mode="equal_compute_training")

    def test_teacher_accounting_required_and_matched(self):
        pair=self.pair(True)
        for r in pair:r["teacher_used"]=True
        with self.assertRaises(ValueError):
            self.audit(pair,trained=True)
        for r in pair:r.update(teacher_checkpoint_id="same-teacher",teacher_accounting_policy_id="same-policy")
        self.audit(pair,trained=True)
        pair[1]["teacher_checkpoint_id"]="different-teacher"
        with self.assertRaises(ValueError):self.audit(pair,trained=True)

    def test_reversed_and_wrong_mode_contrasts_rejected(self):
        with self.assertRaises(ValueError):self.audit(self.pair()[::-1])
        with self.assertRaises(ValueError):self.audit(trained=True,mode="frozen_workload")
        with self.assertRaises(ValueError):self.audit(mode="equal_data_training")
        with self.assertRaises(ValueError):
            audit_pair([],{},contrast="sink_at_equal_budget",mode="frozen_workload")

    def test_plan_delegates_to_current_study_without_second_oracle(self):
        stdout=io.StringIO()
        with patch("sys.stdout",stdout):self.assertEqual(main(["plan"]),0)
        self.assertEqual(json.loads(stdout.getvalue()),json.loads(json.dumps(plan())))

    def test_cli_valid_metadata_and_invalid_root(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/"left.json",Path(d)/"right.json"]
            for path,rec in zip(paths,self.pair()):path.write_text(json.dumps(rec))
            args=["audit-pair","--left",str(paths[0]),"--right",str(paths[1]),
                  "--contrast","sink_at_equal_budget","--mode","frozen_workload"]
            with patch("sys.stdout",io.StringIO()):self.assertEqual(main(args),0)
            paths[0].write_text("[]")
            with patch("sys.stdout",io.StringIO()):self.assertEqual(main(args),1)


if __name__ == "__main__":
    unittest.main()
