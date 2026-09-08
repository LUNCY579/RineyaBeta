"""HLGA-1 comparison-declaration auditor, adapted from the preserved package.

The live study.py owns topology contrasts and inference arithmetic. This module
adds matching checks only: it never loads a model, a corpus, or a training run.
A pass does not authenticate records, license a stage, or prove model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import ARMS, arm_spec
from .provenance import digest
from .study import (CONTRASTS as FROZEN_CONTRASTS, TARGET, SEEDS, MAX_CASES,
                    MAX_BLOCK_SIZE, MAX_NEW_TOKENS, assert_target_config, plan,
                    _positive_int as positive_int)

TRAINED_ARMS = frozenset(("A4_global_gate", "A5_local512_gate", "A6_sink512_gate"))
CONTRASTS = {name: (reference, treatment) for name, reference, treatment in FROZEN_CONTRASTS}
CONTRASTS.update({
    "trained_local_vs_global": ("A4_global_gate", "A5_local512_gate"),
    "trained_sink_vs_global": ("A4_global_gate", "A6_sink512_gate"),
    "trained_sink_vs_local": ("A5_local512_gate", "A6_sink512_gate"),
})
CLOSED = {"licenses_training": False, "licenses_sealed_evaluation": False,
          "quality_improvement_established": False, "timing_causal": False}


COMMON_IDS = (
    "checkpoint_id", "code_identity", "tokenizer_id", "evaluation_manifest_id",
    "prompt_protocol_id", "decoding_id", "environment_id", "measurement_protocol_id",
    "backend", "dtype", "quantization",
)
COMMON_COUNTS = ("context_length", "batch_size", "block_size", "evaluation_cases", "max_new_tokens")
TRAIN_IDS = ("training_manifest_id", "optimizer_id", "schedule_id", "loss_id", "loss_mask_protocol_id", "checkpoint_selection_id", "trainable_allowlist_id")


def _present(record: Mapping[str, Any], field: str, kind: type) -> Any:
    value = record.get(field)
    if type(value) is not kind or (kind is str and not value.strip()):
        raise ValueError(f"Missing/invalid {field}")
    if kind is int and value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def audit_pair(left: Mapping[str, Any], right: Mapping[str, Any], *, contrast: str,
               mode: str, compute_tolerance: float = 0.02) -> dict[str, Any]:
    """Validate declared pairing; this deliberately grants no execution/claim gate.

    Metadata hashes need independently inspected source receipts. This audit does
    not authenticate declarations or constitute a signature/approval mechanism.
    All comparisons, including compute tolerance, must be frozen before outcomes.
    """
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("Each ledger must be a JSON object")
    if type(compute_tolerance) not in (int, float) or not math.isfinite(compute_tolerance):
        raise ValueError("Invalid compute tolerance")
    if not 0 <= compute_tolerance <= 0.02:
        raise ValueError("Compute tolerance must be prespecified and <= 2%")
    if contrast not in CONTRASTS or (left.get("arm"), right.get("arm")) != CONTRASTS[contrast]:
        raise ValueError("Unregistered or incorrectly ordered contrast")
    if mode not in ("frozen_workload", "equal_data_training", "equal_compute_training"):
        raise ValueError("Unknown comparison mode")
    for field in COMMON_IDS + COMMON_COUNTS:
        kind = int if field in COMMON_COUNTS else str
        a, b = _present(left, field, kind), _present(right, field, kind)
        if a != b:
            raise ValueError(f"Unmatched {field}")
    if left["batch_size"] != 1 or left["dtype"] != "float32" or left["quantization"] != "none":
        raise ValueError("Only batch-one, unquantized FP32 is registered in the current HLGA-1 protocol")
    if left["backend"] not in ("math", "sdpa"):
        raise ValueError("Unregistered backend")
    if (left["max_new_tokens"] != MAX_NEW_TOKENS
            or left["context_length"] not in (511, 512, 513, 1023, 1024, 1025)
            or not 1 <= left["block_size"] <= MAX_BLOCK_SIZE
            or not 1 <= left["evaluation_cases"] <= MAX_CASES):
        raise ValueError("Unregistered diagnostic budget")
    for rec in (left, right):
        if rec.get("packing") is not False or rec.get("unpadded") is not True:
            raise ValueError("Packing/padding are not qualified")
        if type(rec.get("target_config")) is not dict:
            raise ValueError("Canonical R2 configuration declaration missing")
        assert_target_config(rec["target_config"])
        if any(type(rec["target_config"][k]) is not type(v) for k, v in TARGET.items()):
            raise ValueError("Invalid target configuration value types")
        if (type(rec.get("prefill_chunk_size")) is not int
                or rec["prefill_chunk_size"] != rec["block_size"]):
            raise ValueError("Registered prefill chunk and attention block sizes must agree")
        expected = arm_spec(rec["arm"], TARGET["num_hidden_layers"],
                            block_size=rec["block_size"], backend=rec["backend"]).to_dict()
        if type(rec.get("spec")) is not dict or digest(rec["spec"]) != digest(expected):
            raise ValueError("Arm label and executed attention specification disagree")
        _present(rec, "source_receipt_id", str)
    if mode == "frozen_workload":
        if contrast.startswith("trained_"):
            raise ValueError("A training contrast is not a frozen-workload result")
        for rec in (left, right):
            if rec.get("trained") is not False or type(rec.get("trainable_parameters")) is not int or rec["trainable_parameters"] != 0:
                raise ValueError("Frozen controls must declare zero trainable parameters")
            if ARMS[rec["arm"]][2] and rec.get("identity_gate_verified") is not True:
                raise ValueError("Untrained gate identity must be separately verified")
    else:
        if not contrast.startswith("trained_") or any(r.get("arm") not in TRAINED_ARMS for r in (left, right)):
            raise ValueError("Training comparisons require two registered gate-only arms")
        for field in TRAIN_IDS + ("seed", "trainable_parameters"):
            kind = int if field in ("seed", "trainable_parameters") else str
            if _present(left, field, kind) != _present(right, field, kind):
                raise ValueError(f"Unmatched {field}")
        for rec in (left, right):
            if rec.get("trained") is not True or rec["trainable_parameters"] != 458752:
                raise ValueError("Only matched canonical R2 gate-only capacity is registered")
            if rec.get("backbone_frozen") is not True:
                raise ValueError("Backbone training is outside this experiment")
            positive_int(rec.get("optimizer_steps"), "optimizer_steps")
            positive_int(rec.get("supervised_tokens"), "supervised_tokens")
            if rec["seed"] not in SEEDS or rec["optimizer_steps"] > 128:
                raise ValueError("Unregistered training seed or update cap")
            for key, value in (("microbatch", 1), ("accumulation", 8), ("sequence_length", 1024)):
                if type(rec.get(key)) is not int or rec[key] != value:
                    raise ValueError(f"Unregistered training geometry: {key}")
            exposure = rec["optimizer_steps"] * 8 * 1024
            if (type(rec.get("input_tokens")) is not int or rec["input_tokens"] != exposure
                    or rec["supervised_tokens"] > exposure):
                raise ValueError("Input/supervised token counts are inconsistent")
            if (mode == "equal_data_training" or rec["arm"] == "A4_global_gate") and rec["optimizer_steps"] != 64:
                raise ValueError("Registered equal-data/global-reference budget is 64 updates")
            _present(rec, "training_order_id", str)
        if any(type(rec.get("teacher_used")) is not bool for rec in (left, right)):
            raise ValueError("Teacher use must be explicit in both training designs")
        if left["teacher_used"] != right["teacher_used"]:
            raise ValueError("Teacher use differs across arms")
        if left["teacher_used"]:
            for field in ("teacher_checkpoint_id", "teacher_accounting_policy_id"):
                if _present(left, field, str) != _present(right, field, str):
                    raise ValueError(f"Unmatched teacher field: {field}")
        if mode == "equal_data_training":
            for field in ("optimizer_steps", "supervised_tokens", "training_order_id"):
                if left[field] != right[field]:
                    raise ValueError(f"Unmatched equal-data field: {field}")
        else:
            for field in ("compute_unit", "compute_accounting_id", "training_stream_id"):
                if _present(left, field, str) != _present(right, field, str):
                    raise ValueError(f"Unmatched compute field: {field}")
            unit = left["compute_unit"]
            if unit not in ("profiled_total_training_flops", "measured_training_device_seconds"):
                raise ValueError("Forward shape/FLOP estimates cannot establish matched training compute")
            costs = []
            for rec in (left, right):
                value = rec.get("compute_spent")
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("compute_spent must be finite and positive")
                costs.append(value)
                _present(rec, "compute_receipt_id", str)
                if rec.get("compute_scope") != "student_forward_backward_optimizer_and_teacher":
                    raise ValueError("Compute accounting omits required training components")
                if rec.get("accounting_complete") is not True or type(rec.get("teacher_used")) is not bool:
                    raise ValueError("Compute/teacher accounting must be explicit")
                if unit == "measured_training_device_seconds" and (
                    rec.get("hardware_isolated") is not True or rec.get("order_randomized") is not True
                ):
                    raise ValueError("Shared/fixed-order timing cannot establish compute matching")
            if abs(costs[0] - costs[1]) / max(costs) > compute_tolerance:
                raise ValueError("Full training compute differs beyond the registered tolerance")
    fingerprint = hashlib.sha256(json.dumps(
        {"left": dict(left), "right": dict(right), "contrast": contrast,
         "mode": mode, "compute_tolerance": compute_tolerance},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return {"status": "matching_declarations_passed_not_efficacy_evidence", "contrast": contrast,
            "mode": mode, "comparison_sha256": fingerprint, "compute_tolerance": compute_tolerance,
            "requires_independent_receipt_review": True, **CLOSED}



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Delegate to the canonical study planner")
    pair = sub.add_parser("audit-pair", help="Check declarations only; never unlock a stage")
    pair.add_argument("--left", type=Path, required=True)
    pair.add_argument("--right", type=Path, required=True)
    pair.add_argument("--contrast", choices=tuple(CONTRASTS), required=True)
    pair.add_argument("--mode", choices=("frozen_workload", "equal_data_training", "equal_compute_training"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan()
        else:
            records = []
            for path in (args.left, args.right):
                if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
                    raise ValueError("Ledger must be a non-symlink JSON file <= 64 KiB")
                records.append(json.loads(path.read_text(encoding="utf-8")))
            result = audit_pair(*records, contrast=args.contrast, mode=args.mode)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc), **CLOSED}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
