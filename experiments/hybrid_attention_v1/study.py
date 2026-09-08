"""Bounded HLGA-1 study contract and analytical accounting, never a trainer.

FLOPs below count specified matrix multiplications, NOT device instructions or
wall time. The rectangular attention work includes masked-out score positions.
"""
from fractions import Fraction
import json
import math
from .config import ARMS, arm_spec

VERSION = 2
MAX_CASES = 72
MAX_NEW_TOKENS = 24
MAX_BLOCK_SIZE = 128
SEEDS = (17, 29, 43)
TARGET = {
    "model_type": "qwen3", "num_hidden_layers": 28, "hidden_size": 1024,
    "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
    "intermediate_size": 3072, "vocab_size": 32768,
    "max_position_embeddings": 40960, "rope_theta": 1000000,
}
# (name, reference, treatment). Positive deltas always favor the treatment.
CONTRASTS = (
    ("primary_frozen_sink_retrofit", "A0_global", "A3_sink512"),
    ("pure_window_damage", "A0_global", "A2_local512"),
    ("sink_at_equal_budget", "A2_local512", "A3_sink512"),
    ("window_budget", "A1_local1024", "A2_local512"),
    ("sink_at_1024", "A1_local1024", "A7_sink1024"),
    ("periodic_global_access", "A8_all_local512", "A2_local512"),
    ("placement_at_equal_work", "A9_frontloaded512", "A2_local512"),
    ("global_gate_identity", "A0_global", "A4_global_gate"),
    ("local_gate_identity", "A2_local512", "A5_local512_gate"),
    ("sink_gate_identity", "A3_sink512", "A6_sink512_gate"),
)


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def assert_target_config(config):
    """Shape lock is NOT a claim that arbitrary same-shaped weights are R2."""
    for key, value in TARGET.items():
        if config.get(key) != value:
            raise ValueError(f"Primary R2 target mismatch: {key}; register other checkpoints separately")


def causal_pairs(tokens, window=None):
    _positive_int(tokens, "tokens")
    if window is None:
        return tokens * (tokens + 1) // 2
    _positive_int(window, "window")
    m = min(tokens, window)
    return m * (m + 1) // 2 + (tokens - m) * window


def _layer_pairs(prompt, generated, block, window, sinks):
    # Match generate(): prefill chunks of block tokens, then generated-1 inputs.
    # The last predicted token is returned, not forwarded through the decoder.
    rectangular = 0
    for start in range(0, prompt, block):
        end = min(start + block, prompt)
        keys = end
        if window is not None:
            recent_start = max(0, start - (window - sinks) + 1)
            prefix_end = min(sinks, end)
            overlap = max(0, prefix_end - recent_start)
            keys = prefix_end + end - recent_start - overlap
        rectangular += (end - start) * keys
    decode = causal_pairs(prompt + generated - 1, window) - causal_pairs(prompt, window)
    return causal_pairs(prompt, window) + decode, rectangular + decode


def inference_workload(spec, config, prompt_tokens, generated_tokens, element_bytes=4):
    """Analytical counts for batch-one generate() with chunk_size=block_size.

    Includes Q/K/V/O, three SwiGLU projections, the LM head and optional gate
    projection. Uses actual Q/KV head widths, which need not equal hidden_size.
    No generic '3x forward' training multiplier is licensed by this function.
    """
    n = _positive_int(prompt_tokens, "prompt_tokens")
    g = _positive_int(generated_tokens, "generated_tokens")
    _positive_int(element_bytes, "element_bytes")
    get = config.get if isinstance(config, dict) else lambda k: getattr(config, k)
    h, kv, d, width, inter, vocab, layers, context = [
        _positive_int(get(k), k) for k in (
            "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size",
            "intermediate_size", "vocab_size", "num_hidden_layers", "max_position_embeddings")]
    if h % kv or layers != spec.layers or n + g > context:
        raise ValueError("Unsupported dimensions or prompt/generation exceeds original context")
    visible = rectangular = 0
    for layer in range(spec.layers):
        window = spec.local_window(layer)
        a, b = _layer_pairs(n, g, spec.block_size, window, spec.sinks if window else 0)
        visible += a
        rectangular += b
    seen = n + g - 1
    # Multiply-add = two FLOPs; embedding lookups and non-matmul work excluded.
    projections = 2 * seen * layers * (width * d * (2*h + 2*kv) + 3*width*inter)
    gate_projection = 2 * seen * layers * width * h if spec.gate else 0
    head_calls = (n + spec.block_size - 1) // spec.block_size + g - 1
    lm_head = 2 * head_calls * width * vocab
    attention = 4 * h * d * rectangular
    return {
        "evidence_class": "analytical_accounting_not_performance_measurement",
        "prompt_tokens": n, "generated_tokens": g, "decoder_tokens": seen,
        "head_projection_calls": head_calls,
        "visible_score_elements": h * visible,
        "issued_score_elements": h * rectangular,
        "attention_matmul_flops": attention,
        "decoder_linear_matmul_flops": projections,
        "gate_projection_matmul_flops": gate_projection,
        "gate_elementwise_multiply_ops": seen * layers * h * d if spec.gate else 0,
        "lm_head_matmul_flops": lm_head,
        "modeled_matmul_flops": projections + gate_projection + lm_head + attention,
        "resident_kv_bytes": spec.cache_bytes(seen, kv, d, element_bytes),
        "excluded": "norm/RoPE/softmax/sigmoid, cache copies, GQA expansion, gathers, allocator, launch, host and backward costs",
        "timing_causal": False, "licenses_training": False,
    }


def validate_execution(spec, config, case, generation):
    """Bind actual per-case shapes/storage to the planner; fail on mismatches."""
    expected = inference_workload(spec, config, case["tokens"], len(generation["token_ids"]))
    stats = generation["attention_stats"]
    for name in ("issued_score_elements", "attention_matmul_flops"):
        if stats.get(name) != expected[name]:
            raise RuntimeError(f"Observed execution/accounting mismatch: {name}")
    cache = generation["cache"]
    if len(cache) != spec.layers:
        raise RuntimeError("Missing layer cache metrics")
    for i, row in enumerate(cache):
        window = spec.local_window(i)
        retained = expected["decoder_tokens"] if window is None else min(expected["decoder_tokens"], window)
        if (row["seen_tokens"] != expected["decoder_tokens"] or row["retained_tokens"] != retained
                or row["logical_kv_bytes"] != row["resident_kv_bytes"]):
            raise RuntimeError("Per-layer resident cache invariant failed")
    if sum(row["resident_kv_bytes"] for row in cache) != expected["resident_kv_bytes"]:
        raise RuntimeError("Resident cache total disagrees with exact slot accounting")
    return expected


def require_study_receipt(receipt):
    # Additional guards after provenance.require_receipt's checkpoint/code binding.
    if (type(receipt.get("study_version")) is not int or receipt["study_version"] != VERSION
            or receipt.get("checkpoint_unchanged") is not True):
        raise ValueError("Current study version and unchanged checkpoint receipt required")
    for name in ("all_local_stream", "frontloaded_stream"):
        if receipt.get("checks", {}).get(name, {}).get("passed") is not True:
            raise ValueError(f"New topology control is not mechanically qualified: {name}")


def contrast_report(all_rows):
    from .diagnostics import paired_result
    report = {}
    for name, reference, treatment in CONTRASTS:
        a, b = all_rows[reference], all_rows[treatment]
        item = {"reference": reference, "treatment": treatment, "metrics": {}}
        for metric in ("strict_exact", "trimmed_exact"):
            for subset in ("all", "answerable", "unanswerable"):
                use = lambda r: subset == "all" or r["answerable"] == (subset == "answerable")
                aa = [dict(r, trimmed_exact=r[metric]) for r in a if use(r)]
                bb = [dict(r, trimmed_exact=r[metric]) for r in b if use(r)]
                item["metrics"][f"{metric}/{subset}"] = paired_result(aa, bb) if aa or bb else None
        report[name] = item
    return report


def matched_update_budget(reference_steps, reference_cost, candidate_cost, *,
                          cost_unit, max_candidate_steps=128):
    """Planning arithmetic only. Costs must later come from a registered profiler.

    Token-matched and compute-matched runs are distinct experiments. Cost may
    denote measured full optimizer-update FLOPs or isolated update seconds, not
    an attention-only ratio substituted for whole-model training work.
    """
    _positive_int(reference_steps, "reference_steps")
    _positive_int(max_candidate_steps, "max_candidate_steps")
    if cost_unit not in ("profiled_update_flops", "isolated_update_seconds"):
        raise ValueError("Specify the measured full-update cost unit")
    for value in (reference_cost, candidate_cost):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Costs must be finite positive numbers, not attention-only estimates")
    total = reference_steps * Fraction(str(reference_cost))
    cost = Fraction(str(candidate_cost))
    ideal = total // cost
    steps = min(ideal, max_candidate_steps)
    remainder = total - steps * cost
    return {"reference_steps": reference_steps, "candidate_steps": steps,
            "cost_unit": cost_unit, "budget": float(total), "spent": float(steps*cost),
            "unspent": float(remainder), "cap_hit": ideal > max_candidate_steps,
            "within_two_percent": steps > 0 and remainder / total <= Fraction(1, 50),
            "same_token_budget": False, "licenses_training": False}


def plan():
    rows = {}
    for name in ARMS:
        spec = arm_spec(name, TARGET["num_hidden_layers"])
        rows[name] = {"spec": spec.to_dict(), "fixed_24_token_envelope": {
            str(n): inference_workload(spec, TARGET, n, MAX_NEW_TOKENS)
            for n in (511, 512, 513, 1023, 1024, 1025)}}
    return {"study_version": VERSION, "primary_target_shape": TARGET,
            "primary_contrast": CONTRASTS[0], "max_cases_per_arm": MAX_CASES,
            "max_arms": len(ARMS), "max_generations": MAX_CASES * len(ARMS),
            "max_new_tokens": MAX_NEW_TOKENS, "max_block_size": MAX_BLOCK_SIZE,
            "arms": rows, "future_training_seeds": SEEDS,
            "future_token_matched_updates": 64, "future_compute_matched_update_cap": 128,
            "future_training_arms": ["A4_global_gate", "A5_local512_gate", "A6_sink512_gate"],
            "licenses_training": False, "licenses_sealed_evaluation": False,
            "established_quality_improvement": False, "timing_causal": False}


if __name__ == "__main__":
    print(json.dumps(plan(), indent=2))
