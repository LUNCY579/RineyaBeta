"""Offline qualification receipt; cannot authorize real-model diagnostics or training."""
import argparse
import importlib.util
import json
import sys
import unittest
import torch
from .attention import CompactKV, blocked_attention, storage_bytes
from .config import arm_spec
from .provenance import ROOT, BASE_COMMIT, code_identity, digest, environment, new_run, write_json


@torch.inference_mode()
def shape_probe():
    """R2 tensor dimensions, synthetic zeros. No checkpoint is loaded."""
    spec = arm_spec("A3_sink512", 28)
    k, v = torch.zeros(1, 8, 1025, 128), torch.ones(1, 8, 1025, 128)
    positions = torch.arange(1025)
    caches = []
    for i in range(28):
        win = spec.local_window(i)
        cache = CompactKV(win, 4 if win else 0)
        cache.retain(*cache.merge(k, v, positions))
        caches.append(cache)
    measured = storage_bytes([t for c in caches for t in (c.k, c.v)])
    expected = spec.cache_bytes(1025, 8, 128, 4)
    if measured != expected:
        raise AssertionError(f"Resident KV storage mismatch: {measured} != {expected}")
    stats = {}
    q = torch.zeros(1, 16, 1025, 128)
    blocked_attention(q, k, v, positions, positions, window=512, sinks=4,
                      block_size=128, stats=stats)
    if stats["max_mask_elements"] > 128*(512+128-1):
        raise AssertionError("Local mask exceeded its registered block bound")
    return {"evidence": "synthetic tensors with R2 dimensions; NOT a Rineya forward",
            "dtype": "float32", "tokens": 1025, "layers": 28,
            "measured_resident_kv_bytes": measured, "expected_kv_bytes": expected,
            "analytical_all_global_kv_bytes": arm_spec("A0_global", 28).cache_bytes(1025, 8, 128, 4),
            "peak_process_memory_measured": False, "attention_stats": stats,
            "layer_cache_metrics": [c.metrics() for c in caches]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--require-hf", action="store_true")
    args = p.parse_args(argv)
    out = new_run(args.run_id)
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT.parents[1]))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    native = importlib.util.find_spec("transformers") is not None
    receipt = {"stage": "cpu_fixture_qualification", "base_commit": BASE_COMMIT,
               "passed": result.wasSuccessful() and (native or not args.require_hf),
               "tests_run": result.testsRun, "tests_passed": result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
               "skipped": [{"test": t.id(), "reason": reason} for t, reason in result.skipped],
               "failures": [{"test": t.id(), "traceback": tb} for t, tb in result.failures],
               "errors": [{"test": t.id(), "traceback": tb} for t, tb in result.errors],
               "environment": environment(), "code_identity": code_identity(),
               "real_rineya_checkpoint_loaded": False, "native_hf_available": native,
               "licenses_training": False, "licenses_sealed_evaluation": False,
               "quality_improvement_established": False, "timing_causal": False}
    try:
        receipt["shape_probe"] = shape_probe()
    except Exception as exc:
        receipt.update(passed=False, shape_probe_error=str(exc))
    receipt["receipt_sha256"] = digest(receipt)
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
