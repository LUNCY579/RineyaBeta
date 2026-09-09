"""CLI: plan -> audit -> qualify -> open diagnostic. Training/sealed stages stay closed."""
import argparse
from dataclasses import replace
import json
import math
import sys
import time
from pathlib import Path
import torch
from .config import ARMS, arm_spec
from .diagnostics import make_cases, paired_result, score, summarize
from .provenance import (ROOT, BASE_COMMIT, checkpoint_inventory, code_identity, digest,
                         environment, new_run, require_receipt, write_json,
                         artifact_inventory, require_artifacts, require_unchanged_code)
from .runtime import HybridRuntime, model_fingerprint
from .study import (VERSION, MAX_BLOCK_SIZE, MAX_CASES, MAX_NEW_TOKENS,
                    assert_target_config, contrast_report, inference_workload,
                    require_study_receipt, validate_execution)


def load_local(path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to(device).eval()
    if model.__class__.__name__ != "Qwen3ForCausalLM":
        raise ValueError("Checkpoint must load through the native Qwen3ForCausalLM implementation")
    # The dtype keyword has been renamed across Transformers majors. Never trust
    # the request: verify the materialized precision, so a silently ignored
    # argument cannot downgrade a registered float32 run to checkpoint dtype.
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.float32}:
        raise ValueError(f"Registered precision is float32; loaded parameters are {dtypes}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    return model, tokenizer


def compare_logits(a, b):
    if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        # Keep failed receipts JSON-serializable rather than losing NaN/Inf failures.
        return {"passed": False, "max_abs_error": None, "argmax_identical": False,
                "error": "shape mismatch or nonfinite logits"}
    # Finite FP32 operands can overflow a FP32 subtraction. Keep the diagnostic
    # magnitude representable and never lose a failed receipt to JSON Infinity.
    error = float((a.to(torch.float64)-b.to(torch.float64)).abs().max())
    if not math.isfinite(error):
        error = None
    equal = bool(torch.equal(a.argmax(-1), b.argmax(-1)))
    return {"passed": bool(torch.allclose(a, b, atol=2e-5, rtol=2e-5) and equal),
            "max_abs_error": error, "argmax_identical": equal}


# Registered mechanical chunk boundaries (protocol.json qualification block).
CHUNKS = ((0, 7), (7, 8), (8, 17))


@torch.inference_mode()
def qualify(model, backend, block_size):
    c = model.config
    g = torch.Generator(device="cpu").manual_seed(17)
    ids = torch.randint(0, c.vocab_size, (1, 17), generator=g).to(next(model.parameters()).device)
    spec = arm_spec("A0_global", c.num_hidden_layers, backend=backend, block_size=block_size)
    runtime = HybridRuntime(model, spec)
    native = model(input_ids=ids, use_cache=False).logits
    base = runtime(ids)
    checks = {"native_global": compare_logits(native, base)}

    def streamed(rt):
        rt.reset()
        out = [rt(ids[:, lo:hi], use_cache=True) for lo, hi in CHUNKS]
        resident = [x.metrics() for x in rt.cache]
        rt.reset()
        return torch.cat(out, dim=1), resident

    def attended(rt, stream):
        rt.start_trace()
        rt.reset()
        if stream:
            for lo, hi in CHUNKS:
                rt(ids[:, lo:hi], use_cache=True)
        else:
            rt(ids)
        trace = [sorted(layer) for layer in rt.trace]
        rt.stop_trace()
        rt.reset()
        return trace

    def bind_schedule_identity(check, rt):
        """Chunked and full prefill issue different-width matmuls, so their
        logits differ by a few ulps at 28 layers and an absolute tolerance
        cannot separate that from a real cache fault. Compare the attended
        absolute key positions per layer instead: exact, no tolerance.
        Strictly additive -- this can only ever withhold a pass."""
        full, stream = attended(rt, False), attended(rt, True)
        identical = full == stream
        check["attended_positions_identical"] = identical
        check["first_divergent_layer"] = next(
            (i for i, (a, b) in enumerate(zip(full, stream)) if a != b), None)
        check["passed"] = bool(check["passed"] and identical)
        return check

    checks["global_stream"] = bind_schedule_identity(
        compare_logits(base, streamed(runtime)[0]), runtime)
    checks["gate_identity"] = compare_logits(base, HybridRuntime(model, replace(spec, gate=True))(ids))
    controls = (("local_stream", "A2_local512", 0), ("sink_stream", "A3_sink512", 2),
                ("all_local_stream", "A8_all_local512", 0),
                ("frontloaded_stream", "A9_frontloaded512", 0))
    for name, arm, sinks in controls:
        local_spec = replace(arm_spec(arm, c.num_hidden_layers, backend=backend,
                                     block_size=block_size), window=8, sinks=sinks)
        local = HybridRuntime(model, local_spec)
        stream, metrics = streamed(local)
        check = compare_logits(local(ids), stream)
        check["cache"] = metrics
        compact = all(m["logical_kv_bytes"] == m["resident_kv_bytes"]
                      and m["retained_tokens"] == (17 if i in local_spec.global_layers else 8)
                      for i, m in enumerate(metrics))
        check["passed"] &= compact
        check["compact_storage"] = compact
        checks[name] = bind_schedule_identity(check, local)
    return checks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "audit", "qualify", "diagnose", "train", "sealed"))
    p.add_argument("--model", type=Path, default=Path("artifacts/rineya-core-en32k-28L"))
    p.add_argument("--run-id")
    p.add_argument("--receipt", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--backend", choices=("math", "sdpa"), default="math")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=MAX_CASES)
    p.add_argument("--cap-mib", type=int, default=6144)
    args = p.parse_args(argv)
    if args.stage in ("train", "sealed"):
        p.error("CLOSED: no reviewed training manifest, matched training registration or sealed approval")
    if args.stage == "plan":
        print((ROOT / "protocol.json").read_text(encoding="utf-8"))
        return 0
    if not args.run_id:
        p.error("--run-id is required; existing runs are never overwritten")
    if (args.threads < 1 or not 1 <= args.block_size <= MAX_BLOCK_SIZE
            or not 1 <= args.max_cases <= MAX_CASES or not 1 <= args.cap_mib <= 6144):
        p.error("Invalid thread/block/case/memory budget; block<=128, cases<=72, cap<=6144 MiB")
    out = new_run(args.run_id)
    receipt = {"stage": args.stage, "passed": False, "base_commit": BASE_COMMIT,
               "study_version": VERSION,
               "licenses_training": False, "licenses_sealed_evaluation": False,
               "timing_causal": False, "quality_improvement_established": False}
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no silent CPU fallback")
            free, total = torch.cuda.mem_get_info()
            if args.cap_mib*2**20 > free:
                raise RuntimeError("Requested allocator cap exceeds currently free device memory")
            torch.cuda.set_per_process_memory_fraction(args.cap_mib*2**20/total)
            torch.backends.cuda.matmul.allow_tf32 = False
            receipt["cuda_preflight"] = {"free_bytes": free, "total_bytes": total,
                                         "allocator_cap_mib": args.cap_mib}
        inventory = checkpoint_inventory(args.model)
        env = environment(args.device)
        receipt.update(checkpoint_id=inventory["id"], environment=env,
                       code_identity=code_identity(), backend=args.backend, block_size=args.block_size)
        write_json(out / "checkpoint_inventory.json", inventory)
        if args.stage == "audit":
            receipt.update(passed=True, stage="local_checkpoint_audit")
        else:
            assert_target_config(inventory["config"])
            if args.stage == "diagnose":
                if args.receipt is None:
                    raise ValueError("--receipt from a passing real-model qualification is required")
                prior = json.loads(args.receipt.read_text(encoding="utf-8"))
                require_receipt(prior, inventory, env, args.backend, args.block_size)
                require_study_receipt(prior)
                require_artifacts(prior, args.receipt.parent)
            model, tokenizer = load_local(args.model, args.device)
            receipt["model_fingerprint"] = model_fingerprint(model)
            if args.stage == "qualify":
                checks = qualify(model, args.backend, args.block_size)
                receipt.update(stage="real_model_qualification", checks=checks,
                               passed=all(v["passed"] for v in checks.values()))
            else:
                cases = make_cases(tokenizer)[:args.max_cases]
                write_json(out / "open_dev_cases.json", cases)
                specs = {name: arm_spec(name, model.config.num_hidden_layers,
                         backend=args.backend, block_size=args.block_size) for name in ARMS}
                envelope = {name: {str(n): inference_workload(spec, model.config, n, MAX_NEW_TOKENS)
                            for n in sorted({c["tokens"] for c in cases})} for name, spec in specs.items()}
                write_json(out / "workload_envelope.json", {
                    "fixed_generation_tokens": MAX_NEW_TOKENS, "arms": envelope,
                    "planned_generations": len(cases)*len(ARMS),
                    "max_emitted_tokens": len(cases)*len(ARMS)*MAX_NEW_TOKENS,
                    "actual_eos_stopped_work_is_not_compute_matched": True})
                all_rows, summaries = {}, {}
                eos = model.config.eos_token_id
                eos = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)
                for name in ARMS:
                    rt = HybridRuntime(model, specs[name])
                    rows, cache_metrics = [], []
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    for case in cases:
                        ids = torch.tensor([case["input_ids"]], device=args.device, dtype=torch.long)
                        gen = rt.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                                          eos_token_ids=eos, chunk_size=args.block_size)
                        work = validate_execution(rt.spec, model.config, case, gen)
                        text_ids = gen["token_ids"][:-1] if gen["stopped_eos"] else gen["token_ids"]
                        text = tokenizer.decode(text_ids, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                        row = score(case, text, gen)
                        row["token_ids"] = gen["token_ids"]
                        rows.append(row)
                        cache_metrics.append({"id": case["id"], "cache": gen["cache"],
                                              "attention_stats": gen["attention_stats"], "workload": work})
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    summary = summarize(rows)
                    summary.update(elapsed_seconds=time.perf_counter()-started, timing_causal=False,
                                   spec=rt.spec.to_dict(), trained_gate=False)
                    if args.device == "cuda":
                        summary.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    all_rows[name], summaries[name] = rows, summary
                    write_json(out / f"{name}.json", {"rows": rows, "summary": summary,
                                                      "cache_metrics": cache_metrics})
                    del rt
                for plain, gated in (("A0_global", "A4_global_gate"),
                                     ("A2_local512", "A5_local512_gate"),
                                     ("A3_sink512", "A6_sink512_gate")):
                    if [r["token_ids"] for r in all_rows[plain]] != [r["token_ids"] for r in all_rows[gated]]:
                        raise RuntimeError(f"Identity gate changed generations: {plain}/{gated}")
                pairs = {k: paired_result(all_rows["A0_global"], v) for k, v in all_rows.items()}
                write_json(out / "summary.json", {"arms": summaries, "paired_vs_A0": pairs,
                            "registered_contrasts": contrast_report(all_rows),
                            "claim": "open synthetic diagnostic only; no preservation or speed claim"})
                receipt.update(stage="open_synthetic_diagnostic", passed=True, n_cases=len(cases),
                               diagnostic_id=digest(cases), quality_improvement_established=False)
        if checkpoint_inventory(args.model)["id"] != inventory["id"]:
            raise RuntimeError("Source checkpoint assets changed during the run")
        receipt["checkpoint_unchanged"] = True
        require_unchanged_code(receipt["code_identity"])
        receipt["source_unchanged"] = True
        receipt["artifacts"] = artifact_inventory(out)
    except Exception as exc:
        receipt.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    receipt["receipt_sha256"] = digest(receipt)
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
