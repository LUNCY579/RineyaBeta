#!/usr/bin/env python3
"""Evaluate SGQR modes with the existing Rineya quality protocol.

The harness measures real generation quality, routing activity, synchronized GPU
latency and peak CUDA memory. It contains no confidence estimator or confidence
threshold. For adaptive execution, a fixed group budget is allocated solely by
static or state-conditioned significance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import memory as mem  # noqa: E402
from scripts.common.quality_evaluation import (  # noqa: E402
    QUALITY_EVALUATION_VERSION,
    aggregate_quality_rows,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)
from scripts.common.response_control import (  # noqa: E402
    build_stopping_criteria,
    finalize_response,
    infer_stop_reason,
)
from scripts.sgqr.install import (  # noqa: E402
    install_sgqr_mlp,
    load_router,
    load_static_scores,
    resolve_decoder_layers,
)

DATASETS = {
    "chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
    "rag_dev": "artifacts/eval_sets/rag_dev.jsonl",
    "rag_test_SEALED": "artifacts/eval_sets/rag_test_SEALED.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(record: dict) -> dict:
    messages = [dict(message) for message in record.get("messages") or []]
    reference = record.get("reference")
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content", ""))
        messages = messages[:-1]
    out = dict(record)
    out["messages"] = messages
    out["reference"] = reference
    out["category"] = str(
        record.get("category")
        or (record.get("metadata") or {}).get("category")
        or (record.get("metadata") or {}).get("reply_class")
        or "uncategorized"
    )
    return out


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return float(ordered[index])


def resolve_budget(group_count: int, groups: int | None, fraction: float | None) -> int:
    if groups is not None and fraction is not None:
        raise ValueError("provide only one of --budget-groups or --budget-fraction")
    if fraction is not None:
        if fraction < 0 or fraction > 1:
            raise ValueError("budget fraction must be in [0, 1]")
        groups = int(math.ceil(group_count * fraction))
    if groups is None:
        groups = max(1, group_count // 2)
    if groups < 0 or groups > group_count:
        raise ValueError(f"budget groups must be within [0, {group_count}]")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="artifacts/rineya-core-en32k-28L")
    parser.add_argument("--mode", required=True, choices=(
        "dense",
        "fp_frozen",
        "deleted",
        "int8_all",
        "int4_all",
        "static_topk",
        "magnitude_topk",
        "random_topk",
        "dynamic_topk",
    ))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--budget-groups", type=int, default=None)
    parser.add_argument("--budget-fraction", type=float, default=None)
    parser.add_argument("--scores", default=None)
    parser.add_argument("--router", default=None)
    parser.add_argument("--router-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cap-mib", type=float, default=5120.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    assets = json.loads(
        (ROOT / "manifests" / "eval_assets.json").read_text(encoding="utf-8")
    )
    protocol = dict(assets["generation_protocol"])
    if args.max_new_tokens is not None:
        protocol["max_new_tokens"] = int(args.max_new_tokens)

    dataset_path = ROOT / DATASETS.get(args.dataset, args.dataset)
    if "SEALED" in dataset_path.name:
        raise SystemExit(
            "refusing sealed-test evaluation during SGQR development; freeze a rung first"
        )
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        normalize(json.loads(line))
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[: args.limit]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    cap = mem.apply_cap(args.cap_mib if use_cuda else None)

    model_path = ROOT / args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = True

    layers = resolve_decoder_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise IndexError(f"layer {args.layer} outside [0, {len(layers)})")
    mlp = layers[args.layer].mlp
    if not hasattr(mlp, "gate_proj"):
        raise TypeError("target layer does not expose a Qwen-style MLP")
    intermediate_size, hidden_size = mlp.gate_proj.weight.shape
    group_count = math.ceil(intermediate_size / args.group_size)
    budget_groups = resolve_budget(
        group_count, args.budget_groups, args.budget_fraction
    )

    bits = 8 if args.mode == "int8_all" else 4 if args.mode == "int4_all" else args.bits
    internal_mode = {
        "int8_all": "quantized_all",
        "int4_all": "quantized_all",
    }.get(args.mode, args.mode)

    scores = None
    if internal_mode == "static_topk":
        if args.scores is None:
            raise SystemExit("--scores is required for static_topk")
        scores = load_static_scores(ROOT / args.scores)

    router = None
    if internal_mode == "dynamic_topk":
        if args.router is None:
            raise SystemExit("--router is required for dynamic_topk")
        router = load_router(
            ROOT / args.router,
            hidden_size=int(hidden_size),
            group_count=group_count,
            rank=args.router_rank,
        )

    installation = install_sgqr_mlp(
        model,
        layer_idx=args.layer,
        mode=internal_mode,
        bits=bits,
        group_size=args.group_size,
        budget_groups=budget_groups,
        static_scores=scores,
        dynamic_router=router,
        retain_dense_reference=False,
        random_seed=args.seed,
    )
    model = model.to(device).eval()

    if use_cuda:
        cuda_sync()
        load_allocated = int(torch.cuda.memory_allocated())
        load_reserved = int(torch.cuda.memory_reserved())
    else:
        load_allocated = load_reserved = 0

    config = load_quality_config()
    corpus_profile = build_corpus_profile(records)
    pad_id = tokenizer.pad_token_id

    # A two-token warm-up initializes kernels without contaminating measured
    # routing and memory statistics.
    if records:
        warm_record = records[0]
        rendered = tokenizer.apply_chat_template(
            warm_record["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        warm_inputs = tokenizer(rendered, return_tensors="pt").to(device)
        with torch.inference_mode():
            model.generate(
                **warm_inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if installation.wrapper is not None:
            installation.wrapper.reset_stats()

    if use_cuda:
        cuda_sync()
        torch.cuda.reset_peak_memory_stats()

    rows = []
    latencies_ms: list[float] = []
    decode_tokens = 0
    started = time.perf_counter()

    for index, record in enumerate(records):
        rendered = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        stopping, criterion = build_stopping_criteria(
            tokenizer,
            int(inputs.input_ids.shape[1]),
            str(protocol["response_control"]),
            int(protocol["min_new_tokens_before_boundary"]),
        )
        generation_kwargs = {
            "max_new_tokens": int(protocol["max_new_tokens"]),
            "do_sample": bool(protocol["do_sample"]),
            "pad_token_id": pad_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if stopping is not None:
            generation_kwargs["stopping_criteria"] = stopping

        cuda_sync()
        case_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        cuda_sync()
        latency_ms = (time.perf_counter() - case_started) * 1000.0
        latencies_ms.append(latency_ms)

        new_tokens = generated[0, inputs.input_ids.shape[1] :]
        decode_tokens += int(new_tokens.numel())
        raw_with_special = tokenizer.decode(
            new_tokens, skip_special_tokens=False
        ).strip()
        ordinary = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        finalized = finalize_response(
            ordinary, str(protocol["response_control"])
        )
        reply = finalized.text
        visible = (
            len(tokenizer(reply, add_special_tokens=False)["input_ids"])
            if reply
            else 0
        )
        stop_reason = infer_stop_reason(
            new_tokens.tolist(),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=int(protocol["max_new_tokens"]),
            criterion=criterion,
        )
        quality = evaluate_response(
            record,
            reply,
            visible,
            visible_model_tokens=visible,
            config=config,
            corpus_profile=corpus_profile,
        )
        rows.append(
            {
                "id": str(record["id"]),
                "category": record["category"],
                "reply": reply,
                "raw_reply": raw_with_special,
                "reference": record.get("reference"),
                "generated_tokens": int(new_tokens.numel()),
                "visible_generated_tokens": visible,
                "prompt_tokens": int(inputs.input_ids.shape[1]),
                "latency_ms": latency_ms,
                "stop_reason": stop_reason,
                "response_control_trimmed": bool(finalized.trimmed),
                **quality,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"{index + 1:>4}/{len(records)}")

    elapsed_s = time.perf_counter() - started
    with (out_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    quality_summary = aggregate_quality_rows(rows)
    if use_cuda:
        cuda_sync()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_allocated = peak_reserved = 0

    routing = (
        installation.wrapper.stats().as_dict()
        if installation.wrapper is not None
        else {
            "forward_calls": None,
            "token_count": None,
            "group_token_invocations": None,
            "available_group_tokens": None,
            "active_group_fraction": (
                0.0 if args.mode == "deleted" else 1.0 if args.mode in {"dense", "fp_frozen"} else None
            ),
        }
    )
    storage = installation.storage_manifest()
    summary = {
        **quality_summary,
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "sgqr": {
            "mode": args.mode,
            "internal_mode": internal_mode,
            "layer_idx": args.layer,
            "bits": bits if installation.wrapper is not None else None,
            "group_size": args.group_size if installation.wrapper is not None else None,
            "group_count": group_count,
            "budget_groups": budget_groups if installation.wrapper is not None else None,
            "routing": routing,
            "storage": storage,
        },
        "performance": {
            "cases": len(rows),
            "generated_tokens": decode_tokens,
            "elapsed_s": elapsed_s,
            "tokens_per_second": decode_tokens / elapsed_s if elapsed_s > 0 else None,
            "mean_case_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
            "p50_case_latency_ms": percentile(latencies_ms, 0.50),
            "p90_case_latency_ms": percentile(latencies_ms, 0.90),
            "p95_case_latency_ms": percentile(latencies_ms, 0.95),
            "load_allocated_bytes": load_allocated,
            "load_reserved_bytes": load_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "format": "sgqr-evaluation-manifest-v1",
        "dataset": str(dataset_path.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": sha256_file(dataset_path),
        "model": args.model,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "mode": args.mode,
        "layer_idx": args.layer,
        "bits": bits,
        "group_size": args.group_size,
        "budget_groups": budget_groups,
        "scores": args.scores,
        "router": args.router,
        "seed": args.seed,
        "protocol": protocol,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "cap_mib": cap,
        "torch": torch.__version__,
    }
    (out_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
