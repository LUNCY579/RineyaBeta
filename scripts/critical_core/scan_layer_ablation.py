#!/usr/bin/env python3
"""Discover catastrophic-removal layers by one-block-at-a-time bypass.

This is a diagnostic scan, not a capability benchmark.  It never opens a sealed
asset, uses a deterministic training-corpus slice, and records both reference
NLL and mechanically observable generation collapse.  A layer is called CRCC
only through the predeclared classifier in the config; unusual text by itself is
not enough.

Usage:
  python scripts/critical_core/scan_layer_ablation.py \
      --config configs/critical_core_scan_v1.yaml
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    SCHEMA_VERSION,
    atomic_write_json,
    checkpoint_inventory,
    load_diagnostic_records,
    load_yaml,
    object_sha256,
    parse_layer_selection,
    read_json,
    repo_relative,
    resolve_repo_path,
    sha256_file,
)
from scripts.critical_core.metrics import (  # noqa: E402
    classify_layer,
    completion_metrics,
    summarize_cases,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dtype(name: str, torch):
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ContractError(f"unsupported dtype {name!r}; choose one of {sorted(table)}") from exc


def _decoder_layers(model):
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    )
    for names in candidates:
        current = model
        for name in names:
            if not hasattr(current, name):
                break
            current = getattr(current, name)
        else:
            if hasattr(current, "__len__") and hasattr(current, "__getitem__"):
                return current
    raise ContractError("could not locate decoder layers on the loaded model")


def _set_determinism(seed: int, torch) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _apply_memory_cap(cap_mib: float | None, torch) -> float | None:
    if cap_mib is None or not torch.cuda.is_available():
        return None
    total_mib = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    if cap_mib <= 0 or cap_mib > total_mib:
        raise ContractError(f"memory.cap_mib must lie in (0, {total_mib:.0f}]")
    torch.cuda.set_per_process_memory_fraction(min(cap_mib / total_mib, 0.99), 0)
    return total_mib


def _render_prompt(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _tokenize_case(
    record: Mapping[str, Any], tokenizer, *, max_total_tokens: int
) -> dict[str, Any]:
    prompt_text = _render_prompt(tokenizer, record["messages"])
    eos = tokenizer.eos_token or "<|im_end|>"
    target_text = str(record["reference"]) + eos
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids

    if len(prompt_ids) + len(target_ids) > max_total_tokens:
        room = max_total_tokens - len(target_ids)
        if room <= 0:
            target_ids = target_ids[:max_total_tokens]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[-room:]
    if len(target_ids) < 2:
        raise ContractError(f"case {record['id']} has too little target signal after tokenization")
    return {
        **dict(record),
        "prompt_ids": [int(value) for value in prompt_ids],
        "target_ids": [int(value) for value in target_ids],
    }


def _reference_nll(model, case: Mapping[str, Any], *, device, torch) -> float:
    input_ids = list(case["prompt_ids"]) + list(case["target_ids"])
    labels = [-100] * len(case["prompt_ids"]) + list(case["target_ids"])
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=ids,
            attention_mask=mask,
            labels=label_tensor,
            use_cache=False,
        )
    value = float(output.loss.float().item())
    return value if math.isfinite(value) else None


def _generate(
    model,
    case: Mapping[str, Any],
    tokenizer,
    generation: Mapping[str, Any],
    *,
    device,
    torch,
) -> dict[str, Any]:
    prompt_ids = list(case["prompt_ids"])
    if not prompt_ids:
        fallback = tokenizer.bos_token_id
        if fallback is None:
            raise ContractError(f"case {case['id']} has an empty prompt and no BOS token")
        prompt_ids = [int(fallback)]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    max_new_tokens = int(generation.get("max_new_tokens", 48))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=mask, **kwargs)
    new_ids = [int(value) for value in output[0, ids.shape[1] :].tolist()]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    eos_id = tokenizer.eos_token_id
    ended_with_eos = bool(eos_id is not None and eos_id in new_ids)
    hit_limit = len(new_ids) >= max_new_tokens
    return {
        "text": text,
        "token_ids": new_ids,
        "metrics": completion_metrics(
            text,
            new_ids,
            ended_with_eos=ended_with_eos,
            hit_max_new_tokens=hit_limit,
        ),
    }


def _evaluate_condition(model, cases, tokenizer, generation, *, device, torch) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(cases, 1):
        nll = _reference_nll(model, case, device=device, torch=torch)
        generated = _generate(
            model, case, tokenizer, generation, device=device, torch=torch
        )
        rows.append(
            {
                "id": case["id"],
                "source": case["source"],
                "category": case["category"],
                "reference_nll": nll,
                "generated_text": generated["text"],
                "generated_token_ids": generated["token_ids"],
                "completion_metrics": generated["metrics"],
            }
        )
        if index % 8 == 0 or index == len(cases):
            print(f"    {index:>3}/{len(cases)} cases")
    return {
        "cases": rows,
        "summary": summarize_cases(rows),
        "elapsed_s": round(time.time() - started, 3),
    }


@contextmanager
def _bypass_layer(layers, layer_index: int, torch) -> Iterator[None]:
    class BypassDecoderLayer(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):  # noqa: ANN001
            if kwargs.get("use_cache"):
                raise RuntimeError("CRCC bypass requires use_cache=False")
            if kwargs.get("output_attentions"):
                return hidden_states, None
            return (hidden_states,)

    original = layers[layer_index]
    layers[layer_index] = BypassDecoderLayer()
    try:
        yield
    finally:
        layers[layer_index] = original


def _validate_completed_result(path: Path, expected_ids: Sequence[str]) -> dict[str, Any]:
    payload = read_json(path)
    ids = [str(row["id"]) for row in payload.get("cases", [])]
    if ids != list(expected_ids):
        raise ContractError(f"completed result {path} does not match this diagnostic set")
    return payload


def _join_layer_profile(summary: dict[str, Any], configured_path: str | None) -> None:
    if not configured_path:
        return
    path = resolve_repo_path(configured_path, must_exist=True)
    profile = read_json(path)
    distances = profile.get("angular_distance", {}).get("1")
    if not isinstance(distances, list) or len(distances) != summary["num_hidden_layers"]:
        raise ContractError(f"layer profile {path} does not match the scanned model depth")
    for row in summary["layers"]:
        row["single_layer_angular_distance"] = float(distances[int(row["layer"])])
    summary["layer_profile"] = {
        "path": repo_relative(path),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="override config.scan.output")
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, must_exist=True)
    config = load_yaml(config_path)
    seed = int(config.get("seed", 0))
    generation_cfg = config.get("generation", {})
    if bool(generation_cfg.get("do_sample", False)):
        raise ContractError("CRCC discovery is greedy; generation.do_sample must be false")
    if bool(generation_cfg.get("enable_thinking", False)):
        raise ContractError("CRCC discovery freezes enable_thinking=false")
    model_path = resolve_repo_path(config["model"]["path"], must_exist=True)
    output = args.out or config.get("scan", {}).get("output")
    if not output:
        raise ContractError("scan.output is required")
    out_dir = resolve_repo_path(output)
    layers_dir = out_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    records, source_provenance = load_diagnostic_records(config)
    print(f"diagnostic rows: {len(records)} from {len(source_provenance)} unsealed source(s)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _set_determinism(seed, torch)
    total_mib = _apply_memory_cap(config.get("memory", {}).get("cap_mib"), torch)
    requested_device = str(config.get("device", "auto")).lower()
    if requested_device not in {"auto", "cpu", "cuda", "cuda:0"}:
        raise ContractError("device must be one of auto, cpu, cuda, or cuda:0")
    if requested_device in {"cuda", "cuda:0"} and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available() and requested_device != "cpu"
        else "cpu"
    )
    requested_dtype = _dtype(str(config["model"].get("dtype", "bfloat16")), torch)
    load_dtype = torch.float32 if device.type == "cpu" else requested_dtype

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized = [
        _tokenize_case(
            record,
            tokenizer,
            max_total_tokens=int(config["data"].get("max_total_tokens", 768)),
        )
        for record in records
    ]
    diagnostic_descriptor = [
        {
            "id": row["id"],
            "source": row["source"],
            "category": row["category"],
            "prompt_tokens": len(row["prompt_ids"]),
            "target_tokens": len(row["target_ids"]),
        }
        for row in tokenized
    ]

    model_identity = checkpoint_inventory(model_path, hash_weights=args.hash_weights)
    code_paths = [
        Path(__file__),
        ROOT / "scripts" / "critical_core" / "common.py",
        ROOT / "scripts" / "critical_core" / "metrics.py",
    ]
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "config_path": repo_relative(config_path),
        "config_sha256": sha256_file(config_path),
        "seed": seed,
        "model": model_identity,
        "sources": source_provenance,
        "diagnostic_cases": diagnostic_descriptor,
        "generation": dict(config.get("generation", {})),
        "criticality": dict(config.get("criticality", {})),
        "code": {repo_relative(path): sha256_file(path) for path in code_paths},
    }
    run_identity_sha = object_sha256(run_identity)
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("run_identity_sha256") != run_identity_sha:
            raise ContractError(
                f"{out_dir} belongs to a different scan identity; choose a new output directory"
            )
    else:
        atomic_write_json(
            manifest_path,
            {
                **run_identity,
                "run_identity_sha256": run_identity_sha,
                "created_at": _utc_now(),
                "device": str(device),
                "torch_version": torch.__version__,
                "transformers_version": __import__("transformers").__version__,
                "cuda_total_mib": round(total_mib, 1) if total_mib else None,
                "command": sys.argv,
            },
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=load_dtype,
        attn_implementation=config["model"].get("attn_implementation", "eager"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(device).eval()
    layers = _decoder_layers(model)
    num_layers = len(layers)
    if int(getattr(model.config, "num_hidden_layers", num_layers)) != num_layers:
        raise ContractError("model config depth and decoder container length disagree")
    selected_layers = (
        sorted(args.layers)
        if args.layers is not None and len(args.layers) > 0
        else parse_layer_selection(
            config.get("scan", {}).get("layers", "all"), num_layers=num_layers
        )
    )
    selected_layers = parse_layer_selection(selected_layers, num_layers=num_layers)
    print(f"model: {repo_relative(model_path)} | {num_layers} layers | device={device}")
    print(f"scan layers: {selected_layers}")
    if args.dry_run:
        print(f"dry-run complete; identity written to {repo_relative(manifest_path)}")
        return

    expected_ids = [str(row["id"]) for row in tokenized]
    baseline_path = out_dir / "baseline.json"
    if baseline_path.exists():
        print("baseline: resume existing")
        baseline = _validate_completed_result(baseline_path, expected_ids)
    else:
        print("baseline: evaluating intact model")
        baseline = _evaluate_condition(
            model,
            tokenized,
            tokenizer,
            config.get("generation", {}),
            device=device,
            torch=torch,
        )
        baseline["condition"] = "intact"
        atomic_write_json(baseline_path, baseline)

    layer_rows: list[dict[str, Any]] = []
    thresholds = config.get("criticality", {}).get("thresholds", {})
    for layer_index in selected_layers:
        result_path = layers_dir / f"layer_{layer_index:02d}.json"
        if result_path.exists():
            print(f"layer {layer_index:02d}: resume existing")
            result = _validate_completed_result(result_path, expected_ids)
        else:
            print(f"layer {layer_index:02d}: bypass and evaluate")
            with _bypass_layer(layers, layer_index, torch):
                result = _evaluate_condition(
                    model,
                    tokenized,
                    tokenizer,
                    config.get("generation", {}),
                    device=device,
                    torch=torch,
                )
            result["condition"] = "single_layer_bypass"
            result["layer"] = layer_index
            atomic_write_json(result_path, result)
        classification = classify_layer(result["summary"], baseline["summary"], thresholds)
        per_source_classification: dict[str, Any] = {}
        for source_label, baseline_source in baseline["summary"]["per_source"].items():
            if source_label not in result["summary"]["per_source"]:
                raise ContractError(
                    f"layer {layer_index} result lacks source group {source_label!r}"
                )
            per_source_classification[source_label] = classify_layer(
                result["summary"]["per_source"][source_label],
                baseline_source,
                thresholds,
            )
        supporting_sources = sorted(
            label
            for label, source_result in per_source_classification.items()
            if source_result["catastrophic"]
        )
        minimum_sources = int(
            config.get("criticality", {}).get("minimum_supporting_sources", 1)
        )
        if minimum_sources < 1 or minimum_sources > len(per_source_classification):
            raise ContractError(
                "criticality.minimum_supporting_sources is outside the available source count"
            )
        automatic_eligible = bool(
            classification["catastrophic"] and len(supporting_sources) >= minimum_sources
        )
        layer_rows.append(
            {
                "layer": layer_index,
                **result["summary"],
                **classification,
                "overall_catastrophic": classification["catastrophic"],
                "catastrophic": automatic_eligible,
                "minimum_supporting_sources": minimum_sources,
                "supporting_sources": supporting_sources,
                "per_source_classification": per_source_classification,
                "result_path": repo_relative(result_path),
                "result_sha256": sha256_file(result_path),
            }
        )

    automatic = sorted(
        int(row["layer"]) for row in layer_rows if bool(row["catastrophic"])
    )
    manual_hypothesis = parse_layer_selection(
        config.get("criticality", {}).get("manual_hypothesis_layers", []),
        num_layers=num_layers,
    ) if config.get("criticality", {}).get("manual_hypothesis_layers") else []
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "catastrophic-removal critical-core discovery",
        "status": "MEASURED_DISCOVERY_ONLY",
        "run_identity_sha256": run_identity_sha,
        "model": model_identity,
        "num_hidden_layers": num_layers,
        "evaluated_layers": selected_layers,
        "baseline": baseline["summary"],
        "layers": sorted(layer_rows, key=lambda row: int(row["layer"])),
        "automatic_critical_layers": automatic,
        "manual_hypothesis_layers": manual_hypothesis,
        "manual_layers_are_not_automatic_evidence": True,
        "thresholds": thresholds,
        "minimum_supporting_sources": int(
            config.get("criticality", {}).get("minimum_supporting_sources", 1)
        ),
        "completed_at": _utc_now(),
    }
    _join_layer_profile(summary, config.get("scan", {}).get("layer_profile"))
    summary["summary_sha256"] = object_sha256(summary)
    atomic_write_json(out_dir / "scan_summary.json", summary)

    print("\nCRCC automatic layers:", automatic or "none")
    if manual_hypothesis:
        print("historical/manual hypothesis layers (separate):", manual_hypothesis)
    print(f"wrote {repo_relative(out_dir / 'scan_summary.json')}")


if __name__ == "__main__":
    main()
