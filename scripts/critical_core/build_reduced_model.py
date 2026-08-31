#!/usr/bin/env python3
"""Materialize one reduced-depth arm while preserving its source-layer map.

Weights are copied from the declared source checkpoint without training.  Decoder
blocks keep source order, attention cache indices are renumbered for the reduced
runtime, and the original indices are stored in both config and layer_map.json so
a recovered donor can later be differenced against the correct full-depth layers.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (  # noqa: E402
    ContractError,
    SCHEMA_VERSION,
    atomic_write_json,
    checkpoint_inventory,
    mapping_source_layers,
    object_sha256,
    read_json,
    repo_relative,
    resolve_repo_path,
    sha256_file,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _layer_parent(model):
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    )
    for names in candidates:
        current = model
        for name in names[:-1]:
            if not hasattr(current, name):
                break
            current = getattr(current, name)
        else:
            attribute = names[-1]
            if hasattr(current, attribute):
                return current, attribute, getattr(current, attribute)
    raise ContractError("could not locate decoder layer container")


def _renumber_runtime_layer(layer, new_index: int) -> None:
    for module in layer.modules():
        if hasattr(module, "layer_idx"):
            try:
                module.layer_idx = new_index
            except (AttributeError, TypeError):
                pass


def _safe_overwrite(output: Path) -> None:
    allowed_root = (ROOT / "runs" / "critical_core").resolve()
    try:
        output.resolve().relative_to(allowed_root)
    except ValueError as exc:
        raise ContractError(
            f"--overwrite is restricted to {allowed_root}; refusing to remove {output}"
        ) from exc
    shutil.rmtree(output)


def _smoke_forward(model, tokenizer, torch) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with OK."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**encoded, use_cache=False)
    finite = bool(torch.isfinite(output.logits).all().item())
    result = {
        "device": str(device),
        "prompt_tokens": int(encoded.input_ids.shape[1]),
        "logits_shape": list(output.logits.shape),
        "finite_logits": finite,
    }
    if not finite:
        raise ContractError("reduced-model smoke forward produced non-finite logits")
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-manifest", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True, help="full-depth source checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto"
    )
    parser.add_argument("--skip-source-weight-hash", action="store_true")
    parser.add_argument("--skip-output-weight-hash", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    arm_manifest_path = resolve_repo_path(args.arm_manifest, must_exist=True)
    arm_manifest = read_json(arm_manifest_path)
    if arm_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("arm manifest schema is not CRCC v1")
    matches = [arm for arm in arm_manifest["arms"] if arm["name"] == args.arm]
    if len(matches) != 1:
        raise ContractError(f"arm {args.arm!r} is absent or duplicated")
    arm = matches[0]
    source_layers = mapping_source_layers(arm["layer_mapping"])

    source = resolve_repo_path(args.model, must_exist=True)
    expected_config_sha = arm_manifest.get("source_model", {}).get("config_sha256")
    actual_config_sha = sha256_file(source / "config.json")
    if expected_config_sha and expected_config_sha != actual_config_sha:
        raise ContractError(
            "source checkpoint config differs from the checkpoint used by discovery: "
            f"expected {expected_config_sha}, got {actual_config_sha}"
        )
    output = resolve_repo_path(args.out)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {output}")
        _safe_overwrite(output)
    output.mkdir(parents=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = None
    if args.dtype != "auto":
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(source)
    load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    parent, attribute, original_layers = _layer_parent(model)
    original_depth = len(original_layers)
    if original_depth != int(arm_manifest["source_depth"]):
        raise ContractError(
            f"source depth {original_depth} != arm manifest {arm_manifest['source_depth']}"
        )
    if source_layers[-1] >= original_depth:
        raise ContractError("arm mapping addresses a layer outside the source model")

    original_params = sum(parameter.numel() for parameter in model.parameters())
    kept_modules = [original_layers[index] for index in source_layers]
    reduced_layers = torch.nn.ModuleList(kept_modules)
    setattr(parent, attribute, reduced_layers)
    for new_index, layer in enumerate(reduced_layers):
        _renumber_runtime_layer(layer, new_index)

    model.config.num_hidden_layers = len(source_layers)
    if hasattr(model.config, "max_window_layers"):
        model.config.max_window_layers = min(
            int(model.config.max_window_layers), len(source_layers)
        )
    if hasattr(model.config, "layer_types"):
        layer_types = list(model.config.layer_types)
        if len(layer_types) == original_depth:
            model.config.layer_types = [layer_types[index] for index in source_layers]
    model.config.crcc_schema_version = SCHEMA_VERSION
    model.config.crcc_arm_name = args.arm
    model.config.crcc_source_depth = original_depth
    model.config.crcc_source_layer_indices = source_layers
    model.config.crcc_arm_design_sha256 = arm_manifest["design_sha256"]

    reduced_params = sum(parameter.numel() for parameter in model.parameters())
    smoke = _smoke_forward(model, tokenizer, torch) if args.smoke else None
    model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output)

    layer_map = {
        "schema_version": SCHEMA_VERSION,
        "arm": args.arm,
        "source_depth": original_depth,
        "reduced_depth": len(source_layers),
        "source_layers": source_layers,
        "mapping": arm["layer_mapping"],
        "arm_design_sha256": arm_manifest["design_sha256"],
    }
    layer_map["mapping_sha256"] = object_sha256(layer_map)
    atomic_write_json(output / "layer_map.json", layer_map)

    source_identity = checkpoint_inventory(
        source, hash_weights=not args.skip_source_weight_hash
    )
    output_identity = checkpoint_inventory(
        output, hash_weights=not args.skip_output_weight_hash
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "BUILT_UNTRAINED_REDUCED_MODEL",
        "created_at": _utc_now(),
        "arm": arm,
        "arm_manifest": {
            "path": repo_relative(arm_manifest_path),
            "sha256": sha256_file(arm_manifest_path),
            "design_sha256": arm_manifest["design_sha256"],
        },
        "source_checkpoint": source_identity,
        "output_checkpoint": output_identity,
        "original_parameter_count": int(original_params),
        "reduced_parameter_count": int(reduced_params),
        "removed_parameter_count": int(original_params - reduced_params),
        "parameter_reduction_fraction": 1.0 - reduced_params / original_params,
        "runtime_layer_indices_renumbered": True,
        "source_mapping_preserved": True,
        "smoke": smoke,
    }
    manifest["build_sha256"] = object_sha256(manifest)
    atomic_write_json(output / "build_manifest.json", manifest)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"arm        : {args.arm}")
    print(f"source     : {repo_relative(source)} ({original_depth} layers)")
    print(f"kept       : {source_layers}")
    print(f"parameters : {original_params:,} -> {reduced_params:,}")
    print(f"wrote      : {repo_relative(output)}")


if __name__ == "__main__":
    main()
