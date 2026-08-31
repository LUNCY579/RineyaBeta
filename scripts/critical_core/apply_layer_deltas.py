#!/usr/bin/env python3
"""Apply recovered reduced-model layer deltas to a compatible target scaffold.

For mapped source layer ``i`` and reduced donor layer ``j`` this implements

    target[i] <- target[i] + alpha * (recovered[j] - untrained_reduced[j])

Only decoder-layer tensors are touched. Embeddings, final norm, LM head, and all
unmapped target layers remain byte-identical in memory until serialization.  The
script supports multiple donors in a deterministic declared order and emits a
complete operation/norm manifest.
"""

from __future__ import annotations

import argparse
import gc
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
            return current
    raise ContractError("could not locate decoder layers")


def _safe_output(output: Path, *, overwrite: bool) -> None:
    if not output.exists():
        output.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    allowed = (ROOT / "runs" / "critical_core").resolve()
    try:
        output.resolve().relative_to(allowed)
    except ValueError as exc:
        raise ContractError(f"refusing to overwrite outside {allowed}: {output}") from exc
    shutil.rmtree(output)
    output.mkdir(parents=True)


def _load_causal_model(path: Path, torch, *, dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )


def _load_recovered(donor: Mapping[str, Any], torch, *, dtype):
    base_path = resolve_repo_path(donor["base_model"], must_exist=True)
    recovered_path = donor.get("recovered_model")
    adapter_path = donor.get("adapter")
    if bool(recovered_path) == bool(adapter_path):
        raise ContractError(
            f"donor {donor['id']} must declare exactly one of recovered_model or adapter"
        )
    base = _load_causal_model(base_path, torch, dtype=dtype)
    if adapter_path:
        from peft import PeftModel

        adapter = resolve_repo_path(adapter_path, must_exist=True)
        recovered_base = _load_causal_model(base_path, torch, dtype=dtype)
        recovered = PeftModel.from_pretrained(recovered_base, adapter)
        recovered = recovered.merge_and_unload(safe_merge=True)
        recovery_identity = {
            "kind": "peft_adapter",
            "path": repo_relative(adapter),
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_weights_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        }
    else:
        recovered_dir = resolve_repo_path(str(recovered_path), must_exist=True)
        recovered = _load_causal_model(recovered_dir, torch, dtype=dtype)
        recovery_identity = {
            "kind": "merged_checkpoint",
            "checkpoint": checkpoint_inventory(recovered_dir, hash_weights=True),
        }
    return base_path, base, recovered, recovery_identity


def _validate_architecture(target_config, donor_config, donor_id: str) -> None:
    fields = (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "rms_norm_eps",
        "attention_bias",
        "rope_theta",
    )
    mismatches = {
        field: (getattr(target_config, field, None), getattr(donor_config, field, None))
        for field in fields
        if getattr(target_config, field, None) != getattr(donor_config, field, None)
    }
    if mismatches:
        raise ContractError(f"target and donor {donor_id} architectures differ: {mismatches}")



def merge_tensor_in_place(
    target_tensor, base_tensor, recovered_tensor, coefficient: float, torch
) -> dict[str, float | int]:
    """Apply one floating tensor delta and return reproducible norm statistics."""

    if base_tensor.shape != recovered_tensor.shape or base_tensor.shape != target_tensor.shape:
        raise ContractError(
            f"tensor shape mismatch: {tuple(base_tensor.shape)}, "
            f"{tuple(recovered_tensor.shape)}, {tuple(target_tensor.shape)}"
        )
    if not (base_tensor.is_floating_point() and recovered_tensor.is_floating_point()):
        raise ContractError("merge_tensor_in_place requires floating tensors")
    delta = recovered_tensor.float() - base_tensor.float()
    if not torch.isfinite(delta).all():
        raise ContractError("non-finite donor delta")
    updated = target_tensor.float() + float(coefficient) * delta
    if not torch.isfinite(updated).all():
        raise ContractError("non-finite merged tensor")
    target_tensor.copy_(updated.to(dtype=target_tensor.dtype))
    delta_sq = float(torch.sum(delta * delta).item())
    return {
        "elements": int(delta.numel()),
        "delta_sq": delta_sq,
        "scaled_delta_sq": delta_sq * float(coefficient) * float(coefficient),
        "delta_max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
    }

def _apply_donor(target, donor: Mapping[str, Any], torch, *, dtype) -> dict[str, Any]:
    coefficient = float(donor["coefficient"])
    if not math.isfinite(coefficient):
        raise ContractError(f"donor {donor['id']} coefficient is non-finite")

    declared_base_path = resolve_repo_path(donor["base_model"], must_exist=True)
    build_manifest_path = declared_base_path / "build_manifest.json"
    if not build_manifest_path.is_file():
        raise ContractError(
            f"donor {donor['id']} lacks build_manifest.json; provenance is incomplete"
        )
    build_manifest = read_json(build_manifest_path)
    if build_manifest.get("status") != "BUILT_UNTRAINED_REDUCED_MODEL":
        raise ContractError(f"donor {donor['id']} has an unexpected build status")
    if not bool(build_manifest.get("output_checkpoint", {}).get("weights_hashed", False)):
        raise ContractError(
            f"donor {donor['id']} base weights were not hashed at construction"
        )

    map_path = resolve_repo_path(donor["layer_map"], must_exist=True)
    mapping_payload = read_json(map_path)
    if mapping_payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"donor {donor['id']} layer map is not CRCC v1")
    source_layers = mapping_source_layers(mapping_payload["mapping"])

    base_path, base, recovered, recovery_identity = _load_recovered(
        donor, torch, dtype=dtype
    )
    base_layers = _decoder_layers(base)
    recovered_layers = _decoder_layers(recovered)
    target_layers = _decoder_layers(target)
    if len(base_layers) != len(source_layers) or len(recovered_layers) != len(source_layers):
        raise ContractError(
            f"donor {donor['id']} depth does not match its layer map: "
            f"base={len(base_layers)} recovered={len(recovered_layers)} map={len(source_layers)}"
        )
    if source_layers[-1] >= len(target_layers):
        raise ContractError(
            f"target has {len(target_layers)} layers but donor maps to {source_layers[-1]}"
        )
    configured = getattr(base.config, "crcc_source_layer_indices", None)
    if configured is not None and [int(value) for value in configured] != source_layers:
        raise ContractError(f"donor {donor['id']} config and layer_map.json disagree")
    _validate_architecture(target.config, base.config, str(donor["id"]))

    tensor_operations: list[dict[str, Any]] = []
    donor_delta_norm_sq = 0.0
    scaled_delta_norm_sq = 0.0
    changed_elements = 0
    with torch.no_grad():
        for reduced_index, target_index in enumerate(source_layers):
            base_state = base_layers[reduced_index].state_dict()
            recovered_state = recovered_layers[reduced_index].state_dict()
            target_state = target_layers[target_index].state_dict()
            if set(base_state) != set(recovered_state) or set(base_state) != set(target_state):
                missing = {
                    "recovered_missing": sorted(set(base_state) - set(recovered_state)),
                    "target_missing": sorted(set(base_state) - set(target_state)),
                    "base_missing_from_target": sorted(set(target_state) - set(base_state)),
                }
                raise ContractError(
                    f"state keys differ for donor {donor['id']} layer {reduced_index}: {missing}"
                )

            layer_delta_sq = 0.0
            layer_scaled_sq = 0.0
            layer_elements = 0
            layer_max_abs = 0.0
            for key in sorted(base_state):
                base_tensor = base_state[key]
                recovered_tensor = recovered_state[key]
                target_tensor = target_state[key]
                if not (base_tensor.is_floating_point() and recovered_tensor.is_floating_point()):
                    if (
                        base_tensor.shape != recovered_tensor.shape
                        or base_tensor.shape != target_tensor.shape
                    ):
                        raise ContractError(
                            f"shape mismatch at donor {donor['id']} layer {reduced_index} {key}"
                        )
                    if not torch.equal(base_tensor, recovered_tensor):
                        raise ContractError(
                            "non-floating donor state changed at "
                            f"{donor['id']}:{reduced_index}:{key}"
                        )
                    continue
                try:
                    stats = merge_tensor_in_place(
                        target_tensor, base_tensor, recovered_tensor, coefficient, torch
                    )
                except ContractError as exc:
                    raise ContractError(
                        f"{donor['id']}:{reduced_index}->{target_index}:{key}: {exc}"
                    ) from exc
                layer_delta_sq += float(stats["delta_sq"])
                layer_scaled_sq += float(stats["scaled_delta_sq"])
                layer_elements += int(stats["elements"])
                layer_max_abs = max(layer_max_abs, float(stats["delta_max_abs"]))

            donor_delta_norm_sq += layer_delta_sq
            scaled_delta_norm_sq += layer_scaled_sq
            changed_elements += layer_elements
            tensor_operations.append(
                {
                    "reduced_layer": reduced_index,
                    "target_layer": target_index,
                    "floating_elements": layer_elements,
                    "delta_l2": math.sqrt(layer_delta_sq),
                    "scaled_delta_l2": math.sqrt(layer_scaled_sq),
                    "delta_max_abs": layer_max_abs,
                }
            )

    result = {
        "id": donor["id"],
        "arm": donor.get("arm"),
        "coefficient": coefficient,
        "base_model": checkpoint_inventory(base_path, hash_weights=False),
        "base_build_manifest": {
            "path": repo_relative(build_manifest_path),
            "sha256": sha256_file(build_manifest_path),
            "build_sha256": build_manifest.get("build_sha256"),
            "hashed_output_inventory": build_manifest.get("output_checkpoint"),
        },
        "recovery": recovery_identity,
        "layer_map": {
            "path": repo_relative(map_path),
            "sha256": sha256_file(map_path),
            "mapping_sha256": mapping_payload.get("mapping_sha256"),
        },
        "source_layers": source_layers,
        "changed_layer_count": len(source_layers),
        "changed_parameter_elements": changed_elements,
        "delta_l2": math.sqrt(donor_delta_norm_sq),
        "scaled_delta_l2": math.sqrt(scaled_delta_norm_sq),
        "layers": tensor_operations,
    }
    del base, recovered
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", default=None, help="override plan.output_model")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-target-weight-hash", action="store_true")
    parser.add_argument("--skip-output-weight-hash", action="store_true")
    args = parser.parse_args()

    plan_path = resolve_repo_path(args.plan, must_exist=True)
    plan = read_json(plan_path)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("merge plan schema is not CRCC v1")
    expected_plan_sha = plan.get("plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256", None)
    if expected_plan_sha != object_sha256(unhashed):
        raise ContractError("merge plan hash does not reproduce")

    target_path = resolve_repo_path(plan["target_model"], must_exist=True)
    output = resolve_repo_path(args.out or plan["output_model"])
    _safe_output(output, overwrite=args.overwrite)
    donors = list(plan.get("donors", []))
    if not donors:
        raise ContractError("merge plan has no donors")
    maximum = float(plan.get("safety", {}).get("max_abs_coefficient", 1.5))
    for donor in donors:
        if abs(float(donor["coefficient"])) > maximum:
            raise ContractError(
                f"donor {donor['id']} coefficient {donor['coefficient']} exceeds {maximum}"
            )

    import torch
    from transformers import AutoTokenizer

    dtype_name = str(plan.get("dtype", "bfloat16"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ContractError(f"unsupported merge dtype {dtype_name!r}")

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    target = _load_causal_model(target_path, torch, dtype=dtype)
    target.eval()
    target_depth = len(_decoder_layers(target))
    donor_results: list[dict[str, Any]] = []
    for donor in donors:
        print(
            f"apply {donor['id']} coefficient={float(donor['coefficient']):g} "
            f"to {plan['candidate_id']}"
        )
        donor_results.append(_apply_donor(target, donor, torch, dtype=dtype))

    target.config.crcc_merge_candidate = plan["candidate_id"]
    target.config.crcc_merge_plan_sha256 = expected_plan_sha
    target.config.crcc_donors = [
        {
            "id": result["id"],
            "coefficient": result["coefficient"],
            "source_layers": result["source_layers"],
        }
        for result in donor_results
    ]
    target.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "MERGED_NOT_EVALUATED",
        "created_at": _utc_now(),
        "candidate_id": plan["candidate_id"],
        "plan": {
            "path": repo_relative(plan_path),
            "sha256": sha256_file(plan_path),
            "plan_sha256": expected_plan_sha,
        },
        "target_before_merge": checkpoint_inventory(
            target_path, hash_weights=not args.skip_target_weight_hash
        ),
        "target_depth": target_depth,
        "donors_applied_in_order": donor_results,
        "untouched_scope": [
            "embeddings",
            "final_norm",
            "lm_head",
            "all decoder layers outside each donor mapping",
        ],
        "output_checkpoint": checkpoint_inventory(
            output, hash_weights=not args.skip_output_weight_hash
        ),
    }
    manifest["merge_sha256"] = object_sha256(manifest)
    atomic_write_json(output / "crcc_merge_manifest.json", manifest)
    print(f"wrote {repo_relative(output)}")


if __name__ == "__main__":
    main()
