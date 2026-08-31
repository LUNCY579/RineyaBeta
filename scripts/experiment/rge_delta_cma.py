#!/usr/bin/env python3
"""Construct exact-rank and rank-matched Delta-CMA adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from cma import (  # noqa: E402
    CMA_IMPLEMENTATION_VERSION,
    concatenate_weighted_deltas,
    rank_compress,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_adapter(out: Path, state: dict[str, torch.Tensor], *, base: str,
                 rank: int, alpha: int, targets: list[str]) -> None:
    from peft import LoraConfig
    out.mkdir(parents=True, exist_ok=False)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                        task_type="CAUSAL_LM", target_modules=targets,
                        base_model_name_or_path=base)
    config.save_pretrained(str(out))
    save_file({key: value.contiguous() for key, value in state.items()},
              str(out / "adapter_model.safetensors"), metadata={"format": "pt"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="artifacts/rge_peft/m4_c_ct_constant_tail_r16")
    parser.add_argument("--steps", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--exact-out", default="artifacts/adapters/rge_m5_cma_exact_r96")
    parser.add_argument("--rank-out", default="artifacts/adapters/rge_m5_cma_rank16")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m5_cma_construction.json")
    args = parser.parse_args()
    run = ROOT / args.run
    paths = [run / "checkpoints" / f"step_{step:09d}.pt" for step in args.steps]
    if any(not path.is_file() for path in paths):
        raise SystemExit("one or more frozen CMA checkpoints are missing")
    exact_out, rank_out = ROOT / args.exact_out, ROOT / args.rank_out
    if exact_out.exists() or rank_out.exists():
        raise SystemExit("CMA output already exists; refusing overwrite")
    payloads = [torch.load(path, map_location="cpu", weights_only=False)
                for path in paths]
    configs = [payload["config"] for payload in payloads]
    lora = configs[0]["lora"]
    if any(config["lora"] != lora for config in configs[1:]):
        raise SystemExit("CMA LoRA configs differ")
    source_states = [payload["adapter_state"] for payload in payloads]
    if any(set(state) != set(source_states[0]) for state in source_states[1:]):
        raise SystemExit("CMA adapter key sets differ")
    rank = int(lora["r"])
    source_scale = float(lora["alpha"]) / rank
    weights = [1.0 / len(payloads)] * len(payloads)
    exact_rank = rank * len(payloads)
    exact_state, compressed_state = {}, {}
    module_diagnostics = {}
    a_keys = sorted(key for key in source_states[0] if key.endswith(".lora_A.weight"))
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        factors = [(state[b_key], state[a_key], source_scale, weight)
                   for state, weight in zip(source_states, weights)]
        bc, ac = concatenate_weighted_deltas(factors)
        exact_state[b_key], exact_state[a_key] = bc, ac
        new_b, new_a, diagnostics = rank_compress(
            bc, ac, rank=rank, output_scale=source_scale)
        compressed_state[b_key], compressed_state[a_key] = new_b, new_a
        module_diagnostics[a_key.removesuffix(".lora_A.weight")] = diagnostics
    if len(exact_state) != len(source_states[0]):
        raise SystemExit("CMA output tensor count differs from source")

    base = str(configs[0]["model"]["path"])
    targets = list(lora["target_modules"])
    save_adapter(exact_out, exact_state, base=base, rank=exact_rank,
                 alpha=exact_rank, targets=targets)
    save_adapter(rank_out, compressed_state, base=base, rank=rank,
                 alpha=int(lora["alpha"]), targets=targets)
    total_energy = sum(item["total_frobenius_energy"]
                       for item in module_diagnostics.values())
    retained_energy = sum(item["retained_frobenius_energy"]
                          for item in module_diagnostics.values())
    compact = {
        "schema_version": 1,
        "status": "PASS",
        "implementation": CMA_IMPLEMENTATION_VERSION,
        "source_run": args.run,
        "source_steps": args.steps,
        "source_checkpoints": [{
            "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
            for path in paths],
        "weights": weights,
        "source_rank": rank,
        "source_scale": source_scale,
        "exact_rank": exact_rank,
        "rank_matched_rank": rank,
        "module_count": len(module_diagnostics),
        "global_retained_energy_fraction": retained_energy / total_energy,
        "global_relative_reconstruction_error": (
            max(total_energy - retained_energy, 0.0) / total_energy) ** 0.5,
        "minimum_module_retained_energy_fraction": min(
            item["retained_energy_fraction"] for item in module_diagnostics.values()),
        "maximum_module_relative_reconstruction_error": max(
            item["relative_reconstruction_error"] for item in module_diagnostics.values()),
        "exact_adapter": args.exact_out,
        "rank_matched_adapter": args.rank_out,
        "exact_adapter_model_sha256": sha256_file(exact_out / "adapter_model.safetensors"),
        "rank_adapter_model_sha256": sha256_file(rank_out / "adapter_model.safetensors"),
        "factor_averaging_used": False,
        "sealed_data_accessed": False,
    }
    detailed = {**compact, "modules": module_diagnostics}
    (exact_out / "cma_manifest.json").write_text(
        json.dumps(detailed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (rank_out / "cma_manifest.json").write_text(
        json.dumps(detailed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = ROOT / args.evidence
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
