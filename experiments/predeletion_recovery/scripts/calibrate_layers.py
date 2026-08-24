#!/usr/bin/env python3
"""Intervention-based layer calibration for low/medium/high deletion targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))

from pdrr.data import collate, load_chat_jsonl, sha256_file  # noqa: E402
from pdrr.hf import install_layer_gate, resolve_decoder_layers  # noqa: E402
from pdrr.losses import token_kl  # noqa: E402
from pdrr.provenance import atomic_write_json, environment_fingerprint, seed_everything  # noqa: E402
from pdrr.surgery import GatedModule  # noqa: E402


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidates", default=None,
                        help="comma-separated zero-based layers; default excludes first/last")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 20260823))
    seed_everything(seed)
    device_name = args.device or str(config.get("device", "auto"))
    device = torch.device("cuda:0" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_path = resolve(str(config["model"]["path"]))
    data_path = resolve(str(config["data"]["train"]))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype,
        attn_implementation=str(config["model"].get("attn_implementation", "eager")))
    model.config.use_cache = False
    model.to(device).eval()
    examples = load_chat_jsonl(
        data_path, tokenizer, max_length=int(config["data"].get("max_length", 384)),
        max_rows=args.limit)
    batches = []
    for start in range(0, len(examples), args.batch_size):
        payload = collate(examples[start:start + args.batch_size], tokenizer.pad_token_id)
        batches.append({key: value.to(device) for key, value in payload.items()
                        if torch.is_tensor(value)})

    _, layers = resolve_decoder_layers(model)
    layer_count = len(layers)
    if args.candidates:
        candidates = [int(value) for value in args.candidates.split(",") if value.strip()]
    else:
        candidates = list(range(1, layer_count - 1))
    bad = [index for index in candidates if not 0 <= index < layer_count]
    if bad:
        raise ValueError(f"candidate layers outside model: {bad}")

    full_outputs: list[tuple[torch.Tensor, float]] = []
    with torch.inference_mode():
        for batch in batches:
            output = model(**batch)
            full_outputs.append((output.logits.detach().cpu(), float(output.loss)))

    rows: list[dict[str, Any]] = []
    for index in candidates:
        original = layers[index]
        gate = install_layer_gate(model, index, 0.0)
        kl_values: list[float] = []
        loss_changes: list[float] = []
        with torch.inference_mode():
            for batch, (full_logits, full_loss) in zip(batches, full_outputs):
                output = model(**batch)
                mask = batch["attention_mask"].float().cpu()
                kl = token_kl(full_logits, output.logits.detach().cpu())
                kl_values.append(float((kl * mask).sum() / mask.sum().clamp_min(1.0)))
                loss_changes.append(float(output.loss) - full_loss)
        layers[index] = gate.module if isinstance(gate, GatedModule) else original
        row = {
            "layer": index,
            "mean_logit_kl": float(np.mean(kl_values)),
            "median_logit_kl": float(np.median(kl_values)),
            "mean_lm_loss_change": float(np.mean(loss_changes)),
            "batches": len(batches),
            "examples": len(examples),
        }
        rows.append(row)
        print(f"layer {index:>2}: KL={row['mean_logit_kl']:.6g} "
              f"loss_delta={row['mean_lm_loss_change']:+.6g}")

    ordered = sorted(rows, key=lambda row: row["mean_logit_kl"])
    def pick(quantile: float) -> int:
        position = round((len(ordered) - 1) * quantile)
        return int(ordered[position]["layer"])
    selection = {"low": pick(0.20), "medium": pick(0.50), "high": pick(0.80)}
    result = {
        "schema_version": 1,
        "model": str(model_path),
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "device": str(device),
        "layer_count": layer_count,
        "candidates": candidates,
        "rows": rows,
        "selection": selection,
        "selection_rule": "20th, 50th, and 80th percentile ranks by mean intact-vs-ablated final-logit KL",
        "environment": environment_fingerprint(),
    }
    out = resolve(args.out)
    atomic_write_json(out, result)
    print(f"selected targets: {selection}; wrote {out}")


if __name__ == "__main__":
    main()
