#!/usr/bin/env python3
"""Scan candidate MLP layers for SGQR deletion damage and low-bit recoverability."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import memory as mem  # noqa: E402
from scripts.sgqr.install import (  # noqa: E402
    install_sgqr_mlp,
    resolve_decoder_layers,
    restore_original_mlp,
)
from scripts.sgqr.utility import marginal_recovery_utility, teacher_kl  # noqa: E402

DATASETS = {
    "chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
    "rag_dev": "artifacts/eval_sets/rag_dev.jsonl",
}


def last_logits(model, inputs):
    kwargs = {key: value for key, value in inputs.items()}
    kwargs["use_cache"] = False
    try:
        outputs = model(**kwargs, logits_to_keep=1)
    except TypeError:
        outputs = model(**kwargs)
    return outputs.logits[:, -1, :]


def normalize_messages(record: dict) -> list[dict]:
    messages = [dict(message) for message in record.get("messages") or []]
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def parse_layers(text: str, layer_count: int) -> list[int]:
    if text == "all":
        return list(range(layer_count))
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or any(value < 0 or value >= layer_count for value in values):
        raise ValueError(f"invalid layer list for model with {layer_count} layers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/rineya-core-en32k-28L")
    parser.add_argument("--dataset", default="chat_dev")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--cap-mib", type=float, default=5120.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = ROOT / args.model
    dataset_path = ROOT / DATASETS.get(args.dataset, args.dataset)
    records = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    mem.apply_cap(args.cap_mib if device.type == "cuda" else None)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
    model.config.use_cache = False

    layers = resolve_decoder_layers(model)
    candidate_layers = parse_layers(args.layers, len(layers))
    tokenized = []
    for record in records:
        rendered = tokenizer.apply_chat_template(
            normalize_messages(record),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenized.append(tokenizer(rendered, return_tensors="pt"))

    teacher_logits = []
    with torch.inference_mode():
        for inputs in tokenized:
            moved = {key: value.to(device) for key, value in inputs.items()}
            teacher_logits.append(last_logits(model, moved).detach().float().cpu())

    results = []
    started = time.time()
    for layer_idx in candidate_layers:
        row = {"layer_idx": layer_idx}
        original_mlp = layers[layer_idx].mlp

        deleted_install = install_sgqr_mlp(
            model,
            layer_idx=layer_idx,
            mode="deleted",
        )
        deleted_logits = []
        with torch.inference_mode():
            for inputs in tokenized:
                moved = {key: value.to(device) for key, value in inputs.items()}
                deleted_logits.append(last_logits(model, moved).detach().float().cpu())
        row["deleted_teacher_kl"] = float(
            np.mean(
                [
                    float(teacher_kl(teacher, deleted, reduction="mean"))
                    for teacher, deleted in zip(teacher_logits, deleted_logits)
                ]
            )
        )
        restore_original_mlp(model, deleted_install)

        for bits in (8, 4):
            installation = install_sgqr_mlp(
                model,
                layer_idx=layer_idx,
                mode="quantized_all",
                bits=bits,
                group_size=args.group_size,
                retain_dense_reference=False,
                random_seed=args.seed,
            )
            quantized_logits = []
            with torch.inference_mode():
                for inputs in tokenized:
                    moved = {key: value.to(device) for key, value in inputs.items()}
                    quantized_logits.append(last_logits(model, moved).detach().float().cpu())
            quantized_kl = [
                float(teacher_kl(teacher, candidate, reduction="mean"))
                for teacher, candidate in zip(teacher_logits, quantized_logits)
            ]
            utility = [
                float(
                    marginal_recovery_utility(teacher, deleted, candidate).mean()
                )
                for teacher, deleted, candidate in zip(
                    teacher_logits, deleted_logits, quantized_logits
                )
            ]
            prefix = f"int{bits}"
            row[f"{prefix}_teacher_kl"] = float(np.mean(quantized_kl))
            row[f"{prefix}_mean_utility"] = float(np.mean(utility))
            row[f"{prefix}_positive_utility_fraction"] = float(
                np.mean(np.asarray(utility) > 0)
            )
            row[f"{prefix}_packed_storage_bytes"] = int(
                installation.storage_manifest()["packed_storage_bytes"]
            )
            restore_original_mlp(model, installation)
            del installation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row["int4_recovered_kl_fraction"] = (
            1.0 - row["int4_teacher_kl"] / row["deleted_teacher_kl"]
            if row["deleted_teacher_kl"] > 0
            else None
        )
        results.append(row)
        print(
            f"layer {layer_idx:>2}: deleted KL {row['deleted_teacher_kl']:.6f}, "
            f"INT4 KL {row['int4_teacher_kl']:.6f}, "
            f"recovered {row['int4_recovered_kl_fraction']}"
        )

        if layers[layer_idx].mlp is not original_mlp:
            raise RuntimeError(f"failed to restore original MLP at layer {layer_idx}")

    ranked = sorted(
        results,
        key=lambda item: float(item["deleted_teacher_kl"]),
    )
    summary = {
        "format": "sgqr-layer-scan-v1",
        "model": args.model,
        "dataset": str(dataset_path.relative_to(ROOT)).replace("\\", "/"),
        "record_count": len(records),
        "group_size": args.group_size,
        "layers": results,
        "low_sensitivity_candidate": ranked[0]["layer_idx"] if ranked else None,
        "moderate_sensitivity_candidate": ranked[len(ranked) // 2]["layer_idx"] if ranked else None,
        "high_sensitivity_candidate": ranked[-1]["layer_idx"] if ranked else None,
        "elapsed_s": round(time.time() - started, 3),
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        },
    }
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
