#!/usr/bin/env python3
"""Test whether learned route mass predicts migrated causal recipient importance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))

from pdrr.data import collate, load_chat_jsonl  # noqa: E402
from pdrr.hf import install_layer_gate, resolve_decoder_layers  # noqa: E402
from pdrr.losses import token_kl  # noqa: E402
from pdrr.provenance import atomic_write_json, environment_fingerprint  # noqa: E402
from pdrr.surgery import GatedModule  # noqa: E402


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def mean_recipient_importance(model, batches, recipients):
    _, layers = resolve_decoder_layers(model)
    base = []
    with torch.inference_mode():
        for batch in batches:
            base.append(model(**batch).logits.detach())
    values = []
    for recipient in recipients:
        original = layers[recipient]
        gate = install_layer_gate(model, recipient, 0.0)
        scores = []
        with torch.inference_mode():
            for batch, reference in zip(batches, base):
                ablated = model(**batch).logits
                mask = batch["attention_mask"].float()
                kl = token_kl(reference, ablated)
                scores.append(float((kl * mask).sum() / mask.sum().clamp_min(1.0)))
        layers[recipient] = gate.module if isinstance(gate, GatedModule) else original
        values.append(float(np.mean(scores)))
    return values


def permutation_spearman(route, gain, seed=20260823, draws=20000):
    from scipy.stats import spearmanr
    observed = float(spearmanr(route, gain).statistic)
    if not np.isfinite(observed):
        return observed, 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    gain = np.asarray(gain)
    for _ in range(draws):
        permuted = rng.permutation(gain)
        value = float(spearmanr(route, permuted).statistic)
        if np.isfinite(value) and abs(value) >= abs(observed) - 1e-15:
            extreme += 1
    return observed, (extreme + 1) / (draws + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovered-model", required=True)
    parser.add_argument("--hard-model", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--route-tail", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    recovered_path = resolve(args.recovered_model)
    hard_path = resolve(args.hard_model)
    manifest = json.loads(resolve(args.run_manifest).read_text(encoding="utf-8"))
    original_recipients = [int(value) for value in manifest["recipients"]]
    target = int(manifest["target_layer"])
    compact_recipients = [index - 1 if index > target else index
                          for index in original_recipients]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(recovered_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    recovered = AutoModelForCausalLM.from_pretrained(recovered_path, dtype=dtype).to(device).eval()
    hard = AutoModelForCausalLM.from_pretrained(hard_path, dtype=dtype).to(device).eval()
    recovered.config.use_cache = False
    hard.config.use_cache = False

    examples = load_chat_jsonl(resolve(args.data), tokenizer, max_length=384,
                               max_rows=args.limit)
    batches = []
    for start in range(0, len(examples), args.batch_size):
        payload = collate(examples[start:start + args.batch_size], tokenizer.pad_token_id)
        batches.append({key: value.to(device) for key, value in payload.items()
                        if torch.is_tensor(value) and key != "labels"})
    hard_importance = mean_recipient_importance(hard, batches, compact_recipients)
    recovered_importance = mean_recipient_importance(recovered, batches, compact_recipients)
    gain = [after - before for after, before in zip(recovered_importance, hard_importance)]

    log_rows = [json.loads(line) for line in resolve(args.train_log).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    route_rows = [row for row in log_rows if any(key.startswith("route_mean_layer_") for key in row)]
    route_rows = route_rows[-args.route_tail:]
    if not route_rows:
        raise RuntimeError("training log contains no per-recipient route masses")
    route = [float(np.mean([row[f"route_mean_layer_{layer}"] for row in route_rows]))
             for layer in original_recipients]
    correlation, permutation_p = permutation_spearman(route, gain)
    top_match = int(np.argmax(route)) == int(np.argmax(gain))

    result = {
        "schema_version": 1,
        "original_recipient_layers": original_recipients,
        "compact_recipient_layers": compact_recipients,
        "mean_route_mass": route,
        "hard_pruned_recipient_importance": hard_importance,
        "recovered_recipient_importance": recovered_importance,
        "causal_importance_gain": gain,
        "route_gain_spearman": correlation,
        "permutation_p": permutation_p,
        "top_route_top_gain_match": top_match,
        "mechanistic_gate_pass": bool(correlation > 0 and permutation_p < 0.05 and top_match),
        "examples": len(examples),
        "environment": environment_fingerprint(),
        "interpretation": (
            "A pass supports selective migration of causal necessity; a quality gain without "
            "this pass supports only behavioral preservation, not job-responsibility transfer."
        ),
    }
    out = resolve(args.out)
    atomic_write_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
