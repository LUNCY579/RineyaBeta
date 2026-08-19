#!/usr/bin/env python3
"""Report the parameter budget of a Qwen3 checkpoint by component.

Also detects a redundant lm_head copy (present in stock Qwen3-0.6B despite
tie_word_embeddings=true) and unused embedding padding slots.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Qwen3-0.6B"

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def classify(key: str) -> str:
    if "embed_tokens" in key:
        return "embed_tokens (tied)"
    if "lm_head" in key:
        return "lm_head (redundant copy)"
    if any(t in key for t in ATTN):
        return "attention proj (all layers)"
    if any(t in key for t in MLP):
        return "MLP (all layers)"
    return "norms / misc"


def main() -> None:
    weights = MODEL / "model.safetensors"
    config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    tok_cfg = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))

    groups: dict[str, int] = defaultdict(int)
    stored = 0
    tied_identical = None

    with safe_open(str(weights), framework="pt") as f:
        keys = set(f.keys())
        for key in keys:
            n = 1
            for dim in f.get_slice(key).get_shape():
                n *= dim
            stored += n
            groups[classify(key)] += n
        if {"lm_head.weight", "model.embed_tokens.weight"} <= keys:
            tied_identical = torch.equal(
                f.get_tensor("lm_head.weight"),
                f.get_tensor("model.embed_tokens.weight"),
            )

    redundant = groups.get("lm_head (redundant copy)", 0)
    resident = stored - redundant if tied_identical else stored

    print(f"checkpoint : {MODEL}")
    print(f"layers     : {config['num_hidden_layers']}   hidden {config['hidden_size']}"
          f"   intermediate {config['intermediate_size']}")
    print(f"vocab_size : {config['vocab_size']}   tie_word_embeddings"
          f" {config['tie_word_embeddings']}\n")

    print(f"{'component':32s} {'params':>14s}  {'share of resident':>18s}")
    for name, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        share = "" if name.startswith("lm_head") and tied_identical else f"{100 * n / resident:17.1f}%"
        print(f"{name:32s} {n:>14,d}  {share:>18s}")
    print(f"{'-' * 68}")
    print(f"{'stored in file':32s} {stored:>14,d}")
    print(f"{'resident (tied)':32s} {resident:>14,d}\n")

    per_layer = (groups["attention proj (all layers)"] + groups["MLP (all layers)"]) // config["num_hidden_layers"]
    print(f"per transformer layer      : {per_layer:>14,d}")
    print(f"embedding row cost         : {config['hidden_size']:>14,d} params/token")

    if tied_identical:
        print(f"\nlm_head.weight is bitwise identical to embed_tokens.weight:"
              f" {redundant:,d} params ({redundant * 2 / 1e6:.0f} MB bf16) of dead disk.")

    base = len(tok_cfg["model"]["vocab"]) + len(tok_cfg.get("added_tokens", []))
    pad = config["vocab_size"] - base
    print(f"real tokens {base:,d}; vocab_size {config['vocab_size']:,d}"
          f" -> {pad:,d} never-addressable padding rows"
          f" ({pad * config['hidden_size']:,d} params).")


if __name__ == "__main__":
    main()
