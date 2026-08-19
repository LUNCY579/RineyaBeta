#!/usr/bin/env python3
"""Rank Qwen3 layers by how much each rewrites the residual stream.

Angular distance d(i, n) = arccos(mean cosine(h_i, h_{i+n})) / pi, averaged over
real Rineya prompts under the real chat template. Low d = the block is close to
an identity map = the cheapest thing to delete. This is the ShortGPT / Gromov
et al. block-importance criterion, measured on this checkpoint and this data
rather than assumed.

Rankings shift after every trim, so re-run this against the checkpoint actually
being pruned -- not against stock Qwen3-0.6B.

Usage: python layer_profile.py [prompts_per_source] [model_dir]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Qwen3-0.6B"          # overridable via argv[2]
OUT = ROOT / "manifests" / "layer_profile.json"

BLOCK_SIZES = (1, 2, 3, 4, 6, 9)
MAX_LEN = 512
SOURCES = (
    ("rag", "Aegis_RAG_v3_train_4410/rag_v3_train.sft.jsonl"),
    ("reasoning", "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/rineya_v3_reasoning_train_50000.jsonl"),
)


def load_prompts(per_source: int) -> list[tuple[str, list[dict]]]:
    out = []
    for label, rel in SOURCES:
        taken = 0
        with (ROOT / rel).open(encoding="utf-8") as handle:
            for line in handle:
                if taken >= per_source:
                    break
                row = json.loads(line)
                msgs = [m for m in row.get("messages") or [] if m.get("role") != "assistant"]
                if msgs:
                    out.append((label, msgs))
                    taken += 1
    return out


def main() -> None:
    global MODEL, OUT
    per_source = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    if len(sys.argv) > 2:
        MODEL = Path(sys.argv[2])
        OUT = ROOT / "manifests" / f"layer_profile_{MODEL.name}.json"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    layers = model.config.num_hidden_layers

    prompts = load_prompts(per_source)
    sums = {n: torch.zeros(layers + 1 - n, dtype=torch.float64) for n in BLOCK_SIZES}

    for _, msgs in prompts:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            hidden = model(**ids, output_hidden_states=True).hidden_states
        for n in BLOCK_SIZES:
            for i in range(layers + 1 - n):
                cos = torch.nn.functional.cosine_similarity(
                    hidden[i][0], hidden[i + n][0], dim=-1).mean()
                sums[n][i] += cos.double()

    count = len(prompts)
    # Average across prompts BEFORE clamping; clamping the accumulated sum first
    # saturates every entry to 1.0 and flattens the whole profile.
    mean_cos = {n: (v / count) for n, v in sums.items()}
    ang = {n: (torch.arccos(v.clamp(-1.0, 1.0)) / math.pi) for n, v in mean_cos.items()}

    d1 = ang[1]
    print(f"{MODEL.name}: {layers} layers | {count} prompts "
          f"({per_source}/source), chat template, enable_thinking=false\n")
    print(f"{'layer':>5s} {'cos':>7s} {'ang.dist':>9s}")
    scale = d1.max().item()
    for i in range(layers):
        print(f"{i:>5d} {mean_cos[1][i].item():>7.4f} {d1[i].item():>9.4f}  "
              f"{'#' * int(60 * d1[i].item() / scale)}")

    order = sorted(range(layers), key=lambda i: d1[i].item())
    print(f"\nmost prunable (single layers) : {order[:12]}")
    print(f"least prunable (never delete)  : {order[-6:]}")

    print("\nbest contiguous block to delete, by ang.dist(h_i, h_i+n):")
    blocks = {}
    for n in BLOCK_SIZES[1:]:
        ranked = sorted(range(layers + 1 - n), key=lambda i: ang[n][i].item())
        blocks[n] = ranked
        alts = ", ".join(f"{i}..{i + n - 1}" for i in ranked[1:4])
        print(f"  n={n}: {ranked[0]}..{ranked[0] + n - 1}"
              f"  d={ang[n][ranked[0]].item():.4f}   next: {alts}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "checkpoint": MODEL.name,
        "num_hidden_layers": layers,
        "prompts": count,
        "prompts_per_source": per_source,
        "max_len": MAX_LEN,
        "mean_cosine": {str(n): v.tolist() for n, v in mean_cos.items()},
        "angular_distance": {str(n): v.tolist() for n, v in ang.items()},
        "single_layer_prunability_order": order,
        "best_blocks": {str(n): r[:4] for n, r in blocks.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
