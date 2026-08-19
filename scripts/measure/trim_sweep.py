#!/usr/bin/env python3
"""Compare trimmed-tokenizer candidates on the trade that actually matters.

A smaller vocabulary saves embedding parameters and shrinks the lm_head GEMM,
but every dropped token inflates the sequence, and inflation is paid on all 28
layers of every forward pass forever. So a candidate is only worth taking if

    inflation  <  break_even = per_token_MACs(stock) / per_token_MACs(trimmed)

Below break-even the model is smaller *and* cheaper. Above it, the parameter
saving is bought with more compute -- which for a model this small is a losing
trade, because the embedding is the one part of the network that costs almost
nothing per token.

Usage: python trim_sweep.py <artifact_dir> [<artifact_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
STOCK = ROOT / "Qwen3-0.6B" / "tokenizer.json"
RESIDENT = 596_049_920
LAYER_MACS = 440_466_432
CHAT_SFT = ROOT / "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/rineya_v3_recommended_combined_124000.jsonl"


def prose_texts() -> list[str]:
    out = []
    for path in sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts):
        try:
            out.append(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    return out


def chat_texts(limit: int = 2000) -> list[str]:
    out = []
    with CHAT_SFT.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= limit:
                break
            for msg in json.loads(line).get("messages") or []:
                content = msg.get("content")
                if isinstance(content, str) and content:
                    out.append(content)
    return out


def count(tok: Tokenizer, texts: list[str]) -> int:
    total = 0
    for i in range(0, len(texts), 1000):
        total += sum(len(e.ids) for e in
                     tok.encode_batch(texts[i:i + 1000], add_special_tokens=False))
    return total


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    stock = Tokenizer.from_file(str(STOCK))
    prose, chat = prose_texts(), chat_texts()
    base_prose, base_chat = count(stock, prose), count(stock, chat)

    print(f"held-out prose: {len(prose)} md files, {base_prose:,d} stock tokens")
    print(f"in-domain chat: 2,000 rows, {base_chat:,d} stock tokens\n")
    print(f"{'candidate':22s} {'vocab':>7s} {'params':>12s} {'vs596M':>8s} "
          f"{'prose':>8s} {'chat':>8s} {'break-even':>10s} {'net cost':>9s} verdict")

    rows = []
    for d in dirs:
        m = json.loads((d / "trim_manifest.json").read_text(encoding="utf-8"))
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        infl_p = count(tok, prose) / base_prose
        infl_c = count(tok, chat) / base_chat
        per_tok = LAYER_MACS + 1024 * m["vocab_size"]
        breakeven = (LAYER_MACS + 1024 * 151_936) / per_tok
        net = infl_p * per_tok / (LAYER_MACS + 1024 * 151_936)
        ok = infl_p < breakeven
        rows.append((d.name, m, infl_p, infl_c, breakeven, net, ok))
        print(f"{d.name:22s} {m['vocab_size']:>7,d} {m['model_total']:>12,d} "
              f"{100 * (m['model_total'] / RESIDENT - 1):>+7.2f}% "
              f"{infl_p:>7.4f}x {infl_c:>7.4f}x {breakeven:>9.4f}x {net:>8.4f}x "
              f"{'PASS' if ok else 'FAIL'}")

    winners = [r for r in rows if r[6]]
    print("\n--- verdict ---")
    if not winners:
        print("No candidate is below break-even: every one trades compute for "
              "parameters.\nIf the objective is parameter count regardless of "
              "throughput, pick on\n'vs596M' and accept the net cost explicitly.")
    else:
        best = min(winners, key=lambda r: r[5])
        print(f"cheapest net compute: {best[0]} at {best[5]:.4f}x "
              f"({100 * (best[5] - 1):+.2f}% compute) for "
              f"{100 * (best[1]['model_total'] / RESIDENT - 1):+.2f}% parameters")
        smallest = min(winners, key=lambda r: r[1]["model_total"])
        print(f"smallest below break-even: {smallest[0]} -> "
              f"{smallest[1]['model_total']:,d} params "
              f"({100 * (smallest[1]['model_total'] / RESIDENT - 1):+.2f}%), "
              f"net {smallest[5]:.4f}x compute")


if __name__ == "__main__":
    main()
