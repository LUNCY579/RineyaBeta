#!/usr/bin/env python3
"""Measure attention mass on the first tokens -- the "attention sink".

The quality sweep found that a sliding window is exactly free while the window
covers the prompt (20/20 cases, delta +0.0000) and catastrophic the moment it
does not -- and that the catastrophe does not grow with how much context is lost:

    prompt tokens beyond the window     n     base   swa-w128     delta
    <= 0                               20   0.8886     0.8886   +0.0000
    1-100                             307   0.6863     0.0197   -0.6666
    101-200                           422   0.6487     0.0068   -0.6419

One token over the edge costs as much as two hundred, and the model emits empty
strings. That is a cliff, not a slope, so it is not gradual information loss.
The standard explanation is the attention sink: transformers dump a large,
content-independent share of attention on the first few positions, and a naive
sliding window evicts exactly those.

This measures it on R2 rather than citing it, so the explanation rests on a
number from this model.

Usage:
    python src/sink_audit.py --cases 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--first-k", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/sink_profile.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = REPO / "artifacts/rineya-core-en32k-28L"
    tok = AutoTokenizer.from_pretrained(path)
    # output_attentions needs eager; fine at these lengths.
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="eager").to(args.device).eval()

    rows = []
    with (REPO / "artifacts/eval_sets/rag_dev.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.cases:
                break

    n_layers = model.config.num_hidden_layers
    sink = torch.zeros(n_layers)
    total = 0
    for rec in rows:
        msgs = [m for m in rec["messages"] if m.get("role") != "assistant"]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**ids, output_attentions=True, use_cache=False)
        # attentions: tuple(n_layers)[batch, heads, q, k]; take the LAST query row,
        # i.e. what the token being generated actually looks at.
        for li, a in enumerate(out.attentions):
            last = a[0, :, -1, :].float()          # [heads, k]
            sink[li] += last[:, :args.first_k].sum(-1).mean().item()
        total += 1
        del out

    sink /= max(total, 1)
    print(f"attention mass on the first {args.first_k} tokens, averaged over "
          f"{total} rag_dev prompts and all 16 query heads\n")
    print(f"{'layer':>6} {'sink share':>11}  {'':<42}")
    for li in range(n_layers):
        bar = "#" * int(round(sink[li].item() * 40))
        print(f"{li:>6} {sink[li].item():>10.1%}  {bar:<42}")
    print(f"\nmean across layers: {sink.mean().item():.1%}")
    hi = [i for i in range(n_layers) if sink[i] > 0.5]
    print(f"layers sending >50% of attention to the first {args.first_k} tokens: "
          f"{len(hi)}/{n_layers}  {hi}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "first_k": args.first_k, "cases": total,
            "per_layer_sink_share": [round(v, 6) for v in sink.tolist()],
            "mean": round(sink.mean().item(), 6),
        }, indent=2) + chr(10), encoding="utf-8")
        print(f"wrote {args.out}")

    print("\nA naive sliding window evicts exactly these positions. That is why the")
    print("collapse is a cliff at the window boundary and not proportional to the")
    print("amount of context lost. The remedy is to pin the first few tokens in the")
    print("window (StreamingLLM-style sinks), which transformers' sliding_window")
    print("does not do.")


if __name__ == "__main__":
    main()
