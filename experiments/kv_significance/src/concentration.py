#!/usr/bin/env python3
"""Phase 0: is attention concentrated enough for eviction to be safe?

This is the cheap test that can kill the expensive one (P7). Everything the
significance-matrix design depends on reduces to one measurable quantity.

THE BOUND. Attention output is a convex combination o = sum_i a_i v_i. Evict a
set, renormalise over what is kept, and with retained mass rho = sum_{i in K} a_i:

    o_hat      = (1/rho) sum_{i in K} a_i v_i
    o - o_hat  = (1 - 1/rho) sum_K a_i v_i + sum_{K complement} a_i v_i
    ||o - o_hat|| <= (1-rho)*V + (1-rho)*V = 2(1-rho)*V,     V = max_i ||v_i||

Output error is bounded by *retained attention mass* and nothing else. If a small
top-k captures most of the mass, eviction is provably near-lossless. If mass is
spread evenly, no eviction policy can be safe and the idea dies here, for the
price of one forward pass.

THREE THINGS MEASURED

  1. rho(k)      retained mass for the top-k keys, per layer.
  2. rho_nosink  the same after zeroing the first 4 positions and renormalising.
                 This is the control that matters: R2 puts 46.3% of its mass on
                 those 4 tokens, so if ALL the concentration is the sink, a
                 significance matrix buys nothing over StreamingLLM (4 sinks plus
                 a recent window) and the extra machinery is not worth building.
  3. persistence overlap of the top-32 set between consecutive query positions.
                 Eviction is irreversible, so a token dropped now must not become
                 important later. Low persistence means a greedy accumulated
                 score is the wrong statistic.

Needs eager attention to see the weights, so it runs at short context -- fine,
because concentration is a property of the distribution, not of the length.

Usage:
    python src/concentration.py --cases 8 --out results/concentration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

MODEL = "artifacts/rineya-core-en32k-28L"
SINK = 4
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def load_prompts(tok, n_cases: int) -> list[str]:
    rows = []
    with (REPO / "artifacts/eval_sets/rag_dev.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs = [m for m in rec["messages"] if m.get("role") != "assistant"]
            rows.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            if len(rows) >= n_cases:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--last-queries", type=int, default=16,
                    help="score using the final N query rows (the observation window)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/concentration.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = REPO / MODEL
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="eager").to(args.device).eval()
    n_layers = model.config.num_hidden_layers

    rho = torch.zeros(n_layers, len(KS))
    rho_ns = torch.zeros(n_layers, len(KS))
    rho_n = torch.zeros(n_layers, len(KS))          # per-cell valid-sample count
    persist = torch.zeros(n_layers)
    n_obs = 0
    seq_lens = []

    for prompt in load_prompts(tok, args.cases):
        ids = tok(prompt, return_tensors="pt").to(args.device)
        seq_lens.append(ids["input_ids"].shape[-1])
        with torch.no_grad():
            out = model(**ids, output_attentions=True, use_cache=False)

        for li, att in enumerate(out.attentions):
            a = att[0].float()                        # [heads, q, k]
            q0 = max(0, a.shape[1] - args.last_queries)
            rows = a[:, q0:, :]                       # [heads, w, k]

            srt, idx = rows.sort(dim=-1, descending=True)
            cum = srt.cumsum(-1)
            for ki, k in enumerate(KS):
                # Only record a cell when the budget actually fits the sequence.
                # Crediting rho = 1.0 for k > n does not measure concentration, it
                # measures "the budget exceeded the context" -- and with sequences
                # of 192-265 tokens that silently fabricated the entire k=256
                # column as exactly 1.0000, which then flattered every downstream
                # allocation computed from these curves.
                if k <= cum.shape[-1]:
                    rho[li, ki] += cum[..., k - 1].mean().item()
                    rho_n[li, ki] += 1

            ns = rows.clone()
            ns[..., :SINK] = 0.0
            ns = ns / ns.sum(-1, keepdim=True).clamp_min(1e-9)
            cum_ns = ns.sort(dim=-1, descending=True)[0].cumsum(-1)
            for ki, k in enumerate(KS):
                if k <= cum_ns.shape[-1]:
                    rho_ns[li, ki] += cum_ns[..., k - 1].mean().item()

            k = min(32, rows.shape[-1])
            top = idx[..., :k]
            if top.shape[1] > 1:
                inter = 0.0
                cnt = 0
                for t in range(top.shape[1] - 1):
                    for h in range(top.shape[0]):
                        s1 = set(top[h, t].tolist())
                        s2 = set(top[h, t + 1].tolist())
                        inter += len(s1 & s2) / k
                        cnt += 1
                persist[li] += inter / cnt
        n_obs += 1
        del out
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    valid = rho_n.clamp_min(1)
    rho /= valid
    rho_ns /= valid
    persist /= n_obs
    unmeasured = [KS[ki] for ki in range(len(KS)) if rho_n[:, ki].max() == 0]
    if unmeasured:
        print(f"NOTE: k={unmeasured} exceeded every sequence and is UNMEASURED;")
        print("      those columns read 0 and must not be used downstream.")
        print()

    print(f"R2, {n_obs} rag_dev prompts, seq len {min(seq_lens)}-{max(seq_lens)}, "
          f"scored on the last {args.last_queries} query rows, mean over 16 heads")
    print()
    print("retained attention mass rho(k), all positions")
    hdr = "".join(f"{'k=' + str(k):>9}" for k in KS)
    print(f"{'layer':>6}{hdr}")
    for li in range(n_layers):
        print(f"{li:>6}" + "".join(f"{rho[li, ki].item():>9.3f}" for ki in range(len(KS))))
    print(f"{'MEAN':>6}" + "".join(f"{rho[:, ki].mean().item():>9.3f}" for ki in range(len(KS))))

    print()
    print("rho(k) with the first 4 positions REMOVED and renormalised")
    print("(if this collapses, the concentration is only the sink, and a")
    print(" significance matrix buys nothing over StreamingLLM)")
    print(f"{'MEAN':>6}" + "".join(f"{rho_ns[:, ki].mean().item():>9.3f}" for ki in range(len(KS))))

    print()
    print(f"top-32 persistence between consecutive queries: mean {persist.mean().item():.3f}"
          f"  min {persist.min().item():.3f} (layer {persist.argmin().item()})"
          f"  max {persist.max().item():.3f}")

    mean_rho = rho.mean(0)
    print()
    print("implied error bound   ||o - o_hat|| <= 2(1-rho) * max||v||")
    for ki, k in enumerate(KS):
        r = mean_rho[ki].item()
        print(f"  budget {k:>4}: rho={r:.4f}   bound={2 * (1 - r):.4f} * max||v||")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": MODEL, "cases": n_obs, "ks": KS, "sink": SINK,
        "last_queries": args.last_queries,
        "seq_len_min": min(seq_lens), "seq_len_max": max(seq_lens),
        "rho_per_layer": [[round(v, 5) for v in row] for row in rho.tolist()],
        "rho_nosink_per_layer": [[round(v, 5) for v in row] for row in rho_ns.tolist()],
        "rho_mean": [round(v, 5) for v in mean_rho.tolist()],
        "rho_nosink_mean": [round(v, 5) for v in rho_ns.mean(0).tolist()],
        "persistence_top32_per_layer": [round(v, 5) for v in persist.tolist()],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
