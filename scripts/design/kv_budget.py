#!/usr/bin/env python3
"""KV-cache feasibility and the iso-memory length frontier.

The long-context experiment is memory-bound, not compute-bound, so which cells
of a (kv_heads x length) grid are even measurable has to be settled before any
GPU time is spent. This computes that from the checkpoint's own config.

Two questions, both answered here:

  1. Matched-length feasibility -- for a given VRAM budget, which (g, L) cells
     fit? Cells that do not fit are not "expensive", they are unmeasurable, and
     a design that depends on them is not executable on this host.

  2. The iso-memory frontier -- at a *fixed* KV byte budget, each config reaches
     a different length: L_max(g) = B / (2 * n_layers * g * head_dim * bytes).
     L_max is linear in 1/g, so halving KV heads exactly doubles reachable
     context. This is the comparison the deployment question actually asks.

Usage:
    python scripts/design/kv_budget.py
    python scripts/design/kv_budget.py --model Qwen3-0.6B --budget-gb 6.5
    python scripts/design/kv_budget.py --json manifests/kv_budget.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Length ladder under evaluation.
LADDER = (4096, 16384, 32768, 65536, 131072)
# KV-head counts under evaluation. 8 is the stock Qwen3-0.6B/1.7B setting.
KV_LEVELS = (8, 4, 2, 1)
# Bytes per cached element, by KV dtype.
DTYPE_BYTES = {"bf16": 2, "int8": 1}


def kv_bytes_per_token(n_layers: int, kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """K and V, every layer, every KV head."""
    return 2 * n_layers * kv_heads * head_dim * dtype_bytes


def eager_scores_bytes(n_q_heads: int, length: int, dtype_bytes: int = 2) -> int:
    """Transient score matrix for one eager-attention layer.

    Quadratic in length. This is why `determinism.py`'s eager requirement cannot
    follow us into long-context evaluation.
    """
    return n_q_heads * length * length * dtype_bytes


def resident_params(cfg: dict, kv_heads: int) -> int:
    """Parameter count with `kv_heads` KV heads, everything else held fixed.

    Assumes tied embeddings (true for both Qwen3-0.6B and 1.7B).
    """
    h = cfg["hidden_size"]
    hd = cfg["head_dim"]
    nq = cfg["num_attention_heads"]
    inter = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]

    attn = h * nq * hd + 2 * (h * kv_heads * hd) + nq * hd * h
    mlp = 3 * h * inter
    norms = 2 * h + 2 * hd  # input/post norms + q/k RMSNorm (Qwen3)
    return cfg["vocab_size"] * h + n_layers * (attn + mlp + norms) + h


def fmt_tokens(n: int) -> str:
    return f"{n / 1024:.0f}K" if n >= 1024 else str(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3-0.6B", help="directory holding config.json")
    ap.add_argument("--budget-gb", type=float, default=6.5,
                    help="usable VRAM in GB after the desktop compositor takes its share")
    ap.add_argument("--overhead-gb", type=float, default=0.6,
                    help="activations, workspace and allocator slack during chunked prefill")
    ap.add_argument("--weight-dtype-bytes", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None, help="also write the feasibility matrix here")
    args = ap.parse_args()

    model_dir = (ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    n_layers = cfg["num_hidden_layers"]
    head_dim = cfg["head_dim"]
    n_q = cfg["num_attention_heads"]
    stock_kv = cfg["num_key_value_heads"]

    print(f"checkpoint     : {model_dir}")
    print(f"shape          : {n_layers}L  {cfg['hidden_size']}d  "
          f"{n_q}Q/{stock_kv}KV  head_dim {head_dim}  vocab {cfg['vocab_size']:,}")
    print(f"rope_theta     : {cfg.get('rope_theta')}   max_position_embeddings "
          f"{cfg.get('max_position_embeddings'):,}")
    print(f"VRAM budget    : {args.budget_gb:.2f} GB usable, {args.overhead_gb:.2f} GB reserved for overhead\n")

    # ---- per-token cost and weight cost -------------------------------------
    print("per-token KV cost and resident parameters, by KV-head count")
    print(f"{'kv_heads':>9} {'KiB/token':>11} {'params':>14} {'weights GB':>11} {'d(params)':>12}")
    base_params = resident_params(cfg, stock_kv)
    weight_gb = {}
    for g in KV_LEVELS:
        p = resident_params(cfg, g)
        wgb = p * args.weight_dtype_bytes / 1e9
        weight_gb[g] = wgb
        per_tok = kv_bytes_per_token(n_layers, g, head_dim, DTYPE_BYTES["bf16"])
        print(f"{g:>9} {per_tok / 1024:>11.0f} {p:>14,} {wgb:>11.3f} {p - base_params:>+12,}")
    print()

    # ---- matched-length feasibility -----------------------------------------
    rows = []
    for dtype, db in DTYPE_BYTES.items():
        print(f"matched-length feasibility, KV in {dtype}  "
              f"(FIT = weights + KV + overhead <= {args.budget_gb:.2f} GB)")
        header = f"{'kv_heads':>9}" + "".join(f"{fmt_tokens(n):>12}" for n in LADDER)
        print(header)
        for g in KV_LEVELS:
            per_tok = kv_bytes_per_token(n_layers, g, head_dim, db)
            cells = []
            for n in LADDER:
                total = weight_gb[g] + per_tok * n / 1e9 + args.overhead_gb
                fits = total <= args.budget_gb
                cells.append(f"{('  ' if fits else ' X') + f'{total:6.2f}GB':>12}")
                rows.append({"kv_heads": g, "kv_dtype": dtype, "length": n,
                             "total_gb": round(total, 3), "fits": fits})
            print(f"{g:>9}" + "".join(cells))
        print("           X = does not fit\n")

    # ---- iso-memory frontier -------------------------------------------------
    print("iso-memory frontier: max reachable length at a fixed KV budget")
    print("(this is the honest comparison -- same bytes of cache, different shape)")
    kv_budget = {g: (args.budget_gb - weight_gb[g] - args.overhead_gb) * 1e9 for g in KV_LEVELS}
    print(f"{'kv_heads':>9} {'KV budget GB':>13} {'L_max bf16':>12} {'L_max int8':>12} {'x vs stock':>11}")
    ref = None
    for g in KV_LEVELS:
        b = kv_budget[g]
        if b <= 0:
            print(f"{g:>9} {'INFEASIBLE':>13}")
            continue
        lmax_bf16 = int(b / kv_bytes_per_token(n_layers, g, head_dim, 2))
        lmax_int8 = int(b / kv_bytes_per_token(n_layers, g, head_dim, 1))
        ref = ref or lmax_bf16
        print(f"{g:>9} {b / 1e9:>13.2f} {lmax_bf16:>12,} {lmax_int8:>12,} {lmax_bf16 / ref:>10.2f}x")
    print()

    # ---- eager-attention ceiling --------------------------------------------
    print("eager-attention score matrix, one layer, transient")
    print("(determinism.py pins eager for bit-exact *training*; evaluation cannot follow)")
    for n in LADDER:
        gb = eager_scores_bytes(n_q, n) / 1e9
        flag = "" if gb <= args.budget_gb else "   <- exceeds the whole card"
        print(f"  {fmt_tokens(n):>5}: {gb:8.1f} GB{flag}")

    if args.json:
        out = {
            "model": str(model_dir.relative_to(ROOT)) if model_dir.is_relative_to(ROOT) else str(model_dir),
            "budget_gb": args.budget_gb,
            "overhead_gb": args.overhead_gb,
            "shape": {"n_layers": n_layers, "head_dim": head_dim,
                      "n_q_heads": n_q, "stock_kv_heads": stock_kv},
            "cells": rows,
            "iso_memory_max_length": {
                str(g): int(kv_budget[g] / kv_bytes_per_token(n_layers, g, head_dim, 2))
                for g in KV_LEVELS if kv_budget[g] > 0
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
