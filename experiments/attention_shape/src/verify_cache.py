#!/usr/bin/env python3
"""Prove the sliding-window cache actually shrinks. (P4: prove comparability first.)

A config flag that changes the attention *mask* but still stores every key and
value would produce identical quality numbers and zero memory saving, and the
whole experiment would be measuring nothing. Reading the transformers source
suggests `DynamicSlidingWindowLayer` truncates to the window; this checks it on
the real model rather than trusting the source.

Three assertions, all against the analytic model in `shapes.py`:

  1. `layer_types` comes out of the config exactly as intended.
  2. After a forward pass of n tokens, sliding layers hold min(n, window) keys,
     full layers hold n.
  3. Summed cache bytes match `Shape.cache_bytes(n)` exactly.

Runs on CPU with a tiny random model by default, so it is fast and needs no GPU.
Pass --model to check a real checkpoint.

Usage:
    python src/verify_cache.py
    python src/verify_cache.py --model artifacts/rineya-core-en32k-28L --length 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes import Shape, gqa, swa_bottom_global, swa_interleaved  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def cache_layers(past) -> list[tuple[int, int]]:
    """Return (n_keys, n_heads) per layer from a transformers Cache object."""
    out = []
    layers = getattr(past, "layers", None)
    if layers is None:  # legacy tuple form
        return [(k.shape[-2], k.shape[1]) for k, _ in past]
    for layer in layers:
        k = layer.keys
        out.append((0, 0) if k is None else (k.shape[-2], k.shape[1]))
    return out


def measured_bytes(past) -> int:
    total = 0
    layers = getattr(past, "layers", None) or []
    for layer in layers:
        for t in (layer.keys, layer.values):
            if t is not None:
                total += t.numel() * t.element_size()
    return total


def build_tiny(window: int | None, n_global: int, n_layers: int = 8,
               kv_heads: int = 4) -> Qwen3ForCausalLM:
    cfg = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=n_layers, num_attention_heads=8,
        num_key_value_heads=kv_heads, head_dim=16,
        max_position_embeddings=8192, rope_theta=1_000_000,
        use_sliding_window=window is not None,
        sliding_window=window, max_window_layers=n_global,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


def check(model, shape: Shape, length: int, label: str, device: str = "cpu") -> bool:
    cfg = model.config
    types = list(cfg.layer_types)
    expected_types = [
        "full_attention" if w is None else "sliding_attention" for w in shape.windows
    ] if shape.windows else ["full_attention"] * shape.n_layers

    ok = True
    if types != expected_types:
        print(f"  FAIL layer_types\n    got      {types}\n    expected {expected_types}")
        ok = False
    else:
        n_slide = sum(t == "sliding_attention" for t in types)
        print(f"  layer_types OK: {n_slide} sliding / {len(types) - n_slide} full")

    ids = torch.randint(0, cfg.vocab_size, (1, length), device=device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values

    held = cache_layers(past)
    want = [
        length if (not shape.windows or shape.windows[i] is None)
        else min(length, shape.windows[i])
        for i in range(shape.n_layers)
    ]
    got = [h for h, _ in held]
    # The sliding layer keeps window-1 cached plus the current step; at prefill
    # the whole window is present. Allow the documented off-by-one.
    mismatch = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if abs(g - w) > 1]
    if mismatch:
        print(f"  FAIL cache occupancy, first 3: {mismatch[:3]}")
        ok = False
    else:
        print(f"  cache occupancy OK: {got[:4]}{'...' if len(got) > 4 else ''} "
              f"(expected {want[:4]}{'...' if len(want) > 4 else ''})")

    meas = measured_bytes(past)
    pred = shape.cache_bytes(length)
    rel = abs(meas - pred) / max(pred, 1)
    verdict = "OK" if rel < 0.01 else "FAIL"
    if rel >= 0.01:
        ok = False
    print(f"  bytes {verdict}: measured {meas / 2**20:8.2f} MiB   "
          f"analytic {pred / 2**20:8.2f} MiB   rel err {rel:.4f}")
    print(f"  -> {label}: {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="real checkpoint, repo-relative")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    passed = []

    if args.model is None:
        n_layers, kv_heads, hd = 8, 4, 16
        # The tiny model is built and run in float32 on CPU; the analytic model
        # must be told so, or it under-predicts by exactly 2x.
        common = dict(kv_heads=kv_heads, n_layers=n_layers, head_dim=hd, bytes_per_elem=4)
        print(f"tiny random Qwen3: {n_layers}L {kv_heads}KV head_dim {hd}, "
              f"float32, length {args.length}\n")

        print(f"[1] baseline, no sliding window")
        m = build_tiny(None, n_layers, n_layers, kv_heads)
        passed.append(check(m, gqa(**common), args.length, "full attention"))

        print(f"[2] sliding window {args.window}, bottom {args.n_global} global")
        m = build_tiny(args.window, args.n_global, n_layers, kv_heads)
        passed.append(check(m, swa_bottom_global(args.window, args.n_global, **common),
                            args.length, "sliding hybrid"))

        print(f"[3] window wider than context -> must equal baseline")
        m = build_tiny(args.length * 4, args.n_global, n_layers, kv_heads)
        passed.append(check(m, swa_bottom_global(args.length * 4, args.n_global, **common),
                            args.length, "window wider than context"))
    else:
        path = REPO / args.model
        cfg = AutoConfig.from_pretrained(path)
        print(f"real checkpoint {args.model}: {cfg.num_hidden_layers}L "
              f"{cfg.num_key_value_heads}KV head_dim {cfg.head_dim}, length {args.length}")
        print(f"  use_sliding_window={cfg.use_sliding_window} "
              f"sliding_window={cfg.sliding_window} max_window_layers={cfg.max_window_layers}\n")
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(args.device).eval()
        elem = 2 if dtype == torch.bfloat16 else 4
        windows = [None if t == "full_attention" else cfg.sliding_window
                   for t in cfg.layer_types]
        shape = Shape(name=args.model, n_layers=cfg.num_hidden_layers,
                      kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                      bytes_per_elem=elem,
                      windows=tuple(windows) if any(w is not None for w in windows) else ())
        passed.append(check(model, shape, args.length, args.model, args.device))

    print("=" * 64)
    print(f"{sum(passed)}/{len(passed)} checks passed")
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
