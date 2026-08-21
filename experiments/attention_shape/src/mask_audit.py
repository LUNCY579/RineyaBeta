#!/usr/bin/env python3
"""Why does sliding window shrink the cache but raise peak memory?

The main sweep measured, on R2 at 16K:

    base-8KV      cache 1792 MiB   peak 3128 MiB
    swa-w1024-g4  cache  352 MiB   peak 3832 MiB
    swa-w256-g4   cache  280 MiB   peak 3832 MiB   <- identical peak, different cache

The peak cannot be explained by the cache, since two variants with different
caches have byte-identical peaks. This instruments a real forward pass and
records what `create_*_mask` actually returns, which is the only reliable way to
answer it.

An earlier version of this script called the mask builders standalone and got
`None` for both, which was wrong: it passed R2's stock config, whose
`layer_types` are all `full_attention`, so the builder took its skip path. The
lesson is worth keeping -- reconstructing a call is not the same as observing
one, and here the reconstruction produced a confidently wrong answer.

Usage:
    python src/mask_audit.py --lengths 4096 16384 --variants base-8KV swa-w1024-g4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compat import force_repeat_kv  # noqa: E402
from variants import catalogue, load_variant  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--variants", nargs="+",
                    default=["base-8KV", "swa-w1024-g4", "swa-w256-g4"])
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs cuda")
    force_repeat_kv(verbose=False)
    if args.cap_mib:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, total) / total)

    import transformers.models.qwen3.modeling_qwen3 as mq
    seen: dict[str, tuple] = {}
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        orig = getattr(mq, name)

        def wrapper(*a, _orig=orig, _name=name, **k):
            r = _orig(*a, **k)
            if isinstance(r, torch.Tensor):
                seen[_name] = (tuple(r.shape), str(r.dtype).replace("torch.", ""),
                               r.numel() * r.element_size() / 2 ** 20)
            else:
                seen[_name] = ("None (is_causal fast path)", "-", 0.0)
            return r

        setattr(mq, name, wrapper)

    cat = {v.name: v for v in catalogue()}
    print(f"{'variant':16}{'n':>8}{'mask shape':>34}{'dtype':>7}{'mask MiB':>10}{'peak MiB':>10}")
    for vname in args.variants:
        model = load_variant(cat[vname], device="cuda:0")
        for n in args.lengths:
            seen.clear()
            ids = torch.randint(0, model.config.vocab_size, (1, n), device="cuda:0")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad():
                    model(ids, use_cache=True, logits_to_keep=1)
                peak = torch.cuda.max_memory_allocated() / 2 ** 20
            except torch.cuda.OutOfMemoryError:
                print(f"{vname:16}{n:>8,}{'OOM':>34}")
                torch.cuda.empty_cache()
                continue
            # Report the sliding mask when present, else the causal one.
            key = ("create_sliding_window_causal_mask"
                   if "create_sliding_window_causal_mask" in seen else "create_causal_mask")
            shape, dtype, mib = seen.get(key, ("<not called>", "-", 0.0))
            print(f"{vname:16}{n:>8,}{str(shape):>34}{dtype:>7}{mib:>10.1f}{peak:>10.1f}")
            del ids
        del model
        torch.cuda.empty_cache()

    print("\n`None` means SDPA took the is_causal fast path and no O(n^2) tensor exists.")
    print("A materialised [1,1,n,n] mask is what a sliding window costs during prefill,")
    print("and it grows with context, not with the window -- which is why a smaller")
    print("window shrinks the cache without shrinking the peak.")


if __name__ == "__main__":
    main()
