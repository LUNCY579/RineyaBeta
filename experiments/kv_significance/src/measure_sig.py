#!/usr/bin/env python3
"""Efficiency of significance-gated eviction: cache bytes, prefill, decode.

Tests two predictions from THEORY.md §6:

  S3  decode throughput improves roughly in proportion to cache reduction,
      because decode is bandwidth-bound on streaming the cache;
  S4  peak prefill memory is NOT improved. Compression happens after the prompt
      cache exists, so the full cache is transiently resident. This is recorded
      as a prediction of *failure* so that a peak-memory win cannot later be
      claimed without the chunked-prefill work that would earn it.

Also reports the significance matrix as a share of the cache it manages, which
THEORY.md §4 puts at 0.78% in fp32, length-independent.

Usage:
    python src/measure_sig.py --lengths 4096 16384 --budgets 0 1024 512
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

import significance as sig  # noqa: E402
from compat import force_repeat_kv  # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"


def timed(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


@torch.no_grad()
def one(model, length: int, budget: int, decode_tokens: int, repeats: int,
        window: int, recent: int, device: str, slack: int = 0) -> dict:
    vocab = model.config.vocab_size
    g = torch.Generator(device="cpu").manual_seed(7)
    ids = torch.randint(0, vocab, (1, length), generator=g).to(device)
    evicting = budget > 0
    policy = (sig.Policy(budget=budget, sinks=4, recent=recent, slack=slack)
              if evicting else None)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sig.reset_scores()
    sig.set_recording(evicting, window=window)

    cache = sig.SignificanceCache(config=model.config, policy=policy)
    out = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    peak = torch.cuda.max_memory_allocated()
    prefill_cache_b = sig.cache_bytes(cache)
    prefill_stored = cache.stored_lengths()[0]

    def prefill():
        sig.reset_scores()
        c = sig.SignificanceCache(config=model.config, policy=policy)
        model(ids, past_key_values=c, use_cache=True, logits_to_keep=1)

    pre = timed(prefill, repeats, 1)

    tok = out.logits[:, -1:].argmax(-1)
    state = {"pkv": cache, "tok": tok}

    def decode_block():
        pkv, t = state["pkv"], state["tok"]
        for _ in range(decode_tokens):
            o = model(t, past_key_values=pkv, use_cache=True, logits_to_keep=1)
            pkv, t = o.past_key_values, o.logits[:, -1:].argmax(-1)
        state["pkv"], state["tok"] = pkv, t

    dec = timed(decode_block, repeats, 1)
    tps = [decode_tokens / (d / 1e3) for d in dec]

    # Measure the cache AFTER decode: compression correctly happens on the first
    # decode step, not during prefill, so a reading taken straight after the
    # prompt pass reports the uncompressed size and makes eviction look inert.
    cache_b = sig.cache_bytes(state["pkv"])
    score_b = sig.score_bytes()
    stored = state["pkv"].stored_lengths()[0]

    sig.set_recording(False)
    del cache, out, state
    torch.cuda.empty_cache()

    return {
        "length": length, "budget": budget, "slack": slack, "stored_kv": stored,
        "prefill_stored_kv": prefill_stored,
        "prefill_cache_mib": round(prefill_cache_b / 2 ** 20, 2),
        "cache_mib": round(cache_b / 2 ** 20, 2),
        "score_mib": round(score_b / 2 ** 20, 3),
        "score_pct_of_cache": round(100 * score_b / max(cache_b, 1), 3),
        "peak_mib": round(peak / 2 ** 20, 1),
        "prefill_ms": round(statistics.median(pre), 1),
        "decode_tok_per_s": round(statistics.median(tps), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 1024, 512, 256],
                    help="0 means eviction off")
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--recent", type=int, default=128)
    ap.add_argument("--slack", type=int, default=0,
                    help="evict only once stored exceeds budget+slack; amortises "
                         "the top-k and gather over many decode steps")
    ap.add_argument("--cap-mib", type=float, default=6144.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "measure_sig.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs cuda")
    force_repeat_kv(verbose=False)
    sig.patch_attention()
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, total) / total)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(args.device).eval()

    print(f"cap {args.cap_mib:.0f} MiB   decode {args.decode_tokens} tok   "
          f"window {args.window}   recent {args.recent}\n")
    print(f"{'len':>7}{'budget':>8}{'stored':>8}{'cache MiB':>11}{'sig MiB':>9}"
          f"{'sig %':>7}{'peak MiB':>10}{'prefill ms':>12}{'tok/s':>8}")
    rows = []
    for L in args.lengths:
        for B in args.budgets:
            if B and B >= L:
                continue
            try:
                r = one(model, L, B, args.decode_tokens, args.repeats,
                        args.window, args.recent, args.device, args.slack)
            except torch.cuda.OutOfMemoryError:
                print(f"{L:>7}{B:>8}   OOM")
                torch.cuda.empty_cache()
                continue
            rows.append(r)
            print(f"{r['length']:>7,}{r['budget']:>8}{r['stored_kv']:>8}"
                  f"{r['cache_mib']:>11.1f}{r['score_mib']:>9.2f}"
                  f"{r['score_pct_of_cache']:>7.2f}{r['peak_mib']:>10.1f}"
                  f"{r['prefill_ms']:>12.1f}{r['decode_tok_per_s']:>8.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
         "decode_tokens": args.decode_tokens, "window": args.window,
         "recent": args.recent, "results": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    for L in args.lengths:
        base = next((r for r in rows if r["length"] == L and r["budget"] == 0), None)
        if not base:
            continue
        print(f"\nlength {L:,} — vs eviction off:")
        for r in rows:
            if r["length"] != L or r["budget"] == 0:
                continue
            print(f"  budget {r['budget']:>5}: cache {base['cache_mib']/r['cache_mib']:.1f}x smaller, "
                  f"decode {r['decode_tok_per_s']/base['decode_tok_per_s']:.2f}x, "
                  f"peak {r['peak_mib']/base['peak_mib']:.2f}x  "
                  f"(S4 predicts peak ~1.0x or worse)")


if __name__ == "__main__":
    main()
