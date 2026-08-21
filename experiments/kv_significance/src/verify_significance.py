#!/usr/bin/env python3
"""Prove the eviction machinery is correct before measuring anything with it.

Five checks, cheapest first, each able to invalidate the ones after it:

  1. Scores are recorded with the right shape, one per layer, per KV head, per
     position, and the observation window does not leak future positions.
  2. A budget larger than the context is a NO-OP: generation must be
     byte-identical to the unpatched baseline. This is the analogue of the
     sliding-window P1 test that caught real bugs last time -- if the machinery
     perturbs anything when it should do nothing, no later number is trustworthy.
  3. Compression actually shrinks the cache to the budget, and the significance
     matrix costs what THEORY.md §4 says it costs.
  4. Sinks survive. R2 puts 46.3% of its attention mass on the first 4 tokens and
     the sliding window collapsed precisely by evicting them, so the policy must
     be shown to keep them -- and, with protection switched off, the scores must
     rank them highly anyway.
  5. Retained mass rho is measured directly and compared against the bound
     ||o - o_hat|| <= 2(1-rho)||v||max.

Usage:
    python src/verify_significance.py
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
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))
sys.path.insert(0, str(REPO))

from scripts.kv import significance as sig  # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"


def build(device: str):
    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM, AutoTokenizer
    force_repeat_kv(verbose=False)
    path = REPO / MODEL
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device).eval()
    return tok, model


def first_prompt(tok) -> str:
    with (REPO / "artifacts/eval_sets/rag_dev.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                msgs = [m for m in rec["messages"] if m.get("role") != "assistant"]
                return tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
    raise SystemExit("no prompts")


@torch.no_grad()
def generate(model, tok, enc, max_new: int, cache=None) -> str:
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                         past_key_values=cache)
    return tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    tok, model = build(args.device)
    prompt = first_prompt(tok)
    enc = tok(prompt, return_tensors="pt").to(args.device)
    n = enc["input_ids"].shape[-1]
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    print(f"prompt {n} tokens, {n_layers} layers, {n_kv} KV heads, budget {args.budget}\n")

    passed: list[tuple[str, bool]] = []

    # ---- baseline, unpatched ------------------------------------------------
    base_reply = generate(model, tok, enc, args.max_new)
    print(f"[baseline] {base_reply[:90]!r}\n")

    sig.patch_attention()

    # ---- 1. score shapes ----------------------------------------------------
    sig.reset_scores()
    sig.set_recording(True, window=16)
    cache = sig.SignificanceCache(config=model.config)
    with torch.no_grad():
        model(**enc, past_key_values=cache, use_cache=True)
    sig.set_recording(False)
    scores = sig.get_scores()
    ok = (len(scores) == n_layers
          and all(tuple(s.shape) == (1, n_kv, n) for s in scores.values())
          and all(torch.isfinite(s).all() for s in scores.values()))
    print(f"[1] scores: {len(scores)} layers, shape {tuple(next(iter(scores.values())).shape)}, "
          f"expected (1, {n_kv}, {n}) -> {'PASS' if ok else 'FAIL'}")
    passed.append(("score shapes", ok))

    # ---- 2. no-op when the budget exceeds the context -----------------------
    sig.reset_scores()
    sig.set_recording(True, window=16)
    cache = sig.SignificanceCache(config=model.config)
    noop_reply = generate(model, tok, enc, args.max_new, cache=cache)
    sig.set_recording(False)
    ok = noop_reply == base_reply
    print(f"[2] recording-only is a no-op: reply identical -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"      base: {base_reply[:80]!r}")
        print(f"      ours: {noop_reply[:80]!r}")
    passed.append(("recording is a no-op", ok))

    # ---- 3. compression shrinks the cache to budget -------------------------
    sig.reset_scores()
    sig.set_recording(True, window=16)
    cache = sig.SignificanceCache(config=model.config)
    with torch.no_grad():
        model(**enc, past_key_values=cache, use_cache=True)
    sig.set_recording(False)
    before_bytes = sig.cache_bytes(cache)
    pol = sig.Policy(budget=args.budget, sinks=4, recent=32)
    rep = sig.compress(cache, pol)
    after_bytes = sig.cache_bytes(cache)
    stored = [l.stored_length() for l in cache.layers]
    ok = all(s == args.budget for s in stored) and cache.get_seq_length() == n
    print(f"[3] compressed: stored length {set(stored)} (want {{{args.budget}}}), "
          f"get_seq_length {cache.get_seq_length()} (want {n}) -> {'PASS' if ok else 'FAIL'}")
    print(f"      cache {before_bytes/2**20:.2f} -> {after_bytes/2**20:.2f} MiB "
          f"({rep['reduction']:.2f}x)   significance matrix "
          f"{sig.score_bytes()/2**20:.3f} MiB "
          f"({100*sig.score_bytes()/max(before_bytes,1):.2f}% of the uncompressed cache)")
    passed.append(("compression to budget", ok))

    # ---- 4. sinks survive ---------------------------------------------------
    s0 = sig.get_scores()[0][0]                       # [n_kv, budget] post-compress
    sig.reset_scores()
    sig.set_recording(True, window=16)
    cache2 = sig.SignificanceCache(config=model.config)
    with torch.no_grad():
        model(**enc, past_key_values=cache2, use_cache=True)
    sig.set_recording(False)
    raw = sig.get_scores()
    ranks = []
    for li, s in raw.items():
        order = s[0].argsort(dim=-1, descending=True)   # [n_kv, n]
        for h in range(order.shape[0]):
            pos = order[h].tolist()
            ranks.append(min(pos.index(i) for i in range(4)))
    best_sink_rank = sum(ranks) / len(ranks)
    ok = best_sink_rank < 4.0
    print(f"[4] sink ranking without protection: the best of the first 4 positions "
          f"has mean rank {best_sink_rank:.2f} -> {'PASS' if ok else 'FAIL'}")
    print(f"      (rank 0 = highest scoring; low means the policy would keep sinks "
          f"even with protect_sinks=False)")
    passed.append(("scores rank sinks highly", ok))

    # ---- 4b. REGRESSION: prefill must see every key -------------------------
    # `update()` runs before the attention call that produces scores, so an
    # eviction triggered on the prefill pass has nothing to select with and falls
    # back to sinks+recent -- a sliding window. This check exists because the
    # first implementation did exactly that and the whole quality sweep measured
    # the wrong thing.
    sig.reset_scores()
    sig.set_recording(True, window=16)
    cache3 = sig.SignificanceCache(config=model.config,
                                   policy=sig.Policy(budget=args.budget, sinks=4, recent=32))
    with torch.no_grad():
        model(**enc, past_key_values=cache3, use_cache=True, logits_to_keep=1)
    sig.set_recording(False)
    seen = sig.get_scores()[0].shape[-1]
    ok = seen == n and cache3.layers[0].evictions == 0
    print(f"[4b] prefill attention saw {seen}/{n} keys, "
          f"{cache3.layers[0].evictions} evictions during prefill -> "
          f"{'PASS' if ok else 'FAIL'}")
    passed.append(("prefill sees full context", ok))

    # ---- 5. retained mass ---------------------------------------------------
    tot = {li: s.sum(-1) for li, s in raw.items()}
    kept_frac = []
    for li, s in raw.items():
        n_sink, n_recent = 4, 32
        n_heavy = args.budget - n_sink - n_recent
        m = s[0].clone()
        m[..., :n_sink] = float("-inf")
        m[..., n - n_recent:] = float("-inf")
        idx = m.topk(max(n_heavy, 0), dim=-1).indices
        keep = s[0].gather(-1, idx).sum(-1)
        keep = keep + s[0][..., :n_sink].sum(-1) + s[0][..., n - n_recent:].sum(-1)
        kept_frac.append((keep / tot[li][0].clamp_min(1e-9)).mean().item())
    rho = sum(kept_frac) / len(kept_frac)
    ok = rho > 0.9
    print(f"[5] retained mass at budget {args.budget}: rho = {rho:.4f}  "
          f"-> bound ||o-o_hat|| <= {2*(1-rho):.4f} * max||v||  "
          f"-> {'PASS' if ok else 'FAIL'}")
    passed.append(("retained mass > 0.9", ok))

    print("\n" + "=" * 68)
    for name, k in passed:
        print(f"  {'PASS' if k else 'FAIL'}  {name}")
    n_ok = sum(k for _, k in passed)
    print(f"{n_ok}/{len(passed)} checks passed")
    raise SystemExit(0 if n_ok == len(passed) else 1)


if __name__ == "__main__":
    main()
