#!/usr/bin/env python3
"""Chunked prefill: make peak KV memory independent of prompt length.

THE IDENTITY. Processing a prompt in chunks of C and compressing the cache to a
budget B between chunks makes the resident cache at most `B + C` tokens at any
instant. The prompt length n does not appear:

    peak_cache = (B + C) * 2 * n_layers * kv_heads * head_dim * bytes

At B=2048, C=4096 on R2 that is 672 MiB, against 14,336 MiB for an un-chunked
128K prompt. Only compute grows with n, not memory. This is the single change
that converts the >32K rungs from arithmetic into something measurable on an
8 GiB card.

WHY THIS IS EASY TO GET SILENTLY WRONG. Three traps, all of which produce a
plausible wrong number rather than an exception:

1. **The mask.** A chunk of q queries attends to `stored + q` keys, and the
   alignment is BOTTOM-RIGHT: query i may see every cached key plus chunk keys
   up to i. SDPA's `is_causal=True` uses TOP-LEFT alignment when L != S, which
   for q=64 against stored=1024 lets query row 0 see one key instead of 1025.
   Acausal in the permissive direction, and it does not raise. This module
   therefore always passes an explicit mask.

2. **Sequence length.** `SignificanceLayer.get_seq_length()` returns the TRUE
   position count, not the stored length, which is what keeps RoPE correct for
   new tokens. The obvious "fix" of returning the stored length instead makes
   transformers build `kv_idx <= q_idx` over SLOT indices, so a query sees
   `true_length - stored` keys it should not. At B=2048 in a 32K context that is
   every key in its own chunk. Do not do it. This module computes positions and
   masks itself and never relies on the inferred values.

3. **Eviction ordering.** `update()` runs BEFORE the attention call that produces
   the scores. Compressing inside the chunk loop before that chunk has been
   scored selects on stale or absent scores and degenerates to a positional
   policy. `significance.py` now raises rather than degrading, and this module
   compresses only *between* chunks, after the chunk has been attended.

The no-op gate below is the thing to run first: with compression disabled,
chunked prefill must reproduce single-shot prefill to within floating-point
associativity. If it does not, nothing built on it means anything.

Usage:
    python scripts/kv/chunked_prefill.py --gate            # no-op proof, fp32
    python scripts/kv/chunked_prefill.py --gate --dtype bf16 --chunks 64 256
"""

from __future__ import annotations

import torch


def causal_lower_right(q_len: int, kv_len: int, device, dtype) -> torch.Tensor:
    """Additive mask, bottom-right aligned: [1, 1, q_len, kv_len].

    Query i (0-indexed within the chunk) may attend to keys
    `0 .. (kv_len - q_len + i)`. With kv_len == q_len this is the ordinary causal
    mask; with kv_len > q_len it is the correct mask for a chunk appended to an
    existing cache, which is NOT what `is_causal=True` computes.
    """
    if kv_len < q_len:
        raise ValueError(f"kv_len {kv_len} < q_len {q_len}")
    offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(-1)
    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    allowed = k_idx <= (q_idx + offset)
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


def stored_length(cache) -> int:
    layers = getattr(cache, "layers", None) or []
    if not layers:
        return 0
    layer = layers[0]
    if not getattr(layer, "is_initialized", False) or layer.keys is None:
        return 0
    return int(layer.keys.shape[-2])


@torch.no_grad()
def chunked_prefill(model, input_ids: torch.Tensor, cache, chunk: int,
                    logits_to_keep: int = 1, compress_fn=None,
                    collect_logits: bool = False):
    """Prefill `input_ids` in chunks of `chunk`, returning the final chunk's output.

    `compress_fn(cache, chunk_index)` is called BETWEEN chunks, never inside one,
    so a chunk is always scored before anything is evicted on its behalf. Pass
    None to disable compression entirely -- that is the configuration the no-op
    gate checks.
    """
    n = input_ids.shape[-1]
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    out = None
    collected = []
    n_chunks = (n + chunk - 1) // chunk

    for ci, start in enumerate(range(0, n, chunk)):
        piece = input_ids[:, start:start + chunk]
        q = piece.shape[-1]
        stored = stored_length(cache)
        mask = causal_lower_right(q, stored + q, device, dtype)
        # TRUE positions, so RoPE is unaffected by anything eviction did.
        pos = torch.arange(start, start + q, device=device).unsqueeze(0)
        out = model(
            piece,
            past_key_values=cache,
            position_ids=pos,
            # A dict short-circuits transformers' own mask construction
            # (Qwen3Model.forward: `if not isinstance(causal_mask_mapping := ...`)
            # so the bottom-right alignment above is used verbatim.
            attention_mask={"full_attention": mask},
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
        if collect_logits:
            collected.append(out.logits.float())
        if compress_fn is not None and ci < n_chunks - 1:
            compress_fn(cache, ci)
    if collect_logits:
        return out, torch.cat(collected, dim=1)
    return out


# --------------------------------------------------------------------------- #
# query-aware scoring
# --------------------------------------------------------------------------- #

def _snapshot(cache):
    return [(l.keys.shape[-2] if getattr(l, "keys", None) is not None else 0,
             getattr(l, "true_length", 0)) for l in getattr(cache, "layers", [])]


def _restore(cache, snap):
    """Truncate the cache back to a snapshot. Must be exact -- see score_with_query."""
    for layer, (n_keys, true_len) in zip(cache.layers, snap):
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys[:, :, :n_keys, :].contiguous()
            layer.values = layer.values[:, :, :n_keys, :].contiguous()
        if hasattr(layer, "true_length"):
            layer.true_length = true_len


@torch.no_grad()
def score_with_query(model, cache, query_ids, position_base: int, sig,
                     verify: bool = False):
    """Score the cached positions against the QUESTION, then undo the pass.

    Prefill-time eviction is query-blind: each chunk is scored against its own
    last query rows, which for every chunk but the last are filler, so a needle
    earns no score before anyone asks for it. Measured consequence: at 64K a
    passkey at depth 0.99 is retrieved at 1.00 and at depth 0.90 at 0.00, and a
    4x larger budget does not change it.

    This runs the question through the model as QUERIES against the current
    cache, records the resulting attention mass, and then truncates the cache
    back so the question is never itself cached. Cost is one forward over
    `len(query)` tokens per chunk -- at 16 question tokens against a 2048-token
    cache that is negligible beside the chunk itself.

    The truncation must be EXACT. If it leaves the question in the cache, every
    later chunk attends to a question it should not see yet, positions shift, and
    the run still produces plausible output -- the failure mode this codebase has
    hit repeatedly. `verify=True` checks byte-equality of every layer before and
    after, and is used by the self-test.
    """
    layers = getattr(cache, "layers", [])
    if not layers or getattr(layers[0], "keys", None) is None:
        return
    snap = _snapshot(cache)
    stored = snap[0][0]
    q = query_ids.shape[-1]
    device = query_ids.device
    dtype = next(model.parameters()).dtype

    before = None
    if verify:
        before = [(l.keys.clone(), l.values.clone()) for l in layers]

    mask = causal_lower_right(q, stored + q, device, dtype)
    pos = torch.arange(position_base, position_base + q, device=device).unsqueeze(0)
    # Score against EVERY question row, not a trailing window -- the whole point
    # is that the question is short and entirely informative.
    sig.set_recording(True, window=q)
    model(query_ids, past_key_values=cache, position_ids=pos,
          attention_mask={"full_attention": mask}, use_cache=True,
          logits_to_keep=1)
    sig.set_recording(False)

    _restore(cache, snap)
    # The scores now cover stored + q positions; the trailing q belong to the
    # question, which is about to be removed from the cache.
    scores = sig.get_scores()
    counts = getattr(sig, "_COUNTS", {})
    for li in list(scores):
        if scores[li].shape[-1] > stored:
            scores[li] = scores[li][..., :stored].contiguous()
    for li in list(counts):
        if counts[li].shape[-1] > stored:
            counts[li] = counts[li][..., :stored].contiguous()

    if verify:
        for i, (l, (k0, v0)) in enumerate(zip(layers, before)):
            if not torch.equal(l.keys, k0) or not torch.equal(l.values, v0):
                raise RuntimeError(
                    f"layer {i}: cache not restored after query scoring. Leaving "
                    f"the question cached makes every later chunk attend to it "
                    f"and still produces plausible output.")


# --------------------------------------------------------------------------- #
# no-op gate
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _gate(model, tok, prompts, chunks, dtype, device, tol):
    from transformers.cache_utils import DynamicCache
    print(f"no-op gate: chunked prefill (compression OFF) vs single-shot\n"
          f"dtype {dtype}, device {device}, tolerance {tol:g}\n")
    print(f"{'chunk':>7}{'prompts':>9}{'max|dlogit|':>14}{'mean|dlogit|':>14}"
          f"{'argmax agree':>14}{'verdict':>10}")
    all_ok = True
    for c in chunks:
        worst, total, npos, agree = 0.0, 0.0, 0, 0
        for text in prompts:
            ids = tok(text, return_tensors="pt").to(device)["input_ids"]
            ref = model(ids, past_key_values=DynamicCache(config=model.config),
                        use_cache=True, logits_to_keep=0).logits.float()
            # Compare EVERY position, not just the final chunk. An error confined
            # to an early chunk would still propagate, but direct verification is
            # what the gate should assert.
            _, got = chunked_prefill(model, ids, DynamicCache(config=model.config),
                                     chunk=c, logits_to_keep=0, collect_logits=True)
            assert got.shape == ref.shape, f"{got.shape} vs {ref.shape}"
            d = (got - ref).abs()
            worst = max(worst, float(d.max()))
            total += float(d.sum())
            npos += d.numel()
            agree += int((got.argmax(-1) == ref.argmax(-1)).sum())
        mean = total / max(npos, 1)
        frac = agree / max(npos // model.config.vocab_size, 1)
        ok = worst < tol and abs(frac - 1.0) < 1e-12
        all_ok &= ok
        print(f"{c:>7}{len(prompts):>9}{worst:>14.3e}{mean:>14.3e}"
              f"{frac:>14.4f}{'PASS' if ok else 'FAIL':>10}")
    return all_ok


def main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--model", default="artifacts/rineya-core-en32k-28L")
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--chunks", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM, AutoTokenizer
    force_repeat_kv(verbose=False)

    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    tol = args.tol if args.tol is not None else (1e-3 if dt == torch.float32 else 5e-1)
    path = REPO / args.model
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(args.device).eval()

    prompts = []
    with (REPO / "artifacts/eval_sets/rag_dev.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                msgs = [m for m in rec["messages"] if m.get("role") != "assistant"]
                prompts.append(tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            if len(prompts) >= args.cases:
                break

    if args.gate:
        ok = _gate(model, tok, prompts, args.chunks, dt, args.device, tol)
        print(f"\n{'GATE PASSED' if ok else 'GATE FAILED'}")
        print("fp32 is the honest gate: in bf16 the final-position logit differs by")
        print("mean 0.481 / max 1.56 on logits of scale ~39 purely from summation")
        print("order, so a bf16 threshold would either pass everything or reject a")
        print("correct implementation.")
        raise SystemExit(0 if ok else 1)

    ap.error("nothing to do; pass --gate")


if __name__ == "__main__":
    main()
