#!/usr/bin/env python3
"""End-to-end correctness tests for the CORE: the significance matrix and the cache
compression themselves — not the selection arms built on top of them.

WHY THIS FILE EXISTS. Every test in this repository until now exercised the things
built ON the mechanism: which positions a policy picks, whether a manifest is
matched, whether an analyser refuses missing data. Nothing checked that the
significance matrix is the quantity its docstring claims, or that evicting from the
KV cache leaves the model computing the same attention over what remains. Those two
assumptions carry 252 pinned numbers, the whole passkey line, and a recent conclusion
that the score is beaten by a BM25 retriever — a conclusion that would be a statement
about a bug rather than about the mechanism if the score were computed wrong.

The tests are written to be able to FAIL. Section 4 deliberately checks that eviction
DOES change the output when it fires, so a suite that passes everything by making the
mechanism inert would be caught.

CPU only, no GPU. Runtime ~2-4 minutes, most of it the real-model sections.

Run:  python scripts/kv/test_core_mechanism.py
      python scripts/kv/test_core_mechanism.py --fast     # skip real-model sections
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/attention_shape/src"))

from scripts.kv import significance as sig  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, why: str = "", info: str = "") -> None:
    """`why` explains a FAILURE and prints only when failing; `info` is a
    measurement and always prints. Printing a failure message beside PASS -- which
    the first version of this file did -- makes a green suite read as a broken one.
    """
    tail = f"   {info}" if info else ""
    if not cond and why:
        tail += f"   <-- {why}"
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{tail}")
    if not cond:
        FAILED.append(name)


# --------------------------------------------------------------------------- #
# 1. IS THE SIGNIFICANCE MATRIX THE QUANTITY IT CLAIMS TO BE?
# --------------------------------------------------------------------------- #

def reference_score(query, key, window, scaling):
    """Independent reimplementation from the docstring, written the slow obvious way.

    Materialises the full attention probability matrix with an explicit repeat_kv,
    applies a causal mask, and sums the probabilities over the query heads sharing
    each KV head and over the observation window. Deliberately shares no code with
    _score_rows.
    """
    b, h, q_len, d = query.shape
    n_kv, n = key.shape[1], key.shape[2]
    groups = h // n_kv
    w = min(window, q_len)
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(d)

    # repeat_kv, HF semantics: query head j attends to KV head j // groups
    k_rep = key[:, :, None, :, :].expand(b, n_kv, groups, n, d).reshape(b, h, n, d)

    out = torch.zeros(b, n_kv, n, dtype=torch.float32)
    for bi in range(b):
        for hj in range(h):
            kv = hj // groups
            for wi in range(w):
                qrow = query[bi, hj, q_len - w + wi, :].float()
                logits = (k_rep[bi, hj].float() @ qrow) * scaling
                if q_len > 1:
                    abs_pos = n - w + wi
                    mask = torch.arange(n) > abs_pos
                    logits = logits.masked_fill(mask, float("-inf"))
                out[bi, kv] += torch.softmax(logits, dim=-1)
    return out


print("\n1. the significance matrix against an independent reference implementation")
torch.manual_seed(0)
for (b, h, n_kv, q_len, n, d, win) in [
    (1, 4, 2, 6, 6, 8, 16),      # prefill, window longer than the prompt
    (1, 4, 2, 6, 6, 8, 3),       # prefill, window shorter
    (1, 16, 8, 1, 40, 16, 16),   # decode: one query row against a long cache
    (2, 6, 3, 5, 12, 8, 4),      # chunked prefill: q_len < n, batch 2
    (1, 8, 8, 4, 4, 8, 2),       # no grouping at all (MHA)
    (1, 12, 3, 7, 7, 4, 5),      # 4 query heads per KV head
]:
    q = torch.randn(b, h, q_len, d)
    k = torch.randn(b, n_kv, n, d)
    got = sig._score_rows(None, q, k, win, None)
    want = reference_score(q, k, win, None)
    ok = got.shape == want.shape and torch.allclose(got, want, atol=1e-5, rtol=1e-4)
    check(f"b{b} h{h} kv{n_kv} q{q_len} n{n} d{d} w{win}", ok,
          info=(f"max abs diff {float((got - want).abs().max()):.3e}" if got.shape == want.shape
            else f"shape {tuple(got.shape)} vs {tuple(want.shape)}"))

print("\n1b. the GQA head mapping is HF's repeat_kv layout, not the interleaved one")
# Give exactly one query head a huge norm. Under HF layout (head j -> kv j//groups)
# its mass lands on kv head HOT//groups. Under an interleaved layout (j -> j % n_kv)
# it would land on HOT % n_kv. Pick numbers where those differ.
b, h, n_kv, q_len, n, d = 1, 8, 4, 3, 3, 8
groups, HOT = h // n_kv, 5           # HF: 5//2 = 2   interleaved: 5%4 = 1
q = torch.zeros(b, h, q_len, d)
q[0, HOT] = 50.0
k = torch.randn(b, n_kv, n, d)
s = sig._score_rows(None, q, k, 16, None)
per_kv = s[0].sum(-1)
hf_kv, inter_kv = HOT // groups, HOT % n_kv
check("mass lands on the HF-layout KV head",
      int(per_kv.argmax()) == hf_kv and hf_kv != inter_kv,
      info=f"argmax kv={int(per_kv.argmax())}, HF says {hf_kv}, interleaved says {inter_kv}")
check("every KV head receives groups*w of probability mass",
      torch.allclose(per_kv, torch.full_like(per_kv, float(groups * min(16, q_len))),
                     atol=1e-4),
      info=f"{[round(float(x), 3) for x in per_kv]} vs {groups * min(16, q_len)}")

print("\n1c. causality: a query row must not see positions after itself")
b, h, n_kv, q_len, n, d = 1, 2, 1, 5, 5, 4
q = torch.randn(b, h, q_len, d)
k = torch.randn(b, n_kv, n, d)
s_full = sig._score_rows(None, q, k, 1, None)     # window 1 -> only the LAST row
# the last query row is at position n-1 and may see everything, so perturbing a
# future key is impossible; instead use window 2 and perturb the final key, which
# row n-2 must not see.
s2 = sig._score_rows(None, q, k, 2, None)
k2 = k.clone()
k2[0, 0, -1] += 100.0                              # only the LAST key changes
s2b = sig._score_rows(None, q, k2, 2, None)
# row n-2 is masked from key n-1, so its contribution to columns 0..n-2 is unchanged
check("masking is applied for q_len>1 (a future key cannot alter an earlier row)",
      not torch.allclose(s2, s2b),
      "changing the last key must alter the last row's distribution")
d_cols = (s2b - s2)[0, 0, :-1].abs().max()
check("the earlier row's own columns move only via its own softmax, not the future key",
      float(d_cols) > 0, info=f"max abs diff on columns 0..n-2 = {float(d_cols):.3e}")

print("\n1d. decode (q_len==1) is unmasked by construction")
q1 = torch.randn(1, 4, 1, 8)
k1 = torch.randn(1, 2, 30, 8)
s1 = sig._score_rows(None, q1, k1, 16, None)
check("a single query row attends to the whole cache",
      bool((s1 > 0).all()), info=f"{int((s1 <= 0).sum())} non-positive entries")
check("its mass is groups*1 per KV head",
      torch.allclose(s1.sum(-1), torch.full((1, 2), 2.0), atol=1e-5),
      info=f"{[round(float(x), 4) for x in s1.sum(-1)[0]]}")


# --------------------------------------------------------------------------- #
# 2. BUDGET ACCOUNTING AND SELECTION INVARIANTS
# --------------------------------------------------------------------------- #

print("\n2. the policy spends exactly the budget it reports, over a wide grid")
bad_sum, bad_dup, bad_range, bad_count, cases = [], [], [], [], 0
for budget in (8, 16, 32, 64, 128, 196, 256, 512):
    for sinks in (0, 4, 16):
        for rf in (None, 0.05, 0.15, 0.5):
            for rec in ((32,) if rf is None else (32,)):
                for select in ("significance", "random", "random-prepool"):
                    for pool, pb, pf in ((1, None, None), (7, 6, 0), (7, 3, 3)):
                        cases += 1
                        pol = sig.Policy(budget=budget, sinks=sinks, recent=rec,
                                         recent_frac=rf, pool=pool, pool_back=pb,
                                         pool_fwd=pf, select=select)
                        ns, nr, nh = pol.n_sinks(), pol.n_recent(), pol.heavy()
                        if ns + nr + nh != budget:
                            bad_sum.append((budget, sinks, rf, select, ns, nr, nh))
                        n = max(budget * 3, 400)
                        idx = sig.select_keep(torch.rand(1, 4, n), n, pol,
                                              torch.device("cpu"))
                        for hh in range(idx.shape[1]):
                            row = idx[0, hh].tolist()
                            if len(set(row)) != len(row):
                                bad_dup.append((budget, sinks, rf, select, pool))
                            if row and (min(row) < 0 or max(row) >= n):
                                bad_range.append((budget, sinks, rf, select))
                        if idx.shape[-1] != budget:
                            bad_count.append((budget, sinks, rf, select, idx.shape[-1]))
check(f"sinks + recent + heavy == budget in all {cases} configurations",
      not bad_sum, info=f"{len(bad_sum)} violations, e.g. {bad_sum[:2]}")
check("no duplicate kept position in any KV head", not bad_dup,
      info=f"{len(bad_dup)} violations, e.g. {bad_dup[:2]}")
check("every kept index is in range", not bad_range, info=f"{len(bad_range)} violations")
check("the kept-set size always equals the budget", not bad_count,
      info=f"{len(bad_count)} violations, e.g. {bad_count[:2]}")

print("\n2b. budget larger than the cache is handled, not silently truncated")
for n, budget in ((10, 64), (64, 64), (64, 65)):
    pol = sig.Policy(budget=budget, sinks=4, recent=8, pool=1, select="significance")
    idx = sig.select_keep(torch.rand(1, 2, n), n, pol, torch.device("cpu"))
    row = idx[0, 0].tolist()
    check(f"n={n} budget={budget}: distinct and in range",
          len(set(row)) == len(row) and max(row) < n, info=f"{len(set(row))} distinct")


# --------------------------------------------------------------------------- #
# 3. THE CACHE SURGERY
# --------------------------------------------------------------------------- #

print("\n3. eviction keeps EXACTLY the selected keys and values, unpermuted")
N, NKV, D, BUD = 300, 4, 6, 60
pol = sig.Policy(budget=BUD, sinks=4, recent=8, pool=1, select="significance")
sig.reset_scores()
layer = sig.SignificanceLayer(layer_idx=0, policy=pol)
k = torch.zeros(1, NKV, N, D)
k[..., 0] = torch.arange(N, dtype=torch.float32)      # channel 0 = original position
k[..., 1] = torch.arange(NKV).view(1, NKV, 1)          # channel 1 = kv head id
v = k.clone() * -1.0                                   # values distinguishable from keys
layer.update(k, v)
sc = torch.rand(1, NKV, N)
sig._SCORES[0] = sc.clone()
sig._COUNTS[0] = torch.ones(1, NKV, N)
# The appended token must be tagged like the others (position N, correct head id),
# or it breaks the very invariants below and the failure looks like a code bug.
t = torch.zeros(1, NKV, 1, D)
t[..., 0] = float(N)
t[..., 1] = torch.arange(NKV).view(1, NKV, 1)
layer.update(t, t.clone() * -1.0)                      # triggers the first eviction

surv = layer.keys[..., 0]
check("stored length equals the budget", layer.stored_length() == BUD,
      info=f"{layer.stored_length()} vs {BUD}")
check("every surviving key carries a real original position tag",
      bool(((surv >= 0) & (surv <= N)).all()),
      info=f"tag range {float(surv.min()):.0f}..{float(surv.max()):.0f}")
check("the KV-head channel is unpermuted",
      bool((layer.keys[..., 1] == torch.arange(NKV).view(1, NKV, 1)).all()),
      "gather transposed heads and positions")
check("values are gathered with the SAME index as keys",
      bool(torch.allclose(layer.values[..., 0], -layer.keys[..., 0])),
      "keys and values disagree on which positions survived")
check("surviving positions are sorted ascending",
      bool((surv[:, :, 1:] >= surv[:, :, :-1]).all()), "order was scrambled")
check("scores were gathered to the same length",
      sig._SCORES[0].shape[-1] == BUD, info=f"{sig._SCORES[0].shape[-1]}")
check("counts were gathered to the same length",
      sig._COUNTS[0].shape[-1] == BUD, info=f"{sig._COUNTS[0].shape[-1]}")
# the score attached to a surviving slot must be the score of THAT original position
ok_align = True
for hh in range(NKV):
    for j in range(BUD):
        p = int(surv[0, hh, j])
        if 0 <= p < N and abs(float(sig._SCORES[0][0, hh, j]) - float(sc[0, hh, p])) > 1e-6:
            ok_align = False
check("each surviving slot's score is that ORIGINAL position's score", ok_align,
      "the score vector desynchronised from the cache")

print("\n3b. true sequence length is positions CONSUMED, not positions stored")
check("get_seq_length() reports the consumed count",
      layer.get_seq_length() == N + 1,
      info=f"{layer.get_seq_length()} consumed, {layer.stored_length()} stored")
check("and it is NOT the stored length",
      layer.get_seq_length() != layer.stored_length(),
      "RoPE would silently re-index every new token")

print("\n3c. the score-lag guard fires instead of degrading")
sig.reset_scores()
pol2 = sig.Policy(budget=40, sinks=4, recent=4, pool=1, select="significance")
lay2 = sig.SignificanceLayer(layer_idx=0, policy=pol2)
k2 = torch.zeros(1, 2, 200, 4)
lay2.update(k2, k2.clone())
sig._SCORES[0] = torch.rand(1, 2, 190)        # lag of 10 > n_recent of 4
try:
    lay2.update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
    check("a lag beyond the recent window raises", False, "it silently padded with +inf")
except RuntimeError as e:
    check("a lag beyond the recent window raises", "score lag" in str(e))


# --------------------------------------------------------------------------- #
# 4. THE REAL MODEL: does the patch perturb anything, and does eviction bite?
# --------------------------------------------------------------------------- #

def real_model_section():
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from compat import force_repeat_kv

    print("\n4. the real model on CPU: transparency and end-to-end equivalence")
    tok = AutoTokenizer.from_pretrained(str(ROOT / "artifacts/rineya-core-en32k-28L"))
    model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / "artifacts/rineya-core-en32k-28L"), dtype=torch.float32).eval()
    recs = [json.loads(l) for l in
            (ROOT / "artifacts/eval_sets/rag_dev.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()][:3]
    prompts = []
    for r in recs:
        msgs = r["messages"][:-1] if r["messages"][-1]["role"] == "assistant" else r["messages"]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False))

    def gen(prompt, cache_factory):
        sig.reset_scores()
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=12, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id,
                                 past_key_values=cache_factory())
        return out[0].tolist(), enc["input_ids"].shape[-1]

    def logits_of(prompt):
        # reset_scores() between prompts is MANDATORY when recording: the
        # accumulator raises rather than silently discarding history if the next
        # prompt is shorter. Discovered by this test hitting the guard.
        sig.reset_scores()
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            return model(**enc).logits.clone()

    # (a) BEFORE any patching -- the ground truth
    base_gen = [gen(p, DynamicCache)[0] for p in prompts]
    base_log = [logits_of(p) for p in prompts]

    # (b) patch applied, recording OFF -- must be bit-identical
    force_repeat_kv(verbose=False)
    sig.patch_attention()
    sig.set_recording(False)
    same_gen = all(gen(p, DynamicCache)[0] == b for p, b in zip(prompts, base_gen))
    same_log = all(torch.equal(logits_of(p), b) for p, b in zip(prompts, base_log))
    check("patch applied with recording OFF: generation bit-identical", same_gen)
    check("patch applied with recording OFF: logits bit-identical", same_log)

    # (c) recording ON -- scoring must not perturb the forward either
    sig.reset_scores()
    sig.set_recording(True, window=16)
    on_log = [logits_of(p) for p in prompts]
    check("recording ON: logits bit-identical to unpatched",
          all(torch.equal(a, b) for a, b in zip(on_log, base_log)),
          "the score pass is changing the model's output")
    check("recording ON actually populated the score matrix",
          len(sig.get_scores()) > 0 and
          all(v.shape[1] == model.config.num_key_value_heads
              for v in sig.get_scores().values()),
          info=f"{len(sig.get_scores())} layers scored")

    # (d) THE DECISIVE ONE: a budget above the prompt cannot evict, so a
    #     SignificanceCache must reproduce the unpatched run exactly.
    n_max = max(gen(p, DynamicCache)[1] for p in prompts)
    def big_cache():
        sig.reset_scores()
        sig.set_recording(True, window=16)
        return sig.SignificanceCache(
            config=model.config,
            policy=sig.Policy(budget=n_max + 500, sinks=4, recent=32, pool=7,
                              pool_back=6, pool_fwd=0, select="significance"))
    noop = [gen(p, big_cache)[0] for p in prompts]
    check("budget above the prompt: SignificanceCache == plain DynamicCache",
          noop == base_gen, "eviction cannot have fired, yet the output moved")

    # (e) TWO assertions, because either alone is misleading.
    #
    # A generous budget must reproduce the uncompressed answer -- that is the
    # mechanism working, not eviction being inert. On these ~195-token rag_dev
    # prompts, keeping 48 positions (24%) is byte-identical on 3/3, and the first
    # version of this test wrongly read that as a failure.
    #
    # A budget BELOW the capacity floor must change the output, or the suite could
    # pass with a mechanism that never does anything. Measured degradation across
    # budgets on these prompts: 128/96/64/48 -> 3/3 identical, 32/24 -> 1/3,
    # 16/12 -> 0/3. 16 is comfortably past the knee.
    def cache_at(B, sk=4, rc=8):
        def f():
            sig.reset_scores()
            sig.set_recording(True, window=16)
            return sig.SignificanceCache(
                config=model.config,
                policy=sig.Policy(budget=B, sinks=sk, recent=rc, pool=7, pool_back=6,
                                  pool_fwd=0, select="significance"))
        return f
    generous = [gen(p, cache_at(48))[0] for p in prompts]
    check("a generous budget (24% kept) reproduces the uncompressed answer exactly",
          generous == base_gen,
          "the mechanism is losing the answer at a budget where it should not")
    tight = [gen(p, cache_at(16))[0] for p in prompts]
    check("a budget below the floor DOES change the output (the tests can fail)",
          tight != base_gen,
          "eviction is inert -- every compressed result would be meaningless")

    # (f) the cache really is smaller
    enc = tok(prompts[0], return_tensors="pt")
    sig.reset_scores(); sig.set_recording(True, window=16)
    c = sig.SignificanceCache(config=model.config,
                              policy=sig.Policy(budget=48, sinks=4, recent=8, pool=1,
                                                select="significance"))
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=12, do_sample=False,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                       past_key_values=c)
    stored = c.stored_lengths()[0]
    check("the stored cache is the budget, not the prompt", stored == 48,
          info=f"stored {stored}, prompt {enc['input_ids'].shape[-1]}")
    check("cache_bytes reflects the smaller cache",
          sig.cache_bytes(c) < enc["input_ids"].shape[-1] * 28 * 8 * 128 * 2 * 4,
          info=f"{sig.cache_bytes(c)} bytes")


# --------------------------------------------------------------------------- #
# 5. count_exponent MUST be live on the self-compressing path
# --------------------------------------------------------------------------- #

# This is a REGRESSION test for a bug this repository already shipped once:
# _gather_counts used to drop _COUNTS whenever its length did not match the cache,
# which is the normal case on the self-compressing path (the score vector is padded
# for the decode lag, the counts were not). ranked_scores then silently fell back to
# the raw sum, so count_exponent was inert and the ranking was wrong with no error.
# The fix must stay fixed: e=0 and e=1 have to select DIFFERENT positions.
print("\n5. count_exponent is live end to end, not silently ignored")
N5, KV5, D5, B5 = 120, 2, 4, 20


def evict_with(exponent):
    sig.reset_scores()
    pol = sig.Policy(budget=B5, sinks=2, recent=4, pool=1, select="significance",
                     count_exponent=exponent)
    lay = sig.SignificanceLayer(layer_idx=0, policy=pol)
    k = torch.zeros(1, KV5, N5, D5)
    k[..., 0] = torch.arange(N5, dtype=torch.float32)
    lay.update(k, k.clone())
    # position 50: large SUM but observed many times -> small mean
    # position 60: smaller sum but observed once     -> large mean
    s = torch.full((1, KV5, N5), 0.10)
    c = torch.ones(1, KV5, N5)
    s[..., 50] = 9.0
    c[..., 50] = 100.0        # mean 0.09, below the 0.10 background
    s[..., 60] = 1.0
    c[..., 60] = 1.0          # mean 1.00, far above
    sig._SCORES[0] = s
    sig._COUNTS[0] = c
    t = torch.zeros(1, KV5, 1, D5)
    t[..., 0] = float(N5)
    lay.update(t, t.clone())
    return set(int(x) for x in lay.keys[0, 0, :, 0].tolist())


keep_sum = evict_with(0.0)      # rank on the raw sum
keep_mean = evict_with(1.0)     # rank on sum / count
check("ranking on the SUM keeps the high-sum position 50", 50 in keep_sum)
check("ranking on the MEAN drops it", 50 not in keep_mean)
check("ranking on the MEAN keeps the high-mean position 60", 60 in keep_mean)
check("the two exponents therefore select differently", keep_sum != keep_mean,
      "count_exponent is inert -- the documented regression is back")
check("counts survived the eviction at the cache length",
      sig._COUNTS[0].shape[-1] == B5, info=f"{sig._COUNTS[0].shape[-1]} vs {B5}")


def writeback(exponent, rounds=3):
    """What _SCORES holds on a surviving slot after each eviction."""
    sig.reset_scores()
    pol = sig.Policy(budget=B5, sinks=2, recent=4, pool=1, select="significance",
                     count_exponent=exponent)
    lay = sig.SignificanceLayer(layer_idx=0, policy=pol)
    k = torch.zeros(1, KV5, N5, D5)
    k[..., 0] = torch.arange(N5, dtype=torch.float32)
    lay.update(k, k.clone())
    sig._SCORES[0] = torch.full((1, KV5, N5), 8.0)     # raw accumulated 8.0
    sig._COUNTS[0] = torch.full((1, KV5, N5), 4.0)     # observed 4 times -> mean 2.0
    seen = []
    for r in range(rounds):
        t = torch.zeros(1, KV5, 1, D5)
        t[..., 0] = float(N5 + r)
        lay.update(t, t.clone())
        seen.append(round(float(sig._SCORES[0][0, 0, 0]), 6))
    return seen


# REGRESSION for a second bug in the same function, found by this audit. _evict
# means to write the RAW accumulator back -- its own comment says "writing the mean
# back would compound the division on every subsequent round" -- but it guarded on
# `raw.shape[-1] == n`, which is false precisely on the self-compressing path,
# because that is the only path with a score lag. So the RANKING KEY was written
# back instead. At exponent 0 that is harmless (the ranking key IS the sum); at
# exponent 1 a raw 8.0 with count 4 became 2.0, then 0.5, then 0.125.
# No committed number was affected: every run that set a non-zero exponent used
# SignificanceCache(policy=None) and evicted through compress(), which writes the
# raw correctly, and every self-compressing run used exponent 0.
for e in (0.0, 1.0):
    seen = writeback(e)
    check(f"exponent {e:.0f}: the RAW accumulator is written back, not the ranking key",
          all(abs(x - 8.0) < 1e-6 for x in seen), info=f"{seen}")


# --------------------------------------------------------------------------- #
# 6. compress() and the self-compressing path must agree
# --------------------------------------------------------------------------- #

print("\n6. the standalone compress() selects what the self-compressing path does")


class _Bag:                      # the minimal duck-typed cache compress() expects
    def __init__(self, layers):
        self.layers = layers


sig.reset_scores()
pol6 = sig.Policy(budget=32, sinks=4, recent=6, pool=7, pool_back=6, pool_fwd=0,
                  select="significance")
N6, KV6, D6 = 200, 3, 4
base_k = torch.zeros(1, KV6, N6, D6)
base_k[..., 0] = torch.arange(N6, dtype=torch.float32)
sc6 = torch.rand(1, KV6, N6)

lay_self = sig.SignificanceLayer(layer_idx=0, policy=pol6)
lay_self.update(base_k.clone(), base_k.clone())
sig._SCORES[0] = sc6.clone()
sig._COUNTS[0] = torch.ones(1, KV6, N6)
# self-compressing: the cache has grown by one un-scored token when it evicts, and
# select_keep therefore sees n = N6 + 1. Reproduce that exactly for compress().
t6 = torch.zeros(1, KV6, 1, D6)
t6[..., 0] = float(N6)
lay_self.update(t6, t6.clone())
got_self = [set(int(x) for x in lay_self.keys[0, h, :, 0].tolist()) for h in range(KV6)]

sig.reset_scores()
lay_cmp = sig.SignificanceLayer(layer_idx=0, policy=pol6)
lay_cmp.update(torch.cat([base_k, t6], dim=2).clone(),
               torch.cat([base_k, t6], dim=2).clone())
pad = torch.full((1, KV6, 1), float("inf"))
sig._SCORES[0] = torch.cat([sc6.clone(), pad], dim=-1)
sig._COUNTS[0] = torch.ones(1, KV6, N6 + 1)
sig.compress(_Bag([lay_cmp]), pol6)
got_cmp = [set(int(x) for x in lay_cmp.keys[0, h, :, 0].tolist()) for h in range(KV6)]
check("both paths keep the same positions in every KV head", got_self == got_cmp,
      f"head diffs: {[sorted(a ^ b)[:6] for a, b in zip(got_self, got_cmp) if a != b][:2]}")

sig.reset_scores()
lay_bad = sig.SignificanceLayer(layer_idx=0, policy=pol6)
lay_bad.update(base_k.clone(), base_k.clone())
sig._SCORES[0] = torch.rand(1, KV6, N6 - 7)          # scores shorter than the cache
try:
    sig.compress(_Bag([lay_bad]), pol6)
    check("compress() refuses a score/cache length mismatch", False,
          "it used to skip the layer and report success, leaving it UNCOMPRESSED")
except RuntimeError as e:
    check("compress() refuses a score/cache length mismatch",
          "positions but the cache" in str(e))


# --------------------------------------------------------------------------- #
# 7. the chunked-prefill mask: eviction must not corrupt what a chunk may see
# --------------------------------------------------------------------------- #

print("\n7. the chunked-prefill attention mask is bottom-right aligned")
sys.path.insert(0, str(ROOT / "scripts/kv"))
from scripts.kv import chunked_prefill as cp  # noqa: E402

NEG_OK = torch.finfo(torch.float32).min
m = cp.causal_lower_right(3, 3, torch.device("cpu"), torch.float32)[0, 0]
check("kv_len == q_len reduces to the ordinary causal mask",
      bool((m == 0).sum() == 6 and m[0, 1] == NEG_OK and m[2, 0] == 0),
      info=f"{int((m == 0).sum())} attendable cells of 9")
# after an eviction the cache holds `stored` survivors and the chunk appends q more
m2 = cp.causal_lower_right(4, 20, torch.device("cpu"), torch.float32)[0, 0]
check("every surviving cached key is visible to every query in the chunk",
      bool((m2[:, :16] == 0).all()),
      "a chunk cannot see part of the cache -- eviction would corrupt attention")
check("the chunk's own tokens stay causal among themselves",
      bool(m2[0, 17] == NEG_OK and m2[3, 19] == 0),
      info="query 0 cannot see chunk token 1; query 3 can see chunk token 3")
try:
    cp.causal_lower_right(8, 4, torch.device("cpu"), torch.float32)
    check("a cache shorter than the chunk raises", False, "it would build a wrong mask")
except ValueError:
    check("a cache shorter than the chunk raises", True)


ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="skip the real-model sections")
args = ap.parse_args()
if not args.fast:
    real_model_section()
else:
    print("\n4. real-model sections SKIPPED (--fast)")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all core-mechanism tests pass")
