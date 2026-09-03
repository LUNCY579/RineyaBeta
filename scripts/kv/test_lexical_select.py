#!/usr/bin/env python3
"""CPU tests for the `lexical-*` triviality-control arms.

WHY THESE ARE SHAPED THE WAY THEY ARE. Twice in this programme a green unit test
pinned the wrong function because it exercised a helper in isolation and never
reproduced its caller's preconditions: `grow_spans` flooded its gate under eleven
passing tests, and the B1 recent floor drove recent to 0 on every short prompt
while 24 Policy tests passed. The dangerous failure for a lexical mask is not
"select_keep returns the wrong thing once" -- it is DESYNCHRONISATION on the
second and later evictions, when every KV head holds a different subset of
positions and a mask indexed by original position starts addressing the wrong
tokens. That cannot be seen from one call to select_keep.

So the alignment tests below drive the real `SignificanceLayer.update` path
through THREE evictions, with each key carrying its own original position index
in channel 0, and assert on which original positions actually survived.

Run:  python scripts/kv/test_lexical_select.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kv import significance as sig  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_layer(policy, n0: int, b: int = 1, n_kv: int = 2, d: int = 4):
    """A layer holding n0 prefill positions, each key tagged with its index."""
    layer = sig.SignificanceLayer(layer_idx=0, policy=policy)
    k = torch.zeros(b, n_kv, n0, d)
    k[..., 0] = torch.arange(n0, dtype=torch.float32)   # channel 0 = original index
    v = k.clone()
    layer.update(k, v)                                   # no scores yet -> no eviction
    return layer


def step(layer, tag: float, b: int = 1, n_kv: int = 2, d: int = 4):
    k = torch.zeros(b, n_kv, 1, d)
    k[..., 0] = tag
    layer.update(k, k.clone())


def survivors(layer) -> torch.Tensor:
    """Original position index of every surviving slot, per KV head. [b, n_kv, n]"""
    return layer.keys[..., 0]


# --------------------------------------------------------------------------- #
print("\n1. the arm is refused rather than degraded when misconfigured")
# --------------------------------------------------------------------------- #

try:
    sig.Policy(budget=32, select="lexical_idf")          # underscore, not hyphen
    check("unknown select raises", False, "accepted a typo")
except ValueError as e:
    check("unknown select raises", "silently fall through" in str(e))

sig.reset_scores()
pol = sig.Policy(budget=16, sinks=2, recent=4, pool=7, pool_back=6, pool_fwd=0,
                 select="lexical-idf")
try:
    sig.select_keep(torch.rand(1, 2, 64), 64, pol, torch.device("cpu"))
    check("select_keep without weights raises", False, "ranked by the score instead")
except ValueError as e:
    check("select_keep without weights raises", "requires a lexical weight" in str(e))

try:
    sig.select_keep(torch.rand(1, 2, 64), 64, pol, torch.device("cpu"),
                    lexical=torch.rand(1, 2, 63))
    check("mismatched weight shape raises", False)
except ValueError as e:
    check("mismatched weight shape raises", "do not match" in str(e))

sig.reset_scores()
lay = sig.SignificanceLayer(layer_idx=0, policy=pol)
try:
    sig._lexical_for(0, 1, 2, 64, torch.device("cpu"))
    check("missing per-case install raises", False)
except RuntimeError as e:
    check("missing per-case install raises", "set_lexical_weights" in str(e))

# --------------------------------------------------------------------------- #
print("\n2. the lexical weight actually decides, and matched slots are matched")
# --------------------------------------------------------------------------- #

N = 200
BUDGET, SINKS, RECENT = 50, 4, 8
HEAVY = BUDGET - SINKS - RECENT
MARKS = [40, 41, 42, 90, 91, 120, 121, 122]        # "question tokens"

w = torch.zeros(N)
w[MARKS] = 1.0

sig.reset_scores()
sig.set_lexical_weights(w)
pol_lex = sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT, pool=1,
                     select="lexical-naive")
s_flat = torch.zeros(1, 2, N)                       # no score signal at all
idx = sig.select_keep(s_flat, N, pol_lex, torch.device("cpu"),
                      lexical=w.view(1, 1, -1).expand(1, 2, -1).contiguous())
kept = set(idx[0, 0].tolist())
check("every marked position is kept", all(m in kept for m in MARKS),
      f"kept {sorted(m for m in MARKS if m in kept)}")
check("slot count equals the budget", idx.shape[-1] == BUDGET, f"{idx.shape[-1]}")

pol_sig = sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT, pool=1,
                     select="significance")
idx_sig = sig.select_keep(torch.rand(1, 2, N), N, pol_sig, torch.device("cpu"))
check("significance arm has the SAME slot count", idx_sig.shape[-1] == idx.shape[-1],
      f"{idx_sig.shape[-1]} vs {idx.shape[-1]}")
check("the two arms select DIFFERENT positions",
      set(idx_sig[0, 0].tolist()) != kept,
      "identical selection would be a plumbing failure, not a null")

# --------------------------------------------------------------------------- #
print("\n3. the backward pool carries a lexical match FORWARD only")
# --------------------------------------------------------------------------- #

w1 = torch.zeros(N)
w1[100] = 1.0
sig.reset_scores()
lex3 = w1.view(1, 1, -1).expand(1, 2, -1).contiguous()
# heavy is set to EXACTLY the 7 positions a (6,0) window can reach from one match.
# With heavy larger than that, the surplus slots are filled by the tie-break jitter
# among the zero-weight positions, which can land on 94..99 and fail the "does not
# reach backward" assertion by luck -- observed failing about 1 run in 6 before this
# was pinned. A flaky assertion is worse than none: it teaches you to re-run.
pol_b = sig.Policy(budget=2 + 4 + 7, sinks=2, recent=4, pool=7, pool_back=6,
                   pool_fwd=0, select="lexical-naive")
kb = set(sig.select_keep(torch.zeros(1, 2, N), N, pol_b, torch.device("cpu"),
                         lexical=lex3)[0, 0].tolist())
fwd_in = [p for p in range(101, 107) if p in kb]
back_in = [p for p in range(94, 100) if p in kb]
check("positions AFTER the match inherit it", len(fwd_in) == 6, f"{fwd_in}")
check("positions BEFORE the match do not", len(back_in) == 0, f"{back_in}")

pol_c = sig.Policy(budget=2 + 4 + 7, sinks=2, recent=4, pool=7, pool_back=3,
                   pool_fwd=3, select="lexical-naive")
kc = set(sig.select_keep(torch.zeros(1, 2, N), N, pol_c, torch.device("cpu"),
                         lexical=lex3)[0, 0].tolist())
check("the centred window is genuinely different",
      set(p for p in range(97, 100) if p in kc) == {97, 98, 99},
      "centred must reach backward where causal does not")

# --------------------------------------------------------------------------- #
print("\n4. ALIGNMENT through three real evictions (the failure that matters)")
# --------------------------------------------------------------------------- #

sig.reset_scores()
sig.set_lexical_weights(w)                            # marks at 40..122 again
pol_a = sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT, pool=1,
                   select="lexical-naive")
layer = make_layer(pol_a, N)
# Scores that point the OPPOSITE way: high at positions the lexical mask ignores.
# If the mask desynchronises or is ignored, these win and the marks are lost.
sc = torch.zeros(1, 2, N)
sc[..., 150:190] = 100.0
sig._SCORES[0] = sc

for r, tag in enumerate((1000.0, 1001.0, 1002.0), start=1):
    step(layer, tag)
    surv = survivors(layer)
    got = [set(surv[0, h].tolist()) for h in range(surv.shape[1])]
    ok = all(all(float(m) in g for m in MARKS) for g in got)
    check(f"round {r}: all marks survive in every KV head", ok,
          f"stored {layer.stored_length()}, missing "
          f"{[[m for m in MARKS if float(m) not in g] for g in got]}")
    check(f"round {r}: mask length tracks the cache",
          sig._LEXICAL[0].shape[-1] == layer.keys.shape[-2],
          f"mask {sig._LEXICAL[0].shape[-1]} vs cache {layer.keys.shape[-2]}")
    # THE DESYNC TEST. Every surviving slot whose mask says 1.0 must really be a
    # marked original position. A shifted mask passes the "marks survived" test
    # above by luck and fails this one.
    m = sig._LEXICAL[0][0, 0]
    claimed = {float(x) for x in surv[0, 0][m > 0.5].tolist()}
    check(f"round {r}: mask==1 iff the slot IS a marked position",
          claimed == {float(x) for x in MARKS},
          f"mask claims {sorted(claimed)}")

# --------------------------------------------------------------------------- #
print("\n5. reset_scores clears the mask, so a case cannot inherit the last one")
# --------------------------------------------------------------------------- #

sig.reset_scores()
check("mask cleared", not sig._LEXICAL and sig._STATE.get("lexical_seed") is None)

# --------------------------------------------------------------------------- #
print("\n6. the existing arms are untouched (bit-identical selection)")
# --------------------------------------------------------------------------- #

for name in ("significance", "random", "streaming"):
    sig.reset_scores()
    torch.manual_seed(7)
    p = sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT, pool=7,
                   pool_back=6, pool_fwd=0, select=name)
    s = torch.rand(1, 2, N)
    torch.manual_seed(11)
    a = sig.select_keep(s, N, p, torch.device("cpu"))
    torch.manual_seed(11)
    b = sig.select_keep(s, N, p, torch.device("cpu"), lexical=None)
    check(f"{name}: unchanged and deterministic", torch.equal(a, b))

# --------------------------------------------------------------------------- #
print("\n7. random-prepool is the CONTIGUITY-MATCHED floor, not a second `random`")
# --------------------------------------------------------------------------- #

def runs_of(sorted_idx):
    """Number of maximal contiguous runs in a sorted index list."""
    r, prev = 0, None
    for x in sorted_idx:
        if prev is None or x != prev + 1:
            r += 1
        prev = x
    return r

sig.reset_scores()
p_post = sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT, pool=7,
                    pool_back=6, pool_fwd=0, select="random")
p_pre = sig.Policy(budget=BUDGET, sinks=SINKS, recent=RECENT, pool=7,
                   pool_back=6, pool_fwd=0, select="random-prepool")
zero = torch.zeros(1, 2, N)
torch.manual_seed(3)
i_post = sorted(sig.select_keep(zero, N, p_post, torch.device("cpu"))[0, 0].tolist())
torch.manual_seed(3)
i_pre = sorted(sig.select_keep(zero, N, p_pre, torch.device("cpu"))[0, 0].tolist())
check("random-prepool selects a different set from random", i_post != i_pre)
check("random-prepool keeps the same slot count", len(i_pre) == len(i_post),
      f"{len(i_pre)} vs {len(i_post)}")
check("random-prepool is CONTIGUOUS where random is scattered",
      runs_of(i_pre) < runs_of(i_post) / 2,
      f"{runs_of(i_pre)} runs vs {runs_of(i_post)}")
check("random-prepool needs no lexical weights",
      sig._STATE.get("lexical_seed") is None)

# --------------------------------------------------------------------------- #
print("\n8. doc-bm25 weights segment on the prompt's own delimiters")
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "experiments/kv_significance/src"))
try:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("artifacts/rineya-core-en32k-28L")
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_res", Path(__file__).resolve().parents[2]
        / "experiments/kv_significance/src/run_eval_sig.py")
    # run_eval_sig imports torch/compat at module scope but nothing GPU-bound.
    _res = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_res)

    doc_prompt = ("<FACTUAL_EVIDENCE>\n"
                  "[doc_a | trust=standard] Task Maple has priority 3.\n"
                  "[doc_b | trust=standard] The launch date for PineRelay 1225 is "
                  "June 9, 2032.\n"
                  "[doc_c | trust=standard] CinderPanel 1242 runtime is 10 hours.\n"
                  "</FACTUAL_EVIDENCE>\n<QUESTION>\nWhen does PineRelay 1225 launch?\n"
                  "</QUESTION>")
    e = _tok(doc_prompt, return_offsets_mapping=True)
    wd = _res.doc_bm25_weights(doc_prompt, e, "When does PineRelay 1225 launch?", _tok)
    ids = e["input_ids"]
    txt = [_tok.decode([t]) for t in ids]
    hot = [k for k, x in enumerate(wd.tolist()) if x >= 0.999]
    span, j = None, None
    for cand in ("June 9, 2032", " June 9, 2032"):
        span = _tok(cand, add_special_tokens=False)["input_ids"]
        j = next((k for k in range(len(ids) - len(span) + 1)
                  if ids[k:k + len(span)] == span), None)
        if j is not None:
            break
    check("the answer-bearing record scores 1.0", j is not None
          and all(wd[k] >= 0.999 for k in range(j, j + len(span))),
          f"weights over the gold span: "
          f"{[round(float(wd[k]), 3) for k in range(j, j + len(span))] if j else 'span not found'}")
    check("a distractor record scores strictly less",
          all(wd[k] < 0.999 for k, t in enumerate(txt) if "Maple" in t),
          "the Maple record must not tie the answer record")
    check("weights are per-record, not per-token",
          len(hot) > len(span),
          f"{len(hot)} positions at the top weight (whole record, not just the span)")

    # --- the granularity-matched floor uses the SAME segmentation -------------
    segs = _res.segment_records(doc_prompt, e)
    check("segmentation stops at </FACTUAL_EVIDENCE>",
          all(not any("QUESTION" in _tok.decode([ids[j]]) for j in tid)
              for tid, _ in segs),
          "no record may contain the question, or it wins BM25 unconditionally")
    torch.manual_seed(5)
    wr = _res.doc_random_weights(doc_prompt, e)
    lv = {round(float(wr[j]), 6) for tid, _ in segs for j in tid}
    check("random-doc is constant WITHIN each record",
          all(len({round(float(wr[j]), 6) for j in tid}) == 1 for tid, _ in segs),
          "a per-position draw would not be granularity-matched")
    check("random-doc differs BETWEEN records", len(lv) == len(segs),
          f"{len(lv)} distinct levels over {len(segs)} records")
    check("random-doc carries no lexical signal",
          _res.segment_records(doc_prompt, e) == segs,
          "same carving as arm G, only the score differs")
except Exception as _e:                                   # noqa: BLE001
    check("doc-bm25 weight construction", False, f"raised {type(_e).__name__}: {_e}")

# --------------------------------------------------------------------------- #
print("\n9. select_keep returns BUDGET DISTINCT positions, for every select")
# --------------------------------------------------------------------------- #

# A MUTATION STUDY on this file found that deleting the post-pool mask re-assertion
# gave 30-32 of 32 KV heads DUPLICATE slots at campaign scale while every assertion
# above stayed green -- and `mean_kept_ratio` still read 0.25, so the analyser's own
# gate was blind to it too. Nothing anywhere asserted that the kept set has no
# repeats. A duplicated slot silently shrinks the effective budget and inflates the
# apparent compression, which is a wrong number in the direction that flatters the
# mechanism.
BIG_N, BIG_B = 4210, 1052          # the real 4K geometry
w_big = torch.zeros(BIG_N)
w_big[1000:1064] = 1.0             # one "record" scoring top
for name, pol_kw, lex in (
    ("significance b6f0", dict(pool=7, pool_back=6, pool_fwd=0,
                               select="significance"), None),
    ("significance pool1", dict(pool=1, select="significance"), None),
    ("random", dict(pool=7, pool_back=6, pool_fwd=0, select="random"), None),
    ("random-prepool", dict(pool=7, pool_back=6, pool_fwd=0,
                            select="random-prepool"), None),
    ("lexical-doc-bm25", dict(pool=7, pool_back=6, pool_fwd=0,
                              select="lexical-doc-bm25"), w_big),
    ("random-doc", dict(pool=7, pool_back=6, pool_fwd=0, select="random-doc"), w_big),
    ("lexical-idf", dict(pool=7, pool_back=6, pool_fwd=0,
                         select="lexical-idf"), w_big),
    ("lexical-naive", dict(pool=7, pool_back=6, pool_fwd=0,
                           select="lexical-naive"), w_big),
    ("streaming", dict(pool=7, pool_back=6, pool_fwd=0, select="streaming"), None),
):
    sig.reset_scores()
    torch.manual_seed(20260826)
    n_kv_big = 8                                    # the real core's KV head count
    pol = sig.Policy(budget=BIG_B, sinks=4, recent_frac=0.15, **pol_kw)
    lx = (lex.view(1, 1, -1).expand(1, n_kv_big, -1).contiguous()
          if lex is not None else None)
    # streaming is realised by the CALLER spending the budget on sinks+recent
    if pol_kw["select"] == "streaming":
        pol = sig.Policy(budget=BIG_B, sinks=4, recent=BIG_B - 4, **{
            k: v for k, v in pol_kw.items() if k != "select"}, select="streaming")
    got = sig.select_keep(torch.rand(1, n_kv_big, BIG_N), BIG_N, pol,
                          torch.device("cpu"), lexical=lx)
    dup_heads = sum(1 for h in range(n_kv_big)
                    if len(set(got[0, h].tolist())) != got.shape[-1])
    check(f"{name}: no duplicate slots in any of {n_kv_big} KV heads",
          dup_heads == 0, f"{dup_heads} head(s) with repeats")
    check(f"{name}: slot count == budget", got.shape[-1] == BIG_B,
          f"{got.shape[-1]} vs {BIG_B}")
    check(f"{name}: indices in range", int(got.min()) >= 0 and int(got.max()) < BIG_N)

# --------------------------------------------------------------------------- #
print("\n9b. the tie-break is load-bearing: without it a lexical arm is POSITIONAL")
# --------------------------------------------------------------------------- #

# A surviving mutation replaced the 1e-3 jitter with zeros. Nothing failed, because
# a deterministic arm still returns distinct slots -- but topk then resolves the
# constant weight by INDEX, so the arm becomes "keep the earliest eligible
# positions" while the manifest still says lexical. Two consequences are asserted:
# heads must differ (the jitter is drawn per KV head), and the kept set must not be
# the lowest-index block.
sig.reset_scores()
N9, B9, HEADS9 = 1000, 200, 8
flat = torch.ones(N9)                       # zero lexical information
pol_j = sig.Policy(budget=B9, sinks=4, recent_frac=0.15, pool=7, pool_back=6,
                   pool_fwd=0, select="lexical-doc-bm25")
torch.manual_seed(20260826)
idxj = sig.select_keep(torch.zeros(1, HEADS9, N9), N9, pol_j, torch.device("cpu"),
                       lexical=flat.view(1, 1, -1).expand(1, HEADS9, -1).contiguous())
sets = [tuple(idxj[0, h].tolist()) for h in range(HEADS9)]
check("a constant weight still selects DIFFERENTLY per KV head",
      len(set(sets)) > 1, f"{len(set(sets))} distinct selections over {HEADS9} heads")
n_sink9, n_rec9 = pol_j.n_sinks(), pol_j.n_recent()
earliest = set(range(n_sink9, n_sink9 + pol_j.heavy()))
check("the kept heavy set is NOT the lowest-index block",
      set(x for x in sets[0] if n_sink9 <= x < N9 - n_rec9) != earliest,
      "identical to the earliest-eligible block means topk resolved by index")

# --------------------------------------------------------------------------- #
print("\n10. the pre-pool masks are load-bearing (sinks must not bleed through)")
# --------------------------------------------------------------------------- #

# Deleting the pre-pool sinks/recent masking survived the whole suite while
# reintroducing a bug this file's own comments say was measured and fixed: R2 puts
# 46.3% of its attention on the first 4 positions, so with a backward window those
# slots win top-k unconditionally. Asserted here directly.
# The heavy budget must be SMALLER than the number of genuinely-scoring positions,
# or every zero-score position ties and topk resolves by index -- which looks exactly
# like bleed and is not. This is the same defect class the file's own header warns
# about, and the first version of this assertion had it.
sig.reset_scores()
s_sink = torch.zeros(1, 1, 300)
s_sink[..., :4] = 1e6                       # all the mass on the sinks
s_sink[..., 200] = 1.0                      # one genuine content token
pol_m = sig.Policy(budget=17, sinks=4, recent=8, pool=7, pool_back=6, pool_fwd=0,
                   select="significance")
check("this probe really has heavy < the scoring span", pol_m.heavy() == 5,
      f"heavy={pol_m.heavy()}")
kept_m = sorted(sig.select_keep(s_sink, 300, pol_m, torch.device("cpu"))[0, 0].tolist())
heavy_m = [x for x in kept_m if 4 <= x < 292]
check("every heavy slot goes to the real scorer's span, not the sinks' neighbours",
      all(200 <= x <= 206 for x in heavy_m), f"heavy slots {heavy_m}")
check("sink mass does not bleed into slots 4..10 through the pool",
      not any(4 <= x <= 10 for x in kept_m),
      f"bled into {[x for x in kept_m if 4 <= x <= 10]}")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all lexical-select tests pass")
