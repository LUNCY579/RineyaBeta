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
pol_b = sig.Policy(budget=20, sinks=2, recent=4, pool=7, pool_back=6, pool_fwd=0,
                   select="lexical-naive")
kb = set(sig.select_keep(torch.zeros(1, 2, N), N, pol_b, torch.device("cpu"),
                         lexical=lex3)[0, 0].tolist())
fwd_in = [p for p in range(101, 107) if p in kb]
back_in = [p for p in range(94, 100) if p in kb]
check("positions AFTER the match inherit it", len(fwd_in) == 6, f"{fwd_in}")
check("positions BEFORE the match do not", len(back_in) == 0, f"{back_in}")

pol_c = sig.Policy(budget=20, sinks=2, recent=4, pool=7, pool_back=3, pool_fwd=3,
                   select="lexical-naive")
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
except Exception as _e:                                   # noqa: BLE001
    check("doc-bm25 weight construction", False, f"raised {type(_e).__name__}: {_e}")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all lexical-select tests pass")
