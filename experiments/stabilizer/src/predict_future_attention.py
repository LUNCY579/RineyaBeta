#!/usr/bin/env python3
"""P1.1 -- can a global matrix predict which cached positions will be needed LATER?

THE REVERSAL. P0.3 asked whether the KV significance matrix helps a global
predictor choose FFN neurons; it does not (+0.0002). This asks the opposite and
better-motivated question: can a global predictor help the significance matrix?

WHY THIS IS THE RIGHT DIRECTION ON THIS HARDWARE. The KV cache is the measured
bottleneck here -- 14,336 MiB at 128K against an 8,188 MiB card -- and the
significance matrix is what addresses it. It also has a measured weakness with a
known cause: it ranks by attention ALREADY RECEIVED, so a position that will
matter later but has not yet been attended has no score to protect it. At 128K
with a 2048 budget the deficits against an uncompressed reference are 0.58 at
depth 0.1 and 0.67 at depth 0.5, against only 0.17 at depth 0.9 -- concentrated
exactly where a backward-looking score is blind.

THE HYPOTHESIS. A position's KEY may carry signal about whether it will be
attended in future, independent of whether it has been attended so far. If so,
one global projection of the key could supplement the accumulated score and
protect needles before they earn protection.

If false, that is worth knowing cheaply and definitively: it would mean the
accumulated score is the only available signal and the remaining gap has to be
closed with budget rather than with ranking.

WHAT IS MEASURED. Split a prompt at time T. For every cached position i:

    past[i]    attention received from queries  <  T   (what we rank by today)
    future[i]  attention received from queries >= T   (what we WISH we ranked by)

Fit one global ridge from the key to `future`, pooled over every (layer, KV head)
so the matrix is genuinely shared. Then compare four rankings by how well each
recovers the true top-B positions by future attention:

    past-only     the current mechanism
    predicted     the global matrix alone
    past+pred     both
    random        the control that has caught two false results in this program

An oracle upper bound (ranking by `future` itself) is reported so the prize is
visible next to the attempts on it.

Usage:
    python src/predict_future_attention.py
    python src/predict_future_attention.py --length 8192 --keep 0.0625
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

MODEL = "artifacts/rineya-core-en32k-28L"
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
PRE = ("There is an important piece of information hidden inside a lot of "
       "irrelevant text. Find it and memorise it.\n\n")
SUF = "\n\nWhat is the pass key? The pass key is"


def build(tok, n_tokens: int, depth: float, key: int) -> str:
    needle = f"\n\nThe pass key is {key}. Remember it. {key} is the pass key.\n\n"
    per = len(tok(FILLER)["input_ids"])
    reps = max(1, (n_tokens - len(tok(PRE + needle + SUF)["input_ids"])) // max(per, 1))
    before = int(reps * depth)
    return PRE + FILLER * before + needle + FILLER * (reps - before) + SUF


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--keep", type=float, default=0.0625)
    ap.add_argument("--split", type=float, default=0.6,
                    help="fraction of the prompt treated as 'past'")
    ap.add_argument("--qblock", type=int, default=512)
    ap.add_argument("--cap-mib", type=float, default=6000.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path,
                    default=EXP / "results" / "predict_future_attention.json")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.integrations import sdpa_attention as sa
    force_repeat_kv(verbose=False)
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, tot) / tot)

    QK: dict[int, tuple] = {}
    _orig = sa.sdpa_attention_forward

    def wrapped(module, query, key, value, attention_mask=None, dropout=0.0,
                scaling=None, is_causal=None, **kw):
        li = getattr(module, "layer_idx", None)
        if li is not None:
            QK[li] = (query.detach(), key.detach(), scaling)
        return _orig(module, query, key, value, attention_mask=attention_mask,
                     dropout=dropout, scaling=scaling, is_causal=is_causal, **kw)

    sa.sdpa_attention_forward = wrapped
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped

    tok = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(args.device).eval()

    text = build(tok, args.length, args.depth, 75526)
    ids = tok(text, return_tensors="pt").to(args.device)["input_ids"]
    n = ids.shape[-1]
    T = int(args.split * n)
    print(f"P1.1 -- future-attention predictability, n={n}, split at {T}, "
          f"keep {args.keep*100:.2f}%\n")

    QK.clear()
    model(ids, use_cache=False)

    KEYS, PAST, FUT = [], [], []
    for li in sorted(QK):
        q, k, scal = QK[li]
        b, h, _, d = q.shape
        nkv = k.shape[1]
        g = h // nkv
        s = scal if scal is not None else 1.0 / math.sqrt(d)
        past = torch.zeros(nkv, n, device=q.device)
        fut = torch.zeros(nkv, n, device=q.device)
        # Column sums of the causal attention matrix, accumulated in query blocks
        # so the full [h, n, n] is never materialised.
        for i0 in range(0, n, args.qblock):
            i1 = min(n, i0 + args.qblock)
            qb = q[:, :, i0:i1, :].float().reshape(b, nkv, g, i1 - i0, d)
            logits = torch.einsum("bgcwd,bgnd->bgcwn", qb, k.float()) * s
            pq = torch.arange(i0, i1, device=q.device).view(1, 1, 1, -1, 1)
            pk = torch.arange(n, device=q.device).view(1, 1, 1, 1, -1)
            p = logits.masked_fill(pk > pq, float("-inf")).softmax(-1)
            colsum = p.sum(dim=(2, 3))[0]              # [nkv, n]
            if i1 <= T:
                past += colsum
            elif i0 >= T:
                fut += colsum
            else:                                       # straddles the split
                cut = T - i0
                past += p[:, :, :, :cut, :].sum(dim=(2, 3))[0]
                fut += p[:, :, :, cut:, :].sum(dim=(2, 3))[0]
            del logits, p
        KEYS.append(k[0].float().cpu())                 # [nkv, n, d]
        PAST.append(past.cpu()); FUT.append(fut.cpu())
        QK[li] = None
        torch.cuda.empty_cache()

    L = len(KEYS)
    # Only positions that exist before the split can be protected by an eviction
    # decision taken at the split, so the problem is defined on i < T.
    feats, tgt, pastv, group = [], [], [], []
    for li in range(L):
        kk, pa, fu = KEYS[li], PAST[li], FUT[li]
        for hh in range(kk.shape[0]):
            feats.append(kk[hh, :T, :])
            tgt.append(fu[hh, :T])
            pastv.append(pa[hh, :T])
            group.append(torch.full((T,), li * kk.shape[0] + hh))
    Xk = torch.cat(feats); Y = torch.cat(tgt); P = torch.cat(pastv); G = torch.cat(group)
    ng = int(G.max().item()) + 1
    print(f"{Xk.shape[0]:,} (layer, head, position) triples over {L} layers, "
          f"{ng // L} KV heads")

    # Held-out by GROUP, not by row: fitting and testing on the same (layer, head)
    # would let the matrix memorise that stream rather than generalise across them.
    tr_groups = set(range(0, ng, 2))
    trm = torch.tensor([int(g_.item()) in tr_groups for g_ in G])
    F_tr = torch.cat([Xk[trm], P[trm, None].log1p(), torch.ones(trm.sum(), 1)], -1)
    W = torch.linalg.solve(
        F_tr.T @ F_tr + 1e-2 * torch.eye(F_tr.shape[1]) * (F_tr.T @ F_tr).diagonal().mean(),
        F_tr.T @ Y[trm])
    Wk = torch.linalg.solve(
        Xk[trm].T @ Xk[trm] + 1e-2 * torch.eye(Xk.shape[1]) * (Xk[trm].T @ Xk[trm]).diagonal().mean(),
        Xk[trm].T @ Y[trm])

    te = ~trm
    B = max(1, int(round(args.keep * T)))
    res = {}
    for name, score in (("random", torch.rand(int(te.sum()))),
                        ("past-only", P[te]),
                        ("predicted (key only)", Xk[te] @ Wk),
                        ("past + predicted", torch.cat(
                            [Xk[te], P[te, None].log1p(), torch.ones(int(te.sum()), 1)], -1) @ W),
                        ("ORACLE (future itself)", Y[te])):
        rec, gs = [], G[te]
        for g_ in gs.unique():
            m = gs == g_
            truth = Y[te][m].topk(min(B, int(m.sum()))).indices
            pick = score[m].topk(min(B, int(m.sum()))).indices
            rec.append(len(set(truth.tolist()) & set(pick.tolist())) / len(truth))
        res[name] = sum(rec) / len(rec)

    print(f"\nRecall of the top-{B} positions by FUTURE attention "
          f"(held-out heads):\n")
    for k_, v in res.items():
        print(f"  {k_:<26} {v:.4f}")
    gain = res["past + predicted"] - res["past-only"]
    print(f"\n  prediction adds over past-only   {gain:+.4f}")
    print(f"  headroom to the oracle            "
          f"{res['ORACLE (future itself)'] - res['past-only']:+.4f}")

    verdict = ("the key predicts future attention and ADDS to the accumulated score"
               if gain > 0.02 else
               "the key adds nothing the accumulated score does not already carry")
    print(f"\nVERDICT: {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": MODEL, "n": n, "split": T, "keep": args.keep, "topB": B,
         "layers": L, "recall": res, "gain_over_past": gain,
         "verdict": verdict}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
