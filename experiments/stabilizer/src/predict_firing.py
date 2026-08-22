#!/usr/bin/env python3
"""P0.3 -- can the firing set be predicted, and does the KV significance matrix help?

P0.2 CHANGED WHAT THIS PROBE IS FOR. A perfect, free selector saves 2.8% of a
decode step on this host, because the FFN matmuls are only 8.9% of decode latency
and even a 1.45x speedup on them is 0.97 ms of 34.70. The speed motivation is
gone. What survives is MEMORY: at 25% density the resident weights fall from 904
to 508 MiB, which is the difference between fitting a larger model on this card
and not.

But that saving is only real if the firing set is STABLE. A set that changes every
token needs every weight available anyway, and nothing is saved. So stability, not
predictability, is now the deciding quantity, and it is measured first.

FOUR ARMS, in increasing order of runtime cost. The cheapest that works wins:

  static      one FIXED set of neurons, chosen by how often each is in the top-k
              across a corpus. Zero runtime cost, zero prediction, and the weights
              can be pruned permanently. If this works, nothing else is needed.
  oracle      per-token true top-k. The unreachable ceiling (P0.1: 0.9586 at 25%).
  global      one persistent matrix mapping the hidden state to predicted
              activation magnitude, shared across all layers -- the proposal.
  global+sig  the same, plus features summarising the attention distribution the
              KV significance matrix already tracks.

THE SIGNIFICANCE HYPOTHESIS, stated so it can fail. Attention output is a weighted
sum of value vectors, so when attention concentrates the MLP input is confined to
a low-dimensional subspace spanned by a few values. The significance matrix knows
that distribution's shape. The claim is NOT that significance predicts neurons
directly -- positions and feature directions are different spaces -- but that the
SHAPE of the attention distribution constrains which neurons can fire. Features:
entropy, mass on sinks, mass on the recent window, and effective attended count.

If `global+sig` does not beat `global`, the shape carries no information the
hidden state does not already contain, and the idea is answered honestly.

Usage:
    python src/predict_firing.py
    python src/predict_firing.py --prompts 16 --keep 0.25
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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--dataset", default="rag_dev")
    ap.add_argument("--cap-mib", type=float, default=6000.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "predict_firing.json")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.integrations import sdpa_attention as sa
    force_repeat_kv(verbose=False)
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, tot) / tot)

    # Capture the attention distribution per layer without disturbing the kernel.
    ATTN: dict[int, torch.Tensor] = {}
    _orig = sa.sdpa_attention_forward

    def wrapped(module, query, key, value, attention_mask=None, dropout=0.0,
                scaling=None, is_causal=None, **kw):
        li = getattr(module, "layer_idx", None)
        if li is not None:
            b, h, q, d = query.shape
            s = scaling if scaling is not None else 1.0 / math.sqrt(d)
            # `key` arrives pre-repeat_kv with H_kv heads against H query heads,
            # which is the grouping eviction acts on. Expand it here so the
            # attention row is per query head, then average over heads below.
            kk = key
            if kk.shape[1] != h:
                kk = kk.repeat_interleave(h // kk.shape[1], dim=1)
            logits = (query.float() @ kk.float().transpose(-1, -2)) * s
            n = key.shape[2]
            pq = torch.arange(n - q, n, device=query.device).view(1, 1, q, 1)
            pk = torch.arange(n, device=query.device).view(1, 1, 1, n)
            logits = logits.masked_fill(pk > pq, float("-inf"))
            ATTN[li] = logits.softmax(-1).mean(1)      # [b, q, n], mean over heads
        return _orig(module, query, key, value, attention_mask=attention_mask,
                     dropout=dropout, scaling=scaling, is_causal=is_causal, **kw)

    sa.sdpa_attention_forward = wrapped
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped

    tok = AutoTokenizer.from_pretrained(REPO / MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(args.device).eval()
    layers = model.model.layers
    L = len(layers)

    ACT: dict[int, torch.Tensor] = {}
    HID: dict[int, torch.Tensor] = {}

    def mk_act(i):
        def h(mod, inp, out):
            ACT[i] = inp[0].detach().float()
        return h

    def mk_hid(i):
        def h(mod, inp, out):
            HID[i] = inp[0].detach().float()
        return h

    for i, lyr in enumerate(layers):
        lyr.mlp.down_proj.register_forward_hook(mk_act(i))
        lyr.mlp.register_forward_hook(mk_hid(i))     # MLP input = post-norm hidden

    path = REPO / "artifacts" / "eval_sets" / f"{args.dataset}.jsonl"
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            recs.append([m for m in d["messages"] if m["role"] != "assistant"])
    recs = recs[:: max(1, len(recs) // args.prompts)][: args.prompts]

    WD = [layers[i].mlp.down_proj.weight.detach().float().cpu() for i in range(L)]
    X, S, A = [], [], []            # hidden, sig-features, activations (layer 0..L-1)
    for li in range(L):
        X.append([]); S.append([]); A.append([])

    for msgs in recs:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        ACT.clear(); HID.clear(); ATTN.clear()
        model(ids, use_cache=False)
        for li in range(L):
            a = ACT[li][0]                      # [seq, 3072]
            x = HID[li][0]                      # [seq, 1024]
            p = ATTN[li][0]                     # [seq, n]
            n = p.shape[-1]
            ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
            sink = p[:, :4].sum(-1)
            rec = p[:, max(0, n - 32):].sum(-1)
            eff = 1.0 / p.pow(2).sum(-1).clamp_min(1e-12)
            feats = torch.stack([ent, sink, rec, eff.log1p()], dim=-1)   # [seq, 4]
            X[li].append(x.cpu()); S[li].append(feats.cpu()); A[li].append(a.cpu())
        ACT.clear(); HID.clear(); ATTN.clear()
        torch.cuda.empty_cache()

    print(f"P0.3 -- firing-set prediction, {len(recs)} prompts, {L} layers, "
          f"keep {args.keep*100:.0f}%\n")

    def ridge(F_, Y, lam=1e-2):
        G = F_.T @ F_
        G += lam * torch.diagonal(G).mean() * torch.eye(G.shape[0])
        return torch.linalg.solve(G, F_.T @ Y)

    rows, stab = [], []
    for li in range(L):
        x = torch.cat(X[li]); s = torch.cat(S[li]); a = torch.cat(A[li])
        T, N = a.shape
        k = max(1, int(round(args.keep * N)))
        true = a.abs().topk(k, dim=-1).indices
        truth = torch.zeros(T, N, dtype=torch.bool).scatter_(1, true, True)

        # --- stability: does the set persist token to token? ------------------
        ov = (truth[:-1] & truth[1:]).sum(-1).float() / k
        # Distant-token overlap: stride 8, with both sides trimmed to a common
        # length. Slicing a[::8] and a[8::8] gives lengths that differ by one
        # whenever T is not a multiple of the stride.
        if T > 16:
            u, v = truth[:-8:8], truth[8::8]
            mlen = min(u.shape[0], v.shape[0])
            far = (u[:mlen] & v[:mlen]).sum(-1).float() / k
        else:
            far = ov
        stab.append((ov.mean().item(), far.mean().item()))

        # --- static: one fixed set, chosen by frequency -----------------------
        freq = truth.float().mean(0)
        static_idx = freq.topk(k).indices
        static = torch.zeros(N, dtype=torch.bool).scatter_(0, static_idx, True)
        rec_static = (truth & static.unsqueeze(0)).sum(-1).float().mean().item() / k

        # --- predictors: hidden state, and hidden state + attention shape -----
        tr = int(0.7 * T)
        Y = a.abs()
        Wd = WD[li]
        full = (a[tr:] @ Wd.T).norm(dim=-1).clamp_min(1e-9)

        def rho_of(idx):
            """Output preserved by a chosen set -- the quantity that matters.

            Recall counts how many of the right neurons were found; rho counts
            how much of the layer OUTPUT survives. They are not the same, and
            only the second is what a reader would care about.
            """
            kept = torch.zeros_like(a[tr:]).scatter_(1, idx, a[tr:].gather(1, idx))
            return ((kept @ Wd.T).norm(dim=-1) / full).mean().item()

        def recall(F_):
            W = ridge(F_[:tr], Y[:tr])
            pred = (F_[tr:] @ W).abs()
            pi = pred.topk(k, dim=-1).indices
            hit = torch.zeros(T - tr, N, dtype=torch.bool).scatter_(1, pi, True)
            return ((truth[tr:] & hit).sum(-1).float().mean().item() / k,
                    rho_of(pi))
        one = torch.ones(T, 1)
        r_glob, rho_glob = recall(torch.cat([x, one], -1))
        r_sig, rho_sig = recall(torch.cat([x, s, one], -1))
        r_rand = k / N
        rho_oracle = rho_of(true[tr:])
        rho_static = rho_of(static_idx.unsqueeze(0).expand(T - tr, -1))

        rows.append({"layer": li, "static": rec_static, "global": r_glob,
                     "global_sig": r_sig, "random": r_rand,
                     "rho_oracle": rho_oracle, "rho_global": rho_glob,
                     "rho_global_sig": rho_sig, "rho_static": rho_static,
                     "stab_adjacent": stab[-1][0], "stab_far": stab[-1][1]})

    def m(key):
        return sum(r[key] for r in rows) / len(rows)

    print("Recall of the true top-k firing set (1.00 = perfect):")
    print(f"  random baseline            {m('random'):.4f}")
    print(f"  STATIC fixed set           {m('static'):.4f}")
    print(f"  global matrix (hidden)     {m('global'):.4f}")
    print(f"  global + significance      {m('global_sig'):.4f}")
    print(f"\n  significance contributes   {m('global_sig') - m('global'):+.4f}")
    print("")
    print("OUTPUT PRESERVED by each selection (rho -- what actually matters):")
    print(f"  oracle top-k               {m('rho_oracle'):.4f}")
    print(f"  global matrix              {m('rho_global'):.4f}")
    print(f"  global + significance      {m('rho_global_sig'):.4f}")
    print(f"  STATIC fixed set           {m('rho_static'):.4f}")
    print("\nFiring-set stability (Jaccard-style overlap of the top-k set):")
    print(f"  adjacent tokens            {m('stab_adjacent'):.4f}")
    print(f"  8 tokens apart             {m('stab_far'):.4f}")

    verdict = []
    if m("global_sig") - m("global") < 0.01:
        verdict.append("significance features add nothing over the hidden state")
    else:
        verdict.append("significance features help")
    if m("static") > m("global") - 0.02:
        verdict.append("a STATIC set is as good as the predictor -- prune, do not predict")
    if m("stab_adjacent") > 0.8:
        verdict.append("the firing set is stable enough to amortise a gather")
    else:
        verdict.append("the firing set is NOT stable; the memory saving does not follow")
    print("\nVERDICT: " + "; ".join(verdict))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": MODEL, "prompts": len(recs), "keep": args.keep, "layers": L,
         "mean": {k: m(k) for k in ("random", "static", "global", "global_sig",
                                    "rho_oracle", "rho_global", "rho_global_sig",
                                    "rho_static", "stab_adjacent", "stab_far")},
         "sig_delta": m("global_sig") - m("global"),
         "verdict": verdict, "per_layer": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
