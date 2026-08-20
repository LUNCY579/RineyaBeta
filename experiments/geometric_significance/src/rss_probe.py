#!/usr/bin/env python3
"""Gate 0 for RSS-P: does profile-space suppression improve the attention output?

The geometric idea survived design review in exactly one form. Embedding cached
positions by their KEY vector is dead -- `redundancy.py` measured the key space
to be strongly anisotropic (mean pairwise cosine 0.661 over all keys) and top-k
already 98% as diverse as random selection in centred space. Embedding them by
their ATTENTION PROFILE -- the softmax row each position receives from the
observation window, [groups*w] numbers per position -- is different: reviewers
measured that field to be heavy-tailed (median cosine 0.383, P99 0.933).

So the open question is not whether near-duplicates exist in profile space. They
do. It is whether suppressing them produces a BETTER ATTENTION OUTPUT than plain
top-k at the same budget. That is measurable directly, without generating
anything, by comparing the exact attention output against its evicted
approximation:

    o_full = softmax(q K^T / sqrt(d)) V           over all n keys
    o_hat  = softmax(q K[idx]^T / sqrt(d)) V[idx] over the kept set
    err    = ||o_full - o_hat|| / ||o_full||

Four policies over identical tensors:

    topk          current policy: sinks + recent + top-B by score
    rss           top-B after suppressing profile-space near-duplicates
    rss-shuffle   THE CONTROL. Permutes the candidate -> profile-row assignment
                  before the Gram. Preserves the edge count, the edge-weight
                  multiset and the number of slots moved EXACTLY, and destroys
                  only the pairing between a key and its claimed duplicates. If
                  rss cannot beat its own shuffle, any gain is near-cutoff
                  jitter rather than redundancy.
    streaming     sinks + recent only, no scoring at all

Queries are HELD OUT: scores come from the prefill window, errors are evaluated
on the queries of subsequently generated tokens. Top-32 importance persistence
between consecutive positions was measured at 0.584, so evaluating on the same
window that built the scores would flatter every policy.

Decision, pre-registered:
    KILL-1  mean_err(rss) >= mean_err(topk)               -> stop, report negative
    KILL-2  rss beats topk but not its own shuffle        -> jitter, stop
    PASS    rss < shuffle AND rss < topk, both p < 0.01,
            relative improvement >= 2%                     -> only then spend GPU-hours

Also reports a mass-compensated variant, which is the one thing an independent
reviewer measured to beat top-k anywhere: a surviving leader inherits the
attention weight of the duplicates it displaced, via an additive logit bias
log(1 + donated/own). It costs ~15 lines over the same tensors.

Usage:
    python src/rss_probe.py --cases 64 --keep 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

MODEL = "artifacts/rineya-core-en32k-28L"
_CAP: dict[int, dict] = {}
_MODE = {"on": False}


def capture() -> None:
    """Record post-RoPE q/K/V per layer, pre-repeat_kv, on every attention call."""
    from transformers.integrations import sdpa_attention as sa
    original = sa.sdpa_attention_forward

    def wrapped(*args, **kw):
        if _MODE["on"] and len(args) >= 4:
            module, q, k, v = args[0], args[1], args[2], args[3]
            li = getattr(module, "layer_idx", None)
            if li is not None:
                _CAP.setdefault(li, {}).setdefault("q", []).append(q.detach())
                _CAP[li]["k"] = k.detach()
                _CAP[li]["v"] = v.detach()
        return original(*args, **kw)

    sa.sdpa_attention_forward = wrapped
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
    except Exception:  # noqa: BLE001
        pass


def profile_and_scores(q: torch.Tensor, k: torch.Tensor, w: int):
    """Observation-window attention. Returns scores [n_kv,n] and profiles [n_kv,n,m]."""
    _, h, q_len, d = q.shape
    n_kv, n = k.shape[1], k.shape[2]
    groups = h // n_kv
    w = min(w, q_len)
    qq = q[:, :, -w:, :].reshape(1, n_kv, groups, w, d).float()
    logits = torch.einsum("bgcwd,bgnd->bgcwn", qq, k.float()) / math.sqrt(d)
    pos_q = torch.arange(n - w, n, device=q.device)
    pos_k = torch.arange(n, device=q.device)
    logits = logits.masked_fill(pos_k[None, :] > pos_q[:, None], float("-inf"))
    a = logits.softmax(-1)                                   # [1,n_kv,groups,w,n]
    s = a.sum(dim=(2, 3))[0]                                 # [n_kv, n]
    prof = a.permute(0, 1, 4, 2, 3).reshape(1, n_kv, n, groups * w)[0]
    return s, prof.contiguous()


def base_regions(n: int, budget: int, sinks: int, recent: int):
    n_sink = min(sinks, budget, n)
    n_recent = min(recent, max(0, budget - n_sink), max(0, n - n_sink))
    n_heavy = max(0, min(budget - n_sink - n_recent, n - n_sink - n_recent))
    return n_sink, n_recent, n_heavy


def select(policy: str, s: torch.Tensor, prof: torch.Tensor, n: int, budget: int,
           sinks: int, recent: int, tau: float, gen: torch.Generator):
    """Return kept indices [n_kv, budget] plus per-head donated mass for compensation."""
    n_kv = s.shape[0]
    dev = s.device
    n_sink, n_recent, n_heavy = base_regions(n, budget, sinks, recent)
    lo, hi = n_sink, n - n_recent
    donated = torch.zeros_like(s)

    if policy == "streaming":
        # sinks + recent, then fill the remainder with the most RECENT middle
        # positions -- position-only, no scores anywhere.
        fill = torch.arange(max(lo, hi - n_heavy), hi, device=dev)
        fill = fill.view(1, -1).expand(n_kv, -1)
    else:
        mid = torch.arange(lo, hi, device=dev)
        s_mid = s[:, lo:hi].clone()                          # [n_kv, C]
        C = s_mid.shape[1]
        if policy in ("rss", "rss-shuffle", "rss-mass") and C > 1 and n_heavy < C:
            p = torch.nn.functional.normalize(prof[:, lo:hi, :].float(), dim=-1)
            if policy == "rss-shuffle":
                perm = torch.stack([torch.randperm(C, generator=gen) for _ in range(n_kv)])
                p = torch.gather(p, 1, perm.to(dev).unsqueeze(-1).expand(-1, -1, p.shape[-1]))
            G = p @ p.transpose(1, 2)                        # [n_kv, C, C]
            eye = torch.eye(C, dtype=torch.bool, device=dev)
            # j dominates i when they are near-duplicates and j scores higher
            dom = (G > tau) & (s_mid[:, None, :] > s_mid[:, :, None]) & ~eye
            suppressed = dom.any(-1)                         # [n_kv, C]
            if policy == "rss-mass":
                # each suppressed follower donates its mass to its best leader
                best = (G.masked_fill(~dom, -1.0)).argmax(-1)             # [n_kv, C]
                give = torch.where(suppressed, s_mid, torch.zeros_like(s_mid))
                acc = torch.zeros_like(s_mid).scatter_add_(1, best, give)
                donated[:, lo:hi] = acc
            # suppressed entries fall to the back of the ranking but stay eligible,
            # so the budget is always exactly filled
            rank = torch.where(suppressed, s_mid - s_mid.max(dim=-1, keepdim=True).values
                               - 1.0, s_mid)
            order = rank.topk(n_heavy, dim=-1).indices
        else:
            order = s_mid.topk(min(n_heavy, s_mid.shape[1]), dim=-1).indices
        fill = mid[order]

    parts = []
    if n_sink:
        parts.append(torch.arange(n_sink, device=dev).view(1, -1).expand(n_kv, -1))
    if fill.numel():
        parts.append(fill)
    if n_recent:
        parts.append(torch.arange(hi, n, device=dev).view(1, -1).expand(n_kv, -1))
    idx = torch.cat(parts, dim=-1).sort(dim=-1).values
    # Mass compensation: a surviving leader inherits the attention weight of the
    # duplicates it displaced, as an additive logit bias log(1 + donated/own).
    # `own` must be the leader's OWN accumulated score, not a placeholder --
    # the ratio is what makes the bias scale-free.
    bias = None
    if policy == "rss-mass":
        own = torch.gather(s, 1, idx).clamp_min(1e-6)
        don = torch.gather(donated, 1, idx)
        bias = torch.log1p(don / own)                         # [n_kv, B]
    return idx, bias


@torch.no_grad()
def attn_err(q_t: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             idx: torch.Tensor, bias: torch.Tensor | None):
    """Relative L2 error of the evicted attention output. q_t [1,h,1,d]."""
    h, d = q_t.shape[1], q_t.shape[3]
    n_kv = k.shape[1]
    groups = h // n_kv
    qq = q_t.reshape(1, n_kv, groups, d).float()
    kk, vv = k.float(), v.float()

    full_logits = torch.einsum("bgcd,bgnd->bgcn", qq, kk) / math.sqrt(d)
    a_full = full_logits.softmax(-1)
    o_full = torch.einsum("bgcn,bgnd->bgcd", a_full, vv)

    gi = idx.unsqueeze(0)                                     # [1,n_kv,B]
    k_s = torch.gather(kk, 2, gi.unsqueeze(-1).expand(-1, -1, -1, d))
    v_s = torch.gather(vv, 2, gi.unsqueeze(-1).expand(-1, -1, -1, d))
    logits = torch.einsum("bgcd,bgnd->bgcn", qq, k_s) / math.sqrt(d)
    if bias is not None:
        # logits is [1, n_kv, groups, B]; bias is [n_kv, B]
        logits = logits + bias.unsqueeze(0).unsqueeze(2)
    a_s = logits.softmax(-1)
    o_hat = torch.einsum("bgcn,bgnd->bgcd", a_s, v_s)

    err = (o_full - o_hat).norm(dim=-1) / o_full.norm(dim=-1).clamp_min(1e-9)
    rho = torch.gather(a_full.sum(2), 2, gi).sum(-1)           # [1,n_kv]
    return err.flatten(), rho.flatten()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=64)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--recent", type=int, default=32)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.93)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/rss_probe.json"))
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM, AutoTokenizer
    force_repeat_kv(verbose=False)
    capture()

    path = REPO / MODEL
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device).eval()

    rows = []
    with (REPO / "artifacts/eval_sets/rag_dev.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    step = max(1, len(rows) // args.cases)
    rows = rows[::step][:args.cases]

    POL = ["topk", "rss", "rss-shuffle", "rss-mass", "streaming"]
    err = {p: [] for p in POL}
    rho = {p: [] for p in POL}
    moved = []
    gen = torch.Generator().manual_seed(0)

    for rec in rows:
        msgs = [m for m in rec["messages"] if m.get("role") != "assistant"]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        enc = tok(prompt, return_tensors="pt").to(args.device)
        n = enc["input_ids"].shape[-1]
        budget = max(8, int(round(args.keep * n)))
        if n <= budget + 4:
            continue

        _CAP.clear()
        _MODE["on"] = True
        out = model.generate(**enc, max_new_tokens=args.decode, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        _MODE["on"] = False

        for li, cap in _CAP.items():
            qs = cap.get("q") or []
            if len(qs) < 2 or qs[0].shape[2] != n:
                continue
            k_pre, v_pre = cap["k"], cap["v"]
            if k_pre.shape[2] < n:
                continue
            k_pre, v_pre = k_pre[:, :, :n, :], v_pre[:, :, :n, :]
            s, prof = profile_and_scores(qs[0], k_pre, args.window)

            sel = {}
            for p in POL:
                g2 = torch.Generator().manual_seed(li)
                sel[p] = select(p, s, prof, n, budget, args.sinks, args.recent,
                                args.tau, g2)
            a_idx = set(sel["topk"][0][0].tolist())
            b_idx = set(sel["rss"][0][0].tolist())
            moved.append(len(a_idx - b_idx))

            for q_t in qs[1:]:                       # held-out decode queries
                if q_t.shape[2] != 1:
                    continue
                for p in POL:
                    idx, bias = sel[p]
                    e, r = attn_err(q_t, k_pre, v_pre, idx, bias)
                    err[p].append(e.cpu())
                    rho[p].append(r.cpu())

    if not err["topk"]:
        raise SystemExit("no samples collected")

    E = {p: torch.cat(v) for p, v in err.items()}
    R = {p: torch.cat(v) for p, v in rho.items()}
    N = E["topk"].numel()
    print(f"R2, {len(rows)} rag_dev prompts, keep={args.keep}, tau={args.tau}, "
          f"{N:,} held-out (layer, head, step) samples")
    print(f"mean slots moved by rss vs topk: {sum(moved)/max(len(moved),1):.2f}\n")
    print(f"{'policy':<14}{'mean err':>10}{'median':>10}{'rho':>9}{'vs topk':>10}")
    for p in POL:
        d = 100 * (E[p].mean() - E["topk"].mean()) / E["topk"].mean()
        print(f"{p:<14}{E[p].mean():>10.4f}{E[p].median():>10.4f}"
              f"{R[p].mean():>9.4f}{d:>9.2f}%")

    def sign_test(a: torch.Tensor, b: torch.Tensor):
        """Fraction of paired samples where a < b, and a normal-approx p."""
        d = (b - a)
        nz = d[d.abs() > 1e-9]
        if nz.numel() == 0:
            return 0.5, 1.0
        wins = int((nz > 0).sum())
        m = nz.numel()
        z = (wins - m / 2) / math.sqrt(m / 4)
        p = math.erfc(abs(z) / math.sqrt(2))
        return wins / m, p

    w_topk, p_topk = sign_test(E["rss"], E["topk"])
    w_shuf, p_shuf = sign_test(E["rss"], E["rss-shuffle"])
    rel_topk = 100 * (E["topk"].mean() - E["rss"].mean()) / E["topk"].mean()
    rel_shuf = 100 * (E["rss-shuffle"].mean() - E["rss"].mean()) / E["rss-shuffle"].mean()
    print(f"\npaired, rss vs topk   : rss better on {w_topk:.1%} of samples, "
          f"p={p_topk:.2e}, relative {rel_topk:+.2f}%")
    print(f"paired, rss vs shuffle: rss better on {w_shuf:.1%} of samples, "
          f"p={p_shuf:.2e}, relative {rel_shuf:+.2f}%")

    if E["rss"].mean() >= E["topk"].mean():
        verdict = ("KILL-1", "suppression does not improve the attention output it "
                             "claims to protect. Stop; report the negative.")
    elif not (p_shuf < 0.01 and rel_shuf >= 2.0):
        verdict = ("KILL-2", "rss beats topk but not its own matched-random shuffle. "
                             "The gain is near-cutoff jitter, not redundancy. Stop.")
    elif p_topk < 0.01 and rel_topk >= 2.0:
        verdict = ("PASS", "rss beats both topk and its shuffle at p<0.01 and >=2%. "
                           "Only now is a quality arm worth GPU-hours.")
    else:
        verdict = ("KILL-2", "improvement over topk too small or not significant.")
    print(f"\nVERDICT {verdict[0]}: {verdict[1]}")

    mass_rel = 100 * (E["topk"].mean() - E["rss-mass"].mean()) / E["topk"].mean()
    print(f"\nmass-compensated variant: {mass_rel:+.2f}% vs topk "
          f"(the one variant an independent reviewer measured to beat top-k)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "cases": len(rows), "keep": args.keep, "tau": args.tau, "samples": N,
        "mean_slots_moved": sum(moved) / max(len(moved), 1),
        "mean_err": {p: float(E[p].mean()) for p in POL},
        "median_err": {p: float(E[p].median()) for p in POL},
        "mean_rho": {p: float(R[p].mean()) for p in POL},
        "rss_vs_topk": {"win_frac": w_topk, "p": p_topk, "rel_pct": float(rel_topk)},
        "rss_vs_shuffle": {"win_frac": w_shuf, "p": p_shuf, "rel_pct": float(rel_shuf)},
        "verdict": verdict[0], "verdict_text": verdict[1],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
