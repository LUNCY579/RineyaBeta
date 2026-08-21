"""What to STORE for each cached position, once you have decided to keep it.

Eviction answers "which positions"; this answers "in what form". Both act on
the same cache and their costs compose, so they have to be measured on the
same scale.

THE SCALE IS CALIBRATION. Attention-output relative error is meaningless on its
own -- 0.01 could be free or fatal. So every candidate is measured alongside the
perturbations whose END-TO-END quality cost is already on record:

    keep 0.75   relerr 0.00937   rag_dev quality -0.01262
    keep 0.50   relerr 0.03729   rag_dev quality -0.05538
    keep 0.25   relerr 0.11381   rag_dev quality -0.33076

A candidate below 0.00937 is cheaper than the eviction setting already shipped;
one near 0.11381 costs a third of the benchmark. That is the whole point of the
design: the reference points and the candidates are computed in one pass, on the
same prompts, at the same layers.

Arms:
  (a) calibration -- int8/int4 at several groupings, low-rank [K_preRoPE ; V],
      and the unhealed GQA 8->2 head merge, against the three reference points
  (b) cross-layer KV sharing (CLA) by ridge regression, train/test split, so a
      memorised fit cannot pass as a prediction
  (c) low-rank damage decomposition: K-only against V-only
  (d) quantisation scale magnitude against context length (fp16 subnormal trap)
  (e) requantisation compounding over repeated compression rounds

Writes results/store_probe.json.

Usage:
    .venv/Scripts/python.exe experiments/kv_significance/src/store_probe.py [n_prompts]
"""
import json, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
MODEL = os.path.join(ROOT, "artifacts", "rineya-core-en32k-28L")
torch.cuda.set_per_process_memory_fraction(0.72)
torch.manual_seed(0)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import sdpa_attention as sa
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

CAP = {}
_orig = sa.sdpa_attention_forward


def wrapped(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None,
            is_causal=None, **kw):
    li = getattr(module, "layer_idx", None)
    if li is not None:
        CAP[li] = (query[:, :, -16:, :].detach().clone(), key.detach().clone(), value.detach().clone())
    return _orig(module, query, key, value, attention_mask=attention_mask, dropout=dropout,
                 scaling=scaling, is_causal=is_causal, **kw)


sa.sdpa_attention_forward = wrapped
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda().eval()
PRE = {}
KPRE = {}
VPRE = {}


def mk_hook(store, i):
    def h(mod, inp, out):
        store[i] = out.detach().clone()
    return h


for i, lyr in enumerate(model.model.layers):
    lyr.self_attn.k_norm.register_forward_hook(mk_hook(PRE, i))
    lyr.self_attn.k_proj.register_forward_hook(mk_hook(KPRE, i))
    lyr.self_attn.v_proj.register_forward_hook(mk_hook(VPRE, i))


def merge_kv(li, cos, sin):
    """Exact gqa-2KV mean merge, evaluated on the UNPERTURBED hidden states."""
    b, seq, _ = KPRE[li].shape
    kp = KPRE[li].view(b, seq, 2, 4, 128).mean(3).repeat_interleave(4, 2)   # [b,seq,8,128]
    w = model.model.layers[li].self_attn.k_norm.weight.float()
    x = kp.float()
    kn = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w).to(torch.bfloat16)
    kn = kn.transpose(1, 2)
    _, kr = apply_rotary_pos_emb(kn[:, :1], kn, cos, sin)
    vm = VPRE[li].view(b, seq, 2, 4, 128).mean(3).repeat_interleave(4, 2).transpose(1, 2)
    return kr, vm

scaling = 128 ** -0.5
LAYERS = list(range(28))


def haystack_ids(tok, n_tokens=4096):
    """A single long context, so per-channel quantisation groups are actually full.

    Short chat prompts leave the g=128 groups mostly padding, which flatters
    per-channel schemes; measuring only on them would overstate int8 and int4.
    """
    for rel in (os.path.join("Rineya_LCR_SFT_v1", "Rineya_LCR_SFT_v1", "data",
                             "lcr_sft_v1.jsonl"),
                os.path.join("artifacts", "eval_sets", "rag_dev.jsonl")):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            break
    else:
        raise SystemExit("no corpus found for the long-context arm")
    texts = []
    with open(path, encoding="utf-8") as f:
        for j, line in enumerate(f):
            if j >= 40:
                break
            d = json.loads(line)
            s = json.dumps(d, ensure_ascii=False)
            for k in ("context", "input", "prompt", "text", "document"):
                if isinstance(d.get(k), str) and len(d[k]) > 200:
                    s = d[k]
                    break
            texts.append(s)
    ids = tok((chr(10) * 2).join(texts), return_tensors="pt").input_ids[:, :n_tokens].cuda()
    if ids.shape[-1] < n_tokens:
        raise SystemExit("corpus too short: %d < %d tokens" % (ids.shape[-1], n_tokens))
    return ids


def relerr(a, b):
    return ((a - b).norm() / b.norm()).item()


def attn_out(q, k, v, keep=None):
    b, H, w, d = q.shape
    Hkv = k.shape[1]
    g = H // Hkv
    npos = k.shape[2]
    if keep is None:
        kk, vv = k, v
        posk = torch.arange(npos, device=q.device).view(1, 1, -1).expand(b, Hkv, -1)
    else:
        gi = keep.unsqueeze(-1).expand(-1, -1, -1, d)
        kk = k.gather(2, gi)
        vv = v.gather(2, gi)
        posk = keep
    kk = kk.repeat_interleave(g, 1).float()
    vv = vv.repeat_interleave(g, 1).float()
    posk = posk.repeat_interleave(g, 1)
    logits = (q.float() @ kk.transpose(-1, -2)) * scaling
    pos_q = torch.arange(npos - w, npos, device=q.device).view(1, 1, w, 1)
    logits = logits.masked_fill(posk.unsqueeze(2) > pos_q, float("-inf"))
    return logits.softmax(-1) @ vv


def qdq(x, bits, axis, group=None, keep_sinks=0, ret_scale=False):
    x = x.float()
    if keep_sinks:
        head, x = x[:, :, :keep_sinks, :], x[:, :, keep_sinks:, :]
    qmax = 2 ** bits - 1
    if axis == "token":
        mn = x.amin(-1, keepdim=True)
        mx = x.amax(-1, keepdim=True)
        s = (mx - mn).clamp_min(1e-12) / qmax
        y = ((x - mn) / s).round().clamp(0, qmax) * s + mn
    else:
        b, h, nn, d = x.shape
        g = min(group or nn, nn)
        pad = (-nn) % g
        xp = torch.cat([x, x[:, :, -1:, :].expand(b, h, pad, d)], 2) if pad else x
        xg = xp.view(b, h, -1, g, d)
        mn = xg.amin(3, keepdim=True)
        mx = xg.amax(3, keepdim=True)
        s = (mx - mn).clamp_min(1e-12) / qmax
        y = (((xg - mn) / s).round().clamp(0, qmax) * s + mn).view(b, h, -1, d)[:, :, :nn, :]
    if keep_sinks:
        y = torch.cat([head, y], 2)
    return (y, s) if ret_scale else y


def sig_scores(q, k):
    b, H, w, d = q.shape
    Hkv = k.shape[1]
    g = H // Hkv
    npos = k.shape[2]
    qq = q.reshape(b, Hkv, g, w, d).float()
    logits = torch.einsum("bgcwd,bgnd->bgcwn", qq, k.float()) * scaling
    pos_q = torch.arange(npos - w, npos, device=q.device)
    pos_k = torch.arange(npos, device=q.device)
    logits = logits.masked_fill(pos_k[None, :] > pos_q[:, None], float("-inf"))
    return logits.softmax(-1).sum(dim=(2, 3))


def keep_idx(s, budget, npos, sinks=4, recent=32):
    b, Hkv = s.shape[0], s.shape[1]
    sinks = min(sinks, budget)
    recent = min(recent, max(0, budget - sinks), max(0, npos - sinks))
    heavy = max(0, budget - sinks - recent)
    mid = s.clone()
    mid[..., :sinks] = float("-inf")
    if recent:
        mid[..., npos - recent:] = float("-inf")
    parts = []
    if sinks:
        parts.append(torch.arange(sinks, device=s.device).view(1, 1, -1).expand(b, Hkv, -1))
    if heavy:
        parts.append(mid.topk(heavy, -1).indices)
    if recent:
        parts.append(torch.arange(npos - recent, npos, device=s.device).view(1, 1, -1).expand(b, Hkv, -1))
    return torch.cat(parts, -1).sort(-1).values


# =========================================================================== #
# (a) calibration on rag_dev
# =========================================================================== #
recs = []
with open(os.path.join(ROOT, "artifacts", "eval_sets", "rag_dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        msgs = [m for m in d["messages"] if m["role"] != "assistant"]
        recs.append(msgs)
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 24
recs = recs[::max(1, len(recs) // NP)][:NP]

TAGS = ["keep0.75", "keep0.50", "keep0.25", "int8", "int4", "int4_notsink", "int8_tokenwise",
        "lowrank256", "lowrank512", "lowrank256_Konly", "lowrank256_Vonly", "gqa2kv_merge"]
acc = {t: [] for t in TAGS}
lens = []

for msgs in recs:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    n = ids.shape[-1]
    lens.append(n)
    CAP.clear()
    PRE.clear()
    with torch.no_grad():
        model(ids, use_cache=False)
    base = {li: (CAP[li][0].clone(), CAP[li][1].clone(), CAP[li][2].clone()) for li in LAYERS}
    pre = {li: PRE[li].clone() for li in LAYERS}
    pos = torch.arange(n, device="cuda")[None]
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(torch.zeros(1, n, 1024, device="cuda",
                                                      dtype=torch.bfloat16), pos)
    for li in LAYERS:
        q, k, v = base[li]
        ref = attn_out(q, k, v)
        s = sig_scores(q, k)
        for frac, tag in ((0.75, "keep0.75"), (0.50, "keep0.50"), (0.25, "keep0.25")):
            b = max(1, int(round(frac * n)))
            acc[tag].append(relerr(attn_out(q, k, v, keep_idx(s, b, n)), ref))
        k8 = qdq(k, 8, "chan", 128).to(torch.bfloat16)
        v8 = qdq(v, 8, "token").to(torch.bfloat16)
        acc["int8"].append(relerr(attn_out(q, k8, v8), ref))
        acc["int8_tokenwise"].append(relerr(attn_out(q, qdq(k, 8, "token").to(torch.bfloat16),
                                                     v8), ref))
        k4 = qdq(k, 4, "chan", 128, keep_sinks=4).to(torch.bfloat16)
        v4 = qdq(v, 4, "token", keep_sinks=4).to(torch.bfloat16)
        acc["int4"].append(relerr(attn_out(q, k4, v4), ref))
        acc["int4_notsink"].append(relerr(attn_out(q, qdq(k, 4, "chan", 128).to(torch.bfloat16),
                                                   qdq(v, 4, "token").to(torch.bfloat16)), ref))
        # low rank on [K_preRoPE ; V]
        Kf = pre[li][0].float().reshape(n, -1)
        Vf = v[0].float().transpose(0, 1).reshape(n, -1)
        X = torch.cat([Kf, Vf], -1)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        for r, tag in ((256, "lowrank256"), (512, "lowrank512")):
            rr = min(r, Vh.shape[0])
            B = Vh[:rr]
            Xr = (X @ B.T) @ B
            Kr = Xr[:, :1024].reshape(n, 8, 128).permute(1, 0, 2)[None]
            Vr = Xr[:, 1024:].reshape(n, 8, 128).permute(1, 0, 2)[None]
            _, Kr = apply_rotary_pos_emb(Kr[:, :1], Kr, cos, sin)
            acc[tag].append(relerr(attn_out(q, Kr.to(torch.bfloat16), Vr.to(torch.bfloat16)), ref))
        # decomposition: K-only / V-only rank 256 (K uses 128 of its own dims, V 128)
        for which, tag in (("K", "lowrank256_Konly"), ("V", "lowrank256_Vonly")):
            M = Kf if which == "K" else Vf
            _, _, Vh2 = torch.linalg.svd(M, full_matrices=False)
            rr = min(128, Vh2.shape[0])
            B2 = Vh2[:rr]
            Mr = (M @ B2.T) @ B2
            if which == "K":
                Kr = Mr.reshape(n, 8, 128).permute(1, 0, 2)[None]
                _, Kr = apply_rotary_pos_emb(Kr[:, :1], Kr, cos, sin)
                acc[tag].append(relerr(attn_out(q, Kr.to(torch.bfloat16), v), ref))
            else:
                Vr = Mr.reshape(n, 8, 128).permute(1, 0, 2)[None]
                acc[tag].append(relerr(attn_out(q, k, Vr.to(torch.bfloat16)), ref))
        mk, mv = merge_kv(li, cos, sin)
        acc["gqa2kv_merge"].append(relerr(attn_out(q, mk, mv), ref))

print("rag_dev prompts: %d, token lengths %d..%d (mean %.0f)"
      % (len(recs), min(lens), max(lens), sum(lens) / len(lens)))
print("\n=== CALIBRATION: attention-output relerr, mean over 28 layers x %d prompts ===" % len(recs))
known = {"keep0.75": "rag_dev quality -0.01262", "keep0.50": "rag_dev quality -0.05538",
         "keep0.25": "rag_dev quality -0.33076", "gqa2kv_merge": "chat_dev 0.835 -> 0.114"}
cal = {}
for t in TAGS:
    x = torch.tensor(acc[t])
    cal[t] = dict(mean=x.mean().item(), p95=x.quantile(0.95).item(), max=x.max().item())
    print("%-18s mean %.5f  p95 %.5f  max %.5f   %s"
          % (t, x.mean(), x.quantile(0.95), x.max(), known.get(t, "")))

# =========================================================================== #
# (b)-(e) on the 4K haystack context
# =========================================================================== #
ids = haystack_ids(tok)
n = ids.shape[-1]
CAP.clear()
PRE.clear()
with torch.no_grad():
    model(ids, use_cache=False)

print("\n=== CLA: ridge prediction of layer l+1 KV from layer l KV (train 3072 / test 1024) ===")
cla = []
for li in range(27):
    A = torch.cat([PRE[li][0].float().reshape(n, -1),
                   CAP[li][2][0].float().transpose(0, 1).reshape(n, -1)], -1)
    Bm = torch.cat([PRE[li + 1][0].float().reshape(n, -1),
                    CAP[li + 1][2][0].float().transpose(0, 1).reshape(n, -1)], -1)
    tr, te = 3072, n - 3072
    At, Bt = A[:tr], Bm[:tr]
    G = At.T @ At
    lam = 1e-3 * torch.diagonal(G).mean()
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device), At.T @ Bt)
    cla.append(dict(pair="%d->%d" % (li, li + 1), train=relerr(At @ W, Bt),
                    test=relerr(A[tr:] @ W, Bm[tr:])))
t = torch.tensor([c["test"] for c in cla])
print("test relerr: mean %.4f  min %.4f (%s)  max %.4f"
      % (t.mean(), t.min(), cla[int(t.argmin())]["pair"], t.max()))
print("  " + "  ".join("%s:%.3f" % (c["pair"], c["test"]) for c in cla))

print("\n=== scale tail vs context length (fp16 min normal 6.104e-05) ===")
for npos in (512, 1024, 2048, 4096):
    for name, bits, axis, group in (("K", 8, "chan", 128), ("K", 4, "chan", 32),
                                    ("V", 8, "token", None)):
        mins, q1 = [], []
        cnt = 0
        for li in LAYERS:
            x = (CAP[li][1] if name == "K" else CAP[li][2])[:, :, :npos, :]
            _, s = qdq(x, bits, axis, group, ret_scale=True)
            mins.append(s.min().item())
            q1.append(s.flatten().quantile(0.0001).item())
            cnt += s.numel()
        print("  n=%5d %s int%d %-5s g%-4s  min %.3e  0.01%%-quantile %.3e  scales stored %d"
              % (npos, name, bits, axis, group, min(mins), sum(q1) / len(q1), cnt))

print("\n=== requantisation compounding: dequant -> drop half -> requant, 5 rounds ===")
for bits in (8, 4):
    seqs = []
    for li in LAYERS:
        cur = CAP[li][1].clone()
        ref = CAP[li][1].float().clone()
        row = []
        for rnd in range(5):
            cur = qdq(cur, bits, "chan", 128, keep_sinks=4).to(torch.bfloat16)
            row.append(relerr(cur.float(), ref))
            m = cur.shape[2]
            sel = torch.arange(0, m, 2, device=cur.device)[: m // 2]
            cur = cur[:, :, sel, :].contiguous()
            ref = ref[:, :, sel, :].contiguous()
        seqs.append(row)
    t = torch.tensor(seqs).mean(0)
    print("  int%d K per-channel g128: %s" % (bits, "  ".join("%.5f" % x for x in t.tolist())))

json.dump({"calibration": cal, "cla": cla}, open(
    os.path.join(ROOT, "experiments", "kv_significance", "results",
                             "store_probe.json"), "w"), indent=1)
print("\npeak %.0f MiB" % (torch.cuda.max_memory_allocated() / 2 ** 20))
