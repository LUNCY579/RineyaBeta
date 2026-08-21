"""Significance-gated MIXED precision: spend bits where the attention goes.

Uniform quantisation prices every cached token identically, but this cache is
not uniform -- 46.3% of attention mass lands on the first 4 tokens, and the
eviction mechanism already computes a per-position, per-KV-head score saying
where the rest goes. That score is free here; it has to be computed anyway.

So: sinks stay bf16, the top-p fraction by score goes int8, everything else
int4. Bytes are counted from the actual layout including scale overhead
(bf16 4096, int8 2112, int4 1088 B per token per layer) rather than from the
bit width alone, so the compression factors are the ones the allocator would
see.

Measured on rag_dev prompts, where quality deltas are on record, AND on a 4K
haystack -- short prompts leave the g=128 per-channel groups mostly padding,
which flatters every per-channel scheme.

Writes results/mixed_precision.json.

Usage:
    .venv/Scripts/python.exe experiments/kv_significance/src/mixed_precision.py [n_prompts]
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


def qdq_full(x, bits, axis, group=None):
    """Quantise every position; caller selects which rows to take."""
    x = x.float()
    qmax = 2 ** bits - 1
    if axis == "token":
        mn = x.amin(-1, keepdim=True)
        mx = x.amax(-1, keepdim=True)
        s = (mx - mn).clamp_min(1e-12) / qmax
        return ((x - mn) / s).round().clamp(0, qmax) * s + mn
    b, h, nn, d = x.shape
    g = min(group or nn, nn)
    pad = (-nn) % g
    xp = torch.cat([x, x[:, :, -1:, :].expand(b, h, pad, d)], 2) if pad else x
    xg = xp.view(b, h, -1, g, d)
    mn = xg.amin(3, keepdim=True)
    mx = xg.amax(3, keepdim=True)
    s = (mx - mn).clamp_min(1e-12) / qmax
    return (((xg - mn) / s).round().clamp(0, qmax) * s + mn).view(b, h, -1, d)[:, :, :nn, :]


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


def mixed(k, v, s, n_hi, sinks=4):
    """bf16 sinks, int8 for the top-n_hi positions by score (per KV head), int4 elsewhere."""
    k8 = qdq_full(k, 8, "chan", 128)
    v8 = qdq_full(v, 8, "token")
    k4 = qdq_full(k, 4, "chan", 128)
    v4 = qdq_full(v, 4, "token")
    npos = k.shape[2]
    m = s.clone()
    m[..., :sinks] = float("-inf")
    hi = torch.zeros_like(s, dtype=torch.bool)
    if n_hi > 0:
        hi.scatter_(2, m.topk(min(n_hi, npos - sinks), -1).indices, True)
    hi_e = hi.unsqueeze(-1)
    kq = torch.where(hi_e, k8, k4)
    vq = torch.where(hi_e, v8, v4)
    keepmask = torch.zeros(npos, dtype=torch.bool, device=k.device)
    keepmask[:sinks] = True
    kq = torch.where(keepmask.view(1, 1, -1, 1), k.float(), kq)
    vq = torch.where(keepmask.view(1, 1, -1, 1), v.float(), vq)
    return kq.to(torch.bfloat16), vq.to(torch.bfloat16)


def bytes_per_tok_layer(frac_hi, n, sinks=4):
    """bf16 4096; int8 layout 2112; int4 layout 1088 bytes per token per layer."""
    fs = sinks / n
    return fs * 4096 + frac_hi * 2112 + max(0.0, 1 - fs - frac_hi) * 1088


recs = []
with open(os.path.join(ROOT, "artifacts", "eval_sets", "rag_dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        recs.append([m for m in d["messages"] if m["role"] != "assistant"])
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 24
recs = recs[::max(1, len(recs) // NP)][:NP]

VARIANTS = [("pure_int8", None), ("pure_int4", 0), ("mix_top64", 64), ("mix_top32", 32),
            ("mix_top16", 16), ("mix_p0.25", "p25"), ("mix_p0.50", "p50")]
acc = {v[0]: [] for v in VARIANTS}
byt = {v[0]: [] for v in VARIANTS}
lens = []
for msgs in recs:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    n = ids.shape[-1]
    lens.append(n)
    CAP.clear()
    with torch.no_grad():
        model(ids, use_cache=False)
    for li in LAYERS:
        q, k, v = CAP[li]
        ref = attn_out(q, k, v)
        s = sig_scores(q, k)
        for name, spec in VARIANTS:
            if spec is None:
                kq = qdq_full(k, 8, "chan", 128).to(torch.bfloat16)
                vq = qdq_full(v, 8, "token").to(torch.bfloat16)
                fh = 1.0
            else:
                nh = {"p25": int(0.25 * n), "p50": int(0.50 * n)}.get(spec, spec if isinstance(spec, int) else 0)
                nh = min(nh, max(0, n - 4))
                kq, vq = mixed(k, v, s, nh)
                fh = nh / n
            acc[name].append(relerr(attn_out(q, kq, vq), ref))
            byt[name].append(bytes_per_tok_layer(fh, n))

print("rag_dev prompts %d, lengths %d..%d mean %.0f" % (len(recs), min(lens), max(lens), sum(lens) / len(lens)))
print("\n=== significance-gated mixed precision, rag_dev ===")
print("reference points: keep0.75 relerr 0.00937 (quality -0.01262) | keep0.50 0.03729 (-0.05538)"
      " | keep0.25 0.11381 (-0.33076)")
out = {}
for name, _ in VARIANTS:
    e = torch.tensor(acc[name])
    b = torch.tensor(byt[name])
    out[name] = dict(relerr=e.mean().item(), p95=e.quantile(0.95).item(),
                     bytes=b.mean().item(), factor=4096.0 / b.mean().item())
    print("%-12s relerr mean %.5f p95 %.5f   %6.0f B/tok/layer  %.2fx vs bf16"
          % (name, e.mean(), e.quantile(0.95), b.mean(), 4096.0 / b.mean()))

# ---- same on a 4K haystack, where per-channel groups are actually full ------
ids = haystack_ids(tok)
n = ids.shape[-1]
CAP.clear()
with torch.no_grad():
    model(ids, use_cache=False)
print("\n=== same, 4K haystack context ===")
acc2 = {v[0]: [] for v in VARIANTS}
byt2 = {v[0]: [] for v in VARIANTS}
for li in LAYERS:
    q, k, v = CAP[li]
    ref = attn_out(q, k, v)
    s = sig_scores(q, k)
    for name, spec in VARIANTS:
        if spec is None:
            kq = qdq_full(k, 8, "chan", 128).to(torch.bfloat16)
            vq = qdq_full(v, 8, "token").to(torch.bfloat16)
            fh = 1.0
        else:
            nh = {"p25": int(0.25 * n), "p50": int(0.50 * n)}.get(spec, spec if isinstance(spec, int) else 0)
            kq, vq = mixed(k, v, s, nh)
            fh = nh / n
        acc2[name].append(relerr(attn_out(q, kq, vq), ref))
        byt2[name].append(bytes_per_tok_layer(fh, n))
for name, _ in VARIANTS:
    e = torch.tensor(acc2[name])
    b = torch.tensor(byt2[name])
    out[name + "@4K"] = dict(relerr=e.mean().item(), bytes=b.mean().item(),
                             factor=4096.0 / b.mean().item())
    print("%-12s relerr mean %.5f max %.5f (L%d)  %6.0f B/tok/layer  %.2fx"
          % (name, e.mean(), e.max(), int(e.argmax()), b.mean(), 4096.0 / b.mean()))

json.dump(out, open(os.path.join(ROOT, "experiments", "kv_significance", "results",
                             "mixed_precision.json"), "w"),
          indent=1)
print("\npeak %.0f MiB" % (torch.cuda.max_memory_allocated() / 2 ** 20))
