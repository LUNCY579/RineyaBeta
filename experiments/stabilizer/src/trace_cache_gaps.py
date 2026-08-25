#!/usr/bin/env python3
"""O13 -- is `back` measured in TOKENS or in CACHE ENTRIES?

THE ANOMALY. At 128K/B=16 with backward reach, depth 0.5 scores 1.00 / 0.42 / 1.00 at
chunk 4096 / 2048 / 1024 while facing 15 / 31 / 61 compression rounds. More rounds is
not monotonically worse, so O7's round-count account does not explain it. Straddling is
ruled out: the needle sits inside one chunk at all three sizes. Seven of twelve failures
are the 4-of-5 truncation, which is the P2 fragmentation mode.

THE HYPOTHESIS THIS TESTS. `select_keep` pools over the score tensor of the CACHE, whose
entries are whatever survived earlier compressions plus the new chunk. Cache entries are
contiguous in *cache index*, not in *original position*. So `pooled[i] = max(s[i-back]
… s[i])` reaches back six CACHE ENTRIES, and how far that is in the text depends on how
compacted the cache was at that moment.

If the hypothesis holds, the effective reach in tokens is not a fixed quantity, and
sections 4.4 and 4.5 -- every reach result -- were measured at chunk 4096 where the
cache is least compacted at the critical compression, i.e. in the most favourable case.

WHAT IS MEASURED. Original position identities are carried through every gather. At each
compression, for the cache entries lying inside the needle's span and for the `back`
entries preceding them, we report:

  cache_gap    the original-position distance between ADJACENT cache entries in the
               window. 1 means the window covers contiguous text; larger means it
               skips.
  reach_tokens how far back in ORIGINAL POSITIONS a `back`-entry window actually
               reaches. This is the number section 4.4 assumes is `back`.

A window whose reach_tokens stays ~= back at chunk 4096 and grows at 2048 confirms the
hypothesis and forces the reach law to be restated in cache coordinates.

Usage:
    python src/trace_cache_gaps.py --chunk 4096
    python src/trace_cache_gaps.py --chunk 2048 --depth 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))
sys.path.insert(0, str(REPO / "experiments/kv_significance/src"))

from scripts.kv import significance as sig  # noqa: E402
from scripts.kv.chunked_prefill import chunked_prefill, score_with_query  # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
QUESTION = "What is the pass key? The pass key is"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=131072)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--back", type=int, default=6)
    ap.add_argument("--key", type=int, default=75526)
    ap.add_argument("--yarn", type=float, default=4.0)
    ap.add_argument("--cap-mib", type=float, default=2200.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (EXP / "results" / f"cache_gaps_chunk{args.chunk}.json")

    from compat import force_repeat_kv
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import passkey as pk

    force_repeat_kv(verbose=False)
    sig.patch_attention()
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, tot) / tot)

    tok = AutoTokenizer.from_pretrained(REPO / MODEL)
    cfg = AutoConfig.from_pretrained(REPO / MODEL)
    if args.yarn:
        orig = int(cfg.max_position_embeddings)
        base = float((getattr(cfg, "rope_parameters", None) or {}).get(
            "rope_theta", 1_000_000.0))
        params = {"rope_type": "yarn", "factor": float(args.yarn),
                  "rope_theta": base, "original_max_position_embeddings": orig}
        cfg.rope_parameters = params
        cfg.rope_scaling = dict(params)
        cfg.max_position_embeddings = int(orig * args.yarn)
    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, config=cfg, dtype=torch.bfloat16).to(args.device).eval()

    L, HKV = cfg.num_hidden_layers, cfg.num_key_value_heads

    KEEP: list[torch.Tensor] = []
    _orig = sig.select_keep

    def spy(s, n, policy, device):
        idx = _orig(s, n, policy, device)
        KEEP.append(idx.detach().clone())
        return idx

    sig.select_keep = spy
    import scripts.kv.significance as _sigmod
    _sigmod.select_keep = spy

    text, _ = pk.build_prompt(tok, args.length, args.depth, args.key)
    ids = tok(text, return_tensors="pt").to(args.device)["input_ids"]
    qids = tok(QUESTION, return_tensors="pt").to(args.device)["input_ids"]
    needle = (f"\n\nThe pass key is {args.key}. Remember it. "
              f"{args.key} is the pass key.\n\n")
    head = text[:text.index(needle)]
    lo = len(tok(head)["input_ids"])
    hi = lo + len(tok(needle)["input_ids"])
    print(f"O13 -- chunk={args.chunk}, depth={args.depth}, back={args.back}")
    print(f"needle at [{lo}, {hi}) of {ids.shape[-1]}\n")

    sig.reset_scores()
    sig.set_recording(False)
    cache = sig.SignificanceCache(config=model.config, policy=None)
    policy = sig.Policy(budget=args.budget, sinks=4, recent=max(4, args.budget // 4),
                        pool=7, pool_back=args.back, pool_fwd=0, count_exponent=1.0)

    orig_ids = None
    state = {"consumed": 0, "round": 0}
    rows = []

    def compress_fn(c, ci):
        nonlocal orig_ids
        stored = c.layers[0].keys.shape[-2]
        if orig_ids is None:
            orig_ids = [torch.arange(stored, device=args.device)
                        .view(1, 1, -1).expand(1, HKV, -1).clone() for _ in range(L)]
        else:
            grown = stored - orig_ids[0].shape[-1]
            if grown > 0:
                prev = state["consumed"]
                add = torch.arange(prev, prev + grown, device=args.device)
                orig_ids = [torch.cat([o, add.view(1, 1, -1).expand(1, HKV, -1)], -1)
                            for o in orig_ids]
        state["consumed"] = c.layers[0].true_length

        # BEFORE the gather: what does a `back`-entry window actually span?
        ob = orig_ids[0][0, 0]                       # layer 0, head 0
        inside = ((ob >= lo) & (ob < hi)).nonzero().flatten()
        if inside.numel():
            last = int(inside[-1].item())
            w0 = max(0, last - args.back)
            span_tokens = int(ob[last].item() - ob[w0].item())
            gaps = (ob[w0 + 1:last + 1] - ob[w0:last]).tolist() if last > w0 else []
            rows.append({"round": state["round"], "stored_before": stored,
                         "entries_in_needle": int(inside.numel()),
                         "reach_tokens": span_tokens,
                         "reach_entries": last - w0,
                         "max_gap": max(gaps) if gaps else 0,
                         "contiguous": all(g == 1 for g in gaps) if gaps else True})

        KEEP.clear()
        score_with_query(model, c, qids, position_base=c.layers[0].true_length, sig=sig)
        sig.compress(c, policy)
        if len(KEEP) == L:
            orig_ids = [orig_ids[i].gather(2, KEEP[i]) for i in range(L)]
        state["round"] += 1

    o = chunked_prefill(model, ids, cache, chunk=args.chunk, logits_to_keep=1,
                        compress_fn=compress_fn)
    got = tok.decode(o.logits[:, -1:].argmax(-1)[0], skip_special_tokens=True)

    sig.select_keep = _orig
    _sigmod.select_keep = _orig

    print(f"{'round':>6} {'stored':>8} {'in needle':>10} {'reach entries':>14}"
          f" {'reach TOKENS':>13} {'max gap':>8} {'contiguous':>11}")
    for r in rows:
        print(f"{r['round']:>6} {r['stored_before']:>8} {r['entries_in_needle']:>10}"
              f" {r['reach_entries']:>14} {r['reach_tokens']:>13} {r['max_gap']:>8}"
              f" {str(r['contiguous']):>11}")
    noncontig = [r for r in rows if not r["contiguous"]]
    print(f"\n  compressions where the window was NOT contiguous in text: "
          f"{len(noncontig)}/{len(rows)}")
    if noncontig:
        worst = max(noncontig, key=lambda r: r["reach_tokens"])
        print(f"  worst: a {worst['reach_entries']}-entry window spanned "
              f"{worst['reach_tokens']} TOKENS (max gap {worst['max_gap']})")
        print("  -> `back` is measured in CACHE ENTRIES, and its reach in text varies.")
    else:
        print("  -> the window was contiguous in text at every compression, so `back`")
        print("     means tokens here and the O13 hypothesis is refuted for this cell.")
    print(f"\n  first token out: {got.strip()[:20]!r}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "chunk": args.chunk, "depth": args.depth,
                               "budget": args.budget, "back": args.back,
                               "needle": [lo, hi], "rounds": rows,
                               "first_token": got}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
