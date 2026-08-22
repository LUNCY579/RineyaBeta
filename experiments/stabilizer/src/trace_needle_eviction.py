#!/usr/bin/env python3
"""P2.1 -- is the needle actually evicted, and at which compression round?

THE PREMISE UNDER TEST. Option 3 from the P1 write-up proposes making eviction
reversible: a second tier that demotes rather than deletes, so an early mistake is
recoverable. That is only worth building if the mistake is real. Two outcomes
falsify it immediately:

  the needle SURVIVES to the end but is still not retrieved
      -> eviction is not the failure. Something else loses it -- position,
         attention dilution, the decode step -- and a demotion tier fixes nothing.

  the needle is deleted in the FIRST round after it arrives
      -> it never accumulates any score at all, so no ranking-based tier would
         have kept it either. The fix would have to be query-aware scoring
         reaching further back, not a second chance.

The informative outcome is a needle that survives several rounds and is then
dropped: that is a position which HAD evidence, lost the competition, and could
have been demoted instead of deleted. In that case the round at which it dies
tells us how large the cold tier must be to have caught it.

WHAT IS MEASURED. A 128K passkey prompt at each depth, run through the real
chunked-prefill path with the real policy. Original position identities are
carried alongside the cache through every gather, so after each compression we
know exactly whether the needle's tokens are still resident, in how many of the
28x8 (layer, KV head) streams, and what its rank was when it died.

Usage:
    python src/trace_needle_eviction.py
    python src/trace_needle_eviction.py --length 32768 --budget 2048
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

from scripts.kv import significance as sig  # noqa: E402
from scripts.kv.chunked_prefill import chunked_prefill, score_with_query  # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
PRE = ("There is an important piece of information hidden inside a lot of "
       "irrelevant text. Find it and memorise it. I will quiz you about it.\n\n")
SUF = "\n\nWhat is the pass key? The pass key is"
QUESTION = "What is the pass key? The pass key is"


def build(tok, n_tokens: int, depth: float, key: int):
    """Prompt plus the token span the needle occupies."""
    needle = f"\n\nThe pass key is {key}. Remember it. {key} is the pass key.\n\n"
    n_pre = len(tok(PRE)["input_ids"])
    n_suf = len(tok(SUF)["input_ids"])
    n_nee = len(tok(needle)["input_ids"])
    per = len(tok(FILLER)["input_ids"])
    reps = max(1, (n_tokens - n_pre - n_suf - n_nee) // max(per, 1))
    before = int(reps * depth)
    head = PRE + FILLER * before
    text = head + needle + FILLER * (reps - before) + SUF
    lo = len(tok(head)["input_ids"])
    return text, lo, lo + n_nee


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=131072)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--recent", type=int, default=512)
    ap.add_argument("--yarn", type=float, default=4.0)
    ap.add_argument("--cap-mib", type=float, default=7400.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path,
                    default=EXP / "results" / "needle_eviction_trace.json")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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

    # Record what select_keep chooses, without duplicating its logic here.
    KEEP: list[torch.Tensor] = []
    _orig_select = sig.select_keep

    def spy(s, n, policy, device):
        idx = _orig_select(s, n, policy, device)
        KEEP.append(idx.detach().clone())
        return idx

    sig.select_keep = spy
    import scripts.kv.significance as _sigmod
    _sigmod.select_keep = spy

    print(f"P2.1 -- needle eviction trace, n={args.length}, B={args.budget}, "
          f"chunk={args.chunk}, YaRN x{args.yarn}\n")
    rows = []
    for depth in args.depths:
        text, lo, hi = build(tok, args.length, depth, 75526)
        ids = tok(text, return_tensors="pt").to(args.device)["input_ids"]
        n = ids.shape[-1]
        qids = tok(QUESTION, return_tensors="pt").to(args.device)["input_ids"]

        sig.reset_scores()
        sig.set_recording(False)
        cache = sig.SignificanceCache(config=model.config, policy=None)
        policy = sig.Policy(budget=args.budget, sinks=4, recent=args.recent,
                            count_exponent=1.0)

        # Original position identity per (layer, head), carried through gathers.
        L = cfg.num_hidden_layers
        HKV = cfg.num_key_value_heads
        orig_ids: list[torch.Tensor] | None = None
        trace = []
        state = {"round": 0}

        def compress_fn(c, ci):
            nonlocal orig_ids
            stored_before = c.layers[0].keys.shape[-2]
            if orig_ids is None:
                orig_ids = [torch.arange(stored_before, device=args.device)
                            .view(1, 1, -1).expand(1, HKV, -1).clone()
                            for _ in range(L)]
            else:
                grown = stored_before - orig_ids[0].shape[-1]
                if grown > 0:
                    base = orig_ids[0].shape[-1]
                    add = torch.arange(base, base + grown, device=args.device)
                    # positions appended since the last compression
                    prev_total = state.get("consumed", base)
                    add = torch.arange(prev_total, prev_total + grown,
                                       device=args.device)
                    orig_ids = [torch.cat(
                        [o, add.view(1, 1, -1).expand(1, HKV, -1)], -1)
                        for o in orig_ids]
            state["consumed"] = c.layers[0].true_length

            KEEP.clear()
            score_with_query(model, c, qids,
                             position_base=c.layers[0].true_length, sig=sig)
            sig.compress(c, policy)

            # KEEP holds one index tensor per layer, in layer order.
            if len(KEEP) == L:
                orig_ids = [orig_ids[i].gather(2, KEEP[i]) for i in range(L)]
            alive, toks = 0, 0
            for i in range(L):
                hit = (orig_ids[i] >= lo) & (orig_ids[i] < hi)
                alive += int(hit.any(-1).sum().item())
                toks += int(hit.sum().item())
            state["round"] += 1
            # `arrived` distinguishes 'not there yet' from 'was there and
            # was dropped'. Without it the first rounds read as evictions.
            trace.append({"round": state["round"],
                          "stored_before": stored_before,
                          "arrived": bool(state["consumed"] > lo),
                          "streams_with_needle": alive,
                          "mean_tokens_kept": toks / (L * HKV),
                          "needle_len": hi - lo,
                          "of": L * HKV})

        out = chunked_prefill(model, ids, cache, chunk=args.chunk,
                              logits_to_keep=1, compress_fn=compress_fn)
        # one greedy step, enough to see whether the key comes out
        got = tok.decode(out.logits[:, -1:].argmax(-1)[0], skip_special_tokens=True)

        post = [t for t in trace if t["arrived"]]
        surv = [t["streams_with_needle"] for t in post]
        first_zero = next((t["round"] for t in post
                           if t["streams_with_needle"] == 0), None)
        keptf = post[-1]["mean_tokens_kept"] / post[-1]["needle_len"] if post else 0.0
        rows.append({"depth": depth, "needle_tokens": [lo, hi], "n": n,
                     "rounds": len(trace), "rounds_after_arrival": len(post),
                     "final_token_frac": keptf,
                     "final_streams": surv[-1] if surv else None,
                     "of": L * HKV, "first_round_fully_gone": first_zero,
                     "trace": trace, "first_token": got})
        print(f"depth {depth:.1f}: {len(trace)} rounds ({len(post)} after the "
              f"needle arrives), needle in {surv[-1] if surv else 0}/{L*HKV} "
              f"streams at the end, {keptf*100:.0f}% of its tokens kept"
              + (f", fully gone after round {first_zero}" if first_zero else
                 ""))
        print(f"   first token out: {got.strip()[:24]!r}")
        if surv:
            step = max(1, len(surv) // 8)
            print("   survival: " + " ".join(f"{v}" for v in surv[::step]))
        del cache, ids
        torch.cuda.empty_cache()

    sig.select_keep = _orig_select
    _sigmod.select_keep = _orig_select

    print("\nREADING THIS. A needle still present in many streams at the end means")
    print("eviction is not what loses it, and a demotion tier would not help. A")
    print("needle gone after round 1 means it never earned a score, and only")
    print("query-aware scoring reaching further back would help. Anything in")
    print("between is the case option 3 was designed for.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": MODEL, "length": args.length, "budget": args.budget,
         "chunk": args.chunk, "yarn": args.yarn, "results": rows},
        indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
