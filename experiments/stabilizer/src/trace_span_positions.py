#!/usr/bin/env python3
"""P2.2 -- WHICH position inside the evidence span is lost?

THE OBSERVATION THIS EXPLAINS. At the transition budget the mechanism fails in a
very specific way: across 32K, 64K and 128K, failures recover a mean of 4.00 of
the key's 5 digits and never fewer than one, and at 128K/B=64 all 26 of 26
failures are correct prefixes. The model finds the span, reproduces it verbatim
including the trailing ". Remember it", and misses precisely the last digit.

WHY THE OBVIOUS EXPLANATION IS WRONG. The first guess was that a centred max-pool
under-protects a span's trailing edge. It cannot: select_keep pools with
kernel=pool, stride=1, padding=pool//2, which is symmetric, so the last digit's
window still reaches back over the earlier digits and inherits their score
through the max. Both edges are covered identically.

THE TWO REMAINING CANDIDATES, which this separates.

  TIE-BREAK AT THE BUDGET BOUNDARY. Pooling flattens a span to near-equal scores,
      so top-k among its positions is close to a tie and the budget boundary can
      cut through the group. Predicts the lost position is ARBITRARY and roughly
      uniform over the span, and that survival is near-equal across offsets.

  A SCORING GRADIENT ALONG THE SPAN. The last digit genuinely earns less score.
      Predicts survival falls monotonically toward the span's end, the same way
      in every stream.

These make opposite predictions about the SHAPE of per-offset survival, so one
trace decides it.

WHAT IS MEASURED. The prompt is built by importing passkey.build_prompt, so it is
byte-identical to the one the floor ladder ran -- not a re-implementation that
could drift. Original position identities are carried alongside the cache through
every gather, and after the final compression we report, for each token offset
inside the needle, how many of the 28x8 (layer, KV head) streams still hold it.
The needle contains the key TWICE ("The pass key is K. Remember it. K is the pass
key."), so both copies are visible: if only the first one matters, that is itself
worth knowing.

Usage:
    python src/trace_span_positions.py --length 131072 --budgets 64 128
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
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.5])
    ap.add_argument("--key", type=int, default=75526)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--yarn", type=float, default=4.0)
    ap.add_argument("--cap-mib", type=float, default=7400.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path,
                    default=EXP / "results" / "span_position_trace.json")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import passkey as pk  # the harness itself, so the prompt cannot drift

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
    _orig_select = sig.select_keep

    def spy(s, n, policy, device):
        idx = _orig_select(s, n, policy, device)
        KEEP.append(idx.detach().clone())
        return idx

    sig.select_keep = spy
    import scripts.kv.significance as _sigmod
    _sigmod.select_keep = spy

    rows = []
    for depth in args.depths:
        for budget in args.budgets:
            text, _ = pk.build_prompt(tok, args.length, depth, args.key)
            ids = tok(text, return_tensors="pt").to(args.device)["input_ids"]
            n = ids.shape[-1]
            qids = tok(QUESTION, return_tensors="pt").to(args.device)["input_ids"]

            needle = (f"\n\nThe pass key is {args.key}. Remember it. "
                      f"{args.key} is the pass key.\n\n")
            head = text[:text.index(needle)]
            lo = len(tok(head)["input_ids"])
            hi = lo + len(tok(needle)["input_ids"])
            flat = ids[0].tolist()
            span_toks = [tok.decode([t]) for t in flat[lo:hi]]
            # VERIFY the offsets: boundary effects could shift them silently.
            joined = "".join(span_toks)
            ok = "pass key is" in joined
            print(f"\n=== depth {depth} budget {budget} ===")
            print(f"needle at [{lo}, {hi}) of {n}; offsets verified: {ok}")
            if not ok:
                print(f"  !! span decodes as {joined[:60]!r} -- offsets unreliable")

            sig.reset_scores()
            sig.set_recording(False)
            cache = sig.SignificanceCache(config=model.config, policy=None)
            policy = sig.Policy(budget=budget, sinks=4,
                                recent=max(4, budget // 4),
                                pool=args.pool, count_exponent=1.0)

            orig_ids = None
            state = {"consumed": 0}

            def compress_fn(c, ci):
                nonlocal orig_ids
                stored = c.layers[0].keys.shape[-2]
                if orig_ids is None:
                    orig_ids = [torch.arange(stored, device=args.device)
                                .view(1, 1, -1).expand(1, HKV, -1).clone()
                                for _ in range(L)]
                else:
                    grown = stored - orig_ids[0].shape[-1]
                    if grown > 0:
                        prev = state["consumed"]
                        add = torch.arange(prev, prev + grown, device=args.device)
                        orig_ids = [torch.cat(
                            [o, add.view(1, 1, -1).expand(1, HKV, -1)], -1)
                            for o in orig_ids]
                state["consumed"] = c.layers[0].true_length
                KEEP.clear()
                score_with_query(model, c, qids,
                                 position_base=c.layers[0].true_length, sig=sig)
                sig.compress(c, policy)
                if len(KEEP) == L:
                    orig_ids = [orig_ids[i].gather(2, KEEP[i]) for i in range(L)]

            out = chunked_prefill(model, ids, cache, chunk=args.chunk,
                                  logits_to_keep=1, compress_fn=compress_fn)
            got = tok.decode(out.logits[:, -1:].argmax(-1)[0],
                             skip_special_tokens=True)

            surv = []
            for off in range(hi - lo):
                pos = lo + off
                alive = sum(int((orig_ids[i] == pos).any(-1).sum().item())
                            for i in range(L))
                surv.append(alive)
            total = L * HKV

            print(f"first token out: {got.strip()[:20]!r}")
            print(f"{'off':>4} {'token':<12} {'streams':>8} {'share':>7}")
            for off, a in enumerate(surv):
                t = span_toks[off].replace("\n", "\\n")
                mark = "  <-- digit" if t.strip().isdigit() else ""
                print(f"{off:>4} {t!r:<12} {a:>8} {a / total:>6.0%}{mark}")

            digits = [(o, span_toks[o].strip(), surv[o])
                      for o in range(hi - lo) if span_toks[o].strip().isdigit()]
            rows.append({"depth": depth, "budget": budget, "n": n,
                         "span": [lo, hi], "tokens": span_toks,
                         "survival": surv, "of": total,
                         "digits": digits, "first_token": got,
                         "offsets_verified": ok})
            del cache, ids
            torch.cuda.empty_cache()

    sig.select_keep = _orig_select
    _sigmod.select_keep = _orig_select

    print("\nREADING THIS. Near-equal survival across the span's offsets means the")
    print("budget boundary cut an effective tie, and the lost position is arbitrary.")
    print("Survival falling monotonically toward the span's end means the score")
    print("itself has a gradient, and the fix belongs in scoring rather than pooling.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": MODEL, "length": args.length, "pool": args.pool,
         "chunk": args.chunk, "yarn": args.yarn, "key": args.key,
         "results": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
