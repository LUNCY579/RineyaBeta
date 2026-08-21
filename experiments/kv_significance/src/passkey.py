#!/usr/bin/env python3
"""Passkey retrieval at length -- the first long-context behaviour measured here.

Everything in this program so far is either (a) a memory identity or (b) a
quality number from prompts of 41-330 tokens. This is the first measurement of
what the model can actually DO at length, and it is only runnable because
chunked prefill removed the memory obstacle.

The task is the standard passkey probe: a 5-digit key is hidden at a controlled
depth inside repetitive filler, and the model is asked to repeat it. Repetitive
filler is deliberate -- it isolates RETRIEVAL from comprehension, so a failure
means "could not find or could not attend", not "could not understand".

TWO VARIABLES ARE CONFOUNDED BY DEFAULT AND THE DESIGN SEPARATES THEM.

  position   R2 has `max_position_embeddings` 40,960 and `rope_scaling: null`,
             so beyond ~41K it is extrapolating with no training. Failure there
             is a positional failure, not a KV-compression failure.
  budget     eviction keeps B of n tokens.

Running the full cache at each length gives the positional ceiling; running
eviction against that ceiling gives the compression cost. Reporting only the
second without the first would attribute positional collapse to eviction.

`--yarn F` applies YaRN RoPE scaling, which is training-free and extends the
usable range to `40960 * F`. Whether it actually works on a 0.6B checkpoint is
an open question this measures rather than assumes.

Usage:
    python src/passkey.py --lengths 2048 4096 8192 16384 --budgets 0
    python src/passkey.py --lengths 32768 65536 --budgets 0 2048 --yarn 4.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

from scripts.kv import significance as sig  # noqa: E402
from scripts.kv.chunked_prefill import (causal_lower_right, chunked_prefill,  # noqa: E402
                                        score_with_query, stored_length)

MODEL = "artifacts/rineya-core-en32k-28L"
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
PREAMBLE = ("There is an important piece of information hidden inside a lot of "
            "irrelevant text. Find it and memorise it. I will quiz you about it.\n\n")
SUFFIX = "\n\nWhat is the pass key? The pass key is"
QUESTION = "What is the pass key? The pass key is"
QUERY_HINT = ("What is the pass key? Watch for it as you read." + chr(10)*2)



def build_prompt(tok, target_tokens: int, depth: float, key: int,
                 query_first: bool = False) -> tuple[str, int]:
    """Filler with the key at `depth` through it, padded to ~target_tokens."""
    needle = f"\n\nThe pass key is {key}. Remember it. {key} is the pass key.\n\n"
    n_pre = len(tok(PREAMBLE)["input_ids"])
    n_suf = len(tok(SUFFIX)["input_ids"])
    n_needle = len(tok(needle)["input_ids"])
    n_fill = max(0, target_tokens - n_pre - n_suf - n_needle)
    per = len(tok(FILLER)["input_ids"])
    reps = max(1, n_fill // max(per, 1))
    before = int(reps * depth)
    head = PREAMBLE
    if query_first:
        # Put the QUESTION at the start as well. Prefill-time eviction scores
        # each chunk against that chunk's own last query rows, which for early
        # chunks are filler -- so a needle can never earn a score before the
        # question is seen. Making the question visible from chunk 0 tests
        # exactly that mechanism and nothing else: the needle, the budget, the
        # depth and the length are unchanged.
        head = QUERY_HINT + PREAMBLE
    text = head + FILLER * before + needle + FILLER * (reps - before) + SUFFIX
    return text, len(tok(text)["input_ids"])


@torch.no_grad()
def answer(model, tok, ids, cache, chunk: int, policy, window: int,
           max_new: int = 12, query_ids=None) -> str:
    """Chunked prefill, then greedy decode with explicit true positions.

    `query_ids` switches scoring from query-BLIND to query-AWARE. Blind scoring
    ranks each chunk by its own trailing query rows, which for every chunk but
    the last are filler, so a needle earns no score before the question is seen.
    Aware scoring runs the question against the cache between chunks and undoes
    the pass, so ranking reflects what was actually asked.
    """
    n = ids.shape[-1]
    sig.reset_scores()
    # With query-aware scoring the chunk's own rows must NOT contribute, or the
    # filler signal is simply added back on top of the question signal.
    sig.set_recording(policy is not None and query_ids is None, window=window)

    def compress_fn(c, ci):
        if query_ids is not None:
            score_with_query(model, c, query_ids,
                             position_base=c.layers[0].true_length, sig=sig)
        sig.compress(c, policy)

    out = chunked_prefill(model, ids, cache, chunk=chunk, logits_to_keep=1,
                          compress_fn=(compress_fn if policy is not None else None))
    tokens = []
    tok_id = out.logits[:, -1:].argmax(-1)
    for i in range(max_new):
        tokens.append(int(tok_id))
        if tok_id.item() in (tok.eos_token_id, tok.pad_token_id):
            break
        stored = stored_length(cache)
        mask = causal_lower_right(1, stored + 1, ids.device,
                                  next(model.parameters()).dtype)
        pos = torch.tensor([[n + i]], device=ids.device)
        out = model(tok_id, past_key_values=cache, position_ids=pos,
                    attention_mask={"full_attention": mask},
                    use_cache=True, logits_to_keep=1)
        tok_id = out.logits[:, -1:].argmax(-1)
    sig.set_recording(False)
    return tok.decode(tokens, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--budgets", type=int, nargs="+", default=[0],
                    help="0 = full cache (the positional ceiling)")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--recent", type=int, default=512)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--count-exponent", type=float, default=None,
                    help="rank on sum/count**e; 0 = raw sum, 1 = mean, between = decay")
    ap.add_argument("--normalize", action="store_true",
                    help="rank on MEAN score (sum / observations) rather than the "
                         "running sum, which otherwise freezes the heavy set")
    ap.add_argument("--query-aware", action="store_true",
                    help="score each chunk against the QUESTION instead of its "
                         "own trailing rows; the question is never cached")
    ap.add_argument("--query-first", action="store_true",
                    help="also state the question at the start of the prompt")
    ap.add_argument("--yarn", type=float, default=None,
                    help="YaRN factor; extends usable range to 40960*F, training-free")
    ap.add_argument("--cap-mib", type=float, default=6144.0,
                    help="bounds RESERVED memory, not allocated. A run can OOM "
                         "with a reported peak well below this: the caching "
                         "allocator holds fragmentation the peak never shows, and "
                         "torch.cat on a growing cache keeps two buffers live. If "
                         "a cell OOMs at a peak under the cap, raise the cap -- "
                         "this default hid the 32K uncompressed reference for the "
                         "whole program. The cap exists at all because Windows/WDDM "
                         "spills to host RAM instead of failing.")
    ap.add_argument("--seed", type=int, default=20260820,
                    help="key sequence; identical in every cell, so depths and "
                         "budgets are paired on needles. Change it for a second "
                         "independent replicate.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "passkey.json")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    force_repeat_kv(verbose=False)
    sig.patch_attention()
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, tot) / tot)

    path = REPO / MODEL
    tok = AutoTokenizer.from_pretrained(path)
    cfg = AutoConfig.from_pretrained(path)
    if args.yarn:
        # transformers 5.x reads `config.rope_parameters`, not `rope_scaling`
        # (_compute_yarn_parameters: `base = rope_parameters_dict["rope_theta"]`).
        # Setting rope_scaling alone leaves rope_parameters at its default and the
        # YaRN initialiser reads base=None, raising inside pow().
        orig = int(cfg.max_position_embeddings)
        base = float((getattr(cfg, "rope_parameters", None) or {}).get(
            "rope_theta", 1_000_000.0))
        params = {"rope_type": "yarn", "factor": float(args.yarn),
                  "rope_theta": base,
                  "original_max_position_embeddings": orig}
        cfg.rope_parameters = params
        cfg.rope_scaling = dict(params)
        cfg.max_position_embeddings = int(orig * args.yarn)

    model = AutoModelForCausalLM.from_pretrained(
        path, config=cfg, dtype=torch.bfloat16).to(args.device).eval()

    print(f"passkey retrieval, R2{'  YaRN x' + str(args.yarn) if args.yarn else ''}")
    print(f"usable positional range: {cfg.max_position_embeddings:,} "
          f"(rope={getattr(cfg, 'rope_parameters', None)})")
    print(f"chunk {args.chunk}, recent {args.recent}, "
          f"{args.trials} trials x {len(args.depths)} depths\n")
    print(f"{'n':>8}{'budget':>8}{'depth':>7}{'acc':>7}{'peak MiB':>10}  examples")

    rows = []
    for n in args.lengths:
        for budget in args.budgets:
            for depth in args.depths:
                # RESET PER CELL, so every cell draws the SAME key sequence and
                # the depths are PAIRED on needles. Running one stream across the
                # whole nested loop gave each cell a different 6 keys, which put
                # key difficulty and depth in the same column: an identical
                # configuration scored 0.17 at depth 0.5 when it drew keys 7-12
                # inside a three-depth sweep and 0.50 when it drew keys 1-6 in a
                # single-depth run. Pairing removes that variance instead of
                # averaging over it.
                rng = random.Random(args.seed)
                hits, seen, ooms, peak, ex, trials = 0, 0, 0, 0.0, [], []
                for _ in range(args.trials):
                    key = rng.randint(10000, 99999)
                    text, real_n = build_prompt(tok, n, depth, key,
                                                query_first=args.query_first)
                    ids = tok(text, return_tensors="pt").to(args.device)["input_ids"]
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    policy = (sig.Policy(budget=budget, sinks=4, recent=args.recent,
                                         normalize=args.normalize,
                                         count_exponent=args.count_exponent)
                              if budget else None)
                    cache = sig.SignificanceCache(config=model.config, policy=None)
                    try:
                        qids = (tok(QUESTION, return_tensors="pt")
                                .to(args.device)["input_ids"]
                                if args.query_aware else None)
                        got = answer(model, tok, ids, cache, args.chunk, policy,
                                     args.window, query_ids=qids)
                    except torch.cuda.OutOfMemoryError:
                        # A trial that ran out of memory measured NOTHING. Scoring
                        # it as a miss is the difference between "the model could
                        # not find the key" and "the run did not happen", and this
                        # host makes that mistake easy: a full-cache 64K reference
                        # needs 8,072 MiB against 8,188 MiB total, so it OOMs at a
                        # plausible length and the table still looks like a result.
                        ooms += 1
                        peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 20)
                        del cache, ids
                        torch.cuda.empty_cache()
                        continue
                    peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 20)
                    seen += 1
                    hit = str(key) in got
                    hits += hit
                    # Per-trial records, so a paired analysis is possible after the
                    # fact. Cell-level accuracy alone cannot support one.
                    trials.append({"key": key, "got": got.strip()[:32], "hit": hit})
                    if len(ex) < 2:
                        ex.append(f"{key}->{got.strip()[:18]!r}")
                    del cache, ids
                    torch.cuda.empty_cache()
                # None, not 0.0, when every trial died: an empty cell must not be
                # readable as a measured failure.
                acc = hits / seen if seen else None
                # Record the EFFECTIVE exponent, not the flag: `--normalize` and
                # `--count-exponent 1.0` are the same run and must not be
                # distinguishable in the results file. (A manifest that stored the
                # raw argument instead of the effective value has already caused
                # one wrong published number in this program.)
                rows.append({"length": n, "real_tokens": real_n, "budget": budget,
                             "depth": depth, "acc": acc, "peak_mib": round(peak, 1),
                             "yarn": args.yarn, "trials_ok": seen, "ooms": ooms,
                             "rank_exponent": (policy.rank_exponent() if policy
                                               else None),
                             "trials": trials})
                shown = f"{acc:>7.2f}" if acc is not None else f"{chr(45)*2:>7}"
                warn = f"  [{ooms}/{args.trials} OOM]" if ooms else ""
                print(f"{n:>8,}{budget:>8}{depth:>7.1f}{shown}{peak:>10.0f}{warn}  "
                      + "  ".join(ex))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": MODEL, "yarn": args.yarn, "chunk": args.chunk,
         "recent": args.recent, "trials": args.trials,
         "count_exponent": args.count_exponent, "normalize": args.normalize,
         "query_aware": args.query_aware, "seed": args.seed,
         "results": rows},
        indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    n_oom = sum(r["ooms"] for r in rows)
    if n_oom:
        print(f"\n{n_oom} trial(s) ran out of memory and were EXCLUDED, not scored.")
        print("Cells shown as -- measured nothing. Re-run them with a smaller")
        print("budget, a smaller chunk, or a higher --cap-mib. Do NOT read an")
        print("empty cell as a failure: that is how a 64K full-cache reference,")
        print("which needs 8,072 MiB against 8,188 MiB total, becomes a result.")

    full = [r for r in rows if r["budget"] == 0 and r["acc"] is not None]
    if full:
        print("\npositional ceiling (full cache) by length:")
        for n in sorted({r['length'] for r in full}):
            a = [r["acc"] for r in full if r["length"] == n]
            print(f"  {n:>8,}: {sum(a)/len(a):.2f}")
        print("Read eviction cost against THIS, not against 1.0 -- a length where")
        print("the full cache already fails cannot show a compression effect.")


if __name__ == "__main__":
    main()
