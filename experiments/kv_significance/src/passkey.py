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
# STAGE 2 of the reasoning programme: SPAN LENGTH.
#
# P2.5 measured that the pool's required backward reach tracks the evidence span --
# a d-digit key is a d-token span, and back=4 rescues d=5 while scoring 0.000 at d=7.
# Digits cap the span at what a plausible key can be. Natural-language evidence runs
# 20-40 tokens, an order of magnitude beyond anything measured, so the span axis has
# to be extended with something other than digits.
#
# Every word below is ONE token for this tokeniser (verified against the vocabulary),
# so W words is exactly a W-token span and span length stays exactly controllable --
# which is the whole point of doing this synthetically before touching real prose.
NEEDLE_WORDS = [
    "amber", "anchor", "barrel", "basket", "beacon", "blanket", "bottle", "bridge",
    "bucket", "button", "camera", "carpet", "ceiling", "cluster", "compass", "copper",
    "cotton", "crystal", "desert", "diamond", "drawer", "engine", "fabric", "forest",
    "garage", "garden", "granite", "hammer", "harbor", "harvest", "helmet", "island",
    "jacket", "kitchen", "lantern", "market", "mirror", "mountain", "needle", "orange",
    "packet", "parcel", "planet", "pocket", "quarry", "river", "rocket", "shadow",
    "shelter", "signal", "silver", "socket", "station", "stream", "sugar", "summit",
    "sunset", "temple", "thread", "ticket", "tomato", "tower", "tunnel", "window",
    "winter", "yellow",
]
SUFFIX_WORDS = chr(10) * 2 + "What is the pass phrase? The pass phrase is"
QUESTION_WORDS = "What is the pass phrase? The pass phrase is"

QUERY_HINT = ("What is the pass key? Watch for it as you read." + chr(10)*2)





def build_chain_prompt(tok, target_tokens: int, depths, values, query_first=False):
    """STAGE 3: a multi-hop chain, where the answer needs SEVERAL spans at once.

    Cornerstone 2's claim is that evidence lives in contiguous spans and that ranking
    positions independently fragments them. Passkey contains exactly ONE span, so the
    claim has never been tested where it makes its sharpest prediction: N hops need N
    intact spans held SIMULTANEOUSLY, which should make the required budget grow with
    the number of hops rather than with the context length (prediction P2).

    The chain is built so hop i is addressed by hop i-1's VALUE:

        The pass key is 75526. Remember it.
        If the code is 75526, then the next code is 41883. Remember it.
        If the code is 41883, then the next code is 90210. Remember it.
        -> What is the final code?  answer: 90210

    That structure matters for a second reason. Only the FIRST segment is addressed by
    words the question contains; every later one is addressed by a number the question
    has never seen. P2.2 measured that query-aware scoring keeps text resembling the
    query -- framing survived in 54% of streams against 35% for the payload -- so the
    later hops have nothing query-shaped to earn score from. If budget alone cannot
    rescue them, the binding resource is query-shaped access rather than capacity, and
    that is a different mechanism from the one stage 1 found.

    Segments are placed at the given depths in order, so hop distance through the
    context is controlled rather than incidental.
    """
    segs = []
    for i, v in enumerate(values):
        if i == 0:
            segs.append(chr(10) * 2 + f"The pass key is {v}. Remember it." + chr(10) * 2)
        else:
            segs.append(chr(10) * 2
                        + f"If the code is {values[i - 1]}, then the next code "
                          f"is {v}. Remember it." + chr(10) * 2)
    suffix = chr(10) * 2 + "What is the final code? The final code is"
    n_pre = len(tok(PREAMBLE)["input_ids"])
    n_suf = len(tok(suffix)["input_ids"])
    n_seg = sum(len(tok(s)["input_ids"]) for s in segs)
    per = len(tok(FILLER)["input_ids"])
    reps = max(1, (target_tokens - n_pre - n_suf - n_seg) // max(per, 1))
    head = QUERY_HINT + PREAMBLE if query_first else PREAMBLE
    text = head
    placed = 0
    for d, s in zip(depths, segs):
        want = int(reps * d)
        text += FILLER * max(0, want - placed) + s
        placed = max(placed, want)
    text += FILLER * max(0, reps - placed) + suffix
    return text, len(tok(text)["input_ids"])


def build_prompt(tok, target_tokens: int, depth: float, key,
                 query_first: bool = False, n_distractors: int = 0,
                 distractor_style: str = "framed") -> tuple[str, int]:
    """Filler with the key at `depth` through it, padded to ~target_tokens."""
    # A string key means STAGE 2: the payload is a word phrase, not digits.
    words = isinstance(key, str)
    label = "pass phrase" if words else "pass key"
    suffix = SUFFIX_WORDS if words else SUFFIX
    needle = (chr(10) * 2 + f"The {label} is {key}. Remember it. "
              + f"{key} is the {label}." + chr(10) * 2)
    # STAGE 1 of the reasoning programme: competition.
    #
    # Passkey's haystack contains no digits at all, so the real key is the only
    # thing of its kind in the entire context and localising it is trivial. P2.2
    # measured that the score keeps text RESEMBLING THE QUERY -- framing survives
    # in 54% of streams against 35% for the payload -- which predicts decoys shaped
    # like the question should be preferentially retained and crowd the answer out.
    # `framed` decoys share the question's sentence frame under a DIFFERENT label,
    # so the true answer stays unambiguous; `bare` decoys carry a number with no
    # frame overlap, isolating mere numeric competition from query similarity.
    decoys = []
    if n_distractors:
        drng = random.Random(f"{key}:decoys")
        labels = ["door code", "locker number", "badge id", "vault code",
                  "seat number", "gate code", "room number", "case file"]
        nd = 0 if words else len(str(key))
        for i in range(n_distractors):
            if words:
                dkey = " ".join(drng.sample(NEEDLE_WORDS, len(key.split())))
            else:
                dkey = drng.randint(10 ** (nd - 1), 10 ** nd - 1)
            if distractor_style == "framed":
                lab = labels[i % len(labels)]
                decoys.append(f"\n\nThe {lab} is {dkey}. Remember it. "
                              f"{dkey} is the {lab}.\n\n")
            else:
                decoys.append(f"\n\nThe reading that morning was {dkey}, "
                              f"noted and filed.\n\n")
    n_pre = len(tok(PREAMBLE)["input_ids"])
    n_suf = len(tok(suffix)["input_ids"])
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
    tail_reps = reps - before
    if decoys:
        # Spread decoys through the remaining filler rather than co-locating them.
        text = head + FILLER * before + needle
        step = max(1, tail_reps // (len(decoys) + 1))
        used = 0
        for dec in decoys:
            text += FILLER * step + dec
            used += step
        text += FILLER * max(0, tail_reps - used) + suffix
    else:
        text = head + FILLER * before + needle + FILLER * tail_reps + suffix
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
    ap.add_argument("--pool", type=int, default=1,
                    help="SnapKV max-pool width over scores before top-k. "
                         "1 disables it, which was the DEFAULT for every "
                         "long-context result in this program -- a needle "
                         "traced at 128K kept only 57-65%% of its tokens, i.e. "
                         "the span survived in fragments. Pooling exists to "
                         "stop exactly that.")
    ap.add_argument("--model", default=MODEL,
                    help="model directory, relative to the repo root. The "
                         "mechanism keys off the ATTENTION SHAPE (28 layers, "
                         "16 query / 8 KV heads, head_dim 128), so any core "
                         "sharing it runs unchanged -- the published Halo "
                         "cores do. Vocabulary and dtype differ and do not "
                         "matter to eviction.")
    ap.add_argument("--min-heavy-frac", type=float, default=0.0,
                    help="B1: floor on the score's share of the budget. See "
                         "run_eval_sig.py; kept here so the two harnesses expose "
                         "the same policy surface.")
    ap.add_argument("--sinks", type=int, default=4,
                    help="always-kept prefix. Fixed at 4 in every result to "
                         "date and never ablated, which matters because at "
                         "B=16 that is 25%% of the budget: the headline claims "
                         "EIGHT significance-chosen positions retrieve a needle, "
                         "and if the sinks carry part of that the count is wrong. "
                         "0 puts the whole budget under the score.")
    ap.add_argument("--select", choices=("significance","random","streaming"),
                    default="significance",
                    help="BASELINE arms. `random` keeps the same number of "
                         "positions chosen uniformly, isolating the score from "
                         "the budget. `streaming` spends the budget on sinks "
                         "plus recent and ranks nothing. Without these the "
                         "mechanism is only ever compared to itself.")
    ap.add_argument("--hops", type=int, default=1,
                    help="STAGE 3: chain length. 1 is ordinary passkey. N>1 "
                         "builds N segments where hop i is addressed by hop "
                         "i-1's VALUE, so only the first is reachable from the "
                         "question's wording. N hops need N intact spans at "
                         "once -- the direct test of the span account.")
    ap.add_argument("--hop-depths", type=float, nargs="+",
                    default=[0.1, 0.4, 0.7],
                    help="where each chain segment sits, in order.")
    ap.add_argument("--needle-words", type=int, default=0,
                    help="STAGE 2: payload is a W-word phrase, not digits. Every "
                         "word is one token so W words is exactly a W-token span "
                         "-- the axis P2.5 could only probe to 7 with digits. "
                         "0 keeps the numeric key.")
    ap.add_argument("--distractors", type=int, default=0,
                    help="decoy keys competing with the real one. Passkey has "
                         "none, which is why 8 heavy slots suffice at B=16.")
    ap.add_argument("--distractor-style", choices=("framed", "bare"),
                    default="framed",
                    help="framed decoys share the question sentence frame under "
                         "a different label; bare ones do not.")
    ap.add_argument("--grow", type=float, default=None,
                    help="DERIVE the pool reach instead of fixing it: geodesic "
                         "dilation gated at this quantile (0.99 = top 1%% may be "
                         "grown into). P2.5 showed a fixed reach cannot serve "
                         "variable span lengths -- back=4 gives 1.000 on a 5-digit "
                         "key and 0.000 on a 7-digit one.")
    ap.add_argument("--grow-max", type=int, default=64,
                    help="iteration bound for --grow.")
    ap.add_argument("--key-digits", type=int, default=5,
                    help="digits in the pass key. Every digit is its own token, so "
                         "this IS the evidence span length. P2.4 found the pool's "
                         "backward reach must cover the span: back=4 rescues a "
                         "5-digit key, back=3 does not. Varying this tests whether "
                         "the threshold tracks span length.")
    ap.add_argument("--pool-back", type=int, default=None,
                    help="score-pool reach BEHIND a position; default pool//2. "
                         "Setting this alone gives a causal window, which credits "
                         "a payload for the query-matching label preceding it "
                         "(P2.2: framing survives at 54%%, payload at 35%%).")
    ap.add_argument("--pool-fwd", type=int, default=None,
                    help="score-pool reach AHEAD of a position; default pool//2.")
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

    path = REPO / args.model
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
                for ti in range(args.trials):
                    # Seed torch per trial so the `random` baseline is
                    # reproducible. A control that cannot be re-run is not a
                    # control, and every other arm here is deterministic.
                    torch.manual_seed(args.seed + ti)
                    if args.hops > 1:
                        # The answer is the LAST link; every earlier value exists
                        # only to be chased through.
                        vals = [rng.randint(10 ** (args.key_digits - 1),
                                            10 ** args.key_digits - 1)
                                for _ in range(args.hops)]
                        key = vals[-1]
                    elif args.needle_words:
                        key = " ".join(rng.sample(NEEDLE_WORDS,
                                                  args.needle_words))
                    else:
                        key = rng.randint(10 ** (args.key_digits - 1),
                                          10 ** args.key_digits - 1)
                    if args.hops > 1:
                        # `depth` shifts the whole chain, so the depth sweep still
                        # varies how many compression rounds the chain must survive.
                        ds = [min(0.95, d + depth - args.hop_depths[0])
                              for d in args.hop_depths[:args.hops]]
                        text, real_n = build_chain_prompt(
                            tok, n, ds, vals, query_first=args.query_first)
                    else:
                        text, real_n = build_prompt(
                            tok, n, depth, key,
                            query_first=args.query_first,
                            n_distractors=args.distractors,
                            distractor_style=args.distractor_style)
                    ids = tok(text, return_tensors="pt").to(args.device)["input_ids"]
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    # budget - SINKS, not budget - 4. Hardcoding 4 here silently
                    # broke streaming for any other sink count: heavy() would be
                    # (sinks - 4) instead of 0, so the baseline would stop being
                    # StreamingLLM without saying so.
                    rec = ((budget - args.sinks) if args.select == 'streaming'
                           else args.recent)
                    policy = (sig.Policy(budget=budget, sinks=args.sinks, recent=rec,
                                         pool=args.pool,
                                         select=args.select,
                                         pool_back=args.pool_back,
                                         pool_fwd=args.pool_fwd,
                                         grow=args.grow,
                                         grow_max=args.grow_max,
                                         normalize=args.normalize,
                                         count_exponent=args.count_exponent)
                              if budget else None)
                    cache = sig.SignificanceCache(config=model.config, policy=None)
                    try:
                        if args.hops > 1:
                            qtext = "What is the final code? The final code is"
                        elif args.needle_words:
                            qtext = QUESTION_WORDS
                        else:
                            qtext = QUESTION
                        qids = (tok(qtext, return_tensors="pt")
                                .to(args.device)["input_ids"]
                                if args.query_aware else None)
                        got = answer(model, tok, ids, cache, args.chunk, policy,
                                     args.window, query_ids=qids,
                                     max_new=max(12, 4 * args.needle_words + 6))
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
                    # Keep enough of the reply to DIAGNOSE a failure, not just
                    # score it. 32 characters suffices for a 5-digit key and
                    # truncates a 20-word phrase (142 chars) to nothing usable,
                    # which made stage 2's failures unclassifiable -- the same
                    # class of defect as the pool=1 default: correct for the case
                    # it was written for, silently wrong for the next one.
                    keep = max(32, len(str(key)) + 24)
                    trials.append({"key": key, "got": got.strip()[:keep],
                                   "hit": hit})
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
                # HOW MANY COMPRESSIONS DID THIS NEEDLE ACTUALLY FACE?
                # chunked_prefill never compresses the final chunk, by design.
                # A needle inside that chunk therefore faces NONE -- and the row
                # used to record no chunk index and no round count, so a vacuous
                # cell was byte-indistinguishable from one that survived thirty
                # rounds. Three published Halo 16K depth-0.9 cells were exactly
                # this: real_tokens 15714, chunk 4096, exempt region >= 12288,
                # needle at ~14142. A policy keeping NOTHING scores 1.00 there.
                # Recorded now so no cell can hide it again.
                _nchunks = max(1, -(-real_n // args.chunk)) if args.chunk else 1
                _faced = max(0, (_nchunks - 1) - (int(depth * real_n) // args.chunk))                     if args.chunk else 0
                rows.append({"length": n, "real_tokens": real_n, "budget": budget,
                             "depth": depth, "acc": acc, "peak_mib": round(peak, 1),
                             "n_chunks": _nchunks, "compressions_faced": _faced,
                             "exempt_from_compression": bool(budget and _faced == 0),
                             "yarn": args.yarn, "trials_ok": seen, "ooms": ooms,
                             "rank_exponent": (policy.rank_exponent() if policy
                                               else None),
                             "trials": trials})
                shown = f"{acc:>7.2f}" if acc is not None else f"{chr(45)*2:>7}"
                warn = f"  [{ooms}/{args.trials} OOM]" if ooms else ""
                print(f"{n:>8,}{budget:>8}{depth:>7.1f}{shown}{peak:>10.0f}{warn}  "
                      + "  ".join(ex))

    b0 = next((r.get("budget") for r in rows if r.get("budget")), 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        # cap_mib and window belong in the manifest because they change the
        # OUTCOME, not only the speed. The cap bounds RESERVED memory, so a cell
        # can OOM at an allocated peak far below it, and a result file that does
        # not record its cap cannot explain its own empty cells. Three memory
        # figures in this program had no recoverable source for exactly this.
        {"model": args.model, "yarn": args.yarn, "chunk": args.chunk,
         # EFFECTIVE recent, not the flag. The streaming baseline overrides it
         # to budget-4 so heavy() reaches 0, and recording args.recent made the
         # manifest claim heavy=8 for a run that had heavy=0. The policy was
         # right and the record was wrong, which is the harder failure to catch.
         "recent": ((b0 - args.sinks) if args.select == "streaming" else args.recent),
         "recent_flag": args.recent, "trials": args.trials, "window": args.window,
         "sinks": args.sinks, "pool": args.pool,
         "pool_back": args.pool_back, "pool_fwd": args.pool_fwd,
         "key_digits": args.key_digits,
         "grow": args.grow, "grow_max": args.grow_max,
         "needle_words": args.needle_words,
         "hops": args.hops, "hop_depths": args.hop_depths,
         "select": args.select,
         "distractors": args.distractors,
         "distractor_style": args.distractor_style,
         "cap_mib": args.cap_mib, "device_total_mib": round(tot, 1),
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
