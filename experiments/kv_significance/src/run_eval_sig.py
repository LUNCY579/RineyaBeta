#!/usr/bin/env python3
"""Quality sweep for significance-gated eviction, protocol-identical to run_gates.py.

Same protocol as the gate baselines and the attention-shape sweep: chat template
with `enable_thinking=False`, `max_new_tokens=32`, greedy, `eos_only`, visible
token count passed to the scorer as both `token_count` and `visible_model_tokens`.

`model.generate` is used unmodified -- eviction happens inside the cache, so this
cannot drift from the reference the way a hand-rolled decode loop would.

Budget can be absolute (`--budget 128`) or a fraction of each prompt's length
(`--keep-frac 0.5`). The fraction form is usually the meaningful one, because
these prompts are 41-330 tokens and a fixed budget means very different
compression ratios across the set.

Run `--budget 0` first: eviction off, which produces the reference every other
run is compared against.

Usage:
    python src/run_eval_sig.py --budget 0 --dataset chat_dev
    python src/run_eval_sig.py --keep-frac 0.5 --dataset chat_dev --compare-to off
    python src/run_eval_sig.py --keep-frac 0.25 --pool 7 --dataset rag_dev --compare-to off
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))
sys.path.insert(0, str(REPO))

from scripts.kv import significance as sig  # noqa: E402
from compat import force_repeat_kv  # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
DATASETS = {
    "chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
    "rag_dev": "artifacts/eval_sets/rag_dev.jsonl",
    # Long-context natural language, built by make_long_rag.py. Same questions,
    # references and response contracts as rag_dev -- so the same validated
    # scorer applies and rag_dev itself is the native-length baseline. Only the
    # amount of real irrelevant prose around the answer changes. These close the
    # 543x gap between this programme's language evidence (median 238 tokens)
    # and its retrieval evidence (125,826 tokens). All fit inside the model's
    # 40,960 native positions, so no YaRN is involved and LENGTH IS THE ONLY
    # VARIABLE against rag_dev.
    "rag_long4k": "artifacts/eval_sets/rag_long4k_d0.5.jsonl",
    "rag_long16k": "artifacts/eval_sets/rag_long16k_d0.5.jsonl",
    "rag_long32k": "artifacts/eval_sets/rag_long32k_d0.5.jsonl",
    "rag_long4k_d0.1": "artifacts/eval_sets/rag_long4k_d0.1.jsonl",
    "rag_long4k_d0.9": "artifacts/eval_sets/rag_long4k_d0.9.jsonl",
    "rag_long16k_d0.1": "artifacts/eval_sets/rag_long16k_d0.1.jsonl",
    "rag_long16k_d0.9": "artifacts/eval_sets/rag_long16k_d0.9.jsonl",
}


def normalize(record: dict) -> dict:
    rec = dict(record)
    msgs = list(rec.get("messages") or [])
    reference = None
    if msgs and msgs[-1].get("role") == "assistant":
        reference = str(msgs[-1].get("content") or "")
        msgs = msgs[:-1]
    rec["messages"] = msgs
    rec["reference"] = rec.get("reference") or reference
    meta = rec.get("metadata") or {}
    rec["category"] = (rec.get("category") or meta.get("category")
                       or meta.get("reply_class") or "uncategorized")
    return rec


DOC_HEADER = re.compile(r"\[doc_[^\]\n]*\]")


def segment_records(prompt: str, enc) -> list[tuple[list[int], list[int]]]:
    """Split the prompt into the records it already prints. [(token idx, ids)].

    STOP AT THE END OF THE EVIDENCE BLOCK. The last record otherwise runs to the end
    of the prompt and swallows `<QUESTION> ... </QUESTION>`, so the trailing
    distractor contains the query verbatim and wins BM25 on every single case -- an
    arm that always selects the last document while looking like retrieval. Caught by
    scripts/kv/test_lexical_select.py, which asserts on the SCORE of the
    answer-bearing record and not merely on the gold span surviving: the span
    survived anyway, at 1.000, so the span-retention proxy could not see the bug.

    Shared by lexical-doc-bm25 and its granularity-matched floor random-doc, so the
    two cannot drift apart in how they carve the prompt -- the only thing that may
    differ between them is the score each record gets.
    """
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]
    heads = [m.start() for m in DOC_HEADER.finditer(prompt)]
    tail = prompt.find("</FACTUAL_EVIDENCE>")
    end = tail if heads and tail > heads[0] else len(prompt)
    heads = [h for h in heads if h < end]
    if len(heads) < 2:
        raise ValueError(
            f"found {len(heads)} record delimiters before the end of the evidence "
            f"block; the record-level arms need the prompt's own structure")
    bounds = heads + [end]
    segments = []
    for i in range(len(heads)):
        a0, z0 = bounds[i], bounds[i + 1]
        tok_ids = [j for j, (a, z) in enumerate(offsets)
                   if a >= a0 and z <= z0 and z > a]
        if tok_ids:
            segments.append((tok_ids, [ids[j] for j in tok_ids]))
    if not segments:
        raise ValueError("record segmentation produced nothing")
    return segments


def doc_random_weights(prompt: str, enc) -> torch.Tensor:
    """THE GRANULARITY-MATCHED FLOOR: one uniform draw per record, broadcast.

    `random-prepool` is matched to the per-position arms' 7-token pooled contiguity,
    not to the ~64-token record contiguity that lexical-doc-bm25 gets for free by
    keeping whole records. Measured on the campaign's own prompts, a record-uniform
    floor retains the required phrase at 0.224 against random-prepool's 0.126, so
    roughly ten points of the document arm's apparent advantage is granularity
    rather than lexical signal. This arm is that granularity with the signal removed,
    and it is the floor the document arm is scored against.

    Uses the RNG seeded per case by the caller, exactly as the other floors do.
    """
    segments = segment_records(prompt, enc)
    w = torch.zeros(len(enc["input_ids"]), dtype=torch.float32)
    for (tok_ids, _), d in zip(segments, torch.rand(len(segments))):
        w[tok_ids] = float(d)
    return w


def instruction_prefix_len(prompt: str, enc) -> int:
    """Tokens before the first evidence delimiter: system prompt plus preamble.

    NOT a constant. It is 100 tokens on 532 of the 600 rag_long4k prompts and
    131-140 on the other 68, because those carry TRUSTED_PROCEDURES content instead
    of "(none)". A fixed `--sinks 100` would therefore under-protect 68 cases and
    quietly reintroduce the abstention collapse on exactly the prompts that carry
    the most instruction text.
    """
    heads = list(DOC_HEADER.finditer(prompt))
    if not heads:
        return 0
    first = heads[0].start()
    return sum(1 for a, z in enc["offset_mapping"] if z <= first and z > a)


def _bm25(segments, qids, k1: float, b: float) -> list[float]:
    """BM25 (Lucene non-negative IDF) of each segment against the question tokens.

    Shared by the delimiter-based and the structure-free selectors so the two differ
    ONLY in how the prompt is carved, never in how a carve is scored.
    """
    n_docs = len(segments)
    df: Counter = Counter()
    for _, toks in segments:
        for t in set(toks):
            df[t] += 1
    avgdl = sum(len(t) for _, t in segments) / n_docs
    out = []
    for _, toks in segments:
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for q in qids:
            f = tf.get(q, 0)
            if f:
                idf = math.log(1 + (n_docs - df[q] + 0.5) / (df[q] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        out.append(s)
    return out


def line_bm25_weights(prompt: str, enc, question: str, tok, chunk: int = 0,
                      k1: float = 1.5, b: float = 0.75) -> torch.Tensor:
    """DELIMITER-FREE BUT BOUNDARY-AWARE: segment on newlines, then BM25.

    Fixed-size chunking fails on this task -- at B=256 it scores 0.7222 on the answer
    stratum against the record-based selector's 0.8056, because a 64-token window
    splits a 63-token record and the answer is cut in half. But the record-based
    selector needs the prompt to print `[doc_...]` headers, which is a property of
    this benchmark and not of prompts in general.

    Newlines are the middle ground: they are universal in real prompts, they are not
    a bespoke schema, and they track the boundaries an author actually wrote. This
    selector needs no schema at all -- only that the text has lines.
    """
    ids = enc["input_ids"]
    n = len(ids)
    offs = enc["offset_mapping"]
    qat = prompt.rfind(question)
    bounds = [0] + [i + 1 for i, ch in enumerate(prompt) if ch == chr(10)] + [len(prompt)]
    bounds = sorted(set(bounds))
    segments = []
    for a0, z0 in zip(bounds, bounds[1:]):
        if z0 - a0 < 2:
            continue
        if qat >= 0 and z0 > qat:
            continue                      # the query's own line is not a candidate
        tid = [j for j, (a, z) in enumerate(offs) if a >= a0 and z <= z0 and z > a]
        if tid:
            segments.append((tid, [ids[j] for j in tid]))
    if len(segments) < 2:
        raise ValueError(f"line-bm25: {len(segments)} lines; this selector needs a "
                         f"prompt with line structure")
    qids = set(tok(question, add_special_tokens=False)["input_ids"])
    qids |= set(tok(" " + question, add_special_tokens=False)["input_ids"])
    scores = _bm25(segments, qids, k1, b)
    top = max(scores)
    w = torch.zeros(n, dtype=torch.float32)
    if top > 0:
        for (tid, _), sc in zip(segments, scores):
            if sc > 0:
                w[tid] = sc / top
    return w


def chunk_bm25_weights(prompt: str, enc, question: str, tok, chunk: int = 64,
                       k1: float = 1.5, b: float = 0.75) -> torch.Tensor:
    """STRUCTURE-FREE. Fixed-size chunks, no delimiters, same BM25.

    The delimiter-based selector only works because rag_long prints
    `[doc_... | trust=... ]` headers. Most prompts do not. This carves the prompt into
    fixed `chunk`-token windows instead and is otherwise identical, so the comparison
    isolates exactly one thing: whether the method needs the prompt to mark its own
    records.

    The chunk containing the question is excluded from scoring. That is not a use of
    prompt structure -- it is what any retriever does with its own query, and the
    question string is already an input to this function. Without it the query chunk
    always ranks first, which is harmless for the ordering (max-normalisation is
    rank-preserving and the question sits inside the protected recent window) but
    wastes ranked slots on the straddling chunk.
    """
    ids = enc["input_ids"]
    n = len(ids)
    qspan = None
    qat = prompt.rfind(question)
    if qat >= 0:
        offs = enc["offset_mapping"]
        qtok = [j for j, (a, z) in enumerate(offs) if z > qat and z > a]
        if qtok:
            qspan = (min(qtok), n)
    segments = []
    for i in range(0, n, chunk):
        idx = list(range(i, min(i + chunk, n)))
        if qspan and idx[-1] >= qspan[0]:
            continue                      # the query's own chunk is not a candidate
        segments.append((idx, [ids[j] for j in idx]))
    if len(segments) < 2:
        raise ValueError(f"chunk-bm25: {len(segments)} chunks at chunk={chunk}; "
                         f"the prompt is too short for this selector")
    qids = set(tok(question, add_special_tokens=False)["input_ids"])
    qids |= set(tok(" " + question, add_special_tokens=False)["input_ids"])
    scores = _bm25(segments, qids, k1, b)
    top = max(scores)
    w = torch.zeros(n, dtype=torch.float32)
    if top > 0:
        for (idx, _), s in zip(segments, scores):
            if s > 0:
                w[idx] = s / top
    return w


def protect_instructions(w: torch.Tensor, prompt: str, enc) -> torch.Tensor:
    """Give every position BEFORE the evidence block the top weight.

    WHY. The record-level selectors write 0.0 everywhere outside the evidence block,
    so the system prompt survives only through the four attention sinks -- 4 of 71
    positions. On rag_long that is catastrophic and measurable: the abstention
    instruction and its verbatim refusal sentence live only there, and
    lexical-doc-bm25 scores 0.0000 against the uncompressed model's 0.8214 on the
    abstain stratum, 0/138 discordant.

    That is a defect of the selector, not a limit of compression, and it is repaired
    by a rule any deployed system would apply anyway: never evict the instructions.
    The rule stays BLIND -- it reads the position of the prompt's own evidence
    delimiter and nothing about the answer.
    """
    heads = list(DOC_HEADER.finditer(prompt))
    if not heads:
        return w
    first = heads[0].start()
    offsets = enc["offset_mapping"]
    for j, (a, z) in enumerate(offsets):
        if z <= first:
            w[j] = 1.0
    return w


def oracle_support_weights(prompt: str, enc, support_ids) -> torch.Tensor:
    """MECHANISM PROBE, NOT AN ARM. Keep only the gold record(s).

    This reads `support_document_ids` and is therefore an ORACLE. It may never be
    reported as a control or compared to the mechanism as though it were one. Its
    only job is to put a CEILING on selection: if keeping exactly the right record
    and discarding all ~64 distractors does not beat the uncompressed model, then
    the uncompressed model's remaining failures are not caused by distractor
    interference and no selector -- clever or trivial -- can fix them.

    Raises if the support record cannot be located, rather than silently degrading
    to an all-zero field, which would make the oracle a random arm wearing an
    oracle's name.
    """
    segments = segment_records(prompt, enc)
    want = [str(x) for x in (support_ids or [])]
    if not want:
        # An ABSTAIN case has no gold record by construction, so "keep only the gold
        # record" is legitimately "keep none of the evidence". That is not the silent
        # degradation the guard below exists to catch, and it cannot contaminate the
        # probe's conclusion: this probe is only ever read on the ANSWER stratum,
        # where every case has a support id (verified 144/144 and 72/72).
        return torch.zeros(len(enc["input_ids"]), dtype=torch.float32)
    offsets = enc["offset_mapping"]
    heads = [m for m in DOC_HEADER.finditer(prompt)]
    w = torch.zeros(len(enc["input_ids"]), dtype=torch.float32)
    hit = 0
    for tok_ids, _ in segments:
        a0 = offsets[tok_ids[0]][0]
        header = next((m.group(0) for m in heads if m.start() <= a0 <= m.end() + 4), "")
        if not header:
            header = next((m.group(0) for m in heads if m.start() >= a0), "")
        if any(sid in header for sid in want):
            w[tok_ids] = 1.0
            hit += 1
    if hit == 0:
        raise ValueError(f"oracle-support: none of {want} matched a printed record "
                         f"header; the oracle would be an all-zero field")
    return w


def doc_bm25_weights(prompt: str, enc, question: str, tok,
                     k1: float = 1.5, b: float = 0.75) -> torch.Tensor:
    """THE STRONGEST TRIVIAL BASELINE: BM25 over the prompt's own record structure.

    The evidence block prints its own delimiters --
    `[doc_v175_fact_225_... | trust=standard | updated_at=...] text` -- so a
    selector needing no forward pass can segment the prompt into records, score
    each record against the question with BM25 (Lucene non-negative IDF) at
    RECORD-level document
    frequency, and broadcast a record's score to all of its positions.

    WHY THIS ARM EXISTS. The per-position `lexical-idf` arm computes df as a term
    count inside ONE document, which is inverse collection frequency on a
    collection of size one -- degenerate, and it is not what IDF means. Measured on
    the exact 200/100 records this campaign runs, per-position IDF retains the gold
    span in 0.736 / 0.681 of answerable cases while this arm retains it in
    **1.000 / 1.000** (144/144 and 72/72). A control that loses to an obvious
    variant of itself cannot support the conclusion "the score does work no lexical
    index reproduces", so the weaker arm alone would have made a SUBSTANTIVE
    verdict unsound.

    Still trivial under the preregistration's own definition: it reads the
    `question` field, the prompt's token ids and the delimiters the prompt already
    prints. It never touches `support_document_ids`, the gold phrase or the depth.
    """
    segments = segment_records(prompt, enc)
    ids = enc["input_ids"]
    qids = set(tok(question, add_special_tokens=False)["input_ids"])
    qids |= set(tok(" " + question, add_special_tokens=False)["input_ids"])
    # Delegated to the shared scorer so the delimiter-based and structure-free arms
    # cannot drift in their BM25 while claiming to differ only in segmentation.
    # Verified identical to the inline version this replaced on all 600 prompts.
    scores = _bm25(segments, qids, k1, b)
    top = max(scores)
    w = torch.zeros(len(ids), dtype=torch.float32)
    if top > 0:
        for (tok_ids, _), s in zip(segments, scores):
            if s > 0:
                w[tok_ids] = s / top
    return w


def lexical_weights(ids: list[int], question: str, tok, variant: str) -> torch.Tensor:
    """Per-position trivial lexical weight in [0, 1]. ANSWER-BLIND by construction.

    Reads only the prompt's token ids and the record's `question` string. It never
    sees the gold phrase, the support document id or the needle depth -- which is
    what makes it a control rather than an oracle.

      lexical-naive  1.0 where the token id occurs in the tokenised question.
                     The literal reading, and badly unselective: on rag_long the
                     question's stopwords and punctuation flag 22.6% of all
                     positions, so this arm keeps nearly every match and ranks
                     almost nothing.
      lexical-idf    log(n/df(t)) normalised to [0, 1], with df the token's count
                     in THIS prompt. Parameter-free -- no threshold is tuned, so
                     there is no knob that could have been fitted to the answer.

    The question is tokenised bare AND with a leading space. Note what that does and
    does not buy: in a byte-level BPE only the FIRST word changes form between the two
    tokenisations -- every later word already carries its leading space in both -- so
    the union adds the space-prefixed variant of the first word and nothing else. An
    earlier version of this docstring claimed both forms were matched for every word,
    which is false; the union is kept because it is strictly better than either
    tokenisation alone, not because it is complete.
    """
    n = len(ids)
    df = Counter(ids)
    qids = set(tok(question, add_special_tokens=False)["input_ids"])
    qids |= set(tok(" " + question, add_special_tokens=False)["input_ids"])
    if variant == "lexical-naive":
        return torch.tensor([1.0 if t in qids else 0.0 for t in ids],
                            dtype=torch.float32)
    w = {t: math.log(n / df[t]) for t in qids if df.get(t)}
    mx = max(w.values(), default=0.0)
    if mx <= 0.0:
        # Every question token occupies every position -- impossible on real
        # prompts, but a zero divisor would produce a NaN field that topk would
        # resolve arbitrarily, so it is refused rather than normalised away.
        raise ValueError("degenerate IDF: no question token is rarer than the "
                         "whole prompt")
    return torch.tensor([w.get(t, 0.0) / mx if t in w else 0.0 for t in ids],
                        dtype=torch.float32)


def load_records(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(normalize(json.loads(line)))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--budget", type=int, default=0, help="0 = eviction off")
    ap.add_argument("--keep-frac", type=float, default=None,
                    help="budget as a fraction of each prompt's length")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--recent", type=int, default=32)
    ap.add_argument("--recent-frac", type=float, default=None,
                    help="recent window as a fraction of the budget. With a fixed "
                         "recent=32 and a small budget the always-kept regions "
                         "consume nearly the whole allowance -- at keep 0.25 on a "
                         "150-token prompt only 5%% of the budget was actually "
                         "significance-selected, making that cell StreamingLLM "
                         "rather than a test of scoring.")
    ap.add_argument("--pool", type=int, default=1)
    # Cornerstone 2 -- directional pooling. These existed in passkey.py from the
    # day the mechanism was found and were never added here, so every
    # natural-language number in this programme was measured with the CENTRED
    # window that cornerstone 2 superseded. Without these flags the mechanism
    # cannot be evaluated on language at all.
    ap.add_argument("--pool-back", type=int, default=None,
                    help="backward reach of the pooling window. Cornerstone 2's "
                         "result is that reach BACKWARD is what decides; forward "
                         "reach is inert. Requires --pool > 1.")
    ap.add_argument("--pool-fwd", type=int, default=None,
                    help="forward reach. Set 0 with --pool-back for the causal "
                         "window. Requires --pool > 1.")
    ap.add_argument("--grow", type=float, default=None,
                    help="derived reach: geodesic dilation with gate "
                         "s > median(s)*mult. Beats a single fixed reach held "
                         "constant across span lengths.")
    ap.add_argument("--grow-max", type=int, default=64,
                    help="dilation cap; only meaningful with --grow.")
    # THE CONTROL THE LANGUAGE RESULTS DID NOT HAVE. Passkey has had this since
    # cornerstone 2 began -- `random` keeps the same NUMBER of positions chosen
    # uniformly, so the score is isolated from the budget at matched sinks,
    # matched recent and matched window. The language arms had no equivalent, so
    # the published "+0.184 over the positional control" compares against an arm
    # differing in THREE things at once: no score, no pooling, AND a recent
    # window 3.7x larger (53.88 against 14.38). That contrast cannot attribute
    # anything to the score.
    # The language harness had MODEL hardcoded and recorded it NOWHERE, so all 37
    # committed rag_dev arms are silently on the base core and a reader cannot
    # tell from the artifact. That also made it impossible to point this harness
    # at a different checkpoint -- which is exactly what testing a long-context
    # fine-tune requires. passkey.py has had --model for weeks.
    ap.add_argument("--model", default=MODEL,
                    help="checkpoint to evaluate. Recorded in the manifest, so a "
                         "result can never again be silent about which model "
                         "produced it.")
    # LoRA/QLoRA adapters. The long-context programme is blocked on a model that
    # can USE long context, and the only route to one here is a fine-tuned
    # adapter over a published core. This harness could not load one.
    #
    # An adapter's config names its own base model, so --adapter alone suffices;
    # --model is only for overriding that deliberately, and the override is
    # recorded, because applying an adapter to a base it was not trained on is a
    # silent way to measure nonsense.
    # A HARD GPU CEILING. passkey.py has had --cap-mib since the campaign began;
    # this harness did not, so a long-context arm could allocate until the driver
    # refused -- which is how three runs died. A cap makes the failure mode
    # "slower, or an honest OOM inside our own process" instead of "the card is
    # exhausted and whatever else is running dies too". Slower beats dying.
    ap.add_argument("--chunk-tokens", type=int, default=64,
                    help="chunk size for the structure-free lexical-chunk-bm25 "
                         "selectors. Only meaningful for those arms.")
    ap.add_argument("--protect-instructions", action="store_true",
                    help="INSTRUCTION SINKS: extend the always-kept prefix from the "
                         "4 attention sinks to cover the whole instruction span, so "
                         "it is held OUTSIDE the ranked budget instead of competing "
                         "with the evidence through a tie-break. The span is measured "
                         "per case -- it is 100 tokens on most rag_long prompts but "
                         "131-140 where TRUSTED_PROCEDURES is non-empty, so a fixed "
                         "--sinks would under-protect exactly those.")
    ap.add_argument("--cap-mib", type=float, default=None,
                    help="hard ceiling on this process's GPU allocation, in MiB. "
                         "Enforced via torch.cuda.set_per_process_memory_fraction, "
                         "so the process OOMs inside its own budget rather than "
                         "taking the card down. Recorded in the manifest.")
    ap.add_argument("--adapter", default=None,
                    help="PEFT adapter directory (trainer checkpoint or saved "
                         "adapter). Merged before evaluation and recorded in the "
                         "manifest with its base model and step.")
    ap.add_argument("--select", choices=("significance", "random", "streaming",
                                         "random-prepool", "random-doc",
                                         "random-doc-sys",
                                         "lexical-idf", "lexical-naive",
                                         "lexical-doc-bm25",
                                         "lexical-doc-bm25-sys",
                                         "lexical-chunk-bm25",
                                         "lexical-chunk-bm25-sys",
                                         "lexical-line-bm25",
                                         "lexical-line-bm25-sys",
                                         "hybrid-doc-sig-sys",
                                         "oracle-support",
                                         "oracle-support-sys"),
                    default="significance",
                    help="BASELINE arms, matching passkey.py. `random` keeps the "
                         "same number of positions chosen uniformly -- the only "
                         "control that isolates the SCORE rather than the budget. "
                         "`streaming` spends everything on sinks plus recent and "
                         "ranks nothing. The two `lexical-*` arms are the "
                         "TRIVIALITY CONTROL: they rank by overlap between a "
                         "position's token id and the question's, which needs no "
                         "forward pass at all, and pass through the same pool, "
                         "top-k and slot count as the score. See "
                         "docs/PREREG_CORNERSTONE2_TRIVIALITY_CONTROL.md.")
    ap.add_argument("--seed", type=int, default=20260826,
                    help="seeds the `random` selection arm so it is reproducible; "
                         "the significance arm is deterministic and ignores it.")
    ap.add_argument("--min-heavy-frac", type=float, default=0.0,
                    help="B1: reserve at least this FRACTION of the budget for "
                         "score-chosen slots. The recent window yields to it. "
                         "Prevents the degenerate case where sinks+recent consume "
                         "the whole budget and the policy is silently StreamingLLM "
                         "-- which fired 17,360 times on rag_dev at keep 0.25.")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--no-sinks", action="store_true", help="disable sink protection")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compare-to", default=None)
    args = ap.parse_args()

    force_repeat_kv(verbose=False)
    sig.patch_attention()
    sig.reset_heavy_zero_count()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.common.quality_evaluation import (  # noqa: E402
        aggregate_quality_rows, build_corpus_profile, evaluate_response,
        load_quality_config)

    # Use the FLAG, not the module constant. Adding --model but still loading
    # MODEL would be the manifest-records-the-flag defect in reverse: the
    # artifact would name a checkpoint the run never opened.
    if args.cap_mib and args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        frac = min(args.cap_mib, total) / total
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"gpu cap              {args.cap_mib:.0f} MiB of {total:.0f} "
              f"({frac:.1%} of the device)")

    adapter_cfg = None
    if args.adapter:
        cfg_path = Path(args.adapter) / "adapter_config.json"
        if not cfg_path.exists():
            raise SystemExit(f"--adapter {args.adapter}: no adapter_config.json there")
        adapter_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if args.model == MODEL:      # not overridden -- trust the adapter's own base
            args.model = adapter_cfg.get("base_model_name_or_path") or args.model

    path = Path(args.model)
    if not path.is_absolute():
        path = REPO / args.model
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device).eval()

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        # MERGE rather than leave the adapter as a side branch. The significance
        # patch wraps attention, and an unmerged PEFT wrapper changes the module
        # tree that patch expects. Merging keeps the evaluated graph identical in
        # shape to every other run in this programme, so a difference in the
        # numbers is the adapter and not the plumbing.
        model = model.merge_and_unload().eval()
        print(f"adapter               {args.adapter}")
        print(f"adapter base          {adapter_cfg.get('base_model_name_or_path')}")

    protocol = json.loads(
        (REPO / "manifests/eval_assets.json").read_text(encoding="utf-8")
    ).get("generation_protocol", {})
    max_new = protocol.get("max_new_tokens", 32)

    records = load_records(REPO / DATASETS[args.dataset], args.limit)
    evicting = bool(args.budget) or args.keep_frac is not None
    tag = args.tag or ("off" if not evicting else
                       (f"frac{args.keep_frac}" if args.keep_frac is not None
                        else f"b{args.budget}")
                       + (f"-pool{args.pool}" if args.pool > 1 else "")
                       + (f"-rf{args.recent_frac}" if args.recent_frac is not None else "")
                       + ("-nosink" if args.no_sinks else ""))

    print(f"variant {tag}   dataset {args.dataset} ({len(records)} cases)   "
          f"device {args.device}")
    print(f"eviction {'ON' if evicting else 'OFF'}"
          + (f"   budget {'frac ' + str(args.keep_frac) if args.keep_frac is not None else args.budget}"
             f"   sinks {0 if args.no_sinks else args.sinks}   recent {args.recent}"
             f"   pool {args.pool}"
             + (f"   window ({args.pool_back},{args.pool_fwd}) CAUSAL"
                if args.pool_fwd == 0 and args.pool_back
                else f"   window ({args.pool_back},{args.pool_fwd})"
                if args.pool_back is not None else "   window CENTRED")
             + (f"   grow {args.grow}" if args.grow else "")
             + f"   attn-window {args.window}" if evicting else ""))

    config = load_quality_config()
    corpus_profile = build_corpus_profile(records)

    rows, t0 = [], time.time()
    kept_ratios = []
    sinks_used = []
    recents_used = []
    windows_used = []
    for i, rec in enumerate(records):
        prompt = tok.apply_chat_template(rec["messages"], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        enc = tok(prompt, return_tensors="pt").to(args.device)
        n_prompt = enc["input_ids"].shape[-1]

        policy = None
        if evicting:
            budget = (max(1, int(round(args.keep_frac * n_prompt)))
                      if args.keep_frac is not None else args.budget)
            recent = (max(1, int(round(args.recent_frac * budget)))
                      if args.recent_frac is not None else args.recent)
            # `streaming` is NOT implemented inside select_keep -- that function
            # only special-cases `random`. passkey.py realises streaming by
            # having the CALLER spend the whole budget on sinks plus recent, so
            # heavy() becomes 0 and nothing is ranked. Doing the same here.
            #
            # Without this, --select streaming produced SIGNIFICANCE selection
            # while writing "streaming" into the manifest: a synthetic probe
            # showed it still keeping the single high-scoring position that
            # streaming must miss. That is the same defect shape as the
            # pool_back/pool_fwd footgun -- a config silently yielding a
            # different mechanism than the caller asked for.
            if args.select == "streaming":
                recent = max(0, budget - (0 if args.no_sinks else args.sinks))
            # INSTRUCTION SINKS. The always-kept prefix is normally 4 attention
            # sinks. With --protect-instructions it is extended to cover the whole
            # instruction span, measured PER CASE, so the instructions are held
            # outside the ranked budget instead of competing with the evidence
            # through a 1e-3 tie-break. Measured per case because the span is 100
            # tokens on 532 of 600 rag_long4k prompts and 131-140 on the rest.
            case_sinks = args.sinks
            if args.protect_instructions:
                _off = tok(prompt, return_offsets_mapping=True)
                case_sinks = max(args.sinks, instruction_prefix_len(prompt, _off))
            sinks_used.append(case_sinks)
            policy = sig.Policy(budget=budget, sinks=case_sinks, recent=recent,
                                protect_sinks=not args.no_sinks, pool=args.pool,
                                pool_back=args.pool_back, pool_fwd=args.pool_fwd,
                                grow=args.grow, grow_max=args.grow_max,
                                min_heavy_frac=args.min_heavy_frac,
                                select=args.select)
            # Record what the policy ACTUALLY used. Writing args.recent here --
            # which is what this file used to do -- makes every --recent-frac run
            # look like a fixed recent=32 run in the manifest, and summarize.py
            # then reported the significance-selected share as 13%/37% when the
            # truth is 63.7%/67.4%. The one column whose purpose is to expose
            # StreamingLLM-in-disguise was reporting the fix for it as a no-op.
            recents_used.append(policy.n_recent())
            windows_used.append(tuple(policy.pool_window()))

        # Seed per case so the `random` arm is reproducible AND independent
        # across cases. passkey.py does the same per trial; without it a random
        # baseline is neither replicable nor properly varied. The lexical arms
        # draw a tie-break from the same generator and need the same treatment.
        if args.select in sig.STOCHASTIC_SELECTS:
            torch.manual_seed(args.seed + i)
        sig.reset_scores()
        if args.select in sig.WEIGHTED_SELECTS:
            # reset_scores() clears the previous case's mask, so this must come
            # after it. Installing before would leave every case ranking on an
            # empty mask and raising -- loudly, which is the intent, but the
            # ordering is load-bearing and worth stating.
            q = rec.get("question")
            if not q:
                raise SystemExit(
                    f"--select {args.select} needs a `question` field and case "
                    f"{rec.get('id')!r} has none. The lexical control is only "
                    f"defined for datasets that carry the question separately "
                    f"from the prompt.")
            if args.select in ("lexical-doc-bm25", "lexical-doc-bm25-sys",
                               "lexical-chunk-bm25", "lexical-chunk-bm25-sys",
                               "lexical-line-bm25", "lexical-line-bm25-sys",
                               "hybrid-doc-sig-sys",
                               "random-doc", "random-doc-sys",
                               "oracle-support", "oracle-support-sys"):
                # A SECOND encoding, for offsets only. The generation path's `enc`
                # is left exactly as every committed run produced it -- adding
                # return_offsets_mapping to it would change the object handed to
                # model.generate, and "the control arm quietly altered the shared
                # prompt path" is not a difference anyone would find later.
                off = tok(prompt, return_offsets_mapping=True)
                if off["input_ids"] != enc["input_ids"][0].tolist():
                    # If the two encodings ever diverge, the weight vector is
                    # indexed against a different tokenisation than the cache and
                    # every position it protects is the wrong one -- silently.
                    raise SystemExit(
                        f"case {rec.get('id')!r}: the offsets encoding "
                        f"({len(off['input_ids'])} ids) does not match the "
                        f"generation encoding ({n_prompt} ids); the doc-bm25 "
                        f"weights would address the wrong positions")
                if args.select.startswith(("lexical-doc-bm25", "hybrid-doc-sig")):
                    w = doc_bm25_weights(prompt, off, q, tok)
                elif args.select.startswith("lexical-line-bm25"):
                    w = line_bm25_weights(prompt, off, q, tok)
                elif args.select.startswith("lexical-chunk-bm25"):
                    w = chunk_bm25_weights(prompt, off, q, tok, args.chunk_tokens)
                elif args.select.startswith("oracle-support"):
                    w = oracle_support_weights(prompt, off,
                                               rec.get("support_document_ids"))
                else:
                    w = doc_random_weights(prompt, off)
                if args.select.endswith("-sys"):
                    w = protect_instructions(w, prompt, off)
            else:
                w = lexical_weights(enc["input_ids"][0].tolist(), q, tok, args.select)
            sig.set_lexical_weights(w)
        sig.set_recording(evicting, window=args.window)
        cache = sig.SignificanceCache(config=model.config, policy=policy)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id,
                                 past_key_values=cache)
        sig.set_recording(False)
        stored = cache.stored_lengths()[0]
        kept_ratios.append(stored / max(n_prompt, 1))

        new_tokens = out[0, n_prompt:]
        reply = tok.decode(new_tokens, skip_special_tokens=True).strip()
        visible = len(tok(reply, add_special_tokens=False)["input_ids"])
        quality = evaluate_response(rec, reply, visible, visible_model_tokens=visible,
                                    config=config, corpus_profile=corpus_profile)
        rows.append({"id": rec.get("id"), "category": rec["category"], "reply": reply,
                     "prompt_tokens": n_prompt, "stored_kv": stored,
                     "visible_generated_tokens": visible, **quality})
        if (i + 1) % 100 == 0:
            # Memory alongside progress. A run of this harness slowed monotonically
            # (233/245/571/528 s per 100 cases) and then died with a CUDA illegal
            # memory access; wall-clock alone could not distinguish a leak from
            # thermal throttling, and there was no allocator number to look at.
            # `reserved` is what matters for fragmentation: allocated can stay
            # flat while reserved climbs, which is the signature that precedes an
            # allocator failure.
            if torch.cuda.is_available():
                al = torch.cuda.memory_allocated() / 2 ** 20
                rs = torch.cuda.memory_reserved() / 2 ** 20
                print(f"  {i + 1}/{len(records)}  {time.time() - t0:.0f}s  "
                      f"alloc {al:.0f} MiB  reserved {rs:.0f} MiB", flush=True)
            else:
                print(f"  {i + 1}/{len(records)}  {time.time() - t0:.0f}s", flush=True)

    summary = aggregate_quality_rows(rows)
    elapsed = time.time() - t0
    mean_keep = sum(kept_ratios) / len(kept_ratios)
    print(f"\nmean_quality_score      {summary['mean_quality_score']:.5f}")
    print(f"quality_pass_rate       {summary['quality_pass_rate']:.5f}")
    print(f"intelligibility         {summary['intelligibility_pass_rate']:.5f}")
    print(f"mean kept / prompt      {mean_keep:.3f}")
    print(f"elapsed                 {elapsed:.1f}s")

    out_dir = EXP / "results" / f"{tag}__{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    per_cat: dict[str, dict] = {}
    for r in rows:
        b = per_cat.setdefault(r["category"], {"n": 0, "total": 0.0})
        b["n"] += 1
        b["total"] += float(r.get("quality_score") or 0.0)
    (out_dir / "per_category.json").write_text(json.dumps(
        {k: {"n": v["n"], "mean_quality_score": v["total"] / v["n"]}
         for k, v in sorted(per_cat.items())}, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "tag": tag, "dataset": args.dataset, "eviction": evicting,
        "model": str(args.model),
        "adapter": (str(args.adapter) if args.adapter else None),
        "cap_mib": args.cap_mib,
        "adapter_base": (adapter_cfg or {}).get("base_model_name_or_path"),
        "budget": args.budget, "keep_frac": args.keep_frac,
        "sinks": (0 if args.no_sinks else
                  (round(sum(sinks_used) / len(sinks_used), 2) if sinks_used
                   else args.sinks)),
        "sinks_flag": args.sinks,
        "protect_instructions": bool(args.protect_instructions),
        "chunk_tokens": (args.chunk_tokens
                         if "chunk" in args.select else None),
        "recent_flag": args.recent,
        "recent": (round(sum(recents_used) / len(recents_used), 2)
                   if recents_used else args.recent),
        "recent_effective_mean": (round(sum(recents_used) / len(recents_used), 2)
                                  if recents_used else None),
        "recent_frac": args.recent_frac,
        "heavy_zero_count": sig.heavy_zero_count(),
        "pool": args.pool, "window": args.window,
        # Record the EFFECTIVE window, not the flags. Policy.pool_window()
        # defaults to (pool//2, pool//2) when --pool-back is unset, so a run
        # that recorded only the flags would write pool_back=null for a centred
        # window AND for a directional one that happened to be defaulted -- the
        # same ambiguity that made a heavy=0 run record as heavy=8. Any pin
        # asserting `pool_back` must be able to tell a cornerstone-2 run from a
        # cornerstone-1 run from the manifest alone.
        "pool_back": (windows_used[0][0] if windows_used else None),
        "pool_fwd": (windows_used[0][1] if windows_used else None),
        "pool_back_flag": args.pool_back, "pool_fwd_flag": args.pool_fwd,
        "pool_window_uniform": (len(set(windows_used)) <= 1) if windows_used else None,
        "grow": args.grow,
        "min_heavy_frac": args.min_heavy_frac,
        "select": args.select, "seed": args.seed,
        "grow_max": (args.grow_max if args.grow else None),
        "directional": bool(windows_used and windows_used[0][1] == 0
                            and windows_used[0][0] > 0),
        "mean_kept_ratio": round(mean_keep, 4),
        "case_count": len(rows), "protocol": protocol,
        "elapsed_s": round(elapsed, 1),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")

    if args.compare_to:
        ref_path = EXP / "results" / f"{args.compare_to}__{args.dataset}" / "results.jsonl"
        if not ref_path.exists():
            raise SystemExit(f"no reference run at {ref_path}")
        ref = {json.loads(l)["id"]: json.loads(l)
               for l in ref_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        common = [r for r in rows if r["id"] in ref]
        same = sum(1 for r in common if r["reply"] == ref[r["id"]]["reply"])
        ref_mean = sum(float(ref[r["id"]].get("quality_score") or 0.0)
                       for r in common) / max(len(common), 1)
        print(f"\nvs {args.compare_to}: {same}/{len(common)} byte-identical "
              f"({100 * same / max(len(common), 1):.1f}%)")
        print(f"  quality {ref_mean:.5f} -> {summary['mean_quality_score']:.5f}  "
              f"({summary['mean_quality_score'] - ref_mean:+.5f})")


if __name__ == "__main__":
    main()
