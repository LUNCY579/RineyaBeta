#!/usr/bin/env python3
"""Build LONG-CONTEXT natural-language benchmarks from rag_dev.

WHY THIS EXISTS. This programme has retrieval evidence at 125,826 tokens
(passkey) and language evidence at a median of 238 tokens (rag_dev), with
NOTHING between them -- a 543x gap. The mechanism's entire value proposition is
long-context memory, so the regime it is actually for is unmeasured. Passkey
cannot close the gap: its filler contains no digits, so a regex scores 1.000 on
it with no model. rag_dev cannot close it either: no prompt exceeds 1K tokens.

WHAT THIS BUILDS. The same rag_dev questions, the same references, the same
response contracts and therefore the SAME VALIDATED SCORER -- with the evidence
block expanded by real evidence documents drawn from other records, and the
supporting document placed at a controlled depth. That is passkey's structure
(needle at depth in a haystack) with passkey's fatal weakness removed: the
haystack is real prose about plausible neighbouring topics, so answering
requires reading the right document rather than matching the only digits
present.

Because the questions and references are unchanged, the existing native-length
runs (~232 tokens) are the length baseline, and a long run differs from them in
exactly one variable: how much irrelevant real text surrounds the answer.

TWO HAZARDS, BOTH HANDLED EXPLICITLY.

1. CONTAMINATION. A distractor from another topic can still happen to contain
   the answer string -- dates and short values repeat across records. Every
   candidate is checked against the normalized reference and rejected if it
   contains it. A benchmark where the haystack answers the question measures
   nothing, and it would fail silently and look like excellent retrieval.

2. PROMPT INJECTION. rag_dev contains 38 `prompt_injection` records whose
   evidence carries adversarial instructions -- that is their purpose as test
   cases. Pasting those blocks into OTHER records' haystacks would inject
   instructions into prompts whose contracts do not expect them, corrupting the
   scores of unrelated cases. They are excluded from the distractor pool. They
   remain available as targets, where the contract accounts for them.

Same-topic documents are also excluded as distractors: they carry related or
deliberately conflicting facts about the same entity, which changes the task
from retrieval to conflict resolution.

Usage:
    python experiments/kv_significance/src/make_long_rag.py --tokens 4096
    python experiments/kv_significance/src/make_long_rag.py --tokens 16384 --depth 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "artifacts/eval_sets/rag_dev.jsonl"
EV = re.compile(r"(<FACTUAL_EVIDENCE>\n)(.*?)(\n</FACTUAL_EVIDENCE>)", re.S)


def normalize_answer(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def evidence_block(rec: dict) -> str | None:
    for m in rec["messages"]:
        if m["role"] == "user":
            g = EV.search(m["content"])
            if g:
                body = g.group(2).strip()
                return body if body and body != "(none)" else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, required=True,
                    help="target prompt length in tokens")
    ap.add_argument("--depth", type=float, default=0.5,
                    help="fractional position of the SUPPORTING document among "
                         "the distractors: 0.0 first, 1.0 last")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="artifacts/rineya-core-en32k-28L")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--same-topic-distractors", action="store_true",
                    help="permit distractors from the target's own topic. The "
                         "default corpus excludes them, which makes document "
                         "BM25 rank the gold record first in 530/530 answerable "
                         "cases -- retrieval is easy BY CONSTRUCTION, and every "
                         "saturation result is conditional on that. Contamination "
                         "and prompt-injection gates still apply.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO / args.model)

    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    # Distractor pool: real evidence blocks, minus the injection cases.
    pool = []
    for r in rows:
        if r.get("category") == "prompt_injection":
            continue
        b = evidence_block(r)
        if b:
            pool.append((r.get("topic_id"), b, len(tok(b)["input_ids"])))
    print(f"distractor pool: {len(pool)} blocks from "
          f"{len({t for t, _, _ in pool})} topics "
          f"({sum(n for _, _, n in pool):,} tokens); "
          f"{sum(1 for r in rows if r.get('category') == 'prompt_injection')} "
          f"injection records excluded")

    out = args.out or (REPO / f"artifacts/eval_sets/rag_long{args.tokens // 1024}k"
                              f"_d{args.depth}.jsonl")
    rng = random.Random(args.seed)
    written = skipped = 0
    contaminated_rejects = 0
    lens = []

    with out.open("w", encoding="utf-8") as fh:
        for rec in rows:
            true_block = evidence_block(rec)
            if not true_block:
                skipped += 1
                continue
            ref = normalize_answer(rec.get("reference"))
            base = len(tok(rec["messages"][1]["content"])["input_ids"])
            need = max(0, args.tokens - base)

            # DEFAULT: same-topic documents are excluded, for the reason in the
            # module docstring. --same-topic-distractors permits them, which
            # makes retrieval genuinely hard rather than trivially easy: on the
            # default corpus document-BM25 ranks the gold record FIRST in 530 of
            # 530 answerable cases, so every "a trivial selector suffices" and
            # "selection is saturated" result in this programme is conditional
            # on this line. The contamination gate below still rejects any
            # distractor containing the reference answer, so the question stays
            # answerable; what changes is that the haystack now contains prose
            # about the same entity.
            if args.same_topic_distractors:
                cand = list(pool)
            else:
                cand = [p for p in pool if p[0] != rec.get("topic_id")]
            rng.shuffle(cand)
            chosen, got = [], 0
            for topic, body, ntok in cand:
                if got >= need:
                    break
                # CONTAMINATION GATE -- see the module docstring.
                if ref and ref in normalize_answer(body):
                    contaminated_rejects += 1
                    continue
                chosen.append(body)
                got += ntok
            if not chosen:
                skipped += 1
                continue

            at = min(len(chosen), max(0, round(args.depth * len(chosen))))
            blocks = chosen[:at] + [true_block] + chosen[at:]
            new_ev = "\n\n".join(blocks)

            rec = json.loads(json.dumps(rec))          # deep copy
            for m in rec["messages"]:
                if m["role"] == "user":
                    m["content"] = EV.sub(
                        lambda g: g.group(1) + new_ev + g.group(3), m["content"], count=1)
                    lens.append(len(tok(m["content"])["input_ids"]))
                    break
            rec["id"] = f"{rec['id']}__L{args.tokens}_d{args.depth}"
            rec.setdefault("metadata", {})
            rec["metadata"]["long_context"] = {
                "target_tokens": args.tokens, "depth": args.depth,
                "n_distractors": len(chosen), "true_doc_index": at,
                "seed": args.seed, "source_id": rec["id"].split("__L")[0],
            }
            fh.write(json.dumps(rec) + "\n")
            written += 1

    lens.sort()
    print(f"wrote {out}")
    print(f"  {written} records, {skipped} skipped (no evidence block or no clean distractors)")
    print(f"  contaminated distractors rejected: {contaminated_rejects}")
    if lens:
        print(f"  prompt tokens: min {lens[0]}  median {lens[len(lens)//2]}  "
              f"max {lens[-1]}  (target {args.tokens})")


if __name__ == "__main__":
    main()
