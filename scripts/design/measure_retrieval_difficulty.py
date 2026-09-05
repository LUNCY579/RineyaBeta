"""How hard is retrieval on this corpus, actually?

Every "a trivial selector suffices" and "selection is saturated" result in this
programme is conditional on one line of `make_long_rag.py`: same-topic documents
are excluded from the distractor pool. Nobody measured what that does to the
baseline it makes look good.

This reports the BM25 rank of the gold record, per corpus. If gold@1 is 1.000
the retrieval problem is degenerate and no statement about the sufficiency of
lexical selection means anything beyond this corpus shape.

It is also the GATE on the same-topic experiment: if the harder corpus still
gives gold@1 near 1.000, the variant did not achieve its purpose and the GPU
arms would measure nothing. Run this before spending the card.

    python scripts/design/measure_retrieval_difficulty.py
    python scripts/design/measure_retrieval_difficulty.py --check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join("experiments", "kv_significance", "results")
MODEL = "artifacts/rineya-core-en32k-28L"
LIMIT = 600

CORPORA = [
    ("default (same-topic EXCLUDED)",
     os.path.join("artifacts", "eval_sets", "rag_long4k_d0.5.jsonl")),
    ("variant (same-topic PERMITTED)",
     os.path.join("artifacts", "eval_sets", "rag_long4k_same_d0.5.jsonl")),
]


def load_selectors():
    path = os.path.join(REPO, "experiments", "kv_significance", "src",
                        "run_eval_sig.py")
    spec = importlib.util.spec_from_file_location("_res", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prompt_of(rec):
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            return m["content"]
    return None


def gold_rank(res, tok, rec):
    """1-based BM25 rank of the gold record, or None if not measurable."""
    support = rec.get("support_document_ids") or []
    if not support:
        return None
    prompt = prompt_of(rec)
    if not prompt:
        return None
    enc = tok(prompt, return_offsets_mapping=True)
    segs = res.segment_records(prompt, enc)
    if len(segs) < 2:
        return None
    w = res.doc_bm25_weights(prompt, enc, rec["question"], tok)

    # Which segments are the gold record? Locate by the [doc_...] header text
    # that the prompt itself prints, the same way oracle_support_weights does.
    offsets = enc["offset_mapping"]
    gold_idx = set()
    for si, (tok_idx, _ids) in enumerate(segs):
        if not tok_idx:
            continue
        a = offsets[tok_idx[0]][0]
        z = offsets[tok_idx[-1]][1]
        span = prompt[a:z]
        if any(d in span for d in support):
            gold_idx.add(si)
    if not gold_idx:
        return None

    scores = []
    for si, (tok_idx, _ids) in enumerate(segs):
        scores.append((float(w[tok_idx[0]]) if tok_idx else -1.0, si))
    # Rank descending; ties broken pessimistically for the gold record so a
    # tie with a distractor is not silently scored as a win.
    scores.sort(key=lambda s: (-s[0], s[1] in gold_idx))
    for pos, (_sc, si) in enumerate(scores, start=1):
        if si in gold_idx:
            return pos
    return None


def measure(res, tok, path):
    ranks = []
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= LIMIT:
                break
            line = line.strip()
            if not line:
                continue
            r = gold_rank(res, tok, json.loads(line))
            if r is not None:
                ranks.append(r)
    n = len(ranks)
    if not n:
        return None
    at1 = sum(1 for r in ranks if r == 1)
    at3 = sum(1 for r in ranks if r <= 3)
    ranks_sorted = sorted(ranks)
    return {
        "measurable_cases": n,
        "gold_at_1": round(at1 / n, 4),
        "gold_at_3": round(at3 / n, 4),
        "mean_rank": round(sum(ranks) / n, 3),
        "median_rank": ranks_sorted[n // 2],
        "worst_rank": ranks_sorted[-1],
        "gold_at_1_count": at1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=os.path.join(RESULTS,
                                                   "retrieval_difficulty.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(REPO, MODEL))
    res = load_selectors()

    out = {"model": MODEL, "limit": LIMIT, "corpora": {}}
    hdr = "{0:34s} {1:>6s} {2:>9s} {3:>9s} {4:>10s} {5:>7s} {6:>6s}".format(
        "corpus", "n", "gold@1", "gold@3", "mean rank", "median", "worst")
    print(hdr)
    print("-" * len(hdr))
    for label, path in CORPORA:
        if not os.path.exists(os.path.join(REPO, path)):
            print("{0:34s} (not generated)".format(label))
            continue
        m = measure(res, tok, path)
        if m is None:
            print("{0:34s} (no measurable cases)".format(label))
            continue
        out["corpora"][path] = dict(m, label=label)
        print("{0:34s} {1:6d} {2:9.4f} {3:9.4f} {4:10.3f} {5:7d} {6:6d}".format(
            label, m["measurable_cases"], m["gold_at_1"], m["gold_at_3"],
            m["mean_rank"], m["median_rank"], m["worst_rank"]))

    vals = list(out["corpora"].values())
    if len(vals) == 2:
        drop = vals[0]["gold_at_1"] - vals[1]["gold_at_1"]
        out["gold_at_1_drop"] = round(drop, 4)
        print()
        print("gold@1 drops by {0:.4f} under same-topic distractors".format(drop))
        print("  GATE: {0}".format(
            "retrieval is now non-trivial -- the GPU arms are worth running"
            if drop >= 0.05 else
            "retrieval is STILL near-degenerate; the variant did not achieve its "
            "purpose and the GPU arms would measure nothing"))

    if args.check:
        if not os.path.exists(args.json):
            print("\nno committed measurement at " + args.json, file=sys.stderr)
            return 1
        with open(args.json, encoding="utf-8") as fh:
            old = json.load(fh)
        if old != out:
            print("\nMISMATCH against committed retrieval-difficulty measurement",
                  file=sys.stderr)
            return 1
        print("\ncommitted retrieval-difficulty measurement reproduces")
        return 0

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
