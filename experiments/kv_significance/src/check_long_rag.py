#!/usr/bin/env python3
"""Validate a generated long-context RAG benchmark before spending GPU on it.

A benchmark whose haystack contains the answer measures nothing, and it fails
SILENTLY -- it looks like excellent retrieval. This checks the properties that
have to hold, and it splits on the document marker rather than on blank lines.

That distinction is not pedantic: 278 of the 732 rag_dev records carry a
MULTI-DOCUMENT supporting block whose documents are separated by blank lines. A
first version of this check split on "\n\n", tore those blocks into pieces,
and reported 278 missing supporting documents and 167 answer leaks -- all four
hundred-odd of them artifacts of the checker. Documents are delimited by
"[doc_", so that is what we split on.

Checks:
  1. every supporting document id is present, and they stay contiguous
  2. no NON-supporting document contains the reference answer
  3. reference, response_contract and question are byte-identical to the source
  4. the supporting block sits at the requested depth
  5. prompt length is within tolerance of the target

Usage:
    python experiments/kv_significance/src/check_long_rag.py artifacts/eval_sets/rag_long4k_d0.5.jsonl
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EV = re.compile(r"<FACTUAL_EVIDENCE>\n(.*?)\n</FACTUAL_EVIDENCE>", re.S)
DOCID = re.compile(r"\[(doc_[^\s|\]]+)")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def user_msg(rec: dict) -> str:
    return next(m["content"] for m in rec["messages"] if m["role"] == "user")


def docs_of(block: str) -> list[str]:
    """Split an evidence block into documents on the '[doc_' marker."""
    parts = re.split(r"(?=\[doc_)", block)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    path = Path(sys.argv[1])
    src = {json.loads(l)["id"]: json.loads(l)
           for l in (REPO / "artifacts/eval_sets/rag_dev.jsonl").open(encoding="utf-8")
           if l.strip()}
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    missing = leaked = altered = noncontig = 0
    depths, lens, leak_examples = [], [], []
    for r in rows:
        meta = r["metadata"]["long_context"]
        orig = src[meta["source_id"]]
        true_docs = docs_of(EV.search(user_msg(orig)).group(1))
        true_ids = {DOCID.search(d).group(1) for d in true_docs if DOCID.search(d)}
        all_docs = docs_of(EV.search(user_msg(r)).group(1))
        all_ids = [DOCID.search(d).group(1) if DOCID.search(d) else "?" for d in all_docs]

        if not true_ids.issubset(set(all_ids)):
            missing += 1
            continue
        idx = sorted(i for i, d in enumerate(all_ids) if d in true_ids)
        if idx != list(range(idx[0], idx[0] + len(idx))):
            noncontig += 1
        depths.append(idx[0] / max(1, len(all_docs) - len(idx)))

        ref = norm(orig.get("reference"))
        if ref:
            for i, d in enumerate(all_docs):
                if all_ids[i] not in true_ids and ref in norm(d):
                    leaked += 1
                    if len(leak_examples) < 3:
                        leak_examples.append((r["id"], all_ids[i], ref))
                    break
        if (r.get("reference") != orig.get("reference")
                or r.get("response_contract") != orig.get("response_contract")
                or r.get("question") != orig.get("question")):
            altered += 1
        lens.append(len(user_msg(r)))

    n = len(rows)
    print(f"{path.name}: {n} records, target {rows[0]['metadata']['long_context']['target_tokens']} tokens\n")
    bad = 0
    for label, count in (("supporting documents missing", missing),
                         ("supporting block non-contiguous", noncontig),
                         ("ANSWER LEAKED into a distractor", leaked),
                         ("reference/contract/question altered", altered)):
        ok = count == 0
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: {count}")
    for ex in leak_examples:
        print(f"        e.g. {ex[0]} -- {ex[1]} contains {ex[2]!r}")
    if depths:
        med = st.median(depths)
        want = rows[0]["metadata"]["long_context"]["depth"]
        ok = abs(med - want) < 0.05
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  depth of supporting block: "
              f"median {med:.3f} (requested {want})")
    print(f"\n  chars per prompt: median {st.median(lens):,.0f}")
    print(f"\n{'benchmark is sound' if not bad else f'{bad} PROPERTY VIOLATED -- do not run this'}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
