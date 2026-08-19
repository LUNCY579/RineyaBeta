#!/usr/bin/env python3
"""Measure how much of the Qwen3 vocabulary the Rineya corpora actually use.

Counts content tokens only (add_special_tokens=False) across every SFT corpus
and the RAG evidence store, then reports the parameter saving available from
trimming the embedding matrix at several candidate vocab sizes.

Emits manifests/vocab_utilization.json with the frequency-ranked keep order so
the trim itself is reproducible from this measurement.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Qwen3-0.6B"
OUT = ROOT / "manifests" / "vocab_utilization.json"

BATCH = 2000
CANDIDATES = (8192, 16384, 24576, 32768, 49152, 65536)

# (label, path, row limit, extraction mode)
SOURCES = (
    ("v3 recommended combined 124k", "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/rineya_v3_recommended_combined_124000.jsonl", 40_000, "messages"),
    ("short-response combined",      "Rineya_Short_Response_Behavior_Corpus_v1.0/data/sft/rineya_short_response_combined_train.jsonl", None, "messages"),
    ("RAG v3 train sft 4,410",       "Aegis_RAG_v3_train_4410/rag_v3_train.sft.jsonl", None, "messages"),
    ("evidence/knowledge 6,000",     "Aegis_RAG_v3_train_4410/evidence/knowledge.jsonl", None, "flat"),
    ("evidence/procedures 200",      "Aegis_RAG_v3_train_4410/evidence/procedures.jsonl", None, "flat"),
)


def strings(path: Path, limit: int | None, mode: str):
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if mode == "flat":
                yield from (v for v in row.values() if isinstance(v, str) and v)
                continue
            for msg in row.get("messages") or []:
                content = msg.get("content")
                if isinstance(content, str) and content:
                    yield content
            for key in ("question", "answer", "approved_answer"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    yield value
            evidence = row.get("evidence")
            if isinstance(evidence, list):
                for item in evidence:
                    if isinstance(item, dict):
                        yield from (v for v in item.values() if isinstance(v, str) and v)
                    elif isinstance(item, str) and item:
                        yield item


def main() -> None:
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    spec = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    vocab, hidden = config["vocab_size"], config["hidden_size"]

    freq: collections.Counter[int] = collections.Counter()
    print(f"{'source':32s} {'tokens':>12s} {'cum unique':>11s} {'new':>8s}")
    for label, rel, limit, mode in SOURCES:
        before, counted = len(freq), 0
        batch: list[str] = []
        for text in strings(ROOT / rel, limit, mode):
            batch.append(text)
            if len(batch) >= BATCH:
                for enc in tok.encode_batch(batch, add_special_tokens=False):
                    freq.update(enc.ids)
                    counted += len(enc.ids)
                batch = []
        for enc in tok.encode_batch(batch, add_special_tokens=False):
            freq.update(enc.ids)
            counted += len(enc.ids)
        print(f"{label:32s} {counted:>12,d} {len(freq):>11,d} {len(freq) - before:>+8,d}")

    total = sum(freq.values())
    print(f"\ncorpus content tokens : {total:,d}")
    print(f"distinct ids used     : {len(freq):,d} / {vocab:,d} = {100 * len(freq) / vocab:.2f}%")
    print(f"never addressed       : {vocab - len(freq):,d} embedding rows\n")

    # Mandatory keeps: complete byte-level alphabet (guarantees any string stays
    # encodable) plus every added/control token the chat template depends on.
    byte_ids = sorted(spec["model"]["vocab"][c] for c in pre_tokenizers.ByteLevel.alphabet())
    special_ids = sorted(a["id"] for a in spec.get("added_tokens", []))
    mandatory = set(byte_ids) | set(special_ids)
    print(f"mandatory keeps: {len(byte_ids)} byte-alphabet ids "
          f"({byte_ids[0]}..{byte_ids[-1]}) + {len(special_ids)} control ids "
          f"({special_ids[0]}..{special_ids[-1]})")

    ranked = [i for i, _ in freq.most_common()]
    keep_order = [i for i in ranked if i not in mandatory]
    counts = dict(freq)

    print(f"\n{'target vocab':>13s} {'content coverage':>17s} {'embed params':>14s} {'saved':>14s} {'model total':>13s}")
    resident = 596_049_920
    rows = []
    for n in CANDIDATES:
        budget = n - len(mandatory)
        kept = keep_order[:max(budget, 0)]
        covered = sum(counts[i] for i in kept) + sum(counts.get(i, 0) for i in mandatory)
        embed, saved = n * hidden, (vocab - n) * hidden
        rows.append({"target_vocab": n, "content_coverage": covered / total,
                     "embed_params": embed, "saved_params": saved})
        print(f"{n:>13,d} {100 * covered / total:>16.4f}% {embed:>14,d} {saved:>14,d}"
              f" {resident - saved:>13,d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "corpus_content_tokens": total,
        "distinct_ids_used": len(freq),
        "vocab_size": vocab,
        "hidden_size": hidden,
        "mandatory_ids": sorted(mandatory),
        "frequency_ranked_keep_order": keep_order,
        "candidates": rows,
        "sources": [{"label": s[0], "path": s[1], "row_limit": s[2]} for s in SOURCES],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
