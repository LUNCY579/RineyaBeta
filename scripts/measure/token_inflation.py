#!/usr/bin/env python3
"""Gate C2: what an English-focused trimmed vocabulary actually costs.

Two questions, in order of importance:

  1. Correctness -- can the trimmed tokenizer still represent every input?
     Byte-level BPE with all 256 byte ids kept should round-trip any string
     exactly, including scripts that were deliberately dropped. A round-trip
     failure is a hard stop, not a cost.

  2. Cost -- how many more tokens does the same text take? Dropped tokens do not
     become unencodable, they decompose into shorter pieces. Inflation on
     held-out human prose is the number that decides the cutoff, because it is
     paid on every forward pass forever.

Held-out set is the repository's own hand-written Markdown -- real English prose
that is not part of any training corpus, so it does not share the synthetic
corpora's vocabulary bias.

Usage: python token_inflation.py <trimmed_tokenizer_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
STOCK = ROOT / "Qwen3-0.6B" / "tokenizer.json"

IN_DOMAIN = (
    ("chat SFT (2,000 rows)", "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/rineya_v3_recommended_combined_124000.jsonl", 2_000),
    ("RAG SFT (2,000 rows)", "Aegis_RAG_v3_train_4410/rag_v3_train.sft.jsonl", 2_000),
    ("RAG evidence (2,000)", "Aegis_RAG_v3_train_4410/evidence/knowledge.jsonl", 2_000),
)

# Deliberately dropped scripts plus symbols the model is known to emit
# (READINESS_ASSESSMENT records it writing "### [check] Final Answer:").
PROBES = (
    ("ascii english", "The final answer is -6, because the third clause overrides the first."),
    ("accented latin", "Beyoncé filed a naïve résumé at the café in Zürich."),
    ("greek / math", "Let α = 0.05, β ≈ 1/3, and ∑ x_i ≤ 42 ∀ i ∈ S."),
    ("emoji", "Done. ✅ Shipped 🚀 and verified."),
    ("cjk (dropped)", "这是一个测试。日本語のテキストもあります。"),
    ("cyrillic (dropped)", "Это проверка кодирования текста."),
    ("arabic (dropped)", "هذا اختبار للترميز."),
    ("json contract", '{"decision": "abstain", "updated_compact_state": null}'),
    ("chat control", "<|im_start|>user\nhi<|im_end|>\n<think>\n\n</think>\n"),
)


def jsonl_texts(path: Path, limit: int):
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in row.get("messages") or []:
                content = msg.get("content")
                if isinstance(content, str) and content:
                    yield content
            if "messages" not in row:
                yield from (v for v in row.values() if isinstance(v, str) and v)


def count(tok: Tokenizer, texts) -> int:
    total, batch = 0, []
    for text in texts:
        batch.append(text)
        if len(batch) >= 1000:
            total += sum(len(e.ids) for e in tok.encode_batch(batch, add_special_tokens=False))
            batch = []
    if batch:
        total += sum(len(e.ids) for e in tok.encode_batch(batch, add_special_tokens=False))
    return total


def main() -> None:
    out_dir = Path(sys.argv[1])
    stock = Tokenizer.from_file(str(STOCK))
    trimmed = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    manifest = json.loads((out_dir / "trim_manifest.json").read_text(encoding="utf-8"))

    print(f"stock vocab   : 151,936")
    print(f"trimmed vocab : {manifest['vocab_size']:,d} "
          f"(rule {manifest['keep_rule']})\n")

    print("--- 1. round-trip fidelity (hard gate) ---")
    print(f"{'probe':22s} {'stock':>6s} {'trim':>6s} {'x':>6s}  exact?")
    failures = []
    for label, text in PROBES:
        a = stock.encode(text, add_special_tokens=False)
        b = trimmed.encode(text, add_special_tokens=False)
        back = trimmed.decode(b.ids, skip_special_tokens=False)
        ok = back == text
        if not ok:
            failures.append((label, text, back))
        ratio = len(b.ids) / len(a.ids) if a.ids else 1.0
        print(f"{label:22s} {len(a.ids):>6d} {len(b.ids):>6d} {ratio:>5.2f}x  "
              f"{'yes' if ok else 'NO'}")

    print("\n--- 2. held-out human prose (repository Markdown) ---")
    md = sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts)
    prose = []
    for path in md:
        try:
            prose.append(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    a, b = count(stock, prose), count(trimmed, prose)
    print(f"{len(md)} files, {sum(len(t) for t in prose):,d} chars")
    print(f"stock {a:,d} tokens -> trimmed {b:,d} tokens = {b / a:.4f}x "
          f"({100 * (b / a - 1):+.2f}%)")

    print("\n--- 3. in-domain corpora ---")
    print(f"{'source':24s} {'stock':>11s} {'trimmed':>11s} {'inflation':>10s}")
    for label, rel, limit in IN_DOMAIN:
        texts = list(jsonl_texts(ROOT / rel, limit))
        sa, sb = count(stock, texts), count(trimmed, texts)
        print(f"{label:24s} {sa:>11,d} {sb:>11,d} {sb / sa:>9.4f}x")

    print("\n--- verdict ---")
    if failures:
        print(f"ROUND-TRIP FAILED on {len(failures)} probe(s) -- STOP:")
        for label, want, got in failures:
            print(f"  {label}: {want!r} -> {got!r}")
        raise SystemExit(1)
    print("round-trip exact on every probe, including dropped scripts.")
    print(f"held-out English prose inflation: {100 * (b / a - 1):+.2f}%")


if __name__ == "__main__":
    main()
