#!/usr/bin/env python3
"""Convert RAG v3 train rows and shuffle-mix with Rineya short-response SFT."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
OUT = Path(__file__).resolve().parent

RAG_TRAIN = (
    ROOT
    / "data"
    / "phase_9_student_distillation"
    / "rag_reasoning_v3_large"
    / "reviewed_splits"
    / "train.reviewed.jsonl"
)
RINEYA_COMBINED = (
    REPO
    / "Rineya_Short_Response_Behavior_Corpus_v1.0"
    / "data"
    / "sft"
    / "rineya_short_response_combined_train.jsonl"
)
RAG_SFT_OUT = OUT / "rag_v3_train.sft.jsonl"
MIX_OUT = OUT / "mix.jsonl"
MANIFEST_OUT = OUT / "mix_manifest.json"
SEED = 20260816


def convert_rag_row(row: dict) -> dict:
    messages = [dict(m) for m in (row.get("messages") or [])]
    if not messages:
        raise ValueError(f"{row.get('id')}: missing messages")
    answer = str(row.get("approved_answer") or row.get("answer") or "").strip()
    if not answer:
        raise ValueError(f"{row.get('id')}: missing approved_answer/answer")
    if messages[-1].get("role") == "assistant":
        messages[-1] = {"role": "assistant", "content": answer}
    else:
        messages.append({"role": "assistant", "content": answer})
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{row.get('id')}: failed to attach assistant target")
    # Keep evidence fields for Phase 9 trainers; Phase 1 uses messages only.
    out = {
        "id": row.get("id"),
        "system_policy_version": "phase9_evidence_grounded_v1",
        "messages": messages,
        "metadata": {
            "reply_class": str(row.get("category") or row.get("case_class") or "evidence"),
            "length_mode": "SHORT",
            "target_token_min": 1,
            "target_token_max": 32,
            "requires_external_fact": True,
            "safety_tier": "normal",
            "tone": ["concise", "evidence_grounded"],
            "source_provenance": str(row.get("source") or "rag_reasoning_v3_large_train"),
            "human_review_status": str(row.get("review_status") or "auto_verified"),
            "split": "train",
            "template_family": str(row.get("topic_id") or row.get("id") or "rag_v3"),
            "corpus": "rag_reasoning_v3_large",
            "intended_split": str(row.get("intended_split") or "train"),
            "decision": row.get("decision"),
            "mode": row.get("mode"),
            "category": row.get("category"),
        },
        # Retained for optional Phase 9 student replay; Phase 1 ignores these.
        "question": row.get("question"),
        "approved_answer": answer,
        "answer": answer,
        "evidence": row.get("evidence") or [],
        "procedures": row.get("procedures") or [],
        "support_document_ids": row.get("support_document_ids") or [],
        "mode": row.get("mode"),
        "decision": row.get("decision"),
        "response_contract": row.get("response_contract"),
    }
    return out


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not RAG_TRAIN.exists():
        raise FileNotFoundError(RAG_TRAIN)
    if not RINEYA_COMBINED.exists():
        raise FileNotFoundError(RINEYA_COMBINED)

    rag_src = read_jsonl(RAG_TRAIN)
    converted: list[dict] = []
    for row in rag_src:
        split = str(row.get("intended_split") or "train").casefold()
        if split and split != "train":
            raise ValueError(f"Non-train row in train.reviewed.jsonl: {row.get('id')} split={split}")
        converted.append(convert_rag_row(row))
    write_jsonl(RAG_SFT_OUT, converted)

    chat_rows = read_jsonl(RINEYA_COMBINED)
    for row in chat_rows:
        meta = row.get("metadata") or {}
        if str(meta.get("split") or "").casefold() not in {"", "train"}:
            raise ValueError(f"Non-train Rineya row: {row.get('id')}")
        msgs = row.get("messages") or []
        if not msgs or msgs[-1].get("role") != "assistant":
            raise ValueError(f"Rineya row missing assistant target: {row.get('id')}")

    mix = [*chat_rows, *converted]
    rng = random.Random(SEED)
    rng.shuffle(mix)
    write_jsonl(MIX_OUT, mix)

    manifest = {
        "seed": SEED,
        "rineya_combined": str(RINEYA_COMBINED),
        "rineya_rows": len(chat_rows),
        "rag_source": str(RAG_TRAIN),
        "rag_converted": str(RAG_SFT_OUT),
        "rag_rows": len(converted),
        "mix": str(MIX_OUT),
        "mix_rows": len(mix),
        "chat_fraction": round(len(chat_rows) / len(mix), 4),
        "evidence_fraction": round(len(converted) / len(mix), 4),
        "note": "RAG rows had assistant targets appended from approved_answer; official reviewed split was not modified.",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
