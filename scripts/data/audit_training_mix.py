#!/usr/bin/env python3
"""Audit every candidate healing corpus before any of it is trained on.

Answers four questions that decide whether a mix is safe:

  1. Inventory    -- how many rows, in which splits, and is the row already in
                     chat form or does it still need conversion?
  2. Overlap      -- the Rineya corpora nest by design (v2 carries a v1
                     preservation slice, v3 carries a v1+v2 one). Concatenating
                     v1+v2+v3 therefore trains the oldest rows several times over.
                     Measured by prompt hash, not assumed from the READMEs.
  3. Contamination-- does any training row share a prompt with an evaluation set?
                     Checked against the frozen long-context demo pack and the
                     puzzle validation/test/challenge splits.
  4. Budget       -- prompt/target/total token lengths under the *trimmed* 32k
                     tokenizer, because that is what will actually be trained,
                     and sequence length is the binding constraint on 8 GB.

Usage: python audit_training_mix.py [rows_for_length_stats]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = ROOT / "artifacts" / "rineya-core-en32k-28L" / "tokenizer.json"
OUT = ROOT / "manifests" / "training_mix_audit.json"
WS = re.compile(r"\s+")

V2 = "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v2.0/data/sft"
V3 = "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft"
V1 = "Rineya_Short_Response_Behavior_Corpus_v1.0/data/sft"
LCR = "Rineya_LCR_SFT_v1/Rineya_LCR_SFT_v1/data"
PZ = "Rineya-Puzzle-Reasoning-51k/data"
LCD = "Rineya_Long_Context_Demo_v1/data"

# (label, path, role) -- role "train" is a healing candidate, "eval" must never
# be trained on and is only used as a contamination reference.
CORPORA = (
    ("v1 short-response combined", f"{V1}/rineya_short_response_combined_train.jsonl", "train"),
    ("v1 preservation slice", f"{V1}/rineya_short_response_preservation_slice.jsonl", "train"),
    ("v2 recommended combined 40k", f"{V2}/rineya_v2_recommended_combined_40000.jsonl", "train"),
    ("v2 new-all", f"{V2}/rineya_v2_new_all_train.jsonl", "train"),
    ("v2 carries v1 slice 8k", f"{V2}/rineya_v1_preservation_slice_8000.jsonl", "train"),
    ("v3 recommended combined 124k", f"{V3}/rineya_v3_recommended_combined_124000.jsonl", "train"),
    ("v3 new-all 110k", f"{V3}/rineya_v3_new_all_train_110000.jsonl", "train"),
    ("v3 carries v1+v2 slice 14k", f"{V3}/rineya_v1_v2_preservation_slice_14000.jsonl", "train"),
    ("RAG v3 train 4,410", "Aegis_RAG_v3_train_4410/rag_v3_train.sft.jsonl", "train"),
    ("LCR-SFT v1 (all splits)", f"{LCR}/lcr_sft_v1.jsonl", "train"),
    ("Puzzle-51k train", f"{PZ}/train.jsonl", "train"),
    ("Puzzle-51k validation", f"{PZ}/validation.jsonl", "eval"),
    ("Puzzle-51k test", f"{PZ}/test.jsonl", "eval"),
    ("Puzzle-51k challenge", f"{PZ}/challenge.jsonl", "eval"),
    ("LongContextDemo (FROZEN EVAL)", f"{LCD}/rineya_long_context_demo_full.jsonl", "eval"),
)


def extract(row: dict) -> tuple[str, str, bool]:
    """-> (prompt text, target text, already_in_chat_form)"""
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        prompt = "\n".join(str(m.get("content", "")) for m in msgs
                           if m.get("role") != "assistant")
        target = next((str(m.get("content", "")) for m in reversed(msgs)
                       if m.get("role") == "assistant"), "")
        return prompt, target, True
    # raw (puzzle) form: needs conversion before it can be trained
    prompt = str(row.get("prompt") or row.get("question") or "")
    target = str(row.get("final_answer") or row.get("answer")
                 or row.get("approved_answer") or "")
    return prompt, target, False


def norm_hash(text: str) -> str:
    return hashlib.sha256(WS.sub(" ", text.strip().lower()).encode()).hexdigest()[:16]


def main() -> None:
    sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    tok = Tokenizer.from_file(str(TOKENIZER))

    inventory, hashes, lengths = {}, {}, {}
    print(f"{'corpus':32s} {'rows':>8s} {'chat':>6s} {'raw':>6s}  splits")
    for label, rel, role in CORPORA:
        path = ROOT / rel
        if not path.exists():
            print(f"{label:32s} {'MISSING':>8s}")
            continue
        seen: set[str] = set()
        splits: Counter = Counter()
        chat = raw = 0
        plens, tlens = [], []
        with path.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt, target, is_chat = extract(row)
                chat += is_chat
                raw += not is_chat
                splits[str(row.get("split", "-"))] += 1
                seen.add(norm_hash(prompt))
                if i < sample_n:
                    plens.append(len(tok.encode(prompt, add_special_tokens=False).ids))
                    tlens.append(len(tok.encode(target, add_special_tokens=False).ids))
        total = chat + raw
        inventory[label] = {"path": rel, "role": role, "rows": total,
                            "chat_form": chat, "raw_form": raw,
                            "splits": dict(splits)}
        hashes[label] = seen
        lengths[label] = (plens, tlens)
        top = ", ".join(f"{k}={v:,d}" for k, v in splits.most_common(4))
        print(f"{label:32s} {total:>8,d} {chat:>6,d} {raw:>6,d}  {top}")

    def pct(part, whole):
        return f"{100 * part / whole:.1f}%" if whole else "-"

    print("\n=== 2. overlap inside the Rineya chain (prompt hash) ===")
    chain = [("v1 short-response combined", "v2 recommended combined 40k"),
             ("v1 short-response combined", "v3 recommended combined 124k"),
             ("v2 recommended combined 40k", "v3 recommended combined 124k"),
             ("v2 new-all", "v3 recommended combined 124k"),
             ("v3 carries v1+v2 slice 14k", "v1 short-response combined"),
             ("v3 carries v1+v2 slice 14k", "v2 recommended combined 40k"),
             ("v2 carries v1 slice 8k", "v1 short-response combined")]
    overlaps = {}
    for a, b in chain:
        if a not in hashes or b not in hashes:
            continue
        shared = hashes[a] & hashes[b]
        overlaps[f"{a} & {b}"] = len(shared)
        print(f"  {a[:30]:30s} & {b[:30]:30s} {len(shared):>7,d}"
              f"  ({pct(len(shared), len(hashes[a]))} of first)")

    print("\n=== 3. contamination: training rows vs EVAL sets ===")
    trains = [l for l, (_, _, r) in zip(inventory, CORPORA) if False]  # placeholder
    train_labels = [l for l in inventory if inventory[l]["role"] == "train"]
    eval_labels = [l for l in inventory if inventory[l]["role"] == "eval"]
    contamination = {}
    for e in eval_labels:
        for t in train_labels:
            shared = hashes[e] & hashes[t]
            if shared:
                contamination[f"{t} -> {e}"] = len(shared)
                print(f"  !! {t} shares {len(shared):,d} prompts with {e}")
    if not contamination:
        print("  clean: no training corpus shares a normalized prompt with any eval set")

    print(f"\n=== 4. sequence budget under the 32k tokenizer (first {sample_n:,d} rows) ===")
    print(f"{'corpus':32s} {'prompt p50':>10s} {'p99':>7s} {'max':>7s} "
          f"{'tgt p50':>8s} {'tgt p99':>8s} {'total p99':>10s}")
    budget = {}
    for label, (plens, tlens) in lengths.items():
        if not plens:
            continue
        p = sorted(plens); t = sorted(tlens)
        tot = sorted(a + b for a, b in zip(plens, tlens))
        q = lambda xs, f: xs[min(int(len(xs) * f), len(xs) - 1)]
        budget[label] = {"prompt_p50": q(p, .5), "prompt_p99": q(p, .99),
                         "prompt_max": max(p), "target_p50": q(t, .5),
                         "target_p99": q(t, .99), "total_p99": q(tot, .99),
                         "total_max": max(tot)}
        b = budget[label]
        print(f"{label:32s} {b['prompt_p50']:>10,d} {b['prompt_p99']:>7,d} "
              f"{b['prompt_max']:>7,d} {b['target_p50']:>8,d} "
              f"{b['target_p99']:>8,d} {b['total_p99']:>10,d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "tokenizer": str(TOKENIZER.relative_to(ROOT)),
        "length_sample_rows": sample_n,
        "inventory": inventory,
        "chain_overlap_prompt_hashes": overlaps,
        "contamination": contamination,
        "sequence_budget": budget,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
