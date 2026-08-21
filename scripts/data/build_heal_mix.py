#!/usr/bin/env python3
"""Build a deterministic healing mix (run B0/B1/B2/B3) from the audited corpora.

Applies every decision from docs/MINI_MODEL_ROADMAP.md §12:

  * v3 recommended combined 124k is the ONLY chat source. v1 and v2 are not
    appended -- v2 already carries a v1 slice and v3 already carries a v1+v2 one,
    so concatenating all three trains ~24,474 duplicate prompts, weighted toward
    the oldest and most brevity-biased rows (§12.2).
  * Puzzle-51k ships raw and is converted `short_sft` style: assistant target is
    the bare `final_answer`. `think_sft` is refused because it collides with the
    no-CoT target rule, the 24-token limit, and `enable_thinking=false` (§12.5).
  * Split filters are enforced, not trusted: LCR-SFT keeps `split == "train"`
    only, Puzzle keeps `split == "train"` only.
  * Rineya_Long_Context_Demo_v1 is never a source. It is the frozen eval pack.
  * Cross-corpus dedup by normalized prompt hash, first occurrence wins in a
    fixed corpus order, so the result does not depend on dict iteration.

Determinism: the output row order is a seeded Fisher-Yates shuffle over a list
built in a fixed order, so the same seed and the same inputs give a byte-identical
mix. The manifest records the sha256 of the output and of every input.

Usage: python build_heal_mix.py <B0|B1|B2|B3> [out_dir]
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "artifacts" / "heal_mixes"
WS = re.compile(r"\s+")
SEED = 20260819

V3 = "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/rineya_v3_recommended_combined_124000.jsonl"
RAG = "Aegis_RAG_v3_train_4410/rag_v3_train.sft.jsonl"
PUZZLE = "Rineya-Puzzle-Reasoning-51k/data/train.jsonl"
LCR = "Rineya_LCR_SFT_v1/Rineya_LCR_SFT_v1/data/lcr_sft_v1.jsonl"

PUZZLE_SYSTEM = "Solve the problem and return only the short final answer requested by the user."

# Refuse to read anything under here: it is the frozen evaluation pack.
FORBIDDEN = ("Rineya_Long_Context_Demo_v1",)

RUNS = {
    "B0": ("chat + grounding control", ("v3", "rag")),
    "B1": ("B0 + puzzle reasoning", ("v3", "rag", "puzzle")),
    "B2": ("B0 + long context", ("v3", "rag", "lcr")),
    "B3": ("everything", ("v3", "rag", "puzzle", "lcr")),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_hash(text: str) -> str:
    return hashlib.sha256(WS.sub(" ", text.strip().lower()).encode()).hexdigest()[:16]


def prompt_of(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages
                     if m.get("role") != "assistant")


def from_chat(row: dict, source: str, category_key: str) -> dict | None:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    if msgs[-1].get("role") != "assistant" or not str(msgs[-1].get("content", "")).strip():
        return None
    meta = row.get("metadata") or {}
    return {
        "messages": msgs,
        "source": source,
        "source_id": row.get("id"),
        "category": str(row.get(category_key) or meta.get(category_key)
                        or row.get("mode") or "unlabelled"),
    }


def load_v3() -> list[dict]:
    out = []
    for row in read_jsonl(ROOT / V3):
        rec = from_chat(row, "v3_chat", "reply_class")
        if rec:
            out.append(rec)
    return out


def load_rag() -> list[dict]:
    out = []
    for row in read_jsonl(ROOT / RAG):
        rec = from_chat(row, "rag_v3", "reply_class")
        if rec:
            rec["decision"] = row.get("decision")
            out.append(rec)
    return out


def load_lcr() -> list[dict]:
    out = []
    for row in read_jsonl(ROOT / LCR):
        if row.get("split") != "train":
            continue
        rec = from_chat(row, "lcr_sft", "task_type")
        if rec:
            rec["variant"] = row.get("variant")
            out.append(rec)
    return out


def load_puzzle() -> list[dict]:
    """Raw -> short_sft chat form. Targets stay bare final answers."""
    out = []
    for row in read_jsonl(ROOT / PUZZLE):
        if row.get("split") != "train":
            continue
        prompt = str(row.get("prompt") or "").strip()
        answer = str(row.get("final_answer") or "").strip()
        if not prompt or not answer:
            continue
        out.append({
            "messages": [
                {"role": "system", "content": PUZZLE_SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "source": "puzzle51k",
            "source_id": row.get("id"),
            "category": str(row.get("family") or "unlabelled"),
        })
    return out


LOADERS = {"v3": load_v3, "rag": load_rag, "puzzle": load_puzzle, "lcr": load_lcr}
PATHS = {"v3": V3, "rag": RAG, "puzzle": PUZZLE, "lcr": LCR}


def read_jsonl(path: Path):
    for part in FORBIDDEN:
        if part in str(path):
            raise SystemExit(f"refusing to read {path}: frozen evaluation pack")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> None:
    run = sys.argv[1].upper()
    if run not in RUNS:
        raise SystemExit(f"unknown run {run}; choose from {sorted(RUNS)}")
    label, sources = RUNS[run]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{run}: {label}\nsources: {', '.join(sources)}\n")
    rows: list[dict] = []
    per_source, dropped = {}, {}
    seen: set[str] = set()
    for name in sources:                      # fixed order -> deterministic dedup
        loaded = LOADERS[name]()
        kept = 0
        dupes = 0
        for rec in loaded:
            h = norm_hash(prompt_of(rec["messages"]))
            if h in seen:
                dupes += 1
                continue
            seen.add(h)
            rows.append(rec)
            kept += 1
        per_source[name] = kept
        dropped[name] = dupes
        print(f"  {name:10s} loaded {len(loaded):>7,d}  kept {kept:>7,d}  "
              f"dropped as duplicate {dupes:>6,d}")

    rng = random.Random(SEED)
    rng.shuffle(rows)

    out_file = out_dir / f"heal_mix_{run.lower()}.jsonl"
    with out_file.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in rows:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    cats: dict[str, dict[str, int]] = {}
    for rec in rows:
        cats.setdefault(rec["source"], {})
        cats[rec["source"]][rec["category"]] = cats[rec["source"]].get(rec["category"], 0) + 1

    manifest = {
        "run": run,
        "label": label,
        "seed": SEED,
        "sources": list(sources),
        "rows": len(rows),
        "rows_per_source": per_source,
        "duplicates_dropped": dropped,
        "categories_per_source": {k: dict(sorted(v.items())) for k, v in sorted(cats.items())},
        "inputs": {name: {"path": PATHS[name], "sha256": sha256_file(ROOT / PATHS[name])}
                   for name in sources},
        "output": {"path": str(out_file.relative_to(ROOT)).replace("\\", "/"),
                   "sha256": sha256_file(out_file), "rows": len(rows)},
        "excluded": {"Rineya_Long_Context_Demo_v1": "frozen evaluation pack",
                     "v1_short_response": "already carried by v2/v3 preservation slices",
                     "v2_combined": "already carried by the v3 preservation slice",
                     "puzzle_think_sft": "conflicts with no-CoT targets and enable_thinking=false"},
    }
    (out_dir / "mix_manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")
    print(f"\nrows          : {len(rows):,d}")
    print(f"output sha256 : {manifest['output']['sha256']}")
    print(f"wrote {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
