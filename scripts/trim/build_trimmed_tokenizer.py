#!/usr/bin/env python3
"""Build an English-focused trimmed Qwen3 tokenizer.

Always kept, unconditionally:

  * ids 0-255          -- the complete byte-level alphabet. Qwen3 has
                          byte_fallback=false and unk_token=null, so these are
                          the only thing standing between an arbitrary input and
                          an unencodable one.
  * the 26 control ids -- <|im_start|>, <|im_end|>, <think>, ... The chat
                          template does not work without them.
  * every id observed  -- anything the Rineya/RAG corpora actually use, so
    in the corpora        in-domain tokenization cannot regress.

Everything else is filled up to a target vocabulary size by one of two rules:

  rank (default)
      Walk the merge list in its original order, keeping a merge A B -> AB when
      A and B are already kept and AB is English-script. BPE merge order *is* a
      frequency ranking learned from Qwen's own training corpus, so this keeps
      the English subwords that are actually common -- at any length -- and drops
      the rare tail. Walking in order also makes the merge list self-consistently
      valid: a kept merge can never reference a dropped token.

  len
      Keep English-script tokens whose decoded form is at most N characters.
      Retained for comparison only. It loses badly: at a matched vocabulary size
      it inflates held-out English prose far more than `rank`, because "short"
      and "frequent" are not the same thing and the model pays inflation on every
      forward pass forever. See docs/MINI_MODEL_ROADMAP.md.

Non-English scripts (CJK, kana, hangul, cyrillic, arabic, hebrew, thai, indic,
emoji, box-drawing) are dropped in both modes. Dropped text still encodes --
byte-level BPE falls back to shorter pieces -- so the cost is token inflation,
never an unrepresentable input. Verified by scripts/measure/token_inflation.py.

Usage:
  python build_trimmed_tokenizer.py rank <target_vocab> [out_dir]
  python build_trimmed_tokenizer.py len  <max_chars>    [out_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Qwen3-0.6B"
MANIFESTS = ROOT / "manifests"

sys.path.insert(0, str(ROOT / "scripts" / "measure"))
from vocab_script_census import byte_decoder  # noqa: E402

ALIGN = 128       # pad vocab_size to this multiple for the lm_head GEMM
RESIDENT = 596_049_920
LAYER_MACS = 440_466_432


def load_inputs() -> tuple[dict, dict, dict]:
    spec = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    census = json.loads((MANIFESTS / "vocab_script_census.json").read_text(encoding="utf-8"))
    usage = json.loads((MANIFESTS / "vocab_utilization.json").read_text(encoding="utf-8"))
    return spec, census, usage


def english_ids(census: dict) -> set[int]:
    out: set[int] = set()
    for cat in census["english_keep_categories"]:
        out |= set(census["ids_by_category"].get(cat, []))
    return out


def decoded_lengths(vocab: dict[str, int]) -> dict[int, int]:
    decoder = byte_decoder()
    lengths: dict[int, int] = {}
    for token, tid in vocab.items():
        try:
            raw = bytes(decoder[c] for c in token)
        except KeyError:
            lengths[tid] = 10**6
            continue
        try:
            lengths[tid] = len(raw.decode("utf-8"))
        except UnicodeDecodeError:
            lengths[tid] = 1  # partial multi-byte piece: cheap, always keep
    return lengths


def select(spec: dict, census: dict, usage: dict, mode: str, budget: int) -> tuple[set[int], dict]:
    vocab: dict[str, int] = spec["model"]["vocab"]
    english = english_ids(census)
    control = set(census["ids_by_category"].get("control", []))
    observed = set(usage["frequency_ranked_keep_order"]) | set(usage["mandatory_ids"])
    kept = set(range(256)) | control | observed

    if mode == "len":
        lengths = decoded_lengths(vocab)
        added = {tid for tid in english if lengths.get(tid, 10**6) <= budget}
        kept |= added
        stats = {"mode": mode, "max_chars": budget, "filled_by_rule": len(added)}
        return kept, stats

    # rank mode: walk merges in original (frequency) order.
    id_of = vocab
    remaining = budget - len(kept)
    filled = 0
    for a, b in spec["model"]["merges"]:
        merged = a + b
        tid = id_of.get(merged)
        if tid is None or tid in kept or tid not in english:
            continue
        if id_of.get(a) not in kept or id_of.get(b) not in kept:
            continue
        if remaining <= 0:
            break
        kept.add(tid)
        remaining -= 1
        filled += 1
    stats = {"mode": mode, "target_vocab": budget, "filled_by_rule": filled}
    return kept, stats


def build(mode: str, budget: int, out_dir: Path) -> dict:
    spec, census, usage = load_inputs()
    vocab: dict[str, int] = spec["model"]["vocab"]
    added_spec = spec.get("added_tokens", [])
    kept, stats = select(spec, census, usage, mode, budget)

    # Base tokens keep their original relative order, so ids 0-255 stay put and
    # the old->new mapping stays auditable. Control tokens sit above the base.
    base = sorted(tid for tid in kept if tid < len(vocab))
    id_of = {tid: new for new, tid in enumerate(base)}
    token_of = {tid: tok for tok, tid in vocab.items()}
    new_vocab = {token_of[tid]: id_of[tid] for tid in base}

    new_added = []
    for offset, entry in enumerate(sorted(added_spec, key=lambda a: a["id"])):
        clone = dict(entry)
        clone["id"] = len(base) + offset
        new_added.append(clone)
        id_of[entry["id"]] = clone["id"]

    real = len(base) + len(new_added)
    padded = ((real + ALIGN - 1) // ALIGN) * ALIGN

    kept_tokens = set(new_vocab)
    merges = [pair for pair in spec["model"]["merges"]
              if pair[0] in kept_tokens and pair[1] in kept_tokens
              and (pair[0] + pair[1]) in kept_tokens]
    reachable = {a + b for a, b in merges}
    dead = sum(1 for tok in kept_tokens if len(tok) > 1 and tok not in reachable)

    out_dir.mkdir(parents=True, exist_ok=True)
    trimmed = dict(spec)
    trimmed["added_tokens"] = new_added
    trimmed["model"] = dict(spec["model"], vocab=new_vocab, merges=merges)
    (out_dir / "tokenizer.json").write_text(
        json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")

    # Legacy companions, regenerated so nothing can read the stale 151k versions.
    (out_dir / "vocab.json").write_text(
        json.dumps(new_vocab, ensure_ascii=False), encoding="utf-8")
    (out_dir / "merges.txt").write_text(
        "#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in merges) + "\n",
        encoding="utf-8")

    saved = (151_936 - padded) * 1024
    manifest = {
        "source": MODEL.name,
        "align": ALIGN,
        "base_tokens": len(base),
        "control_tokens": len(new_added),
        "real_tokens": real,
        "vocab_size": padded,
        "alignment_padding_rows": padded - real,
        "unreachable_vocab_rows": dead,
        "merges_kept": len(merges),
        "merges_dropped": len(spec["model"]["merges"]) - len(merges),
        "keep_rule": stats,
        "embedding_params": padded * 1024,
        "saved_params": saved,
        "model_total": RESIDENT - saved,
        "per_token_macs": LAYER_MACS + 1024 * padded,
        "break_even_inflation": (LAYER_MACS + 1024 * 151_936) / (LAYER_MACS + 1024 * padded),
        "old_to_new_id": {str(k): v for k, v in sorted(id_of.items())},
    }
    (out_dir / "trim_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                encoding="utf-8")
    return manifest


def main() -> None:
    mode, budget = sys.argv[1], int(sys.argv[2])
    default = ROOT / f"artifacts/tokenizer_en_{mode}{budget}"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else default
    m = build(mode, budget, out)
    print(f"rule                    : {m['keep_rule']}")
    print(f"base + control          : {m['base_tokens']:,d} + {m['control_tokens']:,d}"
          f" = {m['real_tokens']:,d}")
    print(f"vocab_size (pad {ALIGN})    : {m['vocab_size']:,d}"
          f"  (+{m['alignment_padding_rows']} alignment rows)")
    print(f"unreachable vocab rows  : {m['unreachable_vocab_rows']:,d}")
    print(f"merges kept / dropped   : {m['merges_kept']:,d} / {m['merges_dropped']:,d}")
    print(f"embedding saving        : {m['saved_params']:,d} params"
          f"  -> model total {m['model_total']:,d}"
          f"  ({100 * (m['model_total'] / RESIDENT - 1):+.2f}%)")
    print(f"break-even inflation    : {m['break_even_inflation']:.4f}x"
          f"  ({100 * (m['break_even_inflation'] - 1):+.2f}%)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
