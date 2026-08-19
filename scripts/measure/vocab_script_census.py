#!/usr/bin/env python3
"""Classify every Qwen3 vocabulary entry by writing system.

Qwen3 stores BPE tokens in GPT-2 byte-level form, so a token string is not the
text it represents. Each character is first mapped back to its raw byte, then
the byte string is UTF-8 decoded (tokens may hold partial multi-byte sequences,
which decode to nothing and are reported as `byte_fragment`).

The census answers: how much of the embedding table is non-English script, and
what does an English-focused vocabulary actually cost?
"""

from __future__ import annotations

import collections
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Qwen3-0.6B"
OUT = ROOT / "manifests" / "vocab_script_census.json"

# Codepoint ranges, checked in order. First match wins.
RANGES: tuple[tuple[str, int, int], ...] = (
    ("cjk", 0x4E00, 0x9FFF),
    ("cjk", 0x3400, 0x4DBF),
    ("cjk", 0x20000, 0x2FA1F),
    ("cjk_punct", 0x3000, 0x303F),
    ("cjk_compat", 0xF900, 0xFAFF),
    ("cjk_fullwidth", 0xFF00, 0xFFEF),
    ("kana", 0x3040, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("cyrillic", 0x0400, 0x052F),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("arabic", 0xFB50, 0xFDFF),
    ("arabic", 0xFE70, 0xFEFF),
    ("hebrew", 0x0590, 0x05FF),
    ("devanagari", 0x0900, 0x097F),
    ("indic_other", 0x0980, 0x0DFF),
    ("thai", 0x0E00, 0x0E7F),
    ("lao_khmer_myanmar", 0x0E80, 0x109F),
    ("lao_khmer_myanmar", 0x1780, 0x17FF),
    ("georgian_armenian", 0x0530, 0x058F),
    ("georgian_armenian", 0x10A0, 0x10FF),
    ("ethiopic_cherokee", 0x1200, 0x139F),
    ("greek", 0x0370, 0x03FF),
    ("greek", 0x1F00, 0x1FFF),
    ("latin_ext", 0x00C0, 0x024F),
    ("latin_ext", 0x1E00, 0x1EFF),
    ("ipa_modifier", 0x0250, 0x02FF),
    ("combining", 0x0300, 0x036F),
    ("punct_general", 0x2000, 0x206F),
    ("currency_symbol", 0x20A0, 0x20CF),
    ("math_symbol", 0x2100, 0x21FF),
    ("math_symbol", 0x2200, 0x22FF),
    ("math_symbol", 0x27C0, 0x27EF),
    ("math_symbol", 0x2A00, 0x2AFF),
    ("box_shape", 0x2500, 0x25FF),
    ("dingbat_symbol", 0x2600, 0x27BF),
    ("emoji", 0x1F300, 0x1FAFF),
    ("emoji", 0x1F000, 0x1F2FF),
    ("private_use", 0xE000, 0xF8FF),
)

# Categories that an English-only daily-use model keeps.
ENGLISH_KEEP = frozenset({
    "ascii", "latin_ext", "punct_general", "currency_symbol", "math_symbol",
    "greek", "combining", "byte_fragment",
})


def byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2 bytes_to_unicode: visible char -> raw byte."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def classify(text: str) -> str:
    """Label a decoded token by the most distinctive script it contains."""
    if not text:
        return "byte_fragment"
    labels = []
    for ch in text:
        cp = ord(ch)
        if cp < 0x80:
            labels.append("ascii")
            continue
        hit = next((name for name, lo, hi in RANGES if lo <= cp <= hi), None)
        labels.append(hit or f"other_{unicodedata.category(ch)}")
    non_ascii = [l for l in labels if l != "ascii"]
    if not non_ascii:
        return "ascii"
    # A mixed token belongs to whichever non-ASCII script dominates it.
    return collections.Counter(non_ascii).most_common(1)[0][0]


def main() -> None:
    spec = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    vocab: dict[str, int] = spec["model"]["vocab"]
    hidden = config["hidden_size"]
    decoder = byte_decoder()

    by_cat: dict[str, list[int]] = collections.defaultdict(list)
    undecodable = 0
    for token, tid in vocab.items():
        try:
            raw = bytes(decoder[c] for c in token)
        except KeyError:
            by_cat["unmappable"].append(tid)
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            undecodable += 1
            by_cat["byte_fragment"].append(tid)
            continue
        by_cat[classify(text)].append(tid)

    for added in spec.get("added_tokens", []):
        by_cat["control"].append(added["id"])

    total = sum(len(v) for v in by_cat.values())
    print(f"vocab entries classified: {total:,d} "
          f"(tokenizer vocab {len(vocab):,d} + {len(spec.get('added_tokens', [])):,d} control)")
    print(f"partial UTF-8 byte fragments: {undecodable:,d}\n")

    print(f"{'category':22s} {'tokens':>8s} {'share':>7s} {'params':>13s}  keep?")
    kept = 0
    for cat, ids in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        keep = cat in ENGLISH_KEEP or cat == "control"
        kept += len(ids) if keep else 0
        print(f"{cat:22s} {len(ids):>8,d} {100 * len(ids) / total:>6.2f}% "
              f"{len(ids) * hidden:>13,d}  {'KEEP' if keep else 'drop'}")

    dropped = total - kept
    print(f"\n{'ENGLISH-ONLY KEEP':22s} {kept:>8,d} {100 * kept / total:>6.2f}% "
          f"{kept * hidden:>13,d}")
    print(f"{'DROP (non-English)':22s} {dropped:>8,d} {100 * dropped / total:>6.2f}% "
          f"{dropped * hidden:>13,d}")

    resident = 596_049_920
    saved = dropped * hidden
    print(f"\nsaving {saved:,d} params -> model total {resident - saved:,d} "
          f"({100 * (resident - saved - resident) / resident:+.2f}%)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "vocab_size": config["vocab_size"],
        "hidden_size": hidden,
        "classified": total,
        "byte_fragments": undecodable,
        "english_keep_categories": sorted(ENGLISH_KEEP),
        "counts": {k: len(v) for k, v in sorted(by_cat.items())},
        "ids_by_category": {k: sorted(v) for k, v in sorted(by_cat.items())},
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
