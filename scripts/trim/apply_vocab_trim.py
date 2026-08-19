#!/usr/bin/env python3
"""Apply a trimmed tokenizer to the Qwen3-0.6B weights -> rung R2.

Two changes, both purely structural. No training, no weight values altered.

  Phase 1 (lossless cleanup)
      Stock model.safetensors stores lm_head.weight as a bitwise-identical copy
      of model.embed_tokens.weight even though tie_word_embeddings is true. The
      copy is not written out; transformers reconstructs the tied head on load.

  Phase 2 (vocabulary trim)
      model.embed_tokens.weight is gathered down to the kept rows, in new-id
      order, so row j of the new matrix is exactly row old_of[j] of the old one.
      Every other tensor is copied byte-for-byte.

Also remaps every hardcoded token id, which is the trap this surgery most easily
fails on:

  config.json            bos_token_id, eos_token_id
  generation_config.json bos_token_id, eos_token_id (a list), pad_token_id
  tokenizer_config.json  the 26 added_tokens_decoder keys

Usage: python apply_vocab_trim.py <tokenizer_dir> <out_dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Qwen3-0.6B"
COPY_AS_IS = ("tokenizer.json", "vocab.json", "merges.txt")


def remap_scalar_or_list(value, mapping: dict[int, int], label: str):
    if isinstance(value, list):
        return [remap_scalar_or_list(v, mapping, label) for v in value]
    if value is None:
        return None
    if value not in mapping:
        raise SystemExit(f"{label}: id {value} was dropped by the trim; refusing to write")
    return mapping[value]


def main() -> None:
    tok_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = json.loads((tok_dir / "trim_manifest.json").read_text(encoding="utf-8"))
    mapping = {int(k): v for k, v in manifest["old_to_new_id"].items()}
    new_vocab = manifest["vocab_size"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # new row j <- old row old_of[j]; unmapped rows are alignment padding.
    old_of = [-1] * new_vocab
    for old, new in mapping.items():
        old_of[new] = old
    padding = [j for j, o in enumerate(old_of) if o < 0]
    if padding and padding != list(range(manifest["real_tokens"], new_vocab)):
        raise SystemExit(f"unmapped rows are not a contiguous tail: {padding[:8]}")

    tensors: dict[str, torch.Tensor] = {}
    dropped_head = False
    with safe_open(str(MODEL / "model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        embed = f.get_tensor("model.embed_tokens.weight")
        if "lm_head.weight" in keys:
            if not torch.equal(f.get_tensor("lm_head.weight"), embed):
                raise SystemExit("lm_head is NOT tied to embed_tokens; trim would "
                                 "silently change the output head")
            dropped_head = True
        index = torch.tensor([o if o >= 0 else 0 for o in old_of], dtype=torch.long)
        new_embed = embed.index_select(0, index).clone()
        for j in padding:
            new_embed[j].zero_()
        tensors["model.embed_tokens.weight"] = new_embed
        for key in sorted(keys - {"model.embed_tokens.weight", "lm_head.weight"}):
            tensors[key] = f.get_tensor(key)

    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    config["vocab_size"] = new_vocab
    config["bos_token_id"] = remap_scalar_or_list(config.get("bos_token_id"), mapping, "config.bos")
    config["eos_token_id"] = remap_scalar_or_list(config.get("eos_token_id"), mapping, "config.eos")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    gen = json.loads((MODEL / "generation_config.json").read_text(encoding="utf-8"))
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if field in gen:
            gen[field] = remap_scalar_or_list(gen[field], mapping, f"generation.{field}")
    (out_dir / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n",
                                                    encoding="utf-8")

    tcfg = json.loads((MODEL / "tokenizer_config.json").read_text(encoding="utf-8"))
    tcfg["added_tokens_decoder"] = {
        str(mapping[int(old)]): entry
        for old, entry in tcfg["added_tokens_decoder"].items()
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tcfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for name in COPY_AS_IS:
        src = tok_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    shutil.copy2(tok_dir / "trim_manifest.json", out_dir / "trim_manifest.json")

    total = sum(t.numel() for t in tensors.values())
    print(f"wrote {out_dir}")
    print(f"  dropped redundant lm_head : {dropped_head}")
    print(f"  embedding rows            : {embed.shape[0]:,d} -> {new_embed.shape[0]:,d}")
    print(f"  zeroed alignment rows     : {len(padding):,d}")
    print(f"  tensors written           : {len(tensors)}")
    print(f"  resident params           : {total:,d}")
    print(f"  file bytes                : {(out_dir / 'model.safetensors').stat().st_size:,d}")
    print(f"  vocab_size                : {new_vocab:,d}")
    print(f"  bos/eos/pad remapped      : {config['bos_token_id']} / "
          f"{config['eos_token_id']} / {gen.get('pad_token_id')}")


if __name__ == "__main__":
    main()
