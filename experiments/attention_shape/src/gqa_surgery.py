#!/usr/bin/env python3
"""Merge KV heads 8 -> 4/2/1 on a Qwen3 checkpoint, with provenance.

Follows the I2 contract the vocab trim already satisfies: every operation emits a
manifest recording exactly what changed, is byte-deterministic, and is verifiable
by an invariant check.

The surgery itself is small. Qwen3 has no attention bias and its q_norm/k_norm
are per-head RMSNorms of size `head_dim`, so they are independent of the head
*count* and are copied untouched. Only `k_proj.weight` and `v_proj.weight`
change shape: `[kv_heads * head_dim, hidden]` becomes `[g * head_dim, hidden]`,
by pooling each contiguous group of `kv_heads / g` heads.

Two merge rules, because the choice is not obviously innocent:

  mean    average each group. The standard GQA construction: it preserves the
          group's mean key direction and is what the GQA paper uptrains from.
  first   keep the first head of each group and discard the rest. A deliberately
          worse-looking baseline that isolates how much the *averaging* matters
          as opposed to the mere reduction in count.

**These checkpoints are not usable models.** Merging KV heads without healing
breaks the learned correspondence between each query head and its key subspace.
Numbers measured here are a *damage floor* -- the worst case before repair --
and are labelled as such everywhere they are reported. They exist to bound the
repair budget, not to be compared against healed variants.

Usage:
    python src/gqa_surgery.py --kv-heads 4 2 1
    python src/gqa_surgery.py --kv-heads 2 --rule first --out-root artifacts/attn_shape
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]
BASE = "artifacts/rineya-core-en32k-28L"
OUT_ROOT = "artifacts/attn_shape"
# Copied verbatim so each variant is a self-contained loadable checkpoint.
SIDECAR = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
           "generation_config.json", "LICENSE")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def merge_kv(w: torch.Tensor, src_heads: int, dst_heads: int, head_dim: int,
             rule: str) -> torch.Tensor:
    """Pool `[src_heads*head_dim, hidden]` down to `[dst_heads*head_dim, hidden]`."""
    if src_heads % dst_heads:
        raise ValueError(f"{dst_heads} does not divide {src_heads}")
    hidden = w.shape[1]
    group = src_heads // dst_heads
    heads = w.reshape(src_heads, head_dim, hidden)
    grouped = heads.reshape(dst_heads, group, head_dim, hidden)
    if rule == "mean":
        # float32 for the reduction so the average does not accumulate in bf16.
        merged = grouped.to(torch.float32).mean(dim=1).to(w.dtype)
    elif rule == "first":
        merged = grouped[:, 0]
    else:
        raise ValueError(f"unknown merge rule {rule!r}")
    return merged.reshape(dst_heads * head_dim, hidden).contiguous()


def surger(base_dir: Path, out_dir: Path, dst_heads: int, rule: str) -> dict:
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    src_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    n_layers = cfg["num_hidden_layers"]

    tensors = load_file(str(base_dir / "model.safetensors"))
    out: dict[str, torch.Tensor] = {}
    touched, params_before, params_after = [], 0, 0

    for key, t in tensors.items():
        if key.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
            merged = merge_kv(t, src_heads, dst_heads, head_dim, rule)
            out[key] = merged
            touched.append(key)
            params_before += t.numel()
            params_after += merged.numel()
        else:
            out[key] = t.clone()
            params_before += t.numel()
            params_after += t.numel()

    expected = 2 * n_layers
    if len(touched) != expected:
        raise SystemExit(f"expected {expected} k/v tensors, merged {len(touched)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"),
              metadata={"format": "pt"})

    new_cfg = dict(cfg)
    new_cfg["num_key_value_heads"] = dst_heads
    (out_dir / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n",
                                         encoding="utf-8")
    for name in SIDECAR:
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    weights_sha = sha256_file(out_dir / "model.safetensors")
    manifest = {
        "operation": "gqa_kv_head_merge",
        "source": str(base_dir.relative_to(REPO)),
        "source_weights_sha256": sha256_file(base_dir / "model.safetensors"),
        "merge_rule": rule,
        "kv_heads_before": src_heads,
        "kv_heads_after": dst_heads,
        "group_size": src_heads // dst_heads,
        "head_dim": head_dim,
        "tensors_modified": sorted(touched),
        "params_before": params_before,
        "params_after": params_after,
        "params_removed": params_before - params_after,
        "kv_bytes_per_token_before": 2 * n_layers * src_heads * head_dim * 2,
        "kv_bytes_per_token_after": 2 * n_layers * dst_heads * head_dim * 2,
        "output_weights_sha256": weights_sha,
        "healed": False,
        "warning": "UNHEALED. Quality numbers from this checkpoint are a damage "
                   "floor, not a measurement of what KV compression costs.",
    }
    (out_dir / "surgery_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(base_dir: Path, out_dir: Path, dst_heads: int, rule: str) -> bool:
    """Structural invariants: shapes right, untouched tensors bit-identical,
    and the merge really is the arithmetic mean of its group."""
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    if cfg["num_key_value_heads"] != dst_heads:
        print("  FAIL config num_key_value_heads")
        return False

    base = load_file(str(base_dir / "model.safetensors"))
    new = load_file(str(out_dir / "model.safetensors"))
    head_dim = cfg["head_dim"]
    src_heads = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))[
        "num_key_value_heads"]

    if set(base) != set(new):
        print("  FAIL tensor key sets differ")
        return False

    bad = 0
    for key, t in new.items():
        if key.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
            if t.shape[0] != dst_heads * head_dim:
                print(f"  FAIL shape {key}: {tuple(t.shape)}")
                bad += 1
            elif rule == "mean":
                want = merge_kv(base[key], src_heads, dst_heads, head_dim, "mean")
                if not torch.equal(t, want):
                    print(f"  FAIL merge not reproducible: {key}")
                    bad += 1
        elif not torch.equal(t, base[key]):
            print(f"  FAIL untouched tensor changed: {key}")
            bad += 1
    if bad:
        return False
    print(f"  verified: shapes correct, merge reproducible, "
          f"{len(base) - 2 * cfg['num_hidden_layers']} tensors bit-identical")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--kv-heads", type=int, nargs="+", default=[4, 2, 1])
    ap.add_argument("--rule", default="mean", choices=("mean", "first"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base_dir = REPO / args.base
    for g in args.kv_heads:
        suffix = "" if args.rule == "mean" else f"-{args.rule}"
        out_dir = REPO / args.out_root / f"r2-gqa{g}kv{suffix}"
        print(f"\n8 -> {g} KV heads ({args.rule})  ->  {out_dir.relative_to(REPO)}")
        if out_dir.exists() and not args.force:
            print("  exists; skipping (use --force to rebuild)")
            continue
        man = surger(base_dir, out_dir, g, args.rule)
        print(f"  params {man['params_before']:,} -> {man['params_after']:,} "
              f"(-{man['params_removed']:,})")
        print(f"  KV/token {man['kv_bytes_per_token_before'] / 1024:.0f} KiB -> "
              f"{man['kv_bytes_per_token_after'] / 1024:.0f} KiB")
        if not verify(base_dir, out_dir, g, args.rule):
            raise SystemExit(f"verification failed for {out_dir}")


if __name__ == "__main__":
    main()
