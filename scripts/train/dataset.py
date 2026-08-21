#!/usr/bin/env python3
"""Pre-tokenised healing dataset with a content-addressed cache.

Tokenisation happens once, up front, for three reasons:

  * The batch plan needs every row's length before training starts. Lengths
    cannot be discovered lazily without making the plan depend on iteration
    order.
  * Re-tokenising 169k rows on every resume would be slow and, worse, would be
    an opportunity for the lengths to change silently if the tokenizer moved.
  * The cache key is a hash of (mix bytes, tokenizer bytes, max_len, template
    settings), so a changed tokenizer or mix produces a cache miss instead of a
    stale hit.

Label construction is exact rather than heuristic. The prompt and the target are
tokenised *separately* and concatenated, so the supervised boundary is known by
construction and never depends on locating a delimiter inside a merged string.
The prompt is rendered with `add_generation_prompt=True, enable_thinking=False`,
matching the evaluation harness -- train and eval see the same prompt shape.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IGNORE_INDEX = -100


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Tokenized:
    ids: np.ndarray          # int32, all rows concatenated
    offsets: np.ndarray      # int64, length n+1
    prompt_lens: np.ndarray  # int32, supervised region starts here
    sources: list[str]
    categories: list[str]
    mix_sha256: str
    cache_key: str

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def total_lengths(self) -> list[int]:
        return (self.offsets[1:] - self.offsets[:-1]).astype(np.int64).tolist()

    @property
    def supervised_lengths(self) -> list[int]:
        """Positions that survive the causal shift and carry a label.

        The shift drops index 0, so a row supervises `len - max(prompt_len, 1)`
        positions. Using max(...,1) keeps this exactly equal to the non-ignore
        count the loss will see, including the rare row whose prompt was trimmed
        away entirely -- otherwise the plan's normaliser would be off by one.
        """
        total = self.offsets[1:] - self.offsets[:-1]
        starts = np.maximum(self.prompt_lens.astype(np.int64), 1)
        return np.maximum(total - starts, 0).astype(np.int64).tolist()

    def row(self, index: int) -> tuple[np.ndarray, int]:
        start, end = self.offsets[index], self.offsets[index + 1]
        return self.ids[start:end], int(self.prompt_lens[index])


def build(mix_path: Path, tokenizer, *, max_len: int, cache_dir: Path,
         tokenizer_dir: Path, verbose: bool = True) -> Tokenized:
    mix_sha = sha256_file(mix_path)
    tok_sha = sha256_file(tokenizer_dir / "tokenizer.json")
    key = hashlib.sha256(
        f"{mix_sha}|{tok_sha}|{max_len}|nothink|v1".encode()).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"tok_{key}.npz"
    meta_path = cache_dir / f"tok_{key}.json"

    if cache.exists() and meta_path.exists():
        blob = np.load(cache)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if verbose:
            print(f"tokenizer cache hit: {cache.name} ({meta['rows']:,d} rows)")
        return Tokenized(ids=blob["ids"], offsets=blob["offsets"],
                         prompt_lens=blob["prompt_lens"],
                         sources=meta["sources"], categories=meta["categories"],
                         mix_sha256=mix_sha, cache_key=key)

    eos = tokenizer.eos_token or "<|im_end|>"
    chunks: list[np.ndarray] = []
    offsets = [0]
    prompt_lens: list[int] = []
    sources: list[str] = []
    categories: list[str] = []
    truncated = 0

    with mix_path.open(encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            row = json.loads(line)
            msgs = row["messages"]
            prompt_text = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            target_text = str(msgs[-1]["content"]) + eos
            p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            t_ids = tokenizer(target_text, add_special_tokens=False).input_ids

            if len(p_ids) + len(t_ids) > max_len:
                truncated += 1
                # Keep the whole target and trim the prompt from the left: the
                # supervised region must stay intact or the row teaches nothing.
                room = max_len - len(t_ids)
                if room <= 0:
                    t_ids = t_ids[:max_len]
                    p_ids = []
                else:
                    p_ids = p_ids[-room:]
            ids = np.asarray(p_ids + t_ids, dtype=np.int32)
            chunks.append(ids)
            offsets.append(offsets[-1] + len(ids))
            prompt_lens.append(len(p_ids))
            sources.append(row.get("source", "?"))
            categories.append(row.get("category", "?"))
            if verbose and (n + 1) % 25_000 == 0:
                print(f"  tokenised {n + 1:,d} rows")

    data = Tokenized(
        ids=np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        prompt_lens=np.asarray(prompt_lens, dtype=np.int32),
        sources=sources, categories=categories, mix_sha256=mix_sha, cache_key=key)

    np.savez(cache, ids=data.ids, offsets=data.offsets, prompt_lens=data.prompt_lens)
    meta_path.write_text(json.dumps({
        "rows": len(data), "mix_sha256": mix_sha, "tokenizer_sha256": tok_sha,
        "max_len": max_len, "truncated_rows": truncated,
        "sources": sources, "categories": categories,
    }), encoding="utf-8")
    if verbose:
        print(f"tokenised {len(data):,d} rows, {int(data.offsets[-1]):,d} tokens"
              f"{f', {truncated:,d} left-trimmed at max_len' if truncated else ''}")
    return data


def collate(data: Tokenized, indices, pad_id: int, max_len: int):
    """Right-pad a micro-batch. Returns numpy arrays; caller moves to device."""
    rows = [data.row(i) for i in indices]
    width = min(max(len(ids) for ids, _ in rows), max_len)
    n = len(rows)
    input_ids = np.full((n, width), pad_id, dtype=np.int64)
    attention = np.zeros((n, width), dtype=np.int64)
    labels = np.full((n, width), IGNORE_INDEX, dtype=np.int64)
    for r, (ids, p_len) in enumerate(rows):
        ids = ids[:width]
        input_ids[r, :len(ids)] = ids
        attention[r, :len(ids)] = 1
        if p_len < len(ids):
            labels[r, p_len:len(ids)] = ids[p_len:]
    return input_ids, attention, labels
