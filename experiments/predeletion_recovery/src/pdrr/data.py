"""Deterministic JSONL chat data preparation for PDRR."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncodedExample:
    example_id: str
    input_ids: list[int]
    labels: list[int]
    category: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _messages_and_target(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = [dict(message) for message in row.get("messages") or []]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("each training row must end with an assistant target")
    target = str(messages[-1].get("content", ""))
    return messages[:-1], target


def load_chat_jsonl(path: Path, tokenizer: Any, *, max_length: int,
                    max_rows: int | None = None) -> list[EncodedExample]:
    examples: list[EncodedExample] = []
    seen_ids: set[str] = set()
    eos = tokenizer.eos_token or ""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages, target = _messages_and_target(row)
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
            target_ids = tokenizer(target + eos, add_special_tokens=False).input_ids
            if len(target_ids) >= max_length:
                target_ids = target_ids[:max_length]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-(max_length - len(target_ids)):]
            ids = prompt_ids + target_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
            example_id = str(row.get("id") or f"line-{line_number}")
            if example_id in seen_ids:
                raise ValueError(f"duplicate example id {example_id!r}")
            seen_ids.add(example_id)
            examples.append(EncodedExample(
                example_id=example_id,
                input_ids=ids,
                labels=labels,
                category=str(row.get("category") or "uncategorized"),
            ))
            if max_rows is not None and len(examples) >= max_rows:
                break
    if not examples:
        raise ValueError(f"no examples loaded from {path}")
    return examples


def deterministic_order(size: int, *, seed: int, epoch: int = 0) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch * 1_000_003)
    return torch.randperm(size, generator=generator).tolist()


def collate(examples: Sequence[EncodedExample], pad_token_id: int) -> dict[str, Tensor]:
    width = max(len(example.input_ids) for example in examples)
    batch = len(examples)
    input_ids = torch.full((batch, width), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, width), dtype=torch.long)
    labels = torch.full((batch, width), IGNORE_INDEX, dtype=torch.long)
    ids = []
    categories = []
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(example.labels)
        ids.append(example.example_id)
        categories.append(example.category)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_ids": ids,
        "categories": categories,
    }
