"""Batching for causal-LM SFT with assistant-only labels and per-example weights.

Kept torch-optional: ``pad_batch`` builds plain Python lists (unit-testable on a
CPU-only machine) and ``to_torch_batch`` converts them. Padded positions are
excluded from attention *and* from the loss.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from rineya_lab.training.chat import IGNORE_INDEX, EncodedExample


@dataclass
class PaddedBatch:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    labels: list[list[int]]
    weights: list[float]
    example_ids: list[str]
    problem_ids: list[str]
    supervised_token_counts: list[int]
    padded_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.input_ids)


def pad_batch(
    encoded: list[EncodedExample],
    *,
    pad_token_id: int,
    weights: list[float] | None = None,
    example_ids: list[str] | None = None,
    problem_ids: list[str] | None = None,
    pad_to_multiple_of: int = 8,
) -> PaddedBatch:
    if not encoded:
        raise ValueError("pad_batch received an empty batch")
    longest = max(len(item) for item in encoded)
    if pad_to_multiple_of > 1:
        remainder = longest % pad_to_multiple_of
        if remainder:
            longest += pad_to_multiple_of - remainder

    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    supervised: list[int] = []
    padded_tokens = 0

    for item in encoded:
        pad_len = longest - len(item.input_ids)
        padded_tokens += pad_len
        input_ids.append([*item.input_ids, *([pad_token_id] * pad_len)])
        attention_mask.append([*([1] * len(item.input_ids)), *([0] * pad_len)])
        labels.append([*item.labels, *([IGNORE_INDEX] * pad_len)])
        supervised.append(sum(1 for label in item.labels if label != IGNORE_INDEX))

    count = len(encoded)
    return PaddedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        weights=list(weights) if weights is not None else [1.0] * count,
        example_ids=list(example_ids) if example_ids is not None else [""] * count,
        problem_ids=list(problem_ids) if problem_ids is not None else [""] * count,
        supervised_token_counts=supervised,
        padded_tokens=padded_tokens,
        meta={"max_length": longest, "batch_size": count},
    )


def to_torch_batch(batch: PaddedBatch, device=None) -> dict[str, Any]:
    import torch

    tensors = {
        "input_ids": torch.tensor(batch.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch.attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch.labels, dtype=torch.long),
        "weights": torch.tensor(batch.weights, dtype=torch.float),
    }
    if device is not None:
        tensors = {key: value.to(device) for key, value in tensors.items()}
    return tensors


def weighted_causal_lm_loss(logits, labels, weights, *, label_smoothing: float = 0.0):
    """Per-example mean token loss, then a weighted mean across the batch.

    Averaging within an example before weighting is what makes
    ``loss_weighting: inverse_transitions`` meaningful: each *source problem*
    then contributes comparable gradient mass in the SPLIT arm regardless of how
    many transitions it expanded into.
    """
    import torch
    import torch.nn.functional as functional

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    token_loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
        label_smoothing=label_smoothing,
    ).view(shift_labels.size())

    valid = (shift_labels != IGNORE_INDEX).to(token_loss.dtype)
    per_example_tokens = valid.sum(dim=-1)
    safe_tokens = per_example_tokens.clamp(min=1.0)
    per_example_loss = (token_loss * valid).sum(dim=-1) / safe_tokens

    weights = weights.to(per_example_loss.dtype)
    # Examples with zero supervised tokens (only possible under pathological
    # truncation) must not silently pull the mean toward zero.
    active = (per_example_tokens > 0).to(per_example_loss.dtype)
    effective_weights = weights * active
    denominator = effective_weights.sum().clamp(min=torch.finfo(per_example_loss.dtype).eps)
    loss = (per_example_loss * effective_weights).sum() / denominator

    return loss, {
        "per_example_loss": per_example_loss.detach(),
        "supervised_tokens": int(per_example_tokens.sum().item()),
        "active_examples": int(active.sum().item()),
    }


def batched(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield fixed-size chunks; the final chunk may be short."""
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
