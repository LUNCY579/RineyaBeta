"""Teacher-forced tokenization with an explicit prompt-only semantic mask."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from dast.data import DASTExample

IGNORE_INDEX = -100


class TokenizationContractError(ValueError):
    """Raised when a tokenizer's full chat rendering is not prompt-prefix stable."""


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_mask: tuple[int, ...]
    training_target_texts: tuple[str, ...]
    canonical_target_texts: tuple[str, ...]
    problem_id: str
    answer: str
    answer_type: str
    category: str
    truncated_prompt_tokens: int


def _as_int_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise TokenizationContractError("expected one tokenized sequence")
        value = value[0]
    return [int(item) for item in value]


def apply_chat_template_ids(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    return _as_int_list(rendered)


def tokenize_example(
    example: DASTExample,
    tokenizer: Any,
    *,
    system_prompt: str,
    max_length: int,
    truncation: str,
) -> TokenizedExample:
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example.record.problem},
    ]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": example.record.full_response},
    ]
    prompt_ids = apply_chat_template_ids(tokenizer, prompt_messages, add_generation_prompt=True)
    full_ids = apply_chat_template_ids(tokenizer, full_messages, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise TokenizationContractError(
            "the tokenizer's full chat template is not prefixed by its generation prompt; "
            "refusing to guess the assistant-loss boundary"
        )
    response_ids = full_ids[len(prompt_ids) :]
    if not response_ids:
        raise TokenizationContractError(f"problem {example.record.problem_id} produced an empty assistant target")

    truncated = 0
    if len(full_ids) > max_length:
        if truncation == "error":
            raise TokenizationContractError(
                f"problem {example.record.problem_id} tokenizes to {len(full_ids)} > max_length={max_length}"
            )
        if truncation != "preserve_target":
            raise ValueError(f"unsupported truncation mode: {truncation}")
        if len(response_ids) >= max_length:
            raise TokenizationContractError(
                f"assistant target for {example.record.problem_id} alone exceeds max_length; cannot preserve it"
            )
        keep_prompt = max_length - len(response_ids)
        truncated = len(prompt_ids) - keep_prompt
        prompt_ids = prompt_ids[-keep_prompt:]
        full_ids = [*prompt_ids, *response_ids]

    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    prompt_mask = [1] * len(prompt_ids) + [0] * len(response_ids)
    if len(full_ids) != len(labels) or len(labels) != len(prompt_mask):
        raise AssertionError("tokenization lengths diverged")
    if not any(label != IGNORE_INDEX for label in labels):
        raise TokenizationContractError("no assistant tokens remain after tokenization")

    return TokenizedExample(
        input_ids=tuple(full_ids),
        labels=tuple(labels),
        prompt_mask=tuple(prompt_mask),
        training_target_texts=example.trajectory.training_texts,
        canonical_target_texts=example.trajectory.canonical_texts,
        problem_id=example.record.problem_id,
        answer=example.record.answer,
        answer_type=example.record.answer_type,
        category=example.record.category,
        truncated_prompt_tokens=truncated,
    )


def stable_group_id(text: str) -> int:
    digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
    # Keep within signed int64 so tensor conversion is portable.
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class CausalBatchCollator:
    """Right-pad causal-LM records and retain a separate semantic prompt mask."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        system_prompt: str,
        max_length: int,
        truncation: str = "preserve_target",
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.truncation = truncation
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is None:
                raise TokenizationContractError("tokenizer requires pad_token_id or eos_token_id")
            pad_token_id = eos
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: Sequence[DASTExample]) -> dict[str, Any]:
        tokenized = [
            tokenize_example(
                example,
                self.tokenizer,
                system_prompt=self.system_prompt,
                max_length=self.max_length,
                truncation=self.truncation,
            )
            for example in examples
        ]
        max_length = max(len(item.input_ids) for item in tokenized)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        prompt_mask: list[list[int]] = []
        for item in tokenized:
            pad = max_length - len(item.input_ids)
            input_ids.append([*item.input_ids, *([self.pad_token_id] * pad)])
            labels.append([*item.labels, *([IGNORE_INDEX] * pad)])
            attention_mask.append([1] * len(item.input_ids) + [0] * pad)
            prompt_mask.append([*item.prompt_mask, *([0] * pad)])

        training_target_texts = [list(item.training_target_texts) for item in tokenized]
        canonical_target_texts = [list(item.canonical_target_texts) for item in tokenized]
        training_group_ids = [
            [stable_group_id(text) for text in item.training_target_texts]
            for item in tokenized
        ]
        canonical_group_ids = [
            [stable_group_id(text) for text in item.canonical_target_texts]
            for item in tokenized
        ]
        result: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_mask": torch.tensor(prompt_mask, dtype=torch.bool),
            "training_target_texts": training_target_texts,
            "canonical_target_texts": canonical_target_texts,
            "problem_ids": [item.problem_id for item in tokenized],
            "answers": [item.answer for item in tokenized],
            "answer_types": [item.answer_type for item in tokenized],
            "categories": [item.category for item in tokenized],
            "truncated_prompt_tokens": [item.truncated_prompt_tokens for item in tokenized],
        }
        if training_group_ids and training_group_ids[0]:
            result["training_group_ids"] = torch.tensor(training_group_ids, dtype=torch.long)
        else:
            result["training_group_ids"] = None
        if canonical_group_ids and canonical_group_ids[0]:
            result["canonical_group_ids"] = torch.tensor(canonical_group_ids, dtype=torch.long)
        else:
            result["canonical_group_ids"] = None
        return result
