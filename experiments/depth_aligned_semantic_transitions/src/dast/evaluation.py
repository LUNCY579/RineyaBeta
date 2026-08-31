"""Generation evaluation and conservative answer extraction."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import torch

from dast.artifacts import atomic_write_json
from dast.config import DASTConfig, assert_split_allowed
from dast.data import ReasoningRecord
from dast.model import DASTModel
from dast.tokenization import apply_chat_template_ids

_FINAL_PATTERNS = (
    re.compile(r"\\boxed\{([^{}]+)\}", re.IGNORECASE),
    re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
)


@dataclass(frozen=True)
class GenerationRecord:
    problem_id: str
    category: str
    answer_type: str
    expected: str
    extracted: str | None
    raw_response: str
    correct: bool
    extraction_route: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "category": self.category,
            "answer_type": self.answer_type,
            "expected": self.expected,
            "extracted": self.extracted,
            "raw_response": self.raw_response,
            "correct": self.correct,
            "extraction_route": self.extraction_route,
        }


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\s\.,;:]+|[\s\.,;:]+$", "", value)
    return value


def _numeric_value(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*/\s*\d+)?", cleaned)
    if not match:
        return None
    token = match.group(0).replace(" ", "")
    try:
        if "/" in token:
            return float(Fraction(token))
        return float(token)
    except (ValueError, ZeroDivisionError):
        return None


def answers_equal(expected: str, predicted: str | None, answer_type: str) -> bool:
    if predicted is None:
        return False
    if answer_type in {"integer", "float", "quantity", "percentage", "fraction", "number"}:
        expected_number = _numeric_value(expected)
        predicted_number = _numeric_value(predicted)
        if expected_number is not None and predicted_number is not None:
            tolerance = max(1e-6, abs(expected_number) * 1e-5)
            return math.isclose(expected_number, predicted_number, rel_tol=1e-5, abs_tol=tolerance)
    return normalize_text(expected) == normalize_text(predicted)


def extract_answer(response: str) -> tuple[str | None, str]:
    for index, pattern in enumerate(_FINAL_PATTERNS):
        matches = list(pattern.finditer(response))
        if matches:
            return matches[-1].group(1).strip(), "boxed" if index == 0 else "labelled"
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        return lines[-1], "last_nonempty_line"
    return None, "empty"


def _decode_generated(tokenizer: Any, generated: torch.Tensor, prompt_lengths: Sequence[int]) -> list[str]:
    responses: list[str] = []
    for row, prompt_length in zip(generated, prompt_lengths):
        continuation = row[prompt_length:]
        responses.append(tokenizer.decode(continuation, skip_special_tokens=True))
    return responses


@torch.no_grad()
def evaluate_generation(
    model: DASTModel,
    tokenizer: Any,
    records: Sequence[ReasoningRecord],
    config: DASTConfig,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    if model.feedback_enabled:
        raise NotImplementedError(
            "cached autoregressive generation with feedback is intentionally unqualified; "
            "use teacher-forced intervention tests until a no-cache generation path is validated"
        )
    if records:
        assert_split_allowed(config, records[0].split)
    device = next(model.parameters()).device
    model.eval()
    rows: list[GenerationRecord] = []
    batch_size = config.evaluation.generation_batch_size

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        token_rows = []
        for record in chunk:
            messages = [
                {"role": "system", "content": config.data.system_prompt},
                {"role": "user", "content": record.problem},
            ]
            token_rows.append(apply_chat_template_ids(tokenizer, messages, add_generation_prompt=True))
        width = max(len(row) for row in token_rows)
        pad_id = int(tokenizer.pad_token_id)
        input_ids = torch.tensor([row + [pad_id] * (width - len(row)) for row in token_rows], device=device)
        attention_mask = input_ids.ne(pad_id).long()
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": config.evaluation.max_new_tokens,
            "do_sample": config.evaluation.temperature > 0,
            "pad_token_id": pad_id,
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if config.evaluation.temperature > 0:
            generation_kwargs.update(
                temperature=config.evaluation.temperature,
                top_p=config.evaluation.top_p,
            )
        generated = model.base_model.generate(**generation_kwargs)
        responses = _decode_generated(tokenizer, generated, [width] * len(chunk))
        for record, response in zip(chunk, responses):
            extracted, route = extract_answer(response)
            rows.append(
                GenerationRecord(
                    problem_id=record.problem_id,
                    category=record.category,
                    answer_type=record.answer_type,
                    expected=record.answer,
                    extracted=extracted,
                    raw_response=response,
                    correct=answers_equal(record.answer, extracted, record.answer_type),
                    extraction_route=route,
                )
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "generation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    accuracy = sum(row.correct for row in rows) / max(1, len(rows))
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_category.setdefault(row.category, {"correct": 0, "total": 0})
        bucket["correct"] += int(row.correct)
        bucket["total"] += 1
    summary = {
        "records": len(rows),
        "accuracy": accuracy,
        "by_category": {
            key: {**value, "accuracy": value["correct"] / value["total"]}
            for key, value in sorted(by_category.items())
        },
        "strict_denominator": True,
        "warning": "This lightweight scorer is deterministic but narrower than the canonical state-sufficiency evaluator.",
    }
    atomic_write_json(output_path / "generation_summary.json", summary)
    return summary
