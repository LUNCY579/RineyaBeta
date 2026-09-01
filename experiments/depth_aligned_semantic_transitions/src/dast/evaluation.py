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
    # "is" matters as much as ":" and "=". The corpus's own full_response ends
    # "N. Therefore, the answer is X.", so after even a short SFT the model adopts
    # that form -- and the colon-only pattern never matched it. Measured on the G1
    # pilot: 200/200 responses fell through to the last-line fallback and strict
    # accuracy read 0.0000 while true capability was ~0.78.
    re.compile(
        r"(?:final answer|answer)\s*(?:is|:|=)\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
)

# "5. Therefore, ..." otherwise contributes 5 as the prediction's first number,
# which then loses every numeric comparison. The trailing \s+ is essential: without
# it this also matches the integer part of a decimal, turning "14.71" into "71" and
# "94." into "". Applied ONLY to the last-line fallback -- a labelled or boxed
# capture is already just the answer, so stripping a leading number there destroys
# legitimate values.
_LEADING_ENUMERATION = re.compile(r"^\s*\d{1,3}\s*[.)]\s+")
_TRAILING_PUNCTUATION = re.compile(r"[\s.]+$")
# Models emit the answer in markdown emphasis (**94**), which survives
# normalize_text and breaks every non-numeric comparison. Deliberately excludes
# the underscore, which is legitimate answer content ("43UZOZ_end").
_EMPHASIS = re.compile(r"[*`]+")


def _clean_extracted(value: str, *, strip_enumeration: bool = False) -> str:
    value = value.strip()
    if strip_enumeration:
        value = _LEADING_ENUMERATION.sub("", value)
    value = _EMPHASIS.sub("", value)
    return _TRAILING_PUNCTUATION.sub("", value).strip()


@dataclass(frozen=True)
class GenerationRecord:
    problem_id: str
    category: str
    # The bootstrap cluster unit. SSOT section 10 promises "problem-cluster
    # bootstrap intervals", and problems sharing a generator share a template, so
    # they are not independent. Without this field on the prediction record,
    # compare.py could only resample problems i.i.d., which is anti-conservative.
    generator: str
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
            "generator": self.generator,
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


# Numeric tolerance previously covered only six answer_type values, i.e. 30.0% of
# the 2,400-problem DEV split. The remaining 70% fell through to exact normalised
# string equality, which scored gold "129 N" against extracted "129" as WRONG even
# though `measurement` alone is 13.3% of DEV. Measured extraction gap from this and
# the set/ordering cases: 0.21 accuracy, stable across generation caps.
_NUMERIC_TYPES = frozenset({
    "integer", "float", "quantity", "percentage", "fraction", "number",
    "measurement", "decimal", "money", "probability", "rate", "statistic",
    "percentage_change", "time",
})
# Answers that are collections: order-insensitive for sets, order-SENSITIVE for
# orderings, so they cannot share one comparison.
#
# `state` belongs with the sets. Its gold form is a bag of key=value bindings, e.g.
# "A=ON, B=OFF, C=OFF, D=OFF", where order carries no meaning. Found by inspecting a
# real discordant pair: a model answered "D=OFF, B=OFF, C=OFF, A=ON" — identical
# content — and exact string equality scored it wrong. Measured effect on the
# escalation run was 1 problem in 800 (0.0013 accuracy), so it does not explain any
# arm difference, but it is a systematic type-level gap that would recur at larger n
# and in `state`-heavy strata.
#
# `ordering` deliberately stays order-SENSITIVE: its gold form is a ranking, e.g.
# "Eli, Kira, Hana, Mina", where a permutation is a different answer.
_SET_TYPES = frozenset({"set", "state"})
_SEQUENCE_TYPES = frozenset({"ordering"})
_BOOLEAN_TRUE = frozenset({"true", "yes", "t", "1"})
_BOOLEAN_FALSE = frozenset({"false", "no", "f", "0"})


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*/\s*\d+)?")


def _all_numbers(value: str) -> list[float]:
    numbers: list[float] = []
    for match in _NUMBER_PATTERN.finditer(value.replace(",", "")):
        token = match.group(0).replace(" ", "")
        try:
            numbers.append(float(Fraction(token)) if "/" in token else float(token))
        except (ValueError, ZeroDivisionError):
            continue
    return numbers


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=max(1e-6, abs(a) * 1e-5))


def _numbers_close(expected: str, predicted: str) -> bool | None:
    """Compare the numeric content of two answers.

    A single-number gold is compared against the prediction's first number, which
    is what gives `129 N` vs `129` and `$14.50` vs `14.50` their intended unit
    tolerance.

    A gold carrying SEVERAL numbers must match elementwise. Comparing only the
    first number silently credited wrong answers for every composite gold in this
    corpus: all 160 DEV `time` answers are `HH:MM`, so `16:20` vs `16:55` compared
    only the hour and scored correct, and the `mean X, range Y` form of `statistic`
    discarded the range. That inflates absolute accuracy and, worse, converts
    genuinely discordant pairs into ties, which is exactly the paired power the
    primary endpoint depends on.
    """

    expected_numbers = _all_numbers(expected)
    predicted_numbers = _all_numbers(predicted)
    if not expected_numbers or not predicted_numbers:
        return None
    if len(expected_numbers) == 1:
        return _close(expected_numbers[0], predicted_numbers[0])
    if len(predicted_numbers) != len(expected_numbers):
        return False
    return all(_close(a, b) for a, b in zip(expected_numbers, predicted_numbers))


def _collection_items(value: str) -> list[str]:
    inner = re.sub(r"^[\s\{\[\(]+|[\s\}\]\)]+$", "", value.strip())
    if not inner:
        return []
    return [normalize_text(part) for part in re.split(r"[,;]| and ", inner) if normalize_text(part)]


def _boolean_value(value: str) -> bool | None:
    token = normalize_text(value)
    if token in _BOOLEAN_TRUE:
        return True
    if token in _BOOLEAN_FALSE:
        return False
    return None


def answers_equal(expected: str, predicted: str | None, answer_type: str) -> bool:
    if predicted is None:
        return False
    if answer_type in _NUMERIC_TYPES:
        verdict = _numbers_close(expected, predicted)
        if verdict is not None:
            return verdict
    elif answer_type in _SET_TYPES:
        expected_items, predicted_items = _collection_items(expected), _collection_items(predicted)
        if expected_items and predicted_items:
            return sorted(expected_items) == sorted(predicted_items)
    elif answer_type in _SEQUENCE_TYPES:
        expected_items, predicted_items = _collection_items(expected), _collection_items(predicted)
        if expected_items and predicted_items:
            return expected_items == predicted_items
    elif answer_type == "boolean":
        expected_bool, predicted_bool = _boolean_value(expected), _boolean_value(predicted)
        if expected_bool is not None and predicted_bool is not None:
            return expected_bool == predicted_bool
    return normalize_text(expected) == normalize_text(predicted)


def extract_answer(response: str) -> tuple[str | None, str]:
    for index, pattern in enumerate(_FINAL_PATTERNS):
        matches = list(pattern.finditer(response))
        if matches:
            # The LAST match, because the answer is stated at the end.
            return _clean_extracted(matches[-1].group(1)), "boxed" if index == 0 else "labelled"
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        return _clean_extracted(lines[-1], strip_enumeration=True), "last_nonempty_line"
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
        # Decoder-only generation requires LEFT padding: with right padding the
        # model is asked to continue after pad positions and the decode window is
        # misaligned. transformers warns about this on every batched call.
        input_ids = torch.tensor(
            [[pad_id] * (width - len(row)) + row for row in token_rows], device=device
        )
        # Build the mask from the known pad counts rather than by value comparison,
        # which would also blank real tokens whenever pad_token_id == eos_token_id.
        attention_mask = torch.ones_like(input_ids)
        for index, row in enumerate(token_rows):
            attention_mask[index, : width - len(row)] = 0
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
                    generator=record.generator,
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
