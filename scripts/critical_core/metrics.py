#!/usr/bin/env python3
"""Model-agnostic collapse metrics and CRCC classification.

No single heuristic is allowed to define a "monster" layer.  Classification
combines teacher-forced reference loss with several generation-collapse signals,
and preserves every component and reason in the output manifest.
"""

from __future__ import annotations

import math
import statistics
import unicodedata
from collections import Counter
from typing import Any, Mapping, Sequence

from .common import ContractError


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _longest_equal_run(values: Sequence[Any]) -> int:
    best = current = 0
    previous: Any = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        best = max(best, current)
    return best


def _ngram_repeat_fraction(values: Sequence[int], n: int) -> float:
    total = len(values) - n + 1
    if total <= 1:
        return 0.0
    grams = [tuple(values[index : index + n]) for index in range(total)]
    return 1.0 - len(set(grams)) / total


def completion_metrics(
    text: str,
    token_ids: Sequence[int],
    *,
    ended_with_eos: bool,
    hit_max_new_tokens: bool,
) -> dict[str, Any]:
    characters = list(text)
    visible_characters = [char for char in characters if not char.isspace()]
    weird = 0
    punctuation = 0
    replacement = text.count("\ufffd")
    for char in characters:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            punctuation += 1
        if category in {"Cc", "Cs", "Co", "Cn"} and char not in {"\n", "\r", "\t"}:
            weird += 1

    token_count = len(token_ids)
    token_counter = Counter(int(token) for token in token_ids)
    dominant_token_fraction = (
        max(token_counter.values()) / token_count if token_count >= 8 else 0.0
    )
    unique_token_fraction = len(token_counter) / token_count if token_count else 0.0
    token_run_fraction = (
        _longest_equal_run([int(token) for token in token_ids]) / token_count
        if token_count >= 4
        else 0.0
    )
    character_run_fraction = (
        _longest_equal_run(visible_characters) / len(visible_characters)
        if len(visible_characters) >= 8
        else 0.0
    )
    denominator = max(len(characters), 1)
    weird_unicode_rate = (weird + replacement) / denominator
    punctuation_symbol_rate = punctuation / denominator
    repeated_bigram_fraction = _ngram_repeat_fraction(
        [int(token) for token in token_ids], 2
    )
    repeated_trigram_fraction = _ngram_repeat_fraction(
        [int(token) for token in token_ids], 3
    )

    empty = not text.strip()
    components = {
        "empty": 1.0 if empty else 0.0,
        "weird_unicode": _clip(weird_unicode_rate / 0.03),
        "dominant_token": _clip((dominant_token_fraction - 0.35) / 0.45),
        "token_run": _clip((token_run_fraction - 0.20) / 0.55),
        "character_run": _clip((character_run_fraction - 0.20) / 0.55),
        "repeated_bigram": _clip((repeated_bigram_fraction - 0.35) / 0.60),
        "no_eos_at_limit": 1.0 if hit_max_new_tokens and not ended_with_eos else 0.0,
    }
    collapse_score = (
        0.30 * components["empty"]
        + 0.15 * components["weird_unicode"]
        + 0.15 * components["dominant_token"]
        + 0.10 * components["token_run"]
        + 0.10 * components["character_run"]
        + 0.10 * components["repeated_bigram"]
        + 0.10 * components["no_eos_at_limit"]
    )

    return {
        "character_count": len(characters),
        "visible_character_count": len(visible_characters),
        "token_count": token_count,
        "empty": empty,
        "ended_with_eos": bool(ended_with_eos),
        "hit_max_new_tokens": bool(hit_max_new_tokens),
        "weird_unicode_rate": weird_unicode_rate,
        "replacement_character_count": replacement,
        "punctuation_symbol_rate": punctuation_symbol_rate,
        "dominant_token_fraction": dominant_token_fraction,
        "unique_token_fraction": unique_token_fraction,
        "longest_token_run_fraction": token_run_fraction,
        "longest_character_run_fraction": character_run_fraction,
        "repeated_bigram_fraction": repeated_bigram_fraction,
        "repeated_trigram_fraction": repeated_trigram_fraction,
        "collapse_components": components,
        "collapse_score": collapse_score,
        "intelligible_by_mechanical_screen": bool(not empty and collapse_score < 0.50),
    }


def _summarize_flat(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw_losses = [case.get("reference_nll") for case in cases]
    finite_losses = [
        float(value)
        for value in raw_losses
        if value is not None and math.isfinite(float(value))
    ]
    completions = [case["completion_metrics"] for case in cases]
    collapse = [float(item["collapse_score"]) for item in completions]
    intelligible = [bool(item["intelligible_by_mechanical_screen"]) for item in completions]
    return {
        "case_count": len(cases),
        "mean_reference_nll": _safe_mean(finite_losses) if finite_losses else None,
        "median_reference_nll": (
            float(statistics.median(finite_losses)) if finite_losses else None
        ),
        "nonfinite_reference_nll_rate": 1.0 - len(finite_losses) / len(raw_losses),
        "mean_collapse_score": _safe_mean(collapse),
        "collapse_case_rate": _safe_mean([value >= 0.50 for value in collapse]),
        "intelligibility_rate": _safe_mean(intelligible),
        "empty_rate": _safe_mean([bool(item["empty"]) for item in completions]),
        "eos_rate": _safe_mean([bool(item["ended_with_eos"]) for item in completions]),
        "max_length_without_eos_rate": _safe_mean(
            [
                bool(item["hit_max_new_tokens"] and not item["ended_with_eos"])
                for item in completions
            ]
        ),
        "mean_generated_tokens": _safe_mean(
            [float(item["token_count"]) for item in completions]
        ),
        "mean_weird_unicode_rate": _safe_mean(
            [float(item["weird_unicode_rate"]) for item in completions]
        ),
        "mean_dominant_token_fraction": _safe_mean(
            [float(item["dominant_token_fraction"]) for item in completions]
        ),
        "mean_repeated_bigram_fraction": _safe_mean(
            [float(item["repeated_bigram_fraction"]) for item in completions]
        ),
    }


def summarize_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ContractError("cannot summarize an empty case list")
    summary = _summarize_flat(cases)
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        by_source.setdefault(str(case["source"]), []).append(case)
    summary["per_source"] = {
        label: _summarize_flat(rows) for label, rows in sorted(by_source.items())
    }
    return summary

def classify_layer(
    layer_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify a deletion as catastrophic while retaining the causal signals."""

    if baseline_summary.get("mean_reference_nll") is None:
        raise ContractError("intact baseline produced no finite reference NLL")
    baseline_nll = max(float(baseline_summary["mean_reference_nll"]), 1e-12)
    layer_nll = (
        float(layer_summary["mean_reference_nll"])
        if layer_summary.get("mean_reference_nll") is not None
        else math.inf
    )
    nll_ratio = layer_nll / baseline_nll if math.isfinite(layer_nll) else math.inf

    baseline_collapse = float(baseline_summary["collapse_case_rate"])
    layer_collapse = float(layer_summary["collapse_case_rate"])
    baseline_intelligibility = float(baseline_summary["intelligibility_rate"])
    layer_intelligibility = float(layer_summary["intelligibility_rate"])
    baseline_eos = float(baseline_summary["eos_rate"])
    layer_eos = float(layer_summary["eos_rate"])

    nll_component = (
        1.0
        if not math.isfinite(nll_ratio)
        else _clip(math.log2(max(nll_ratio, 1.0)) / 3.0)
    )
    collapse_component = _clip((layer_collapse - baseline_collapse) / 0.75)
    intelligibility_component = _clip(
        (baseline_intelligibility - layer_intelligibility) / 0.75
    )
    eos_component = _clip((baseline_eos - layer_eos) / 0.75)
    composite = (
        0.45 * nll_component
        + 0.25 * collapse_component
        + 0.20 * intelligibility_component
        + 0.10 * eos_component
    )

    score_threshold = float(thresholds.get("composite_score", 0.60))
    hard_nll_ratio = float(thresholds.get("hard_nll_ratio", 4.0))
    hard_min_collapse = float(thresholds.get("hard_min_collapse_rate", 0.25))
    hard_intelligibility_drop = float(thresholds.get("hard_intelligibility_drop", 0.60))
    hard_nonfinite_rate = float(thresholds.get("hard_nonfinite_nll_rate", 0.0))

    reasons: list[str] = []
    if composite >= score_threshold:
        reasons.append(f"composite>={score_threshold:g}")
    if nll_ratio >= hard_nll_ratio and layer_collapse >= hard_min_collapse:
        reasons.append(
            f"nll_ratio>={hard_nll_ratio:g} with collapse_rate>={hard_min_collapse:g}"
        )
    if baseline_intelligibility - layer_intelligibility >= hard_intelligibility_drop:
        reasons.append(f"intelligibility_drop>={hard_intelligibility_drop:g}")
    if float(layer_summary["nonfinite_reference_nll_rate"]) > hard_nonfinite_rate:
        reasons.append(f"nonfinite_nll_rate>{hard_nonfinite_rate:g}")

    return {
        "criticality_score": composite,
        "catastrophic": bool(reasons),
        "classification_reasons": reasons,
        "signals": {
            "reference_nll_ratio": nll_ratio if math.isfinite(nll_ratio) else None,
            "nll_component": nll_component,
            "collapse_rate_delta": layer_collapse - baseline_collapse,
            "collapse_component": collapse_component,
            "intelligibility_delta": layer_intelligibility - baseline_intelligibility,
            "intelligibility_component": intelligibility_component,
            "eos_rate_delta": layer_eos - baseline_eos,
            "eos_component": eos_component,
        },
    }
