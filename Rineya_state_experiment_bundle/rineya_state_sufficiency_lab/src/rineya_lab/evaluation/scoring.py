"""Typed deterministic scorers.

These synthetic tasks have verifiable canonical answers, so correctness is
decided by deterministic comparison, not by a model judge. A judge can be added
later as a supplementary qualitative diagnostic; it must not replace this.

Every score record carries the scorer that decided it and the normalised forms
that were compared, so any disputed item can be re-adjudicated by hand from the
stored record alone.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from rineya_lab.config import ScoringConfig

# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------

_UNICODE_MAP = {
    "−": "-",   # minus sign
    "–": "-",   # en dash
    "—": "-",   # em dash
    "×": "*",
    "÷": "/",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    " ": " ",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "²": "^2",
    "³": "^3",
}

_CURRENCY = "$€£¥₹"

_TRAILING_PUNCT = ".,;:!"


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for source, target in _UNICODE_MAP.items():
        text = text.replace(source, target)
    return text


def normalize_text(text: str, *, case_sensitive: bool = False, strip_trailing_punctuation: bool = True) -> str:
    value = normalize_unicode(str(text)).strip()
    if strip_trailing_punctuation:
        value = value.rstrip(_TRAILING_PUNCT).strip()
    value = re.sub(r"\s+", " ", value)
    if not case_sensitive:
        value = value.lower()
    return value


def strip_thousands(text: str) -> str:
    """Remove digit-group separators without touching decimal points."""
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_number(text: str) -> float | None:
    candidate = strip_thousands(normalize_unicode(str(text)).strip())
    candidate = candidate.lstrip(_CURRENCY).strip()
    candidate = candidate.replace("%", "").strip()
    try:
        return float(candidate)
    except ValueError:
        pass
    # A fraction such as "5/32"
    fraction = parse_fraction(candidate)
    if fraction is not None:
        return float(fraction)
    match = _NUMBER_RE.search(candidate)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def parse_fraction(text: str) -> Fraction | None:
    candidate = strip_thousands(normalize_unicode(str(text)).strip()).replace(" ", "")
    candidate = candidate.lstrip(_CURRENCY)
    match = re.fullmatch(r"([-+]?\d+)\s*/\s*([-+]?\d+)", candidate)
    if match:
        denominator = int(match.group(2))
        if denominator == 0:
            return None
        return Fraction(int(match.group(1)), denominator)
    # mixed number "1 3/4"
    mixed = re.fullmatch(r"([-+]?\d+)\s+(\d+)/(\d+)", normalize_unicode(str(text)).strip())
    if mixed:
        whole, numerator, denominator = (int(g) for g in mixed.groups())
        if denominator == 0:
            return None
        sign = -1 if whole < 0 else 1
        return Fraction(whole) + sign * Fraction(numerator, denominator)
    try:
        return Fraction(candidate)
    except (ValueError, ZeroDivisionError):
        return None


def numbers_equal(left: float, right: float, cfg: ScoringConfig) -> bool:
    if math.isnan(left) or math.isnan(right):
        return False
    if left == right:
        return True
    delta = abs(left - right)
    if delta <= cfg.numeric_tolerance_abs:
        return True
    scale = max(abs(left), abs(right))
    return scale > 0 and delta / scale <= cfg.numeric_tolerance_rel


_QUANTITY_RE = re.compile(
    r"^\s*(?P<currency>[" + _CURRENCY + r"])?\s*"
    r"(?P<number>[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?\d+\s*/\s*\d+)\s*"
    r"(?P<unit>.*?)\s*$"
)


def parse_quantity(text: str) -> tuple[float, str] | None:
    """Split "129 N" / "$2853.52" / "5 km/h" into (value, normalised unit)."""
    candidate = normalize_unicode(str(text)).strip().rstrip(_TRAILING_PUNCT).strip()
    match = _QUANTITY_RE.match(candidate)
    if not match:
        return None
    number = parse_number(match.group("number"))
    if number is None:
        return None
    unit = (match.group("unit") or "").strip().lower()
    if match.group("currency"):
        unit = (match.group("currency") + " " + unit).strip() if unit else match.group("currency")
    unit = re.sub(r"\s+", " ", unit)
    return number, unit


def _units_match(left: str, right: str, cfg: ScoringConfig) -> bool:
    def canon(unit: str) -> str:
        unit = cfg.unit_synonyms.get(unit, unit)
        unit = unit.strip().lower()
        # tolerate singular/plural of the trailing word only
        return re.sub(r"(\w)s$", r"\1", unit)

    return canon(left) == canon(right)


def split_collection(text: str) -> list[str]:
    value = normalize_unicode(str(text)).strip()
    value = value.strip("{}[]()")
    if not value.strip():
        return []
    parts = [part.strip() for part in re.split(r"[,;]", value)]
    if len(parts) == 1:
        parts = [part for part in re.split(r"\s+", parts[0]) if part]
    return [part for part in parts if part]


# --------------------------------------------------------------------------------------
# Score result
# --------------------------------------------------------------------------------------


@dataclass
class ScoreResult:
    correct: bool
    scorer: str
    normalized_prediction: str | None = None
    normalized_gold: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "scorer": self.scorer,
            "normalized_prediction": self.normalized_prediction,
            "normalized_gold": self.normalized_gold,
            "notes": self.notes,
        }


Scorer = Callable[[str, str, ScoringConfig], ScoreResult]


# --------------------------------------------------------------------------------------
# Individual scorers
# --------------------------------------------------------------------------------------


def score_integer(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    left, right = parse_number(prediction), parse_number(gold)
    if left is None or right is None:
        return ScoreResult(False, "integer", str(prediction), str(gold), ["non-numeric input"])
    correct = numbers_equal(left, right, cfg) and float(right).is_integer()
    if not float(right).is_integer():
        correct = numbers_equal(left, right, cfg)
    return ScoreResult(correct, "integer", str(left), str(right))


def score_float(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    left, right = parse_number(prediction), parse_number(gold)
    if left is None or right is None:
        return ScoreResult(False, "float", str(prediction), str(gold), ["non-numeric input"])
    return ScoreResult(numbers_equal(left, right, cfg), "float", repr(left), repr(right))


def score_scientific(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    result = score_float(prediction, gold, cfg)
    result.scorer = "scientific"
    return result


def score_fraction(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    left, right = parse_fraction(prediction), parse_fraction(gold)
    if left is not None and right is not None:
        return ScoreResult(left == right, "fraction", str(left), str(right))
    numeric = score_float(prediction, gold, cfg)
    numeric.scorer = "fraction"
    numeric.notes.append("compared numerically; at least one side was not a rational literal")
    return numeric


def score_percentage(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    def as_percent(text: str) -> float | None:
        raw = normalize_unicode(str(text)).strip()
        value = parse_number(raw)
        if value is None:
            return None
        # "0.42" and "42%" are only equal if the gold side declares the same form,
        # so no implicit rescaling is performed here.
        return value

    left, right = as_percent(prediction), as_percent(gold)
    if left is None or right is None:
        return ScoreResult(False, "percentage", str(prediction), str(gold), ["non-numeric input"])
    correct = abs(left - right) <= cfg.percent_tolerance_abs or numbers_equal(left, right, cfg)
    return ScoreResult(correct, "percentage", str(left), str(right))


_TRUE_WORDS = {"true", "yes", "y", "t", "1", "correct"}
_FALSE_WORDS = {"false", "no", "n", "f", "0", "incorrect"}


def score_boolean(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    def as_bool(text: str) -> bool | None:
        value = normalize_text(text).strip(".")
        if value in _TRUE_WORDS:
            return True
        if value in _FALSE_WORDS:
            return False
        return None

    left, right = as_bool(prediction), as_bool(gold)
    if right is None:
        return ScoreResult(False, "boolean", str(prediction), str(gold), ["gold is not boolean"])
    if left is None:
        return ScoreResult(False, "boolean", normalize_text(prediction), str(right), ["prediction is not boolean"])
    return ScoreResult(left == right, "boolean", str(left), str(right))


def score_text_exact(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    left = normalize_unicode(str(prediction)).strip()
    right = normalize_unicode(str(gold)).strip()
    if cfg.strip_trailing_punctuation:
        left, right = left.rstrip(_TRAILING_PUNCT), right.rstrip(_TRAILING_PUNCT)
    return ScoreResult(left == right, "text_exact", left, right)


def score_text_normalized(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    left = normalize_text(prediction, strip_trailing_punctuation=cfg.strip_trailing_punctuation)
    right = normalize_text(gold, strip_trailing_punctuation=cfg.strip_trailing_punctuation)
    return ScoreResult(left == right, "text_normalized", left, right)


_CHOICE_RE = re.compile(r"^\(?([A-Za-z])[).:]?$")


def score_multiple_choice(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    def letter(text: str) -> str | None:
        value = normalize_text(text).strip()
        match = _CHOICE_RE.match(value)
        if match:
            return match.group(1).upper()
        head = value.split()[0] if value.split() else ""
        match = _CHOICE_RE.match(head)
        return match.group(1).upper() if match else None

    left, right = letter(prediction), letter(gold)
    if right is None:
        return ScoreResult(False, "multiple_choice", str(prediction), str(gold), ["gold is not a choice letter"])
    return ScoreResult(left == right, "multiple_choice", str(left), str(right))


def score_ordered_collection(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    left = [normalize_text(item) for item in split_collection(prediction)]
    right = [normalize_text(item) for item in split_collection(gold)]
    return ScoreResult(left == right, "ordered_collection", json.dumps(left), json.dumps(right))


def score_unordered_collection(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    def canon(text: str) -> list[str]:
        items = [normalize_text(item) for item in split_collection(text)]
        # Numeric items sort numerically so {2, 10} == {10, 2} regardless of format.
        def key(item: str):
            value = parse_number(item)
            return (0, value, "") if value is not None else (1, 0.0, item)

        return sorted(items, key=key)

    left, right = canon(prediction), canon(gold)
    return ScoreResult(left == right, "unordered_collection", json.dumps(left), json.dumps(right))


def score_json(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    from rineya_lab.evaluation.parsing import locate_json_object

    left, _ = locate_json_object(str(prediction))
    try:
        right = json.loads(str(gold))
    except (json.JSONDecodeError, ValueError):
        right = None
    if left is None or right is None:
        return ScoreResult(False, "json", str(prediction)[:200], str(gold)[:200], ["unparseable JSON"])
    canon_left = json.dumps(left, sort_keys=True, separators=(",", ":"))
    canon_right = json.dumps(right, sort_keys=True, separators=(",", ":"))
    return ScoreResult(canon_left == canon_right, "json", canon_left, canon_right)


_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE)


def score_datetime(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    def canon(text: str) -> str | None:
        value = normalize_text(text)
        match = _TIME_RE.match(value)
        if not match:
            return None
        hour, minute, second, meridiem = match.groups()
        hour_i = int(hour) % 12 if meridiem else int(hour)
        if meridiem and meridiem.lower() == "pm":
            hour_i += 12
        suffix = value[match.end():].strip()
        base = f"{hour_i:02d}:{int(minute):02d}" + (f":{int(second):02d}" if second else "")
        return f"{base} {suffix}".strip()

    left, right = canon(prediction), canon(gold)
    if right is None:
        return score_text_normalized(prediction, gold, cfg)
    if left is None:
        return ScoreResult(False, "datetime", normalize_text(prediction), right, ["prediction is not a time"])
    return ScoreResult(left == right, "datetime", left, right)


def score_symbolic(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    """Symbolic equality via sympy when available, else normalised relation text."""
    left_raw = normalize_unicode(str(prediction)).strip().rstrip(_TRAILING_PUNCT)
    right_raw = normalize_unicode(str(gold)).strip().rstrip(_TRAILING_PUNCT)
    quick = score_relation(left_raw, right_raw, cfg)
    if quick.correct:
        quick.scorer = "symbolic"
        quick.notes.append("matched by normalised relation form")
        return quick
    if not cfg.symbolic_enabled:
        return ScoreResult(False, "symbolic", left_raw, right_raw, ["sympy disabled by config"])
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        result = score_text_normalized(left_raw, right_raw, cfg)
        result.scorer = "symbolic"
        result.notes.append("sympy not installed; fell back to normalised text")
        return result
    try:
        relations = ("<=", ">=", "!=", "<", ">", "=")
        left_rel = next((r for r in relations if r in left_raw), None)
        right_rel = next((r for r in relations if r in right_raw), None)
        if left_rel != right_rel:
            return ScoreResult(False, "symbolic", left_raw, right_raw, ["relation operators differ"])
        if left_rel:
            left_lhs, left_rhs = left_raw.split(left_rel, 1)
            right_lhs, right_rhs = right_raw.split(right_rel, 1)
            equal = sympy.simplify(
                (parse_expr(left_lhs) - parse_expr(left_rhs)) - (parse_expr(right_lhs) - parse_expr(right_rhs))
            ) == 0
        else:
            equal = sympy.simplify(parse_expr(left_raw) - parse_expr(right_raw)) == 0
        return ScoreResult(bool(equal), "symbolic", left_raw, right_raw)
    except Exception as exc:
        return ScoreResult(False, "symbolic", left_raw, right_raw, [f"sympy could not compare: {exc}"])


def score_relation(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    """Normalised comparison for answers like ``x = 6`` or ``x >= 36``."""
    def canon(text: str) -> str:
        value = normalize_unicode(str(text)).strip().rstrip(_TRAILING_PUNCT)
        value = strip_thousands(value)
        value = re.sub(r"\s*(<=|>=|!=|<|>|=)\s*", r"\1", value)
        return value.lower().replace(" ", "")

    left, right = canon(prediction), canon(gold)
    if left == right:
        return ScoreResult(True, "relation", left, right)
    # Compare the numeric right-hand sides when the operators agree.
    pattern = re.compile(r"^(?P<lhs>[^<>=!]*)(?P<op><=|>=|!=|<|>|=)(?P<rhs>.*)$")
    left_match, right_match = pattern.match(left), pattern.match(right)
    if left_match and right_match and left_match.group("op") == right_match.group("op"):
        if left_match.group("lhs") == right_match.group("lhs"):
            left_value = parse_number(left_match.group("rhs"))
            right_value = parse_number(right_match.group("rhs"))
            if left_value is not None and right_value is not None:
                return ScoreResult(numbers_equal(left_value, right_value, cfg), "relation", left, right)
    return ScoreResult(False, "relation", left, right)


def score_quantity(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    """Number plus unit, e.g. ``129 N``, ``$2853.52``, ``246 bolts``."""
    exact = score_text_normalized(prediction, gold, cfg)
    if exact.correct:
        exact.scorer = "quantity"
        return exact
    left, right = parse_quantity(prediction), parse_quantity(gold)
    if left is None or right is None:
        exact.scorer = "quantity"
        exact.notes.append("could not parse a number+unit on at least one side")
        return exact
    value_ok = numbers_equal(left[0], right[0], cfg)
    unit_ok = _units_match(left[1], right[1], cfg)
    notes: list[str] = []
    if value_ok and not unit_ok:
        notes.append(f"numeric values agree but units differ ({left[1]!r} vs {right[1]!r})")
    return ScoreResult(
        value_ok and unit_ok,
        "quantity",
        f"{left[0]} {left[1]}".strip(),
        f"{right[0]} {right[1]}".strip(),
        notes,
    )


def make_regex_scorer(pattern: str) -> Scorer:
    """Build a scorer that credits a prediction matching a configured regex."""
    compiled = re.compile(pattern)

    def _scorer(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
        value = normalize_unicode(str(prediction)).strip()
        match = compiled.search(value)
        extracted = (match.group(1) if match and match.groups() else match.group(0)) if match else None
        if extracted is None:
            return ScoreResult(False, f"regex[{pattern}]", value, str(gold), ["prediction did not match the pattern"])
        return ScoreResult(
            normalize_text(extracted) == normalize_text(gold),
            f"regex[{pattern}]",
            normalize_text(extracted),
            normalize_text(gold),
        )

    return _scorer


def score_auto(prediction: str, gold: str, cfg: ScoringConfig) -> ScoreResult:
    """Fallback chain used when no scorer is declared for an answer type."""
    attempts: list[ScoreResult] = []
    exact = score_text_normalized(prediction, gold, cfg)
    if exact.correct:
        exact.scorer = "auto:text_normalized"
        return exact
    attempts.append(exact)

    gold_text = normalize_text(gold)
    if gold_text in _TRUE_WORDS | _FALSE_WORDS:
        result = score_boolean(prediction, gold, cfg)
        result.scorer = "auto:boolean"
        return result
    if re.search(r"[<>=]", gold_text):
        result = score_relation(prediction, gold, cfg)
        if result.correct:
            result.scorer = "auto:relation"
            return result
        attempts.append(result)
    if gold_text.startswith("{") or "," in gold_text:
        unordered = score_unordered_collection(prediction, gold, cfg)
        ordered = score_ordered_collection(prediction, gold, cfg)
        chosen = ordered if gold_text.startswith("{") is False else unordered
        if chosen.correct:
            chosen.scorer = f"auto:{chosen.scorer}"
            return chosen
        attempts.append(chosen)
    if parse_fraction(gold) is not None and "/" in gold_text:
        result = score_fraction(prediction, gold, cfg)
        if result.correct:
            result.scorer = "auto:fraction"
            return result
        attempts.append(result)
    if parse_quantity(gold) is not None:
        result = score_quantity(prediction, gold, cfg)
        if result.correct:
            result.scorer = "auto:quantity"
            return result
        attempts.append(result)

    failed = attempts[-1]
    return ScoreResult(
        False,
        "auto",
        failed.normalized_prediction,
        failed.normalized_gold,
        ["no scorer in the auto chain accepted the prediction"],
    )


SCORERS: dict[str, Scorer] = {
    "integer": score_integer,
    "float": score_float,
    "scientific": score_scientific,
    "fraction": score_fraction,
    "percentage": score_percentage,
    "boolean": score_boolean,
    "text_exact": score_text_exact,
    "text_normalized": score_text_normalized,
    "multiple_choice": score_multiple_choice,
    "ordered_collection": score_ordered_collection,
    "unordered_collection": score_unordered_collection,
    "json": score_json,
    "datetime": score_datetime,
    "symbolic": score_symbolic,
    "relation": score_relation,
    "quantity": score_quantity,
    "auto": score_auto,
}

# Defaults for the 21 answer types present in the Rineya dataset. Overridable via
# ``evaluation.scoring.answer_type_scorers`` in any config.
DEFAULT_ANSWER_TYPE_SCORERS: dict[str, str] = {
    "integer": "integer",
    "number": "integer",
    "program_output": "auto",
    "table_result": "integer",
    "decimal": "quantity",
    "money": "quantity",
    "measurement": "quantity",
    "quantity": "quantity",
    "rate": "quantity",
    "statistic": "auto",
    "fraction": "fraction",
    "probability": "fraction",
    "boolean": "boolean",
    "equation": "relation",
    "inequality": "relation",
    "set": "auto",
    "ordering": "ordered_collection",
    "text_transformation": "ordered_collection",
    "string": "text_exact",
    "time": "datetime",
    "state": "text_normalized",
}


def resolve_scorer(answer_type: str | None, cfg: ScoringConfig) -> tuple[Scorer, str]:
    """Pick the scorer for an answer type, honouring config overrides."""
    name = None
    if answer_type:
        name = cfg.answer_type_scorers.get(answer_type) or DEFAULT_ANSWER_TYPE_SCORERS.get(answer_type)
    name = name or cfg.default_scorer or "auto"
    if name.startswith("regex:"):
        return make_regex_scorer(name.split(":", 1)[1]), name
    if name in cfg.regex_tasks:
        return make_regex_scorer(cfg.regex_tasks[name]), f"regex[{name}]"
    scorer = SCORERS.get(name)
    if scorer is None:
        raise KeyError(f"Unknown scorer {name!r}. Available: {sorted(SCORERS)}")
    return scorer, name


def score_answer(
    prediction: str | None,
    gold: str,
    *,
    answer_type: str | None = None,
    cfg: ScoringConfig | None = None,
) -> ScoreResult:
    """Score one predicted answer against the canonical answer."""
    cfg = cfg or ScoringConfig()
    if prediction is None or not str(prediction).strip():
        return ScoreResult(False, "none", None, str(gold), ["no prediction to score"])
    scorer, name = resolve_scorer(answer_type, cfg)
    try:
        result = scorer(str(prediction), str(gold), cfg)
    except Exception as exc:  # pragma: no cover - defensive: a scorer must never crash a run
        return ScoreResult(False, name, str(prediction)[:200], str(gold)[:200], [f"scorer raised {type(exc).__name__}: {exc}"])
    result.notes.append(f"scorer_selected={name}")
    return result


def states_equivalent(left: str | None, right: str | None, cfg: ScoringConfig | None = None) -> bool:
    """Compare two compact states for the first-error-depth analysis.

    Text equality after normalisation, with a numeric-tolerant fallback so a
    state differing only in decimal formatting is not counted as an error.
    """
    if left is None or right is None:
        return False
    cfg = cfg or ScoringConfig()
    if normalize_text(left) == normalize_text(right):
        return True
    left_numbers = _NUMBER_RE.findall(strip_thousands(normalize_unicode(left)))
    right_numbers = _NUMBER_RE.findall(strip_thousands(normalize_unicode(right)))
    if left_numbers and len(left_numbers) == len(right_numbers):
        words_left = re.sub(r"[-+]?[\d.,]+", "#", normalize_text(left))
        words_right = re.sub(r"[-+]?[\d.,]+", "#", normalize_text(right))
        if words_left == words_right:
            return all(
                numbers_equal(float(a), float(b), cfg) for a, b in zip(left_numbers, right_numbers)
            )
    return False
