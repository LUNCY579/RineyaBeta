"""The response-parsing contract.

The evaluator never assumes the model obeys the requested format. Every raw
response is retained, and extraction proceeds through an ordered set of routes.
The route that produced an answer is recorded on every record, which is what
makes "the model knew it but formatted it badly" separable from "the model was
wrong".

Two tracks
----------
**Strict** (primary): credit only for an unambiguous, contract-compliant answer
-- structured JSON, an explicit tag, a boxed value or an explicit label -- with
no conflicting candidates.

**Lenient** (diagnostic): also accepts prose-tail heuristics such as the final
line, and resolves conflicts by taking the last stated candidate.

A response that cannot be interpreted safely is *not* discarded. It stays in the
denominator and scores as a failure, with a failure type attached.

See docs/RESPONSE_PARSING_CONTRACT.md for the normative description.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------------------
# Route names (recorded verbatim in every evaluation record)
# --------------------------------------------------------------------------------------

ROUTE_JSON_EXACT = "json_exact"
ROUTE_JSON_FENCED = "json_fenced"
ROUTE_JSON_EMBEDDED = "json_embedded"
ROUTE_JSON_REPAIRED = "json_repaired"
ROUTE_TAG = "tag_final_answer"
ROUTE_BOXED = "boxed_latex"
ROUTE_LABELED = "labeled_answer"
ROUTE_LAST_LINE = "last_line_fallback"
ROUTE_PLAIN_TEXT = "plain_text_step"
ROUTE_EMPTY = "empty_output"
ROUTE_REFUSAL = "refusal"
ROUTE_NONE = "no_answer_found"

STRICT_ROUTES = frozenset(
    {ROUTE_JSON_EXACT, ROUTE_JSON_FENCED, ROUTE_JSON_EMBEDDED, ROUTE_JSON_REPAIRED, ROUTE_TAG, ROUTE_BOXED,
     ROUTE_LABELED, ROUTE_PLAIN_TEXT}
)

# Statuses
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_REFUSAL = "refusal"
STATUS_MALFORMED = "malformed"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_MISSING_FIELDS = "missing_fields"
STATUS_INVALID_FLAG = "invalid_completion_flag"
STATUS_NOT_FOUND = "not_found"
STATUS_BACKEND_FAILURE = "backend_failure"
STATUS_OVERLONG = "overlong"

REQUIRED_TRANSITION_KEYS = ("next_reasoning_step", "updated_compact_state", "done")

_REFUSAL_PATTERNS = (
    r"\bi (?:can(?:no|')t|cannot|am unable to|won'?t)\b",
    r"\bi'?m (?:sorry|unable|not able)\b",
    r"\bas an ai\b",
    r"\bi (?:do not|don'?t) (?:have|feel comfortable)\b",
    r"\bunable to (?:assist|help|comply)\b",
)

_TAG_PATTERNS = (
    r"<final[_ ]?answer>\s*(?P<value>.*?)\s*</final[_ ]?answer>",
    r"\[\s*final[_ ]?answer\s*\]\s*(?P<value>.*?)(?:\n|$)",
    r"<answer>\s*(?P<value>.*?)\s*</answer>",
)

# Real models decorate their answer lines: markdown headings, bullets, emphasis and
# emoji ("### ✅ Final Answer: **False**"), and they usually write "The final answer
# is: X" rather than a bare "Final answer: X". Patterns that miss those forms
# silently score a correct answer as a failure, which biases the whole comparison
# toward whichever arm happens to be more terse.
_DECORATION = r"[^\w\n]{0,16}"

# A separator is a colon, or a dash ONLY when whitespace-surrounded. A bare "-"
# must never count, or "The final answer is -6" loses its minus sign.
_SEP_OPTIONAL = r"\s*(?::|\s-\s)?\s*"
_SEP_REQUIRED = r"\s*(?::|\s-\s)\s*"

# Emphasis that closes after the label ("**Final Answer:** 42"). Only skipped when
# the run is followed by whitespace, so the opening "**" of "**Ivan**" is left in
# place for the symmetric-wrapper stripper to handle.
_EMPHASIS_TAIL = r"(?:[*_`]{1,2}(?=\s))?\s*"

_LABEL_PATTERNS = (
    rf"^{_DECORATION}(?:the\s+)?final\s+answer\s*(?:is)?{_SEP_OPTIONAL}{_EMPHASIS_TAIL}(?P<value>\S.*?)\s*$",
    rf"^{_DECORATION}answer\s*(?:is)?{_SEP_REQUIRED}{_EMPHASIS_TAIL}(?P<value>\S.*?)\s*$",
    rf"\b(?:so|thus|hence|therefore),?\s+the\s+(?:final\s+)?answer\s+is{_SEP_OPTIONAL}(?P<value>[^\n]+?)\s*[.]?\s*$",
    rf"\bthe\s+(?:final\s+)?answer\s+is{_SEP_OPTIONAL}(?P<value>[^\n]+?)\s*[.]?\s*$",
)

_MAX_REASONABLE_CHARS = 20_000

# Reasoning-scratch blocks emitted by "thinking" chat templates (Qwen3, R1-style).
# Their contents are working notes, not answers: mining them for candidates
# manufactures ambiguity and can extract a value the model later discarded.
_THINK_BLOCK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<(think|thinking|reasoning)>", re.IGNORECASE)


def strip_reasoning_blocks(text: str) -> tuple[str, dict[str, Any]]:
    """Remove <think>...</think> scratch, returning the remainder and a note record.

    An *unterminated* opener means the response ran out of budget inside its own
    reasoning and never reached an answer. Everything from that opener onward is
    discarded, which correctly leaves nothing to extract rather than pulling a
    half-formed guess out of the scratchpad.
    """
    cleaned, closed = _THINK_BLOCK_RE.subn("", text)
    unterminated = False
    match = _THINK_OPEN_RE.search(cleaned)
    if match:
        cleaned = cleaned[: match.start()]
        unterminated = True
    info = {"reasoning_blocks_removed": closed, "unterminated_reasoning_block": unterminated}
    return cleaned.strip(), info


@dataclass
class AnswerParse:
    """Result of extracting a final answer from an arbitrary response."""

    raw: str
    route: str = ROUTE_NONE
    status: str = STATUS_NOT_FOUND
    answer: str | None = None
    candidates: list[str] = field(default_factory=list)
    strict_ok: bool = False
    lenient_ok: bool = False
    ambiguous: bool = False
    lenient_answer: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "status": self.status,
            "answer": self.answer,
            "lenient_answer": self.lenient_answer,
            "candidate_count": len(self.candidates),
            "candidates": self.candidates[:8],
            "strict_ok": self.strict_ok,
            "lenient_ok": self.lenient_ok,
            "ambiguous": self.ambiguous,
            "notes": self.notes,
        }


@dataclass
class TransitionParse:
    """Result of parsing a stateful-transition response."""

    raw: str
    route: str = ROUTE_NONE
    status: str = STATUS_NOT_FOUND
    next_reasoning_step: str | None = None
    updated_compact_state: str | None = None
    done: bool | None = None
    final_answer: str | None = None
    payload: dict[str, Any] | None = None
    strict_ok: bool = False
    lenient_ok: bool = False
    state_coerced: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "status": self.status,
            "next_reasoning_step": self.next_reasoning_step,
            "updated_compact_state": self.updated_compact_state,
            "done": self.done,
            "final_answer": self.final_answer,
            "state_coerced": self.state_coerced,
            "strict_ok": self.strict_ok,
            "lenient_ok": self.lenient_ok,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------


def looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _REFUSAL_PATTERNS)


def strip_code_fences(text: str) -> list[str]:
    """Return the contents of every fenced block, in order."""
    return [m.group("body") for m in re.finditer(r"```(?:[A-Za-z0-9_+-]*)\s*\n(?P<body>.*?)```", text, re.DOTALL)]


def find_balanced_json_objects(text: str, *, limit: int = 8) -> list[str]:
    """Extract balanced ``{...}`` spans, respecting strings and escapes."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : index + 1])
                    if len(spans) >= limit:
                        break
    return spans


def repair_json(text: str) -> str | None:
    """Conservative repair of nearly-valid JSON.

    Only transformations that cannot change the meaning of already-valid JSON
    are applied. Anything more aggressive risks manufacturing an answer the
    model never produced, which would corrupt the primary metric.
    """
    candidate = text.strip()
    if not candidate:
        return None

    # Drop any prose before the first brace and after the last brace.
    first = candidate.find("{")
    last = candidate.rfind("}")
    if first > 0:
        candidate = candidate[first:]
    if last != -1 and last >= first:
        tail_cut = candidate.rfind("}")
        candidate = candidate[: tail_cut + 1]

    # Python/JS literals that appear when a model free-forms the object.
    candidate = re.sub(r"(?<![\"\w])True(?![\"\w])", "true", candidate)
    candidate = re.sub(r"(?<![\"\w])False(?![\"\w])", "false", candidate)
    candidate = re.sub(r"(?<![\"\w])(?:None|NULL|Null|undefined)(?![\"\w])", "null", candidate)

    # Single-quoted object when no double quotes are present at all.
    if '"' not in candidate and "'" in candidate:
        candidate = candidate.replace("'", '"')

    # A truncated object often ends mid-list with a dangling comma. Drop it before
    # balancing, otherwise the appended closer produces `... ,}` which is still invalid.
    candidate = re.sub(r",\s*$", "", candidate)

    # Unbalanced closers: append what is missing (never remove content).
    opens = candidate.count("{") - candidate.count("}")
    if opens > 0:
        candidate += "}" * opens
    brackets = candidate.count("[") - candidate.count("]")
    if brackets > 0:
        candidate += "]" * brackets

    # Trailing commas before a closer (including any closer just appended).
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


def _load_object(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def locate_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Try every JSON route in priority order. Returns (object, route)."""
    stripped = text.strip()
    obj = _load_object(stripped)
    if obj is not None:
        return obj, ROUTE_JSON_EXACT

    for block in strip_code_fences(text):
        obj = _load_object(block.strip())
        if obj is not None:
            return obj, ROUTE_JSON_FENCED
        repaired = repair_json(block)
        if repaired:
            obj = _load_object(repaired)
            if obj is not None:
                return obj, ROUTE_JSON_REPAIRED

    for span in find_balanced_json_objects(text):
        obj = _load_object(span)
        if obj is not None:
            return obj, ROUTE_JSON_EMBEDDED

    repaired = repair_json(text)
    if repaired:
        obj = _load_object(repaired)
        if obj is not None:
            return obj, ROUTE_JSON_REPAIRED
    return None, ROUTE_NONE


_WRAPPERS = (("**", "**"), ("__", "__"), ('"', '"'), ("'", "'"), ("*", "*"), ("_", "_"), ("`", "`"))


def _clean_value(value: str) -> str:
    """Undo answer decoration without altering the answer itself.

    Applied repeatedly because the layers nest: ``**Ivan**.`` needs the trailing
    period removed before the bold markers become symmetric and strippable.
    Only symmetric wrappers are removed, so a value that legitimately starts with
    an asterisk or quote is left alone.
    """
    value = value.strip()
    for _ in range(4):
        before = value
        value = value.strip().rstrip(".,;:!").strip()
        for opener, closer in _WRAPPERS:
            if len(value) > len(opener) + len(closer) and value.startswith(opener) and value.endswith(closer):
                value = value[len(opener) : -len(closer)].strip()
                break
        if value == before:
            break
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_boxed(text: str) -> list[str]:
    """Extract ``\\boxed{...}`` payloads with brace balancing."""
    out: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        index = match.end() - 1
        depth = 0
        for position in range(index, len(text)):
            char = text[position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[index + 1 : position])
                    break
    return out


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def extract_final_answer(
    text: str | None, *, expect_json: bool = False, strip_thinking: bool = True
) -> AnswerParse:
    """Extract a final answer from an arbitrary response."""
    if text is None:
        return AnswerParse(raw="", route=ROUTE_EMPTY, status=STATUS_BACKEND_FAILURE,
                           notes=["backend returned None"])
    parse = AnswerParse(raw=text)
    if not text.strip():
        parse.route, parse.status = ROUTE_EMPTY, STATUS_EMPTY
        parse.notes.append("empty output")
        return parse
    if strip_thinking:
        text, think_info = strip_reasoning_blocks(text)
        if think_info["reasoning_blocks_removed"]:
            parse.notes.append(f"removed {think_info['reasoning_blocks_removed']} reasoning block(s) before extraction")
        if think_info["unterminated_reasoning_block"]:
            parse.notes.append("response ended inside an unterminated reasoning block")
        if not text.strip():
            parse.route = ROUTE_EMPTY
            parse.status = STATUS_EMPTY if not think_info["unterminated_reasoning_block"] else STATUS_NOT_FOUND
            parse.notes.append("nothing remained outside the reasoning block")
            return parse
    if len(text) > _MAX_REASONABLE_CHARS:
        parse.notes.append(f"response exceeds {_MAX_REASONABLE_CHARS} characters; truncating for extraction")
        text = text[:_MAX_REASONABLE_CHARS]
        parse.status = STATUS_OVERLONG

    candidates: list[tuple[str, str]] = []  # (route, value)

    obj, json_route = locate_json_object(text)
    if obj is not None:
        for key in ("final_answer", "answer", "result"):
            if key in obj and obj[key] not in (None, ""):
                candidates.append((json_route, str(obj[key])))
                break

    for pattern in _TAG_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            value = _clean_value(match.group("value"))
            if value:
                candidates.append((ROUTE_TAG, value))

    for value in extract_boxed(text):
        cleaned = _clean_value(value)
        if cleaned:
            candidates.append((ROUTE_BOXED, cleaned))

    for pattern in _LABEL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            value = _clean_value(match.group("value"))
            if value:
                candidates.append((ROUTE_LABELED, value))

    if candidates:
        # Route priority, then position: an explicit tag beats a prose label.
        priority = {ROUTE_JSON_EXACT: 0, ROUTE_JSON_FENCED: 0, ROUTE_JSON_EMBEDDED: 0, ROUTE_JSON_REPAIRED: 1,
                    ROUTE_TAG: 2, ROUTE_BOXED: 3, ROUTE_LABELED: 4}
        candidates.sort(key=lambda item: priority.get(item[0], 9))
        best_route, best_value = candidates[0]
        distinct = {value for _, value in candidates}
        parse.route = best_route
        parse.answer = best_value
        parse.candidates = [value for _, value in candidates]
        parse.status = STATUS_OK
        parse.strict_ok = True
        parse.lenient_ok = True
        if len(distinct) > 1:
            parse.ambiguous = True
            parse.status = STATUS_AMBIGUOUS
            # Conflicting answers are never unambiguous: strict withholds credit,
            # lenient takes the last stated candidate.
            parse.strict_ok = False
            parse.answer = candidates[0][1]
            parse.notes.append(f"{len(distinct)} conflicting answer candidates")
            last_stated = max(distinct, key=lambda value: text.rfind(value))
            parse.notes.append(f"lenient selection: {last_stated!r}")
            parse.lenient_answer = last_stated
        return parse

    if looks_like_refusal(text):
        parse.route, parse.status = ROUTE_REFUSAL, STATUS_REFUSAL
        parse.notes.append("response matched a refusal pattern")
        return parse

    if expect_json:
        parse.route, parse.status = ROUTE_NONE, STATUS_MALFORMED
        parse.notes.append("JSON contract requested but no parseable object was found")

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        value = _clean_value(lines[-1])
        if value:
            parse.route = ROUTE_LAST_LINE
            parse.answer = value
            parse.candidates = [value]
            parse.lenient_ok = True
            parse.strict_ok = False
            # "malformed" is reserved for a broken *requested structure*. For a
            # free-text protocol nothing was malformed -- the response simply never
            # marked an answer, which is `not_found`. Conflating the two would make
            # the failure distribution unreadable for the FULL arm, whose contract
            # is prose.
            if parse.status == STATUS_NOT_FOUND and expect_json:
                parse.status = STATUS_MALFORMED
            parse.notes.append("fell back to the final non-empty line; strict track withholds credit")
            return parse

    parse.route, parse.status = ROUTE_NONE, STATUS_NOT_FOUND
    return parse


def lenient_answer_of(parse: AnswerParse) -> str | None:
    """The answer the lenient track should score."""
    return parse.lenient_answer or parse.answer


def parse_transition_response(text: str | None, *, strip_thinking: bool = True) -> TransitionParse:
    """Parse a stateful transition response against the JSON contract."""
    if text is None:
        return TransitionParse(raw="", route=ROUTE_EMPTY, status=STATUS_BACKEND_FAILURE,
                               notes=["backend returned None"])
    parse = TransitionParse(raw=text)
    if not text.strip():
        parse.route, parse.status = ROUTE_EMPTY, STATUS_EMPTY
        parse.notes.append("empty output")
        return parse
    if strip_thinking:
        text, think_info = strip_reasoning_blocks(text)
        if think_info["reasoning_blocks_removed"]:
            parse.notes.append(f"removed {think_info['reasoning_blocks_removed']} reasoning block(s) before parsing")
        if think_info["unterminated_reasoning_block"]:
            parse.notes.append("response ended inside an unterminated reasoning block")
        if not text.strip():
            parse.route, parse.status = ROUTE_EMPTY, STATUS_NOT_FOUND
            parse.notes.append("nothing remained outside the reasoning block")
            return parse
    if looks_like_refusal(text) and "{" not in text:
        parse.route, parse.status = ROUTE_REFUSAL, STATUS_REFUSAL
        return parse

    obj, route = locate_json_object(text)
    if obj is None:
        parse.route, parse.status = ROUTE_NONE, STATUS_MALFORMED
        parse.notes.append("no JSON object could be recovered")
        # Lenient salvage: treat the whole response as the reasoning step so a
        # rollout can continue and error propagation is still measurable.
        cleaned = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
        if cleaned:
            parse.next_reasoning_step = cleaned
            parse.lenient_ok = True
            parse.notes.append("lenient salvage: whole response treated as the next reasoning step")
        return parse

    parse.route = route
    parse.payload = obj
    missing = [key for key in REQUIRED_TRANSITION_KEYS if key not in obj]

    step = obj.get("next_reasoning_step")
    state = obj.get("updated_compact_state")
    parse.next_reasoning_step = str(step).strip() if isinstance(step, (str, int, float)) and str(step).strip() else None

    if isinstance(state, (str, int, float)) and str(state).strip():
        parse.updated_compact_state = str(state).strip()
    elif isinstance(state, (dict, list)) and state:
        # The contract asks for a string state. A structured state is a contract
        # violation and loses strict credit -- but it is still usable text, so a
        # rollout continues from it. Aborting here would turn the error-propagation
        # measurement into a formatting-compliance measurement.
        parse.updated_compact_state = json.dumps(state, ensure_ascii=False, sort_keys=True)
        parse.state_coerced = True
        parse.notes.append(f"updated_compact_state was a {type(state).__name__}; serialised to continue the rollout")
    else:
        parse.updated_compact_state = None

    raw_done = obj.get("done")
    if isinstance(raw_done, bool):
        parse.done = raw_done
    elif isinstance(raw_done, str) and raw_done.strip().lower() in ("true", "false", "yes", "no"):
        parse.done = raw_done.strip().lower() in ("true", "yes")
        parse.notes.append(f"coerced non-boolean done={raw_done!r}")
        parse.status = STATUS_INVALID_FLAG
    elif raw_done is None and "done" in obj:
        parse.notes.append("done was null")
        parse.status = STATUS_INVALID_FLAG
    elif "done" in obj:
        parse.notes.append(f"invalid done value {raw_done!r}")
        parse.status = STATUS_INVALID_FLAG

    final_answer = obj.get("final_answer")
    if final_answer is not None and str(final_answer).strip():
        parse.final_answer = str(final_answer).strip()

    if missing:
        parse.notes.append(f"missing required keys: {missing}")
        if parse.status == STATUS_NOT_FOUND:
            parse.status = STATUS_MISSING_FIELDS
        parse.lenient_ok = parse.next_reasoning_step is not None
        return parse

    contract_ok = (
        parse.next_reasoning_step is not None
        and parse.updated_compact_state is not None
        and not parse.state_coerced
        and isinstance(raw_done, bool)
    )
    if parse.state_coerced and parse.status == STATUS_NOT_FOUND:
        parse.status = STATUS_MISSING_FIELDS
    if contract_ok and parse.done and not parse.final_answer:
        parse.status = STATUS_MISSING_FIELDS
        parse.notes.append("done=true but final_answer is empty (premature completion)")
        parse.strict_ok = False
        parse.lenient_ok = True
        return parse
    if contract_ok and not parse.done and parse.final_answer:
        parse.notes.append("final_answer supplied while done=false")

    if contract_ok:
        parse.status = STATUS_OK if parse.status == STATUS_NOT_FOUND else parse.status
        parse.strict_ok = parse.status == STATUS_OK
        parse.lenient_ok = True
    else:
        if parse.status == STATUS_NOT_FOUND:
            parse.status = STATUS_MISSING_FIELDS
        parse.lenient_ok = parse.next_reasoning_step is not None
    return parse


def parse_next_step_response(text: str | None, *, strip_thinking: bool = True) -> TransitionParse:
    """Parse a next-step-only response.

    The ``split_next_step`` ablation is trained to emit a bare reasoning step, not
    JSON. Scoring it against the stateful JSON contract would score its *format*
    rather than its reasoning and make the ablation meaningless, so plain prose is
    the contract-compliant answer here.
    """
    if text is None:
        return TransitionParse(raw="", route=ROUTE_EMPTY, status=STATUS_BACKEND_FAILURE,
                               notes=["backend returned None"])
    parse = TransitionParse(raw=text)
    if not text.strip():
        parse.route, parse.status = ROUTE_EMPTY, STATUS_EMPTY
        return parse
    if strip_thinking:
        text, think_info = strip_reasoning_blocks(text)
        if think_info["reasoning_blocks_removed"]:
            parse.notes.append(f"removed {think_info['reasoning_blocks_removed']} reasoning block(s) before parsing")
        if not text.strip():
            parse.route, parse.status = ROUTE_EMPTY, STATUS_NOT_FOUND
            parse.notes.append("nothing remained outside the reasoning block")
            return parse
    if looks_like_refusal(text):
        parse.route, parse.status = ROUTE_REFUSAL, STATUS_REFUSAL
        return parse

    # A model that volunteers the stateful JSON anyway is honoured, not penalised.
    obj, route = locate_json_object(text)
    if obj is not None and "next_reasoning_step" in obj:
        structured = parse_transition_response(text)
        structured.notes.append("structured output accepted under the next-step-only contract")
        if structured.next_reasoning_step:
            structured.strict_ok = True
            structured.lenient_ok = True
            structured.status = STATUS_OK
        return structured

    cleaned = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    parse.route = ROUTE_PLAIN_TEXT
    parse.status = STATUS_OK
    parse.next_reasoning_step = cleaned
    parse.strict_ok = True
    parse.lenient_ok = True
    return parse
