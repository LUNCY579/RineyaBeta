"""The formal failure taxonomy.

Every scored item that is not correct receives exactly one failure type. The
taxonomy separates *format* failures from *reasoning* failures, because the
central claim of the experiment ("compact states are sufficient") is about
reasoning, and a formatting regression in one arm would otherwise masquerade as
a capability difference.

See docs/FAILURE_TAXONOMY.md for the normative definitions.
"""

from __future__ import annotations

# -- format / protocol failures -------------------------------------------------------
EMPTY_OUTPUT = "empty_output"
REFUSAL = "refusal"
MALFORMED_STRUCTURE = "malformed_structure"
MISSING_FIELDS = "missing_fields"
INVALID_COMPLETION_FLAG = "invalid_completion_flag"
AMBIGUOUS_ANSWER = "ambiguous_answer"
NO_ANSWER_FOUND = "no_answer_found"
OVERLONG_OUTPUT = "overlong_output"
BACKEND_FAILURE = "backend_failure"

# -- rollout control failures ---------------------------------------------------------
NONTERMINATION = "nontermination"
LOOP = "loop"
STAGNATION = "stagnation"
PREMATURE_COMPLETION = "premature_completion"

# -- reasoning failures ---------------------------------------------------------------
WRONG_ANSWER = "wrong_answer"
WRONG_TRANSITION = "wrong_transition"
STATE_DRIFT = "state_drift"

CORRECT = "correct"

FORMAT_FAILURES = frozenset(
    {
        EMPTY_OUTPUT,
        REFUSAL,
        MALFORMED_STRUCTURE,
        MISSING_FIELDS,
        INVALID_COMPLETION_FLAG,
        AMBIGUOUS_ANSWER,
        NO_ANSWER_FOUND,
        OVERLONG_OUTPUT,
        BACKEND_FAILURE,
    }
)

CONTROL_FAILURES = frozenset({NONTERMINATION, LOOP, STAGNATION, PREMATURE_COMPLETION})

REASONING_FAILURES = frozenset({WRONG_ANSWER, WRONG_TRANSITION, STATE_DRIFT})

ALL_FAILURES = FORMAT_FAILURES | CONTROL_FAILURES | REASONING_FAILURES | {CORRECT}


_STATUS_TO_FAILURE = {
    "empty": EMPTY_OUTPUT,
    "refusal": REFUSAL,
    "malformed": MALFORMED_STRUCTURE,
    "missing_fields": MISSING_FIELDS,
    "invalid_completion_flag": INVALID_COMPLETION_FLAG,
    "ambiguous": AMBIGUOUS_ANSWER,
    "not_found": NO_ANSWER_FOUND,
    "overlong": OVERLONG_OUTPUT,
    "backend_failure": BACKEND_FAILURE,
}


def classify_from_status(status: str, *, correct: bool, default: str = WRONG_ANSWER) -> str:
    """Map a parse status plus a correctness verdict onto one failure type."""
    if correct:
        return CORRECT
    return _STATUS_TO_FAILURE.get(status, default)


def is_format_failure(failure_type: str) -> bool:
    return failure_type in FORMAT_FAILURES
