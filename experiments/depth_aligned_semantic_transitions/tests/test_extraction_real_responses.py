"""Answer extraction pinned against REAL post-SFT model output.

The G1 pilot reported strict accuracy 0.0000 on 200 DEV problems for both arms
while the raw responses were visibly correct. Cause: short SFT taught the model the
corpus's own `full_response` format, which ends

    "5. Therefore, the answer is 246 bolts."

The shipped pattern required a colon or equals after "answer", so all 200/200
responses fell through to the last-line fallback, and the list enumeration `5.`
became the prediction's first number, losing every numeric comparison.

True capability on that run was ~0.78. So the endpoint was discarding 100% of the
model's correct answers, and every arm of a Phase-1 matrix would have tied at zero.

The cases below are verbatim final lines from `runs/g1_pilot_state_ordered`, one per
answer_type, so this file fails if extraction regresses on output a real fine-tuned
Qwen3-0.6B actually produced.
"""

from __future__ import annotations

import pytest

from dast.evaluation import answers_equal, extract_answer

# (answer_type, gold answer, verbatim final line of the real response)
REAL_RESPONSES = [
    ("quantity", "246 bolts", "5. Therefore, the answer is 246 bolts."),
    ("integer", "94", "5. Therefore, the answer is 94."),
    ("number", "792", "3. Therefore, the answer is 792."),
    ("measurement", "129 N", "3. Therefore, the answer is 129 N."),
    (
        "set",
        "{6, 7, 9, 10, 11, 14, 17, 18, 20, 21, 26}",
        "3. Therefore, the answer is {6, 7, 9, 10, 11, 14, 17, 18, 20, 21, 26}.",
    ),
    ("state", "A=OFF, B=OFF, C=ON, D=OFF", "5. Therefore, the answer is A=OFF, B=OFF, C=ON, D=OFF."),
    ("boolean", "False", "4. Therefore, the answer is False."),
    ("ordering", "Faye", "4. Therefore, the answer is Faye."),
    ("inequality", "x ≥ 36", "3. Therefore, the answer is x ≥ 36."),
    ("probability", "1/6", "4. Therefore, the answer is 1/6."),
    ("equation", "x = 40", "3. Therefore, the answer is x = 40."),
    ("time", "09:44", "6. Therefore, the answer is 09:44."),
]


@pytest.mark.parametrize("answer_type,gold,line", REAL_RESPONSES)
def test_real_post_sft_responses_are_scored_correct(answer_type, gold, line):
    extracted, route = extract_answer(line)
    assert route == "labelled", f"expected the labelled route, got {route!r} for {line!r}"
    assert answers_equal(gold, extracted, answer_type), (
        f"gold={gold!r} extracted={extracted!r} type={answer_type}"
    )


def test_answer_is_without_punctuation_is_recognised():
    """The single change that moved the pilot from 0.0000 to 0.7850."""

    assert extract_answer("the answer is 12")[0] == "12"
    assert extract_answer("Therefore, the answer is 246 bolts.")[0] == "246 bolts"
    # The older colon and equals forms must keep working.
    assert extract_answer("Final answer: 94")[0] == "94"
    assert extract_answer("answer = 3.5")[0] == "3.5"
    assert extract_answer(r"so \boxed{77}")[0] == "77"


def test_markdown_emphasis_is_stripped_but_underscores_are_not():
    """`**94**` broke every non-numeric comparison; stripping `_` broke answers
    that legitimately contain one."""

    assert extract_answer("Final answer: **94**")[0] == "94"
    assert extract_answer("the answer is `abc`")[0] == "abc"
    assert extract_answer("the answer is 43UZOZ_end")[0] == "43UZOZ_end"


def test_enumeration_stripping_never_eats_a_decimal():
    """`^\\d+[.)]` without a trailing space turns "14.71" into "71" and "94." into
    "". The labelled route must not strip enumeration at all."""

    assert extract_answer("the answer is 14.71")[0] == "14.71"
    assert extract_answer("the answer is 94.")[0] == "94"
    assert extract_answer("the answer is 111.34 liters")[0] == "111.34 liters"
    # In the fallback route, a genuine enumeration prefix IS stripped.
    value, route = extract_answer("premise line\n7) some trailing sentence")
    assert route == "last_nonempty_line"
    assert value == "some trailing sentence"


def test_fallback_still_returns_something_useful():
    value, route = extract_answer("no marker at all\njust a final line 42")
    assert route == "last_nonempty_line"
    assert value == "just a final line 42"
    assert extract_answer("")[1] == "empty"
