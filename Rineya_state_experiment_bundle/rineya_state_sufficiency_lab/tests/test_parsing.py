from __future__ import annotations

import json

import pytest

from rineya_lab.evaluation.parsing import (
    ROUTE_BOXED,
    ROUTE_EMPTY,
    ROUTE_JSON_EXACT,
    ROUTE_JSON_FENCED,
    ROUTE_JSON_REPAIRED,
    ROUTE_LABELED,
    ROUTE_LAST_LINE,
    ROUTE_REFUSAL,
    ROUTE_TAG,
    extract_boxed,
    extract_final_answer,
    find_balanced_json_objects,
    lenient_answer_of,
    parse_transition_response,
    repair_json,
)

# -- final answer extraction ----------------------------------------------------------


def test_exact_json_answer():
    parse = extract_final_answer(json.dumps({"final_answer": "42"}))
    assert parse.route == ROUTE_JSON_EXACT and parse.answer == "42" and parse.strict_ok


def test_fenced_json_answer():
    parse = extract_final_answer('Here you go:\n```json\n{"final_answer": "7 cm"}\n```\nDone.')
    assert parse.route == ROUTE_JSON_FENCED and parse.answer == "7 cm"


def test_embedded_json_answer():
    parse = extract_final_answer('I computed it. {"final_answer": "x = 6"} That is all.')
    assert parse.answer == "x = 6" and parse.strict_ok


def test_tag_answer():
    parse = extract_final_answer("Reasoning...\n<final_answer>246 bolts</final_answer>")
    assert parse.route == ROUTE_TAG and parse.answer == "246 bolts"


def test_boxed_answer_with_nested_braces():
    parse = extract_final_answer(r"Therefore \boxed{\frac{1}{2}} follows.")
    assert parse.route == ROUTE_BOXED and parse.answer == r"\frac{1}{2}"


def test_labeled_answer():
    parse = extract_final_answer("1. add\n2. divide\nFinal answer: 568")
    assert parse.route == ROUTE_LABELED and parse.answer == "568" and parse.strict_ok


def test_last_line_is_lenient_only():
    parse = extract_final_answer("Some working here\n568")
    assert parse.route == ROUTE_LAST_LINE
    assert parse.lenient_ok and not parse.strict_ok


def test_empty_output():
    parse = extract_final_answer("   \n  ")
    assert parse.route == ROUTE_EMPTY and not parse.lenient_ok and not parse.strict_ok


def test_none_output_is_backend_failure():
    parse = extract_final_answer(None)
    assert parse.status == "backend_failure"


def test_refusal_detected():
    parse = extract_final_answer("I'm sorry, I cannot help with that.")
    assert parse.route == ROUTE_REFUSAL and not parse.strict_ok


def test_conflicting_answers_lose_strict_credit():
    parse = extract_final_answer("Final answer: 10\nActually the answer is 12")
    assert parse.ambiguous
    assert not parse.strict_ok
    assert parse.lenient_ok
    assert lenient_answer_of(parse) == "12"


def test_overlong_output_flagged():
    parse = extract_final_answer("x " * 20000 + "\nFinal answer: 3")
    assert parse.status in ("overlong", "ok", "ambiguous")


# -- JSON repair ----------------------------------------------------------------------


def test_repair_python_literals_and_trailing_comma():
    broken = '{"a": True, "b": None,'
    repaired = repair_json(broken)
    assert repaired is not None and json.loads(repaired) == {"a": True, "b": None}


def test_repair_strips_surrounding_prose():
    repaired = repair_json('Here it is: {"a": 1} hope that helps')
    assert json.loads(repaired) == {"a": 1}


def test_repair_refuses_hopeless_input():
    assert repair_json("no json at all") is None


def test_repair_does_not_change_valid_json():
    payload = '{"a": [1, 2], "b": {"c": "d"}}'
    assert json.loads(repair_json(payload)) == json.loads(payload)


def test_balanced_scanner_ignores_braces_in_strings():
    spans = find_balanced_json_objects('{"a": "} not a closer {"} tail')
    assert len(spans) == 1 and json.loads(spans[0]) == {"a": "} not a closer {"}


def test_extract_boxed_multiple():
    assert extract_boxed(r"\boxed{1} and \boxed{2}") == ["1", "2"]


# -- transition contract --------------------------------------------------------------


CONTRACT = {
    "next_reasoning_step": "Divide both sides by 7: x = 42 / 7 = 6.",
    "updated_compact_state": "Solved value: x = 6.",
    "done": False,
    "final_answer": None,
}


def test_transition_contract_ok():
    parse = parse_transition_response(json.dumps(CONTRACT))
    assert parse.strict_ok and parse.done is False
    assert parse.updated_compact_state == "Solved value: x = 6."


def test_transition_missing_field():
    payload = dict(CONTRACT)
    payload.pop("updated_compact_state")
    parse = parse_transition_response(json.dumps(payload))
    assert not parse.strict_ok and parse.status == "missing_fields"
    assert parse.lenient_ok  # the step is still recoverable


def test_transition_invalid_done_flag_is_coerced_and_flagged():
    payload = dict(CONTRACT, done="yes")
    parse = parse_transition_response(json.dumps(payload))
    assert parse.done is True
    assert parse.status == "invalid_completion_flag"
    assert not parse.strict_ok


def test_transition_premature_completion():
    payload = dict(CONTRACT, done=True, final_answer=None)
    parse = parse_transition_response(json.dumps(payload))
    assert parse.status == "missing_fields"
    assert not parse.strict_ok
    assert "premature completion" in " ".join(parse.notes)


def test_transition_repaired_route():
    broken = 'Here:\n{"next_reasoning_step": "Add 1.", "updated_compact_state": "Sum: 3.", "done": False, "final_answer": None,'
    parse = parse_transition_response(broken)
    assert parse.route == ROUTE_JSON_REPAIRED
    assert parse.updated_compact_state == "Sum: 3."
    assert parse.strict_ok


def test_transition_prose_salvage_is_lenient_only():
    parse = parse_transition_response("Just add three to the running total.")
    assert not parse.strict_ok and parse.lenient_ok
    assert parse.updated_compact_state is None


def test_transition_empty_and_refusal():
    assert parse_transition_response("").status == "empty"
    assert parse_transition_response("I cannot do that.").status == "refusal"


def test_plain_text_step_is_contract_compliant():
    from rineya_lab.evaluation.parsing import ROUTE_PLAIN_TEXT, parse_next_step_response

    parse = parse_next_step_response("Divide both sides by 7: x = 6.")
    assert parse.route == ROUTE_PLAIN_TEXT and parse.strict_ok
    assert parse.next_reasoning_step == "Divide both sides by 7: x = 6."
    assert parse.updated_compact_state is None


def test_plain_text_parser_still_accepts_volunteered_json():
    from rineya_lab.evaluation.parsing import parse_next_step_response

    parse = parse_next_step_response(json.dumps(CONTRACT))
    assert parse.strict_ok and parse.next_reasoning_step == CONTRACT["next_reasoning_step"]


def test_plain_text_parser_flags_empty_and_refusal():
    from rineya_lab.evaluation.parsing import parse_next_step_response

    assert parse_next_step_response("").status == "empty"
    assert parse_next_step_response("I'm sorry, I cannot help.").status == "refusal"
    assert not parse_next_step_response("").strict_ok


def test_prose_without_a_marker_is_not_called_malformed():
    """`malformed` must mean a broken requested structure, not 'prose had no label'."""
    parse = extract_final_answer("Some working here\n568", expect_json=False)
    assert parse.route == ROUTE_LAST_LINE
    assert parse.status == "not_found"
    assert parse.lenient_ok and not parse.strict_ok


def test_missing_json_when_json_was_requested_is_malformed():
    parse = extract_final_answer("Some working here\n568", expect_json=True)
    assert parse.status == "malformed"


# -- answer-label forms observed from real models --------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("The final answer is: **Ivan**.", "Ivan"),
    ("### ✅ Final Answer: **False**.", "False"),
    ("- **Final Answer:** `246 bolts`", "246 bolts"),
    ("**Answer:** 09:44", "09:44"),
    ("Final answer: 568", "568"),
    ("Answer: x = 6", "x = 6"),
    ("Answer - 42", "42"),
    ("So the final answer is 42.", "42"),
    ("Therefore, the answer is 3/4.", "3/4"),
    ("The final answer is **x ≥ 36**", "x ≥ 36"),
    ("Final answer: {2, 10, 12}", "{2, 10, 12}"),
    ("**Final Answer:** $2853.52", "$2853.52"),
    ("Answer: (3, 4)", "(3, 4)"),
    ("The final answer is: 1 cup of oats", "1 cup of oats"),
])
def test_decorated_answer_labels_are_strict_credit(raw, expected):
    parse = extract_final_answer(raw)
    assert parse.answer == expected, parse.as_dict()
    assert parse.strict_ok


@pytest.mark.parametrize("raw,expected", [
    ("The final answer is -6", "-6"),
    ("Final answer: -6", "-6"),
    ("Answer: -12.5", "-12.5"),
])
def test_a_bare_hyphen_is_never_treated_as_a_separator(raw, expected):
    """A dash only separates when whitespace-surrounded, or minus signs vanish."""
    assert extract_final_answer(raw).answer == expected


def test_label_extraction_does_not_fire_on_ordinary_prose():
    parse = extract_final_answer("I answered the question about the final results yesterday.")
    assert parse.route in (ROUTE_LAST_LINE, "no_answer_found")
    assert not parse.strict_ok


def test_structured_state_is_usable_but_loses_strict_credit():
    """A dict state violates the contract, but a rollout must still be able to continue.

    Aborting here would turn the error-propagation measurement into a
    formatting-compliance measurement.
    """
    payload = dict(CONTRACT, updated_compact_state={"Tara": "before Seth"})
    parse = parse_transition_response(json.dumps(payload))
    assert parse.state_coerced
    assert parse.updated_compact_state == '{"Tara": "before Seth"}'
    assert not parse.strict_ok
    assert parse.lenient_ok
    assert parse.status == "missing_fields"


def test_plain_string_state_keeps_strict_credit():
    parse = parse_transition_response(json.dumps(CONTRACT))
    assert not parse.state_coerced and parse.strict_ok


def test_empty_structured_state_is_not_coerced():
    payload = dict(CONTRACT, updated_compact_state={})
    parse = parse_transition_response(json.dumps(payload))
    assert parse.updated_compact_state is None and not parse.state_coerced
