from __future__ import annotations

import pytest

from rineya_lab.config import ScoringConfig
from rineya_lab.evaluation.scoring import (
    DEFAULT_ANSWER_TYPE_SCORERS,
    parse_quantity,
    resolve_scorer,
    score_answer,
    states_equivalent,
)

CFG = ScoringConfig()


@pytest.mark.parametrize(
    "prediction,gold,answer_type",
    [
        ("568", "568", "integer"),
        ("1,024", "1024", "integer"),
        ("$2853.52", "$2853.52", "money"),
        ("2853.52 dollars", "2853.52 dollars", "money"),
        ("129 N", "129 N", "measurement"),
        ("246 bolts", "246 bolt", "quantity"),
        ("5/32", "5/32", "probability"),
        ("10/20", "1/2", "fraction"),
        ("True", "True", "boolean"),
        ("yes", "True", "boolean"),
        ("x = 6", "x = 6", "equation"),
        ("x>=36", "x ≥ 36", "inequality"),
        ("{2, 10, 12}", "{12, 2, 10}", "set"),
        ("Eli, Kira, Hana", "Eli, Kira, Hana", "ordering"),
        ("09:44", "09:44", "time"),
        ("9:44 am", "09:44", "time"),
        ("A=OFF, B=ON", "a=off, b=on", "state"),
        ("35ILIM!", "35ILIM!", "string"),
        ("brushes, doors, skies", "brushes, doors, skies", "text_transformation"),
    ],
)
def test_equivalent_answers_score_correct(prediction, gold, answer_type):
    result = score_answer(prediction, gold, answer_type=answer_type, cfg=CFG)
    assert result.correct, f"{prediction!r} vs {gold!r} ({answer_type}) -> {result.as_dict()}"


@pytest.mark.parametrize(
    "prediction,gold,answer_type",
    [
        ("569", "568", "integer"),
        ("$2853.53", "$2853.52", "money"),
        ("129 J", "129 N", "measurement"),
        ("5/33", "5/32", "probability"),
        ("False", "True", "boolean"),
        ("x = 7", "x = 6", "equation"),
        ("x <= 36", "x >= 36", "inequality"),
        ("{2, 10}", "{2, 10, 12}", "set"),
        ("Kira, Eli, Hana", "Eli, Kira, Hana", "ordering"),
        ("09:45", "09:44", "time"),
        ("35ilim!", "35ILIM!", "string"),
    ],
)
def test_different_answers_score_incorrect(prediction, gold, answer_type):
    assert not score_answer(prediction, gold, answer_type=answer_type, cfg=CFG).correct


def test_empty_prediction_is_incorrect_not_an_error():
    result = score_answer("", "568", answer_type="integer", cfg=CFG)
    assert not result.correct and result.scorer == "none"


def test_none_prediction_is_incorrect():
    assert not score_answer(None, "568", answer_type="integer", cfg=CFG).correct


def test_case_sensitive_string_type():
    assert not score_answer("abc", "ABC", answer_type="string", cfg=CFG).correct
    assert score_answer("abc", "ABC", answer_type="state", cfg=CFG).correct


def test_unit_synonyms_are_configurable():
    cfg = ScoringConfig(unit_synonyms={"litres": "liters"})
    assert score_answer("5 litres", "5 liters", answer_type="measurement", cfg=cfg).correct


def test_numeric_tolerance_is_configurable():
    tight = ScoringConfig(numeric_tolerance_abs=0.0, numeric_tolerance_rel=0.0)
    loose = ScoringConfig(numeric_tolerance_abs=0.01, numeric_tolerance_rel=0.0)
    assert not score_answer("3.14", "3.145", answer_type="decimal", cfg=tight).correct
    assert score_answer("3.14", "3.145", answer_type="decimal", cfg=loose).correct


def test_regex_scorer_from_config():
    cfg = ScoringConfig(answer_type_scorers={"code": "regex:OUTPUT=(\\S+)"})
    assert score_answer("OUTPUT=42", "42", answer_type="code", cfg=cfg).correct
    assert not score_answer("nothing here", "42", answer_type="code", cfg=cfg).correct


def test_json_scorer_ignores_key_order():
    assert score_answer('{"b":2,"a":1}', '{"a":1,"b":2}', answer_type="doc", cfg=ScoringConfig(
        answer_type_scorers={"doc": "json"})).correct


def test_symbolic_scorer_handles_equivalent_expressions():
    result = score_answer("2*x + 2", "2*(x + 1)", answer_type="expr", cfg=ScoringConfig(
        answer_type_scorers={"expr": "symbolic"}))
    assert result.correct


def test_scorer_records_its_own_name():
    result = score_answer("568", "568", answer_type="integer", cfg=CFG)
    assert any(note.startswith("scorer_selected=") for note in result.notes)


def test_resolve_scorer_defaults_to_auto_for_unknown_types():
    _, name = resolve_scorer("something_new", CFG)
    assert name == "auto"


def test_quantity_parser():
    assert parse_quantity("$2853.52") == (2853.52, "$")
    assert parse_quantity("5 km/h") == (5.0, "km/h")
    assert parse_quantity("not a quantity") is None


def test_state_equivalence_tolerates_number_formatting():
    assert states_equivalent("Sum: 3.0", "Sum: 3", CFG)
    assert not states_equivalent("Sum: 3", "Sum: 4", CFG)
    assert not states_equivalent(None, "Sum: 3", CFG)


def test_every_dataset_answer_type_has_a_registered_scorer():
    for answer_type, scorer_name in DEFAULT_ANSWER_TYPE_SCORERS.items():
        scorer, resolved = resolve_scorer(answer_type, CFG)
        assert callable(scorer), answer_type
        assert resolved == scorer_name


def test_scorer_never_raises_on_hostile_input():
    for bad in ["\x00\x01", "{" * 500, "∞", "-", "1/0"]:
        result = score_answer(bad, "568", answer_type="integer", cfg=CFG)
        assert result.correct is False
