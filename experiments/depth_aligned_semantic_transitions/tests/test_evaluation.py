from dast.evaluation import answers_equal, extract_answer


def test_answer_extraction_prefers_explicit_final_label():
    answer, route = extract_answer("Work: 2 + 3 = 5.\nFinal answer: 5")
    assert answer == "5"
    assert route == "labelled"


def test_numeric_scoring_accepts_equivalent_quantity_text():
    assert answers_equal("1,920 cells", "1920", "quantity")
    assert not answers_equal("1920 cells", "1919", "quantity")
