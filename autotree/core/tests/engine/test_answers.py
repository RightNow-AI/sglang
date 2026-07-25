from autotree_core.engine.answers import extract_final_answer


def test_extract_final_answer_prefers_boxed_value() -> None:
    assert extract_final_answer(r"Work says 7, but the result is \boxed{42}.") == "42"


def test_extract_final_answer_uses_last_number() -> None:
    assert extract_final_answer("First 12, then the answer is 19.") == "19"


def test_extract_final_answer_normalizes_commas_and_trailing_period() -> None:
    assert extract_final_answer("Final answer: 1,234.") == "1234"


def test_extract_final_answer_returns_none_without_a_number() -> None:
    assert extract_final_answer("No numeric answer is available.") is None
