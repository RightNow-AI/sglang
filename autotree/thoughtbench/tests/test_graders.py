import pytest

from thoughtbench.graders import grade_exact, grade_numeric


def test_exact_match_strips_edges_but_preserves_case_and_internal_space() -> None:
    assert grade_exact("fixture-ok", "\nfixture-ok  ")
    assert not grade_exact("fixture-ok", "Fixture-ok")
    assert not grade_exact("fixture ok", "fixture  ok")


def test_numeric_uses_final_number_not_an_earlier_matching_number() -> None:
    assert not grade_numeric("42", "I first guessed 42, but the final answer is 41.")


@pytest.mark.parametrize(
    ("expected", "response"),
    [
        ("42", r"After checking, \\boxed{42}."),
        ("42", "Candidate 41. Final answer: 42."),
        ("1/2", r"The result is \\boxed{\\frac{1}{2}}."),
        ("1234.5", "Answer: 1,234.5000000001"),
        ("0.125", "The last value is 1.25e-1."),
        ("0.25", "Final answer: 25%."),
    ],
)
def test_numeric_accepts_supported_final_answer_conventions(
    expected: str, response: str
) -> None:
    assert grade_numeric(expected, response)


@pytest.mark.parametrize(
    "response",
    [
        "There is no numeric result.",
        "Final answer: 0/0",
        "42 is tempting, but final answer: -42",
        r"\\boxed{41} after mentioning 42",
    ],
)
def test_numeric_rejects_missing_invalid_or_wrong_final_values(response: str) -> None:
    assert not grade_numeric("42", response)
