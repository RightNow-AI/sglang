import pytest

from thoughtbench.answers import extract_answer, grade_answer, majority_vote, normalize_answer


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("work\n#### 1,234.", "1234"),
        (r"first 4 then \\boxed{42}", "42"),
        ("candidate 8, final answer: -7", "-7"),
        ("There were 12 and then 18", "18"),
        ("no numeric answer", None),
    ],
)
def test_extract_answer_mirrors_engine_conventions(response, expected) -> None:
    assert extract_answer(response) == expected


def test_normalization_and_grading_are_deliberately_narrow() -> None:
    assert normalize_answer(" 1,234. ") == "1234"
    assert grade_answer("Answer: 1,234", "1234") == ("1234", True)
    assert grade_answer("Answer: 0.5", "1/2") == ("0.5", False)


def test_majority_vote_is_client_side_and_first_seen_breaks_ties() -> None:
    assert majority_vote(["Answer: 7", "Answer: 8", "#### 7"]) == "7"
    assert majority_vote(["Answer: 8", "Answer: 7"]) == "8"
    assert majority_vote(["none", "still none"]) is None
