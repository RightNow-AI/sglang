from __future__ import annotations

import pytest

from autotree_core.engine.consensus import ConsensusConfig, consensus_scores


def test_answer_groups_score_by_live_branch_fraction() -> None:
    scores = consensus_scores(
        {
            3: "The answer is 9.",
            1: "Final answer: 1,000.",
            2: r"Therefore, \boxed{1000}.",
        },
        ConsensusConfig(min_survivors=2),
    )

    assert list(scores) == [1, 2, 3]
    assert scores == pytest.approx({1: 2 / 3, 2: 2 / 3, 3: 1 / 3})


def test_lexical_fallback_uses_mean_trailing_token_overlap() -> None:
    scores = consensus_scores(
        {
            1: "alpha beta gamma",
            2: "alpha beta delta",
            3: "zeta eta theta",
        },
        ConsensusConfig(min_survivors=2, trailing_window=3),
    )

    assert scores == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 0.0})


def test_lexical_fallback_ignores_tokens_before_trailing_window() -> None:
    scores = consensus_scores(
        {
            1: "old path shared finish",
            2: "different route shared finish",
        },
        ConsensusConfig(min_survivors=1, trailing_window=2),
    )

    assert scores == pytest.approx({1: 1.0, 2: 1.0})


def test_min_survivors_uses_branch_id_to_break_equal_score_ties() -> None:
    scores = consensus_scores(
        {
            5: "Answer: 11.",
            2: "Answer: 7.",
            4: "Answer: 10.",
            1: "Answer: 7.",
            3: "Answer: 9.",
        },
        ConsensusConfig(min_survivors=3),
    )

    assert list(scores) == [1, 2, 3, 4, 5]
    assert scores == pytest.approx({1: 0.4, 2: 0.4, 3: 0.4, 4: 0.2, 5: 0.2})


def test_all_branches_agree_without_collapsing_survivors() -> None:
    scores = consensus_scores(
        {
            3: "Final answer: 7.",
            1: r"\boxed{7}",
            2: "The answer is 7.",
        },
        ConsensusConfig(min_survivors=2),
    )

    assert scores == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})


@pytest.mark.parametrize(
    "config, message",
    [
        (ConsensusConfig(min_survivors=0), "min_survivors"),
        (ConsensusConfig(trailing_window=0), "trailing_window"),
    ],
)
def test_consensus_config_rejects_invalid_values(
    config: ConsensusConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        consensus_scores({1: "work"}, config)
