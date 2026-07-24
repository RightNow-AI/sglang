"""Unit tests for confidence-weighted winner selection (selection.py)."""

import math
import os

from sglang.srt.tree import selection


def test_weighted_picks_confident_minority():
    # class "A": 3 branches at low confidence; class "B": 2 branches at high
    # confidence. Weighted mass of B can exceed A even though A has more votes.
    items = [
        ("A", -3.0),
        ("A", -3.0),
        ("A", -3.0),
        ("B", -0.1),
        ("B", -0.1),
    ]
    a_mass = 3 * math.exp(-3.0)
    b_mass = 2 * math.exp(-0.1)
    assert b_mass > a_mass  # sanity: the fixture is set up for B to win
    assert selection.weighted_winning_key(items, beta=1.0) == "B"


def test_weighted_matches_plurality_when_confidence_uniform():
    items = [("A", -1.0), ("A", -1.0), ("B", -1.0)]
    assert selection.weighted_winning_key(items, beta=1.0) == "A"


def test_weighted_falls_back_to_plurality_without_logprobs():
    # missing logprob on any voting branch -> plurality by count
    items = [("A", None), ("A", None), ("B", -0.01)]
    assert selection.weighted_winning_key(items) == "A"


def test_empty_and_none_keys_ignored():
    assert selection.weighted_winning_key([(None, -1.0), ("", -1.0)]) is None
    assert selection.weighted_winning_key([]) is None


def test_vote_mode_and_beta_env(monkeypatch=None):
    old_mode = os.environ.get("AUTOTREE_VOTE")
    old_beta = os.environ.get("AUTOTREE_VOTE_BETA")
    try:
        os.environ.pop("AUTOTREE_VOTE", None)
        assert selection.vote_mode() == "plurality"
        os.environ["AUTOTREE_VOTE"] = "weighted"
        assert selection.vote_mode() == "weighted"
        os.environ["AUTOTREE_VOTE_BETA"] = "2.5"
        assert selection.vote_beta() == 2.5
        os.environ["AUTOTREE_VOTE_BETA"] = "not-a-float"
        assert selection.vote_beta() == 1.0
    finally:
        if old_mode is None:
            os.environ.pop("AUTOTREE_VOTE", None)
        else:
            os.environ["AUTOTREE_VOTE"] = old_mode
        if old_beta is None:
            os.environ.pop("AUTOTREE_VOTE_BETA", None)
        else:
            os.environ["AUTOTREE_VOTE_BETA"] = old_beta


def test_env_float_empty_string_uses_default():
    old = os.environ.get("AUTOTREE_TEST_MARGIN")
    try:
        os.environ["AUTOTREE_TEST_MARGIN"] = ""
        assert selection.env_float("AUTOTREE_TEST_MARGIN", 0.8) == 0.8
        os.environ["AUTOTREE_TEST_MARGIN"] = "0.4"
        assert selection.env_float("AUTOTREE_TEST_MARGIN", 0.8) == 0.4
        os.environ["AUTOTREE_TEST_MARGIN"] = "junk"
        assert selection.env_float("AUTOTREE_TEST_MARGIN", 0.8) == 0.8
        os.environ.pop("AUTOTREE_TEST_MARGIN")
        assert selection.env_float("AUTOTREE_TEST_MARGIN", 0.8) == 0.8
    finally:
        if old is None:
            os.environ.pop("AUTOTREE_TEST_MARGIN", None)
        else:
            os.environ["AUTOTREE_TEST_MARGIN"] = old
