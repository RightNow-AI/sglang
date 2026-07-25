"""Deterministic content agreement scores for live tree branches."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from sglang.srt.tree.answers import extract_answer_text


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    """Pure scoring configuration independent of scheduler state."""

    min_survivors: int = 2
    trailing_window: int = 64


def consensus_scores(
    partial_texts: Mapping[int, str],
    config: ConsensusConfig,
) -> dict[int, float]:
    """Score live branches by answer agreement or trailing token overlap."""

    _validate_config(config)
    branch_ids = sorted(partial_texts)
    if not branch_ids:
        return {}

    answers = {
        branch_id: extract_answer_text(partial_texts[branch_id])
        for branch_id in branch_ids
    }
    if any(answer is not None for answer in answers.values()):
        scores = _answer_group_scores(branch_ids, answers)
    else:
        scores = _lexical_scores(
            branch_ids,
            partial_texts,
            config.trailing_window,
        )
    return _protect_minimum_survivors(scores, config.min_survivors)


def _validate_config(config: ConsensusConfig) -> None:
    if (
        isinstance(config.min_survivors, bool)
        or not isinstance(config.min_survivors, int)
        or config.min_survivors <= 0
    ):
        raise ValueError("min_survivors must be a positive integer")
    if (
        isinstance(config.trailing_window, bool)
        or not isinstance(config.trailing_window, int)
        or config.trailing_window <= 0
    ):
        raise ValueError("trailing_window must be a positive integer")


def _answer_group_scores(
    branch_ids: list[int],
    answers: Mapping[int, str | None],
) -> dict[int, float]:
    groups: dict[tuple[str, str | int], list[int]] = {}
    for branch_id in branch_ids:
        answer = answers[branch_id]
        key: tuple[str, str | int]
        if answer is None:
            key = ("branch", branch_id)
        else:
            key = ("answer", answer.casefold())
        groups.setdefault(key, []).append(branch_id)

    live_count = len(branch_ids)
    return {
        branch_id: len(groups[key]) / live_count
        for key in sorted(groups, key=lambda item: (item[0], str(item[1])))
        for branch_id in sorted(groups[key])
    }


def _lexical_scores(
    branch_ids: list[int],
    partial_texts: Mapping[int, str],
    trailing_window: int,
) -> dict[int, float]:
    tokens = {
        branch_id: tuple(_TOKEN_RE.findall(partial_texts[branch_id].casefold()))[
            -trailing_window:
        ]
        for branch_id in branch_ids
    }
    if len(branch_ids) == 1:
        return {branch_ids[0]: 1.0}

    pair_scores: dict[tuple[int, int], float] = {}
    for left_index, left in enumerate(branch_ids):
        for right in branch_ids[left_index + 1 :]:
            pair_scores[(left, right)] = _token_overlap(tokens[left], tokens[right])

    return {
        branch_id: sum(
            pair_scores[(min(branch_id, other), max(branch_id, other))]
            for other in branch_ids
            if other != branch_id
        )
        / (len(branch_ids) - 1)
        for branch_id in branch_ids
    }


def _token_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    shared = sum((Counter(left) & Counter(right)).values())
    return shared / max(len(left), len(right))


def _protect_minimum_survivors(
    scores: Mapping[int, float],
    min_survivors: int,
) -> dict[int, float]:
    if not scores:
        return {}
    best_score = max(scores.values())
    ranked = sorted(scores, key=lambda branch_id: (-scores[branch_id], branch_id))
    protected = set(ranked[: min(min_survivors, len(ranked))])
    return {
        branch_id: best_score if branch_id in protected else scores[branch_id]
        for branch_id in sorted(scores)
    }
