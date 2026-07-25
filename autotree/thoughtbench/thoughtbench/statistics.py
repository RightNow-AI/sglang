"""Honest summary statistics for one benchmark arm."""

from __future__ import annotations

import math
from statistics import fmean, median
from typing import Sequence

from .bench_models import ResultSummary, TaskResult


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive")
    if correct < 0 or correct > total:
        raise ValueError("correct must be between zero and total")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize(results: Sequence[TaskResult]) -> ResultSummary:
    if not results:
        raise ValueError("cannot summarize an empty result set")
    correct = sum(result.correct for result in results)
    low, high = wilson_interval(correct, len(results))
    tokens = [result.tokens for result in results]
    all_tokens_measured = all(value is not None for value in tokens)
    measured = [value for value in tokens if value is not None]
    return ResultSummary(
        accuracy=correct / len(results),
        ci_low=low,
        ci_high=high,
        mean_tokens=fmean(measured) if all_tokens_measured else None,
        median_tokens=median(measured) if all_tokens_measured else None,
        tokens_per_correct=(sum(measured) / correct if all_tokens_measured and correct else None),
        n_tasks=len(results),
    )
