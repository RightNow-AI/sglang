"""Hand-auditable ThoughtBench metric formulas."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable, Sequence

from .models import AggregateValue, MetricSet, NumericStats, SampleResult

SUPPORTED_K = (1, 4, 16, 64)


def _validate_outcomes(outcomes: Sequence[Sequence[bool]], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not outcomes:
        raise ValueError("at least one task is required")
    for index, samples in enumerate(outcomes):
        if len(samples) < k:
            raise ValueError(
                f"task index {index} has fewer than k samples: {len(samples)} < {k}"
            )


def accuracy_at_k(outcomes: Sequence[Sequence[bool]], k: int) -> float:
    """Fraction of tasks with at least one correct answer in the first k."""

    _validate_outcomes(outcomes, k)
    return fmean(1.0 if any(samples[:k]) else 0.0 for samples in outcomes)


def pass_power_k(outcomes: Sequence[Sequence[bool]], k: int) -> float:
    """Fraction of tasks for which every one of the first k samples passes."""

    _validate_outcomes(outcomes, k)
    return fmean(1.0 if all(samples[:k]) else 0.0 for samples in outcomes)


def percentile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolated percentile using a zero-indexed rank."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_stats(values: Iterable[float | None]) -> NumericStats:
    """Summarize available values while retaining an explicit count."""

    present = [value for value in values if value is not None]
    if not present:
        return NumericStats(count=0)
    return NumericStats(
        count=len(present),
        mean=fmean(present),
        minimum=min(present),
        maximum=max(present),
        spread=max(present) - min(present),
        p50=percentile(present, 0.5),
        p95=percentile(present, 0.95),
    )


def _task_outcomes(samples: Sequence[SampleResult]) -> list[list[bool]]:
    grouped: dict[str, list[SampleResult]] = {}
    for sample in samples:
        grouped.setdefault(sample.task_id, []).append(sample)
    return [
        [sample.correct for sample in sorted(group, key=lambda item: item.sample_index)]
        for _task_id, group in sorted(grouped.items())
    ]


def compute_metric_set(
    samples: Sequence[SampleResult],
    *,
    metric_ks: Sequence[int],
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> MetricSet:
    """Compute one seed/budget cell from raw samples."""

    if not samples:
        raise ValueError("metrics require at least one sample")
    outcomes = _task_outcomes(samples)
    accuracy = {str(k): accuracy_at_k(outcomes, k) for k in metric_ks}
    pass_power = {str(k): pass_power_k(outcomes, k) for k in metric_ks}
    input_tokens = sum(sample.prompt_tokens for sample in samples)
    output_tokens = sum(sample.completion_tokens for sample in samples)
    correct_count = sum(sample.correct for sample in samples)
    total_tokens = input_tokens + output_tokens
    total_cost = (
        input_tokens * input_cost_per_million
        + output_tokens * output_cost_per_million
    ) / 1_000_000
    return MetricSet(
        task_count=len(outcomes),
        sample_count=len(samples),
        correct_sample_count=correct_count,
        accuracy_at_k=accuracy,
        pass_power_k=pass_power,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_per_correct=(total_tokens / correct_count if correct_count else None),
        total_cost_usd=total_cost,
        cost_per_correct_usd=(total_cost / correct_count if correct_count else None),
        latency_seconds=numeric_stats(sample.latency_seconds for sample in samples),
        ttft_seconds=numeric_stats(sample.ttft_seconds for sample in samples),
        tokens_per_second=numeric_stats(sample.tokens_per_second for sample in samples),
        rollout_throughput_per_hour=numeric_stats(
            sample.rollout_throughput_per_hour for sample in samples
        ),
        kv_reuse_ratio=numeric_stats(sample.kv_reuse_ratio for sample in samples),
        useful_token_ratio=numeric_stats(sample.useful_token_ratio for sample in samples),
    )


def aggregate_values(values: Sequence[float | None]) -> AggregateValue:
    """Aggregate a scalar across protocol seeds."""

    present = [value for value in values if value is not None]
    if not present:
        return AggregateValue(count=0)
    return AggregateValue(
        count=len(present),
        mean=fmean(present),
        minimum=min(present),
        maximum=max(present),
        spread=max(present) - min(present),
    )
