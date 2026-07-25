import pytest

from thoughtbench.metrics import (
    accuracy_at_k,
    aggregate_values,
    compute_metric_set,
    numeric_stats,
    pass_power_k,
    percentile,
)
from thoughtbench.models import SampleResult


def test_accuracy_at_k_rejects_fewer_than_k_samples() -> None:
    with pytest.raises(ValueError, match="fewer than k"):
        accuracy_at_k([[True, False]], 4)


def test_accuracy_and_pass_power_k_match_hand_computed_outcomes() -> None:
    outcomes = [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
        [True, True, True, True],
    ]

    assert accuracy_at_k(outcomes, 1) == pytest.approx(0.5)
    assert accuracy_at_k(outcomes, 4) == pytest.approx(0.75)
    assert pass_power_k(outcomes, 1) == pytest.approx(0.5)
    assert pass_power_k(outcomes, 4) == pytest.approx(0.25)


def _sample(
    task_id: str,
    index: int,
    correct: bool,
    latency: float,
    *,
    kv_reuse_ratio: float | None = None,
) -> SampleResult:
    return SampleResult(
        sample_key=f"{task_id}-{index}",
        task_id=task_id,
        protocol_seed=1,
        request_seed=index,
        budget_name="hand",
        sample_index=index,
        response_text="response",
        expected_answer="answer",
        grader="exact-match",
        tags=["fixture"],
        correct=correct,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_seconds=latency,
        ttft_seconds=latency / 10,
        tokens_per_second=5 / latency,
        rollout_throughput_per_hour=3600 / latency,
        kv_reuse_ratio=kv_reuse_ratio,
    )


def test_metric_set_cost_tokens_and_stats_match_hand_calculation() -> None:
    samples = [
        *[_sample("a", index, index == 0, index + 1) for index in range(4)],
        *[_sample("b", index, index == 1, index + 1) for index in range(4)],
    ]

    metrics = compute_metric_set(
        samples,
        metric_ks=(1, 4),
        input_cost_per_million=1,
        output_cost_per_million=2,
    )

    assert metrics.accuracy_at_k == {"1": 0.5, "4": 1.0}
    assert metrics.pass_power_k == {"1": 0.5, "4": 0.0}
    assert metrics.input_tokens == 80
    assert metrics.output_tokens == 40
    assert metrics.correct_sample_count == 2
    assert metrics.tokens_per_correct == 60
    assert metrics.total_cost_usd == pytest.approx(0.00016)
    assert metrics.cost_per_correct_usd == pytest.approx(0.00008)
    assert metrics.latency_seconds.mean == pytest.approx(2.5)
    assert metrics.ttft_seconds.p95 == pytest.approx(0.4)
    assert metrics.kv_reuse_ratio.count == 0


def test_percentiles_and_seed_spread_are_explicit() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert numeric_stats([None, 1, 3]).spread == 2
    aggregate = aggregate_values([0.5, 0.75, 1.0])
    assert aggregate.mean == pytest.approx(0.75)
    assert aggregate.spread == pytest.approx(0.5)


def test_tokens_and_cost_per_correct_are_null_without_a_correct_sample() -> None:
    metrics = compute_metric_set(
        [_sample("a", 0, False, 1)],
        metric_ks=(1,),
        input_cost_per_million=1,
        output_cost_per_million=2,
    )

    assert metrics.tokens_per_correct is None
    assert metrics.cost_per_correct_usd is None


def test_core_logical_over_physical_kv_reuse_ratio_is_preserved() -> None:
    metrics = compute_metric_set(
        [_sample("a", 0, True, 1, kv_reuse_ratio=5.0)],
        metric_ks=(1,),
        input_cost_per_million=0,
        output_cost_per_million=0,
    )

    assert metrics.kv_reuse_ratio.mean == 5.0
    assert metrics.kv_reuse_ratio.minimum == 5.0
    assert metrics.kv_reuse_ratio.maximum == 5.0
