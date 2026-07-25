import pytest

from thoughtbench.bench_models import TaskResult
from thoughtbench.statistics import summarize, wilson_interval


@pytest.mark.parametrize(
    ("correct", "total", "low", "high"),
    [
        (0, 10, 0.0, 0.277533),
        (5, 10, 0.236593, 0.763407),
        (10, 10, 0.722467, 1.0),
    ],
)
def test_wilson_interval_matches_known_95_percent_cases(correct, total, low, high) -> None:
    actual_low, actual_high = wilson_interval(correct, total)
    assert actual_low == pytest.approx(low, abs=1e-6)
    assert actual_high == pytest.approx(high, abs=1e-6)


def test_summary_uses_null_when_any_token_measurement_is_missing() -> None:
    results = [
        TaskResult(id="a", seed=1, correct=True, answer="1", gold="1", tokens=10, wall_s=0.1),
        TaskResult(id="b", seed=1, correct=False, answer="2", gold="1", tokens=None, wall_s=0.2),
    ]

    summary = summarize(results)

    assert summary.accuracy == 0.5
    assert summary.mean_tokens is None
    assert summary.median_tokens is None
    assert summary.tokens_per_correct is None
    assert summary.n_tasks == 2
