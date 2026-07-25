import json

import pytest
from jsonschema import ValidationError

from thoughtbench.bench_models import BenchmarkResults, ResultMeta, ResultSummary, TaskResult
from thoughtbench.harness_schema import benchmark_results_schema, validate_benchmark_results


def _results() -> BenchmarkResults:
    return BenchmarkResults(
        meta=ResultMeta(
            engine_label="engine",
            model="model",
            base_url_redacted="http://endpoint.test",
            arm="single",
            params={"max_tokens": 8},
            seeds=[1],
            git_sha=None,
            started_at="2026-01-01T00:00:00+00:00",
        ),
        tasks=[TaskResult(id="a", seed=1, correct=True, answer="1", gold="1", tokens=2, wall_s=0.1)],
        summary=ResultSummary(
            accuracy=1,
            ci_low=0.2,
            ci_high=1,
            mean_tokens=2,
            median_tokens=2,
            tokens_per_correct=2,
            n_tasks=1,
        ),
    )


def test_results_schema_is_serializable_and_validates_complete_artifact() -> None:
    payload = _results().model_dump(mode="json")
    validate_benchmark_results(payload)
    assert "thoughtbench.results.v1" in json.dumps(benchmark_results_schema())


def test_results_schema_rejects_missing_task_measurements() -> None:
    payload = _results().model_dump(mode="json")
    del payload["tasks"][0]["wall_s"]
    with pytest.raises(ValidationError):
        validate_benchmark_results(payload)
