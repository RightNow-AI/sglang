import json

import pytest

from thoughtbench.models import SampleResult
from thoughtbench.runner import PartialStore, ResumeError


def _sample() -> SampleResult:
    return SampleResult(
        sample_key="key",
        task_id="task",
        protocol_seed=1,
        request_seed=2,
        budget_name="tiny",
        sample_index=0,
        response_text="x",
        expected_answer="y",
        grader="exact-match",
        tags=["fixture"],
        correct=False,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        latency_seconds=0.1,
        tokens_per_second=10,
        rollout_throughput_per_hour=36000,
    )


def test_partial_store_round_trips_completed_samples(tmp_path) -> None:
    store = PartialStore(tmp_path / "partial.jsonl", "a" * 64)
    store.append(_sample())

    assert store.load() == {"key": _sample()}


def test_partial_store_rejects_a_different_run_fingerprint(tmp_path) -> None:
    path = tmp_path / "partial.jsonl"
    PartialStore(path, "a" * 64).append(_sample())

    with pytest.raises(ResumeError, match="fingerprint"):
        PartialStore(path, "b" * 64).load()


def test_partial_store_ignores_only_a_truncated_crash_tail(tmp_path) -> None:
    path = tmp_path / "partial.jsonl"
    store = PartialStore(path, "a" * 64)
    store.append(_sample())
    with path.open("ab") as handle:
        handle.write(b'{"run_fingerprint":')

    assert store.load() == {"key": _sample()}


def test_partial_store_rejects_corruption_before_the_tail(tmp_path) -> None:
    path = tmp_path / "partial.jsonl"
    path.write_text("not-json\n" + json.dumps({"x": 1}) + "\n", encoding="utf-8")

    with pytest.raises(ResumeError, match="line 1"):
        PartialStore(path, "a" * 64).load()
