import importlib.util
import threading
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_PROFILE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "tree"
    / "profile.py"
)


def _load_profile(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv("AUTOTREE_PROFILE", "1")
    else:
        monkeypatch.delenv("AUTOTREE_PROFILE", raising=False)
    spec = importlib.util.spec_from_file_location(
        "_autotree_profile_under_test", _PROFILE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disabled_profile_is_empty_and_no_op(monkeypatch):
    profile = _load_profile(monkeypatch, enabled=False)

    with profile.span("disabled"):
        profile.incr("counter", 7)

    assert profile.dump() == {}


def test_enabled_spans_accumulate_time_and_counts(monkeypatch):
    profile = _load_profile(monkeypatch, enabled=True)
    readings = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(profile, "_perf_counter", lambda: next(readings))

    with profile.span("phase"):
        pass
    with profile.span("phase"):
        pass
    profile.incr("tokens", 3)

    result = profile.dump()
    assert result["counters"] == {"tokens": 3}
    assert result["spans"]["phase"]["count"] == 2
    assert result["spans"]["phase"]["total_seconds"] == pytest.approx(0.75)


def test_concurrent_incr_is_thread_safe(monkeypatch):
    profile = _load_profile(monkeypatch, enabled=True)
    thread_count = 8
    increments_per_thread = 2000

    def worker():
        for _ in range(increments_per_thread):
            profile.incr("concurrent")

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert profile.dump()["counters"]["concurrent"] == (
        thread_count * increments_per_thread
    )
