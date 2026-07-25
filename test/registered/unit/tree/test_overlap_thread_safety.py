"""The forward thread must never mutate tree state.

SGLang's default overlap scheduler runs the model forward on a different thread
from the scheduler loop. get_active() reaps departed parents, which finalizes
runs and mutates runtime.runs. When ForwardBatch construction called it, the
forward thread was mutating the same dict the scheduler thread iterates.

peek_active() is the read-only accessor the forward path must use. These tests
pin that contract from both sides: peek never mutates, get still reaps, and the
forward-path call sites do not import the reaping variant.
"""

import re
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

from sglang.srt.tree import tree_runtime

register_cpu_ci(est_time=2, suite="per-commit-cpu")


def make_runtime_with_a_departed_parent(monkeypatch):
    """A runtime holding one run whose parent request has finished."""
    runtime = tree_runtime.SchedulerTreeRuntime(SimpleNamespace())
    parent_req = SimpleNamespace(rid="parent-0", finished=lambda: True)
    branch = SimpleNamespace(req=parent_req)
    run = SimpleNamespace(
        branches={"0": branch},
        forked=False,
        finalized=False,
        parent_rid="parent-0",
    )
    runtime.runs = {"parent-0": run}
    reaped = []
    monkeypatch.setattr(
        runtime, "_cleanup_run",
        lambda r, reason: reaped.append(reason), raising=False)
    monkeypatch.setattr(tree_runtime, "_ACTIVE", runtime, raising=False)
    return runtime, run, reaped


def test_peek_active_does_not_reap(monkeypatch):
    """The forward thread must observe state without changing it."""
    runtime, run, reaped = make_runtime_with_a_departed_parent(monkeypatch)
    runs_before = dict(runtime.runs)

    result = tree_runtime.peek_active()

    assert result is runtime
    assert reaped == [], "peek_active reaped; that is scheduler-thread work"
    assert runtime.runs == runs_before, "peek_active mutated runtime.runs"
    assert run.finalized is False


def test_peek_active_is_none_when_no_runtime(monkeypatch):
    monkeypatch.setattr(tree_runtime, "_ACTIVE", None, raising=False)
    assert tree_runtime.peek_active() is None


def test_get_active_still_reaps(monkeypatch):
    """The scheduler thread keeps the reaping behavior it relies on."""
    _, _, reaped = make_runtime_with_a_departed_parent(monkeypatch)
    tree_runtime.get_active()
    assert reaped == ["parent_left"]


def test_forward_path_does_not_import_the_reaping_accessor():
    """Pin the call site: ForwardBatch construction runs on the forward thread.

    A future edit that swaps peek_active back to get_active reintroduces the
    race silently, because nothing crashes; it only corrupts under concurrency.
    """
    # tree_runtime.py lives at .../srt/tree/tree_runtime.py, so parents[1] is srt
    srt = Path(tree_runtime.__file__).resolve().parents[1]
    fbi = srt / "model_executor" / "forward_batch_info.py"
    text = fbi.read_text(encoding="utf-8", errors="replace")

    assert "peek_active" in text, "forward path lost the read-only accessor"
    bad = re.findall(r"^\s*get_active\b|import get_active|get_active as ",
                     text, flags=re.M)
    assert not bad, f"forward path imports the reaping accessor: {bad}"
