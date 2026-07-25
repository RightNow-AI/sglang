"""AutoTree OFF must be inert: the fork's central promise, previously untested.

The fork's adoption story is that changing one base_url is safe, because with no
tree parameters the engine behaves exactly like stock SGLang. The scheduler
calls the AutoTree hooks unconditionally on the hot path
(batch_result_processor.py:239, :718, :725), so "OFF" really means "the hooks
are a no-op for every request AutoTree does not own". None of the existing 91
tests checked that, which left the fork's core promise unpinned.

These tests pin it against the PRODUCTION path, which is
tree_runtime.get_active() plus the SchedulerTreeRuntime hooks.
"""

from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

from sglang.srt.tree import tree_runtime

register_cpu_ci(est_time=2, suite="per-commit-cpu")


def stock_req(rid="stock-request-1"):
    """A request that never carried tree params."""
    return SimpleNamespace(rid=rid, output_ids=[], origin_input_ids=[1, 2, 3])


def fresh_runtime():
    return tree_runtime.SchedulerTreeRuntime(SimpleNamespace())


def test_get_active_is_none_when_no_runtime_installed(monkeypatch):
    """With AutoTree never engaged the scheduler must skip the hooks entirely."""
    monkeypatch.setattr(tree_runtime, "_ACTIVE", None, raising=False)
    assert tree_runtime.get_active() is None


def test_get_active_is_none_when_runtime_has_no_live_runs(monkeypatch):
    runtime = fresh_runtime()
    runtime.runs = {}
    monkeypatch.setattr(tree_runtime, "_ACTIVE", runtime, raising=False)
    assert tree_runtime.get_active() is None


def test_on_token_is_a_no_op_for_a_stock_request():
    """The hot path: every sampled token of every stock request reaches this."""
    runtime = fresh_runtime()
    req = stock_req()
    before = dict(vars(req))

    for token in range(8):
        assert runtime.on_token(req, [token], -0.1) is None

    assert dict(vars(req)) == before, "hook mutated a non-tree request"
    assert runtime.branch_index == {}


def test_on_request_finished_is_a_no_op_for_a_stock_request():
    runtime = fresh_runtime()
    req = stock_req()
    before = dict(vars(req))
    assert runtime.on_request_finished(req) is None
    assert dict(vars(req)) == before


def test_on_prefill_done_is_a_no_op_for_a_stock_request():
    runtime = fresh_runtime()
    req = stock_req()
    before = dict(vars(req))
    assert runtime.on_prefill_done(req) is None
    assert dict(vars(req)) == before


def test_stock_requests_never_accumulate_runtime_state():
    """A long-lived server serving only stock traffic must not grow state."""
    runtime = fresh_runtime()
    for i in range(50):
        req = stock_req(f"stock-{i}")
        runtime.on_prefill_done(req)
        runtime.on_token(req, [i], -0.5)
        runtime.on_request_finished(req)
    assert runtime.branch_index == {}
    assert runtime.runs == {}

