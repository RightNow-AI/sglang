import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.tree import tree_runtime
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, skip_special_tokens=True):
        answer = next((token for token in token_ids if token != self.eos_token_id), 0)
        return f"#### {answer}"


class FakeReq:
    def __init__(
        self,
        rid,
        output_ids=(),
        *,
        max_new_tokens=164,
        ignore_eos=True,
        finished=False,
    ):
        self.rid = rid
        self.output_ids = list(output_ids)
        self.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
        )
        self.customized_info = None
        self.send_token_offset = 0
        self.snapshot_before_finish = None
        self._finished = finished
        self._to_finish = None

    @property
    def to_finish(self):
        return self._to_finish

    @to_finish.setter
    def to_finish(self, value):
        if value is not None:
            values = (self.customized_info or {}).get("autotree", [])
            self.snapshot_before_finish = values[-1] if values else None
        self._to_finish = value

    def finished(self):
        return self._finished


def install_fake_finish_reason(monkeypatch):
    module = types.ModuleType("sglang.srt.managers.schedule_batch")
    module.FINISH_LENGTH = lambda *, length: ("length", length)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def make_run(*, parent_ids, sibling_ids, sibling_finished, budget_tokens=0):
    runtime = tree_runtime.SchedulerTreeRuntime(
        SimpleNamespace(tokenizer=FakeTokenizer())
    )
    run = tree_runtime._TreeRun(
        "parent",
        {"branches": 2, "budget_tokens": budget_tokens, "policy": "beam"},
    )
    run.orig_sampling = (100, False)
    run.forked = True

    parent_req = FakeReq("parent", parent_ids)
    sibling_req = FakeReq(
        "parent#tree1",
        sibling_ids,
        max_new_tokens=100,
        ignore_eos=False,
        finished=sibling_finished,
    )
    parent = tree_runtime._BranchState("parent", 0, parent_req)
    sibling = tree_runtime._BranchState("parent#tree1", 1, sibling_req)
    parent.tokens = len(parent_ids)
    sibling.tokens = len(sibling_ids)
    parent.score = -float(parent.tokens)
    sibling.score = -float(sibling.tokens)

    run.branches = {"0": parent, "1": sibling}
    run.branches_by_rid = {parent.rid: parent, sibling.rid: sibling}
    runtime.runs[run.parent_rid] = run
    runtime.branch_index = {parent.rid: run, sibling.rid: run}
    return runtime, run, parent_req, sibling_req


def test_parent_hold_parks_at_eos_before_compatibility_tail(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    monkeypatch.delenv("AUTOTREE_EARLY_PARENT_STOP", raising=False)
    runtime = tree_runtime.SchedulerTreeRuntime(
        SimpleNamespace(tokenizer=FakeTokenizer())
    )
    run = tree_runtime._TreeRun("parent", {"branches": 2})
    run.orig_sampling = (100, False)
    parent = FakeReq(
        "parent",
        output_ids=[7],
        max_new_tokens=100,
        ignore_eos=False,
    )

    runtime._hold_parent(run, parent)

    assert parent.sampling_params.max_new_tokens == 164
    assert parent.sampling_params.ignore_eos is True
    branch = tree_runtime._BranchState("parent", 0, parent)
    run.branches = {"0": branch}
    run.branches_by_rid = {"parent": branch}
    run.forked = True
    parent.output_ids.append(99)

    runtime._account_tokens(run, parent, [99], -0.1)

    assert parent.to_finish == ("length", 2)
    assert parent.output_ids == [7, 99]
    runtime._attach_snapshot(run)
    assert "tail_tokens_saved" not in parent.customized_info["autotree"][-1]


def test_naturally_finished_parent_output_waits_without_more_decode(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    output_streamer = SimpleNamespace(stream_output=Mock())
    runtime, run, parent, sibling = make_run(
        parent_ids=[7, 99],
        sibling_ids=[7],
        sibling_finished=False,
    )
    runtime.scheduler.output_streamer = output_streamer
    parent._finished = True

    assert runtime.should_defer_parent_output(parent) is True
    assert run.parent_output_deferred is True
    assert parent.sampling_params.max_new_tokens == 164
    assert len(parent.output_ids) == 2

    sibling._finished = True
    runtime.on_request_finished(sibling)

    assert run.finalized is True
    assert run.parent_output_deferred is False
    output_streamer.stream_output.assert_called_once_with([parent], False)
    assert len(parent.output_ids) == 2


def test_early_stop_waits_for_parent_eos_even_when_siblings_are_done(monkeypatch):
    monkeypatch.setenv("AUTOTREE_EARLY_PARENT_STOP", "1")
    runtime, run, parent, sibling = make_run(
        parent_ids=[7],
        sibling_ids=[8],
        sibling_finished=True,
    )

    runtime.on_request_finished(sibling)
    runtime._account_tokens(run, parent, [7], -1.0)

    assert run.finished_sibling_rids == {sibling.rid}
    assert run.finalized is False
    assert parent.to_finish is None


def test_early_stop_publishes_final_snapshot_before_finishing_parent(monkeypatch):
    monkeypatch.setenv("AUTOTREE_EARLY_PARENT_STOP", "1")
    install_fake_finish_reason(monkeypatch)
    runtime, run, parent, sibling = make_run(
        parent_ids=[7, 99],
        sibling_ids=[7],
        sibling_finished=True,
    )

    runtime.on_request_finished(sibling)

    assert run.finalized is True
    assert parent.to_finish == ("length", len(parent.output_ids))
    snapshot = parent.snapshot_before_finish
    assert snapshot is not None
    assert snapshot["winner_is_final"] is True
    assert snapshot["branches"]["0"]["output_ids"] == [7, 99]
    assert snapshot["branches"]["1"]["output_ids"] == [7]
    assert snapshot["tail_tokens_saved"] == 162
    assert run.tail_tokens_saved == 162


def test_parent_without_eos_still_finishes_at_existing_budget(monkeypatch):
    monkeypatch.setenv("AUTOTREE_EARLY_PARENT_STOP", "1")
    install_fake_finish_reason(monkeypatch)
    runtime, run, parent, _ = make_run(
        parent_ids=[7, 8],
        sibling_ids=[],
        sibling_finished=False,
        budget_tokens=2,
    )
    run.spent = 1
    run.branches["0"].tokens = 1

    runtime._account_tokens(run, parent, [8], -1.0)

    assert run.finalized is True
    assert parent.to_finish == ("length", len(parent.output_ids))
    assert run.tail_tokens_saved == 0
    assert parent.snapshot_before_finish["tail_tokens_saved"] == 0
