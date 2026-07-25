import sys
import types
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol_tree import TreeParameters
from sglang.srt.tree import tree_runtime
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeSampling:
    def __init__(self, *, max_new_tokens=8, ignore_eos=False, seed=17):
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos
        self.seed = seed


class FakeTokenized(msgspec.Struct):
    rid: str
    input_ids: object
    sampling_params: object


class FakeReq:
    def __init__(self, rid, output_ids=(), *, finished=False):
        self.rid = rid
        self.origin_input_ids = array("q", [10, 11])
        self.output_ids = list(output_ids)
        self.sampling_params = FakeSampling()
        self.customized_info = None
        self.send_token_offset = 0
        self.return_logprob = False
        self.to_finish = None
        self._finished = finished

    def finished(self):
        return self._finished or self.to_finish is not None


class FakeScheduler:
    def __init__(self):
        self.requests = []
        self.abort_request = Mock(return_value="aborted")
        self.output_streamer = SimpleNamespace(stream_output=Mock())
        self.tokenizer = SimpleNamespace(eos_token_id=99)

    def handle_generate_request(self, request):
        self.requests.append(request)
        return request


def install_fake_finish_reasons(monkeypatch):
    module = types.ModuleType("sglang.srt.managers.schedule_batch")
    module.FINISH_LENGTH = lambda *, length: ("length", length)
    module.FINISH_ABORT = lambda: ("abort",)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def make_run(runtime, parent_rid, *, branches=2, budget_tokens=8):
    run = tree_runtime._TreeRun(
        parent_rid,
        {
            "policy": "beam",
            "branches": branches,
            "budget_tokens": budget_tokens,
        },
    )
    for branch_id in range(branches):
        rid = parent_rid if branch_id == 0 else f"{parent_rid}#tree{branch_id}"
        req = FakeReq(rid)
        branch = tree_runtime._BranchState(rid, branch_id, req)
        runtime._register_branch(run, branch)
    run.forked = branches > 1
    runtime.runs[parent_rid] = run
    return run


def test_abort_soak_cleans_all_runtime_refs_and_finishes_every_branch(monkeypatch):
    install_fake_finish_reasons(monkeypatch)
    scheduler = FakeScheduler()
    original_abort = scheduler.abort_request
    runtime = tree_runtime.install(scheduler)
    runs = [make_run(runtime, f"tree-{index}", branches=4) for index in range(40)]
    deferred = runs[0]
    deferred_parent = deferred.branches["0"].req
    deferred_parent._finished = True
    assert runtime.should_defer_parent_output(deferred_parent) is True

    for run in runs:
        scheduler.abort_request(SimpleNamespace(rid=run.parent_rid, abort_all=False))

    assert original_abort.call_count == 40
    assert runtime.runs == {}
    assert runtime.branch_index == {}
    scheduler.output_streamer.stream_output.assert_called_once_with(
        [deferred_parent], False
    )
    assert deferred_parent.to_finish == ("abort",)
    assert all(
        branch.req.to_finish is not None
        for run in runs
        for branch in run.branches.values()
    )


def test_budget_soak_stops_exactly_at_budget(monkeypatch):
    install_fake_finish_reasons(monkeypatch)
    runtime = tree_runtime.SchedulerTreeRuntime(FakeScheduler())
    run = make_run(runtime, "budget-tree", branches=2, budget_tokens=12)

    offered = ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14])
    for step, token_chunk in enumerate(offered):
        branch = run.branches[str(step % 2)]
        token_chunk, logprobs = runtime.trim_tokens_to_budget(
            branch.req,
            token_chunk,
            [-0.1] * len(token_chunk),
        )
        branch.req.output_ids.extend(token_chunk)
        runtime.on_token(branch.req, token_chunk, logprobs)
        if step < 2:
            assert run.finalized is False

    assert run.spent == 12
    assert sum(len(branch.req.output_ids) for branch in run.branches.values()) == 12
    assert run.finalized is True
    assert all(branch.req.to_finish is not None for branch in run.branches.values())


def test_concurrent_mixed_load_keeps_plain_requests_unchanged(monkeypatch):
    install_fake_finish_reasons(monkeypatch)
    runtime = tree_runtime.SchedulerTreeRuntime(FakeScheduler())
    first = make_run(runtime, "first", branches=2, budget_tokens=6)
    second = make_run(runtime, "second", branches=3, budget_tokens=9)
    plain = FakeReq("plain", [100])
    first_parent = first.branches["0"].req
    first_parent.output_ids[:] = [7, 99]
    first_parent._finished = True
    assert runtime.should_defer_parent_output(first_parent) is True

    for step in range(9):
        runtime.on_token(plain, [200 + step], -0.2)
        for run in (first, second):
            if run.finalized:
                continue
            if run is first and step == 5:
                first.branches["1"].req._finished = True
                runtime.on_request_finished(first.branches["1"].req)
                continue
            if run is first:
                continue
            branch = run.branches[str(step % len(run.branches))]
            branch.req.output_ids.append(step)
            runtime.on_token(branch.req, [step], -0.1)

    assert plain.output_ids == [100]
    assert plain.to_finish is None
    assert first.finalized is True
    assert first.parent_output_deferred is False
    assert runtime.runs["second"] is second
    assert second.finalized is True
    assert second.spent == 9


def test_single_branch_tree_uses_the_plain_intake_and_sampling_path():
    scheduler = FakeScheduler()
    runtime = tree_runtime.SchedulerTreeRuntime(scheduler)
    sampling = FakeSampling(max_new_tokens=13, ignore_eos=False, seed=5)
    base = FakeTokenized("single", array("q", [1, 2]), sampling)
    wrapped = tree_runtime.TokenizedTreeGenerateReqInput(
        base,
        {"policy": "beam", "branches": 1, "budget_tokens": 13},
    )

    result = runtime.handle_tree_request(wrapped)
    req = FakeReq("single", [7])
    runtime.on_prefill_done(req)

    assert result is base
    assert scheduler.requests == [base]
    assert sampling.max_new_tokens == 13
    assert sampling.ignore_eos is False
    assert sampling.seed == 5
    assert list(runtime.runs["single"].branches) == ["0"]
    assert list(runtime.branch_index) == ["single"]


@pytest.mark.parametrize(
    ("tree", "message"),
    [
        ({"policy": "greedy", "branches": 2, "budget_tokens": 8}, "policy"),
        ({"policy": "beam", "branches": 0, "budget_tokens": 8}, "branches"),
        ({"policy": "beam", "branches": True, "budget_tokens": 8}, "branches"),
        ({"policy": "beam", "branches": 2, "budget_tokens": 0}, "budget_tokens"),
        (
            {"policy": "beam", "branches": 2, "budget_tokens": 8, "scorer": 7},
            "scorer",
        ),
        (
            {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 8,
                "fork_at_text": "",
            },
            "fork_at_text",
        ),
        (
            {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 8,
                "fork_at_entropy": False,
            },
            "fork_at_entropy",
        ),
        (
            {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 8,
                "fork_at_text": "Answer:",
                "fork_at_entropy": 0.5,
            },
            "mutually exclusive",
        ),
        (
            {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 8,
                "adaptive_width": 2,
            },
            "adaptive_width",
        ),
        (
            {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 8,
                "mystery": 1,
            },
            "unknown tree parameter",
        ),
    ],
)
def test_malformed_tree_params_are_rejected_before_scheduler_intake(tree, message):
    with pytest.raises(ValueError, match=message):
        tree_runtime.TokenizedTreeGenerateReqInput(SimpleNamespace(rid="bad"), tree)


def test_http_tree_params_reject_coerced_integer_strings():
    with pytest.raises(ValidationError, match="branches"):
        TreeParameters(policy="beam", branches="2", budget_tokens=8)
