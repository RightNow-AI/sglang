from array import array
from typing import Any
from unittest.mock import Mock

import msgspec
import pytest

from sglang.srt.tree import tree_runtime
from sglang.srt.tree.params import MAX_BRANCHES, TreeParams
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeSampling:
    def __init__(self, seed=7):
        self.max_new_tokens = 32
        self.ignore_eos = False
        self.seed = seed


class FakeTokenized(msgspec.Struct):
    rid: str
    input_ids: Any
    sampling_params: Any


class FakeReq:
    def __init__(self, rid, output_ids, *, finished=True):
        self.rid = rid
        self.origin_input_ids = array("q", [10, 11])
        self.output_ids = array("q", output_ids)
        self._finished = finished

    def finished(self):
        return self._finished


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, answers):
        self.answers = answers

    def decode(self, token_ids, skip_special_tokens=True):
        return f"#### {self.answers[tuple(token_ids)]}"


class FakeScheduler:
    def __init__(self, answers):
        self.tokenizer = FakeTokenizer(answers)
        self.requests = []

    def handle_generate_request(self, request):
        self.requests.append(request)
        return request


def make_run(*, answers, adaptive_width=5, budget_tokens=100):
    scheduler = FakeScheduler(answers)
    runtime = tree_runtime.SchedulerTreeRuntime(scheduler)
    run = tree_runtime._TreeRun(
        "parent",
        {
            "branches": len(answers),
            "adaptive_width": adaptive_width,
            "budget_tokens": budget_tokens,
        },
    )
    run.base_tokenized = FakeTokenized(
        rid="parent",
        input_ids=array("q", [10, 11]),
        sampling_params=FakeSampling(),
    )
    run.branches = {
        str(branch_id): tree_runtime._BranchState(
            "parent" if branch_id == 0 else f"parent#tree{branch_id}",
            branch_id,
            FakeReq(
                "parent" if branch_id == 0 else f"parent#tree{branch_id}",
                [branch_id + 1, 99] if branch_id == 0 else [branch_id + 1],
            ),
        )
        for branch_id in range(len(answers))
    }
    return runtime, scheduler, run


def test_tree_params_validate_adaptive_width_bounds():
    params = TreeParams(branches=3, adaptive_width=5)
    params.validate()
    assert params.to_runtime_dict()["adaptive_width"] == 5

    for invalid in (True, 3, 2, MAX_BRANCHES + 1):
        with pytest.raises(ValueError, match="adaptive_width"):
            TreeParams(branches=3, adaptive_width=invalid).validate()


def test_adaptive_width_unset_preserves_fixed_width_behavior(monkeypatch):
    monkeypatch.setattr(tree_runtime, "ADAPT_MARGIN", 2.0)
    runtime, scheduler, run = make_run(
        answers={(1,): "7", (2,): "8", (3,): "9"},
        adaptive_width=None,
    )
    runtime._fork_branches = Mock()
    runtime._finalize = Mock()

    runtime._maybe_majority_lock(run)

    runtime._fork_branches.assert_not_called()
    runtime._finalize.assert_not_called()
    assert scheduler.requests == []
    assert len(run.branches) == 3


def test_disagreement_spawns_fresh_siblings_up_to_adaptive_width(monkeypatch):
    monkeypatch.setattr(tree_runtime, "ADAPT_MARGIN", 2.0)
    runtime, scheduler, run = make_run(
        answers={(1,): "7", (2,): "8", (3,): "9"},
        adaptive_width=5,
    )
    run.spent = 12

    runtime._maybe_majority_lock(run)

    assert len(run.branches) == 5
    assert [request.rid for request in scheduler.requests] == [
        "parent#tree3",
        "parent#tree4",
    ]
    assert [request.sampling_params.seed for request in scheduler.requests] == [
        10,
        11,
    ]
    assert all(
        list(request.input_ids) == [10, 11] for request in scheduler.requests
    )
    assert run.shared_prefix_group.rids == [
        "parent",
        "parent#tree1",
        "parent#tree2",
        "parent#tree3",
        "parent#tree4",
    ]


@pytest.mark.parametrize("guard", ["margin", "finalized", "budget"])
def test_adaptive_width_guard_rails_do_not_spawn(monkeypatch, guard):
    monkeypatch.setattr(tree_runtime, "ADAPT_MARGIN", 2.0)
    answers = (
        {(1,): "7", (2,): "7", (3,): "7", (4,): "9"}
        if guard == "margin"
        else {(1,): "7", (2,): "8", (3,): "9"}
    )
    runtime, _, run = make_run(
        answers=answers,
        adaptive_width=len(answers) + 2,
        budget_tokens=12,
    )
    if guard == "finalized":
        run.finalized = True
    if guard == "budget":
        run.spent = 12
    runtime._fork_branches = Mock()
    runtime._finalize = Mock()

    runtime._maybe_majority_lock(run)

    runtime._fork_branches.assert_not_called()
    if guard == "margin":
        runtime._finalize.assert_called_once_with(run, reason="majority_locked")
    else:
        runtime._finalize.assert_not_called()
