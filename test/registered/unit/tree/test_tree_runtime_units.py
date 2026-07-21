import importlib.util
import pickle
import sys
import types
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.tree import tree_runtime
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeReq:
    def __init__(self, output_ids=(), *, finished=False):
        self.output_ids = array("l", output_ids)
        self._finished = finished
        self.customized_info = None
        self.to_finish = None

    def finished(self):
        return self._finished


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, decoded):
        self.decoded = decoded

    def decode(self, token_ids, skip_special_tokens=True):
        return self.decoded[tuple(token_ids)]


def make_branch(branch_id, *, tokens=0, mean=0.0, req=None):
    req = req or FakeReq()
    branch = tree_runtime._BranchState(f"req-{branch_id}", branch_id, req)
    branch.tokens = tokens
    branch.score = mean * tokens
    return branch


def install_fake_finish_reason(monkeypatch):
    module = types.ModuleType("sglang.srt.managers.schedule_batch")
    module.FINISH_LENGTH = lambda *, length: ("length", length)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def track_run(runtime, run):
    runtime.runs[run.parent_rid] = run
    for branch in run.branches.values():
        runtime.branch_index[branch.rid] = run


def test_branch_state_mean_logprob_handles_empty_and_scored_branches():
    branch = make_branch(0)
    assert branch.mean_logprob() == 0.0

    branch.tokens = 4
    branch.score = -2.0
    assert branch.mean_logprob() == -0.5


def test_tree_run_starts_with_isolated_empty_state():
    first = tree_runtime._TreeRun("parent-1", {"policy": "beam"})
    second = tree_runtime._TreeRun("parent-2", {"policy": "mcts"})

    first.branches["0"] = make_branch(0)
    first.spent = 7

    assert second.branches == {}
    assert second.spent == 0
    assert second.finalized is False
    assert second.winner_branch_id is None
    assert second.last_value_check == 0


def test_get_active_only_exposes_runtime_during_unfinished_tree_run(monkeypatch):
    monkeypatch.setattr(tree_runtime, "_ACTIVE", None)
    scheduler = SimpleNamespace()
    runtime = tree_runtime.install(scheduler)

    assert scheduler.tree_runtime is runtime
    assert tree_runtime.get_active() is None

    run = tree_runtime._TreeRun("parent", {})
    runtime.runs[run.parent_rid] = run
    assert tree_runtime.get_active() is runtime

    run.finalized = True
    assert tree_runtime.get_active() is None


def test_non_tree_get_active_hook_is_byte_identical(monkeypatch):
    monkeypatch.setattr(tree_runtime, '_ACTIVE', None)
    runtime = tree_runtime.install(SimpleNamespace())
    plain_req = FakeReq([1, 2, 3])
    before = pickle.dumps(plain_req, protocol=pickle.HIGHEST_PROTOCOL)

    active = tree_runtime.get_active()
    if active is not None:
        active.on_token(plain_req, [4], -0.25)

    assert pickle.dumps(plain_req, protocol=pickle.HIGHEST_PROTOCOL) == before
    assert runtime.runs == {}
    assert runtime.branch_index == {}


def test_parent_abort_cleanup_finalizes_children_and_forgets_run(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    monkeypatch.setattr(tree_runtime, '_ACTIVE', None)
    original_abort = Mock(return_value='aborted')
    scheduler = SimpleNamespace(abort_request=original_abort)
    runtime = tree_runtime.install(scheduler)
    run = tree_runtime._TreeRun('parent', {'budget_tokens': 1000})
    run.branches = {
        '0': make_branch(0, req=FakeReq([10])),
        '1': make_branch(1, req=FakeReq([20])),
        '2': make_branch(2, req=FakeReq([30])),
    }
    track_run(runtime, run)
    abort = SimpleNamespace(rid='parent', abort_all=False)

    assert scheduler.abort_request(abort) == 'aborted'

    original_abort.assert_called_once_with(abort)
    assert run.finalized is True
    assert all(
        branch.req.to_finish == ('length', len(branch.req.output_ids))
        for branch in run.branches.values()
    )
    assert runtime.runs == {}
    assert runtime.branch_index == {}


def test_parent_leave_cleans_never_finalized_run(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    monkeypatch.setattr(tree_runtime, '_ACTIVE', None)
    runtime = tree_runtime.install(SimpleNamespace())
    run = tree_runtime._TreeRun('parent', {'budget_tokens': 0})
    run.branches = {
        '0': make_branch(0, req=FakeReq([10], finished=True)),
        '1': make_branch(1, req=FakeReq([20], finished=False)),
        '2': make_branch(2, req=FakeReq([30], finished=False)),
    }
    track_run(runtime, run)

    assert tree_runtime.get_active() is None

    assert run.finalized is True
    assert run.branches['1'].req.to_finish == ('length', 1)
    assert run.branches['2'].req.to_finish == ('length', 1)
    assert runtime.runs == {}
    assert runtime.branch_index == {}


def test_tree_branch_cap_rejects_above_limit_and_accepts_at_or_under(
    monkeypatch,
):
    monkeypatch.setattr(tree_runtime, 'MAX_BRANCHES', 2, raising=False)
    base = SimpleNamespace(rid='parent')

    for branches in (1, 2):
        wrapped = tree_runtime.TokenizedTreeGenerateReqInput(
            base, {'branches': branches}
        )
        assert wrapped.base is base
        assert wrapped.tree['branches'] == branches

    with pytest.raises(ValueError, match=r'branches=3.*maximum=2'):
        tree_runtime.TokenizedTreeGenerateReqInput(base, {'branches': 3})


def test_value_prune_honors_margin_and_min_keep(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    monkeypatch.setattr(tree_runtime, "VALUE_WARMUP_TOKENS", 8)
    monkeypatch.setattr(tree_runtime, "VALUE_MARGIN", 0.5)
    monkeypatch.setattr(tree_runtime, "VALUE_MIN_KEEP", 2)

    run = tree_runtime._TreeRun("parent", {})
    run.branches = {
        "0": make_branch(0, tokens=8, mean=-0.1),
        "1": make_branch(1, tokens=8, mean=-0.6),
        "2": make_branch(2, tokens=8, mean=-1.0),
    }

    tree_runtime.SchedulerTreeRuntime(SimpleNamespace())._maybe_value_prune(run)

    assert run.branches["2"].state == "pruned"
    assert run.branches["1"].state == "active"
    assert run.pruned == 1
    assert sum(b.state == "active" for b in run.branches.values()) == 2


def test_value_prune_waits_for_warmup(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    monkeypatch.setattr(tree_runtime, "VALUE_WARMUP_TOKENS", 8)
    monkeypatch.setattr(tree_runtime, "VALUE_MARGIN", 0.1)
    monkeypatch.setattr(tree_runtime, "VALUE_MIN_KEEP", 1)

    run = tree_runtime._TreeRun("parent", {})
    run.branches = {
        "0": make_branch(0, tokens=7, mean=-0.1),
        "1": make_branch(1, tokens=7, mean=-2.0),
    }

    tree_runtime.SchedulerTreeRuntime(SimpleNamespace())._maybe_value_prune(run)

    assert all(b.state == "active" for b in run.branches.values())


def test_value_prune_stops_at_min_keep(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    monkeypatch.setattr(tree_runtime, "VALUE_WARMUP_TOKENS", 1)
    monkeypatch.setattr(tree_runtime, "VALUE_MARGIN", 0.0)
    monkeypatch.setattr(tree_runtime, "VALUE_MIN_KEEP", 2)

    run = tree_runtime._TreeRun("parent", {})
    run.branches = {
        "0": make_branch(0, tokens=2, mean=-0.1),
        "1": make_branch(1, tokens=2, mean=-3.0),
    }

    tree_runtime.SchedulerTreeRuntime(SimpleNamespace())._maybe_value_prune(run)

    assert run.pruned == 0


def test_majority_lock_requires_more_than_half_of_all_branches():
    scheduler = SimpleNamespace(
        tokenizer=FakeTokenizer(
            {
                (1,): "#### 7",
                (2,): "#### 7",
                (3,): "#### 7",
                (4,): "#### 9",
            }
        )
    )
    runtime = tree_runtime.SchedulerTreeRuntime(scheduler)
    runtime._finalize = Mock()
    run = tree_runtime._TreeRun("parent", {})
    run.branches = {
        "0": make_branch(0, req=FakeReq([1, 99], finished=False)),
        "1": make_branch(1, req=FakeReq([2], finished=True)),
        "2": make_branch(2, req=FakeReq([3], finished=False)),
        "3": make_branch(3, req=FakeReq([4], finished=False)),
    }

    runtime._maybe_majority_lock(run)
    runtime._finalize.assert_not_called()

    run.branches["2"].req._finished = True
    runtime._maybe_majority_lock(run)
    runtime._finalize.assert_called_once_with(run, reason="majority_locked")


def test_majority_lock_waits_for_held_parent_eos():
    scheduler = SimpleNamespace(
        tokenizer=FakeTokenizer({(1,): "#### 7", (2,): "#### 7"})
    )
    runtime = tree_runtime.SchedulerTreeRuntime(scheduler)
    runtime._finalize = Mock()
    run = tree_runtime._TreeRun("parent", {})
    parent = make_branch(0, req=FakeReq([1], finished=False))
    sibling = make_branch(1, req=FakeReq([2], finished=True))
    other = make_branch(2, req=FakeReq([], finished=False))
    run.branches = {"0": parent, "1": sibling, "2": other}

    runtime._maybe_majority_lock(run)
    runtime._finalize.assert_not_called()

    parent.req.output_ids.append(99)
    runtime._maybe_majority_lock(run)
    runtime._finalize.assert_called_once_with(run, reason="majority_locked")


def test_majority_lock_does_not_finalize_disagreement():
    scheduler = SimpleNamespace(
        tokenizer=FakeTokenizer({(1,): "#### 1", (2,): "#### 2", (3,): "#### 3"})
    )
    runtime = tree_runtime.SchedulerTreeRuntime(scheduler)
    runtime._finalize = Mock()
    run = tree_runtime._TreeRun("parent", {})
    run.branches = {
        str(i): make_branch(i, req=FakeReq([i + 1], finished=True)) for i in range(3)
    }

    runtime._maybe_majority_lock(run)

    runtime._finalize.assert_not_called()


def test_attach_snapshot_is_token_aligned_at_last_parent_index():
    runtime = tree_runtime.SchedulerTreeRuntime(SimpleNamespace())
    run = tree_runtime._TreeRun("parent", {"policy": "beam", "budget_tokens": 20})
    parent = make_branch(0, tokens=3, mean=-0.25, req=FakeReq([10, 11, 12]))
    run.branches = {"0": parent}
    run.spent = 3

    runtime._attach_snapshot(run)

    values = parent.req.customized_info["autotree"]
    assert len(values) == len(parent.req.output_ids)
    assert values[:-1] == [None, None]
    assert values[-1]["spent_tokens"] == 3
    assert values[-1]["branches"]["0"]["tokens"] == 3


def test_finalize_never_mutates_parent_output_ids(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    runtime = tree_runtime.SchedulerTreeRuntime(SimpleNamespace())
    run = tree_runtime._TreeRun("parent", {"policy": "beam"})
    parent = make_branch(0, tokens=2, mean=-1.0, req=FakeReq([10, 11]))
    child = make_branch(1, tokens=2, mean=-0.1, req=FakeReq([20, 21]))
    run.branches = {"0": parent, "1": child}
    original = array("l", parent.req.output_ids)

    runtime._finalize(run, reason="test")

    assert parent.req.output_ids == original
    assert run.winner_branch_id == 1
    assert parent.req.customized_info["autotree"][-1]["winner_is_final"] is True


def test_environment_knobs_override_defaults(monkeypatch):
    runtime_path = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "tree"
        / "tree_runtime.py"
    )
    monkeypatch.setenv("AUTOTREE_VALUE_CHECK_INTERVAL", "5")
    monkeypatch.setenv("AUTOTREE_VALUE_WARMUP_TOKENS", "6")
    monkeypatch.setenv("AUTOTREE_VALUE_MARGIN", "1.25")
    monkeypatch.setenv("AUTOTREE_VALUE_MIN_KEEP", "3")

    monkeypatch.setenv('AUTOTREE_MAX_BRANCHES', '7')

    spec = importlib.util.spec_from_file_location("tree_runtime_env_test", runtime_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert [
        module.VALUE_CHECK_INTERVAL,
        module.VALUE_WARMUP_TOKENS,
        module.VALUE_MARGIN,
        module.VALUE_MIN_KEEP,
        module.MAX_BRANCHES,
    ] == [5, 6, 1.25, 3, 7]
