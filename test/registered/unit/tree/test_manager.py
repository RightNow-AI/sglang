from array import array
from types import SimpleNamespace

import pytest

from sglang.srt.tree.manager import TreeRunManager, TreeTokenBudget
from sglang.srt.tree.params import TreeGenerateReqInput, TreeParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


def test_tree_budget_stops_exactly_at_limit_across_branches():
    budget = TreeTokenBudget(5)

    updates = [
        budget.consume(branch_id) for branch_id in ("0", "1", "0", "2", "1", "2")
    ]

    assert [update.accepted for update in updates] == [True] * 5 + [False]
    assert [update.spent for update in updates] == [1, 2, 3, 4, 5, 5]
    assert [update.exhausted for update in updates] == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert budget.spent == 5
    assert budget.spent_per_branch == {"0": 2, "1": 2, "2": 1}


class FakeTreeCache:
    def __init__(self):
        self.root_node = SimpleNamespace(lock_ref=0)
        self.prefix_node = SimpleNamespace(lock_ref=1)

    def cache_unfinished_req(self, parent):
        parent.prefix_indices = list(range(len(parent.get_fill_ids())))
        parent.last_node = self.prefix_node

    def inc_lock_ref(self, node):
        node.lock_ref += 1

    def dec_lock_ref(self, node):
        node.lock_ref -= 1


class ScriptedPolicy:
    instances = []

    def __init__(self, config):
        self.config = config
        self.events = []
        self.commands = []
        self.__class__.instances.append(self)

    def feed_event(self, event):
        self.events.append(event)
        if event["branch"] == 2:
            self.commands = [{"type": "kill", "branch": 2, "reason": "beam"}]

    def poll_commands(self):
        commands, self.commands = self.commands, []
        return commands

    def drain(self):
        self.commands = [{"type": "finalize", "branch": 1}]


def make_parent():
    parent = SimpleNamespace(
        rid="tree-parent",
        origin_input_ids=array("q", [10, 11]),
        output_ids=array("q", [12]),
        prefix_indices=[],
        last_node=None,
    )
    parent.get_fill_ids = lambda: array("q", [10, 11, 12])
    return parent


def make_branch(rid, decoded_text=""):
    return SimpleNamespace(
        rid=rid,
        decoded_text=decoded_text,
        output_ids=array("q"),
    )


def test_manager_assembles_events_summary_and_winner_under_scripted_policy():
    manager = TreeRunManager(
        tree_cache=FakeTreeCache(), scheduler_factory=ScriptedPolicy
    )
    tree_input = TreeGenerateReqInput(
        base=SimpleNamespace(rid="tree-parent"),
        tree=TreeParams(policy="beam", branches=3, budget_tokens=4, scorer="logprob"),
    )

    assert manager.start_tree(tree_input) is tree_input.base
    plan = manager.on_parent_prefill_done(make_parent())
    branches = [make_branch(child.rid) for child in plan.children]

    assert manager.on_branch_token(branches[0], 20, -0.5) == ()
    assert manager.on_branch_token(branches[1], 21, -0.2) == ()
    assert manager.on_branch_token(branches[2], 22, -1.5) == (
        {"type": "kill", "branch": 2, "reason": "beam"},
    )
    branches[1].decoded_text = "winner text"
    commands = manager.on_branch_token(branches[1], 23, -0.1)

    assert commands == (
        {"type": "finalize", "branch": 1},
        {"type": "kill", "branch": 0, "reason": "budget"},
    )

    result = manager.finalize_tree("tree-parent", "1")
    assert result.winner_text == "winner text"
    assert result.winner_token_ids == [21, 23]
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4
    assert result.summary.branch_count == 3
    assert result.summary.pruned_count == 2
    assert result.summary.merged_count == 0
    assert result.summary.winner_branch_id == "1"
    assert result.summary.tokens_spent_per_branch == {"0": 1, "1": 2, "2": 1}
    assert result.summary.final_scores == pytest.approx(
        {"0": -0.5, "1": -0.3, "2": -1.5}
    )
    assert result.summary.kv_reuse_ratio == pytest.approx(13 / 5)
    assert result.finish_reason == "length"
    assert result.counters.logical_tokens == 13
    assert result.counters.physical_tokens == 5
    assert result.counters.useful_tokens == 5
    assert [event.event for event in result.branch_events] == [
        "forked",
        "forked",
        "forked",
        "token",
        "token",
        "token",
        "pruned",
        "token",
        "finalized",
        "pruned",
    ]
    assert manager.completed_results["tree-parent"] is result


def test_manager_rejects_unknown_branch_and_unknown_winner():
    manager = TreeRunManager(
        tree_cache=FakeTreeCache(), scheduler_factory=ScriptedPolicy
    )
    tree_input = TreeGenerateReqInput(
        base=SimpleNamespace(rid="tree-parent"),
        tree=TreeParams(branches=1, budget_tokens=1),
    )
    manager.start_tree(tree_input)
    manager.on_parent_prefill_done(make_parent())

    with pytest.raises(KeyError, match="unknown tree branch"):
        manager.on_branch_token(make_branch("missing"), 1, -0.1)
    with pytest.raises(KeyError, match="unknown winner branch"):
        manager.finalize_tree("tree-parent", "99")
