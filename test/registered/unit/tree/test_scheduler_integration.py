"""Full-lifecycle CPU test for the scheduler tree bridge."""

from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

from sglang.srt.tree.params import TreeGenerateReqInput, TreeParams
from sglang.srt.tree.scheduler_integration import SchedulerTreeBridge

register_cpu_ci(est_time=5, suite="per-commit-cpu")


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
    def __init__(self, config):
        self.commands = []

    def feed_event(self, event):
        if event["branch"] == 1:
            self.commands = [{"type": "kill", "branch": 1, "reason": "beam"}]

    def poll_commands(self):
        commands, self.commands = self.commands, []
        return commands

    def drain(self):
        self.commands = [{"type": "finalize", "branch": 0}]


def make_parent():
    parent = SimpleNamespace(
        rid="tree-parent",
        origin_input_ids=array("q", [10, 11]),
        output_ids=array("q", [12]),
        prefix_indices=[],
        last_node=None,
        sampling_params=SimpleNamespace(),
    )
    parent.get_fill_ids = lambda: array("q", [10, 11, 12])
    return parent


def build_bridge():
    enqueued = []
    finished = []
    bridge = SchedulerTreeBridge(
        tree_cache=FakeTreeCache(),
        enqueue=enqueued.append,
        req_factory=lambda child, parent: SimpleNamespace(
            rid=child.rid,
            branch_id=child.branch_id,
            output_ids=array("q"),
            decoded_text=f"branch-{child.branch_id}",
            to_finish=False,
        ),
        mark_finish=finished.append,
        scheduler_factory=ScriptedPolicy,
    )
    return bridge, enqueued, finished


def test_bridge_runs_fork_prune_finalize_lifecycle():
    bridge, enqueued, finished = build_bridge()
    tree_input = TreeGenerateReqInput(
        base=SimpleNamespace(rid="tree-parent"),
        tree=TreeParams(policy="beam", branches=2, budget_tokens=64),
    )

    bridge.handle_tree_request(tree_input)
    parent = make_parent()
    bridge.register_parent(parent)
    assert bridge.is_tree_parent("tree-parent")

    spawned = bridge.on_parent_prefill_done(parent)
    assert spawned == 2
    assert len(enqueued) == 2
    assert all(bridge.is_tree_branch(req.rid) for req in enqueued)

    survivor, doomed = enqueued
    # Token on the doomed branch triggers the scripted kill command.
    result = bridge.on_branch_token(doomed, token=7, logprob=-0.5)
    assert result is None
    assert doomed in finished

    # Budget-driven finalize: survivor decodes until the policy finalizes it.
    result = None
    for token in range(70):
        result = bridge.on_branch_token(survivor, token=token, logprob=-0.1)
        if result is not None:
            break

    assert result is not None
    assert result.summary.winner_branch_id == "0"
    assert result.summary.pruned_count == 1
    assert bridge.take_result("tree-parent") is not None
    assert bridge.take_result("tree-parent") is None
    assert survivor in finished
