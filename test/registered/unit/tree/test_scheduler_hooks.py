from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


torch = pytest.importorskip(
    "torch", reason="tensor-backed radix-cache tests require optional torch"
)

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.tree.params import TreeParams
from sglang.srt.tree.scheduler_hooks import (
    RustSchedulerAdapter,
    apply_kill,
    on_branch_token,
    on_parent_prefill_done,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


class MockParentReq:
    def __init__(self, token_ids, req_to_token):
        self.rid = "tree-parent"
        self.origin_input_ids = array("q", token_ids[:-1])
        self.output_ids = array("q", token_ids[-1:])
        self.full_untruncated_fill_ids = array("q", token_ids)
        self.extend_range = SimpleNamespace(start=0, end=len(token_ids))
        self.req_pool_idx = 0
        self.cache_protected_len = 0
        self.last_node = None
        self.extra_key = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.priority = 0
        req_to_token[0, : len(token_ids)] = torch.tensor(
            [100 + token for token in token_ids], dtype=torch.int64
        )

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_range.end]


def make_cache():
    allocator = MagicMock()
    allocator.device = torch.device("cpu")
    req_to_token = torch.zeros((4, 32), dtype=torch.int64)
    req_pool = SimpleNamespace(req_to_token=req_to_token)
    req_pool.write = lambda index, values: req_to_token.__setitem__(index, values)
    cache = RadixCache.create_simulated(mock_allocator=allocator, page_size=1)
    cache.req_to_token_pool = req_pool
    return cache, allocator, req_to_token


def release_for_test(req, tree_cache, *, is_insert):
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert,
        kv_len_to_handle=req.effective_kv_committed_len(),
    )
    tree_cache.req_to_token_pool.free(req)
    req.kv = None


def test_fork_locks_shared_prefix_until_every_branch_releases_it():
    cache, allocator, req_to_token = make_cache()
    parent = MockParentReq([1, 2, 3, 4], req_to_token)
    parent.last_node = cache.root_node

    plan = on_parent_prefill_done(parent, branch_count=3, tree_cache=cache)

    assert len(plan.children) == 3
    assert plan.prefix_length == 4
    assert plan.prefix_node.lock_ref == 3
    assert all(child.last_node is plan.prefix_node for child in plan.children)
    assert all(list(child.input_ids) == [1, 2, 3, 4] for child in plan.children)

    cache.dec_lock_ref(plan.children[0].last_node)
    assert plan.prefix_node.lock_ref == 2
    assert cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted == 0

    cache.dec_lock_ref(plan.children[1].last_node)
    cache.dec_lock_ref(plan.children[2].last_node)
    assert plan.prefix_node.lock_ref == 0
    assert cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted == 4

    nonempty_frees = [
        call.args[0].tolist()
        for call in allocator.free.call_args_list
        if call.args and len(call.args[0])
    ]
    assert nonempty_frees == [[101, 102, 103, 104]]


def test_prune_reclaims_only_branch_suffix_once_and_releases_its_lock():
    cache, allocator, req_to_token = make_cache()
    cache.req_to_token_pool.free = lambda req: setattr(req, "req_pool_idx", None)
    parent = MockParentReq([1, 2, 3], req_to_token)
    parent.last_node = cache.root_node
    plan = on_parent_prefill_done(parent, branch_count=2, tree_cache=cache)

    branch = SimpleNamespace(
        rid=plan.children[0].rid,
        origin_input_ids=array("q", [1, 2, 3]),
        output_ids=array("q", [8, 9]),
        req_pool_idx=1,
        cache_protected_len=3,
        last_node=plan.prefix_node,
        extra_key=None,
        priority=0,
        kv=SimpleNamespace(kv_allocated_len=5),
        to_finish=None,
    )
    branch.effective_kv_committed_len = lambda: 5
    req_to_token[1, :5] = torch.tensor([101, 102, 103, 208, 209])

    assert apply_kill(
        branch,
        tree_cache=cache,
        reason="policy",
        release_fn=release_for_test,
        finish_reason_factory=lambda reason: f"pruned:{reason}",
    )
    assert branch.to_finish == "pruned:policy"
    assert plan.prefix_node.lock_ref == 1
    assert branch.req_pool_idx is None
    assert branch.kv is None

    assert not apply_kill(
        branch,
        tree_cache=cache,
        reason="policy",
        release_fn=release_for_test,
        finish_reason_factory=lambda reason: f"pruned:{reason}",
    )
    nonempty_frees = [
        call.args[0].tolist()
        for call in allocator.free.call_args_list
        if call.args and len(call.args[0])
    ]
    assert nonempty_frees == [[208, 209]]


class FakeRustScheduler:
    instances = []

    def __init__(self, config):
        self.config = config
        self.events = []
        self.commands = [
            {"type": "continue", "branch": 1},
            {"type": "kill", "branch": 2, "reason": "beam"},
        ]
        self.drained = False
        self.__class__.instances.append(self)

    def feed_event(self, event):
        self.events.append(event)

    def poll_commands(self):
        commands, self.commands = self.commands, []
        return commands

    def drain(self):
        self.drained = True
        self.commands = [{"type": "finalize", "branch": 1}]


def test_rust_scheduler_adapter_runs_with_injected_stub_without_wheel():
    adapter = RustSchedulerAdapter(
        TreeParams(
            policy="best_first", branches=3, budget_tokens=17, scorer="logprob"
        ),
        scheduler_cls=FakeRustScheduler,
    )

    commands = on_branch_token(adapter, 1, 42, -0.25, eos=False)

    fake = FakeRustScheduler.instances[-1]
    assert fake.config == {
        "policy": "best_first",
        "branches": 3,
        "budget_tokens": 17,
        "scorer": "logprob",
    }
    assert fake.events == [
        {
            "type": "token_sampled",
            "branch": 1,
            "token": 42,
            "logprob": -0.25,
            "eos": False,
        }
    ]
    assert commands == (
        {"type": "continue", "branch": 1},
        {"type": "kill", "branch": 2, "reason": "beam"},
    )
    assert adapter.drain() == ({"type": "finalize", "branch": 1},)
