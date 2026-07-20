from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.tree.scheduler_hooks import on_parent_prefill_done
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
