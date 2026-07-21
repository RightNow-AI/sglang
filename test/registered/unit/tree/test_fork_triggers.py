from array import array
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import msgspec
import pytest
import torch
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol_tree import TreeParameters
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.tree.params import TreeParams
from sglang.srt.tree.tree_runtime import (
    SchedulerTreeRuntime,
    TokenizedTreeGenerateReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class FakeSampling:
    def __init__(self, max_new_tokens=32, ignore_eos=False, seed=7):
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos
        self.seed = seed


class FakeTokenized(msgspec.Struct):
    rid: str
    input_ids: Any
    sampling_params: Any
    extra_key: Optional[str] = None


class PieceTokenizer:
    eos_token_id = 99

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.pieces.get(int(token_id), "") for token_id in token_ids)


class RecordingCache:
    supports_tree_fork_namespaces = True
    disable = False

    def __init__(self):
        self.inserted = []

    def cache_unfinished_req(self, req):
        self.inserted.append(list(req.get_fill_ids()))


class UnsafeCache:
    disable = False

    def cache_unfinished_req(self, req):
        raise AssertionError("unsafe cache must never receive generated KV")


class FakeScheduler:
    def __init__(self, tokenizer, tree_cache):
        self.tokenizer = tokenizer
        self.tree_cache = tree_cache
        self.requests = []

    def handle_generate_request(self, request):
        self.requests.append(request)
        return request


class FakeReq:
    def __init__(self, tokenized, first_output=()):
        self.rid = tokenized.rid
        self.origin_input_ids = array("q", tokenized.input_ids)
        self.output_ids = array("q", first_output)
        self.full_untruncated_fill_ids = array("q")
        self.extend_range = SimpleNamespace(start=0, end=len(self.origin_input_ids))
        self.kv_committed_len = len(self.origin_input_ids)
        self.extra_key = tokenized.extra_key
        self.sampling_params = tokenized.sampling_params
        self.output_token_logprobs_val = []
        self.customized_info = None
        self.to_finish = None
        self._finished = False

    def _refresh_fill_ids(self):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def set_extend_range(self, start, end):
        self.extend_range = SimpleNamespace(start=start, end=end)

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_range.end]

    def finished(self):
        return self._finished

    def append_decode_token(self, token_id):
        self.output_ids.append(token_id)
        self.kv_committed_len = len(self.origin_input_ids) + len(self.output_ids) - 1


def start_runtime(*, fork_at_text, branches=3, cache=None, pieces=None, rid="parent"):
    cache = cache or RecordingCache()
    scheduler = FakeScheduler(PieceTokenizer(pieces or {}), cache)
    runtime = SchedulerTreeRuntime(scheduler)
    sampling = FakeSampling()
    base = FakeTokenized(
        rid=rid,
        input_ids=array("q", [10, 11]),
        sampling_params=sampling,
        extra_key="tenant",
    )
    envelope = TokenizedTreeGenerateReqInput(
        base,
        {
            "policy": "beam",
            "branches": branches,
            "budget_tokens": 100,
            "scorer": None,
            "fork_at_text": fork_at_text,
        },
    )
    runtime.handle_tree_request(envelope)
    return runtime, scheduler, cache


def test_protocol_validates_optional_fork_delimiter_and_runtime_dict():
    base = {"policy": "beam", "branches": 2, "budget_tokens": 32}

    assert TreeParameters(**base).fork_at_text is None
    assert TreeParameters(**base, fork_at_text="x" * 64).fork_at_text == "x" * 64
    for invalid in ("", "x" * 65):
        with pytest.raises(ValidationError):
            TreeParameters(**base, fork_at_text=invalid)

    params = TreeParams(branches=2, budget_tokens=32, fork_at_text="</plan>")
    params.validate()
    assert params.to_runtime_dict()["fork_at_text"] == "</plan>"


def test_delimiter_spanning_tokens_forks_with_extended_child_input():
    runtime, scheduler, cache = start_runtime(
        fork_at_text="</plan>", pieces={1: "prefix</", 2: "plan", 3: ">"}
    )
    parent_base = scheduler.requests[0]
    parent = FakeReq(parent_base, first_output=[1])

    runtime.on_prefill_done(parent)
    parent.append_decode_token(2)
    runtime.on_token(parent, [2], -0.2)
    assert len(scheduler.requests) == 1

    parent.append_decode_token(3)
    runtime.on_token(parent, [3], -0.3)

    run = runtime.runs[parent.rid]
    assert cache.inserted == [[10, 11, 1, 2]]
    assert len(scheduler.requests) == 3
    assert all(
        list(child.input_ids) == [10, 11, 1, 2, 3]
        for child in scheduler.requests[1:]
    )
    assert all(child.extra_key == parent.extra_key for child in scheduler.requests[1:])
    assert run.forked is True
    assert run.spent == 0
    assert all(branch.tokens == 0 for branch in run.branches.values())
    assert parent.sampling_params.max_new_tokens == 96
    assert parent.sampling_params.ignore_eos is True
    assert all(child.sampling_params.max_new_tokens == 32 for child in scheduler.requests[1:])


def test_delimiter_never_seen_finishes_as_single_parent_branch():
    runtime, scheduler, cache = start_runtime(
        fork_at_text="</plan>", pieces={1: "thinking", 2: " still"}
    )
    parent = FakeReq(scheduler.requests[0], first_output=[1])

    runtime.on_prefill_done(parent)
    parent.append_decode_token(2)
    runtime.on_token(parent, [2], -0.1)
    parent._finished = True
    runtime.on_request_finished(parent)

    run = runtime.runs[parent.rid]
    snapshot = parent.customized_info["autotree"][-1]
    assert len(scheduler.requests) == 1
    assert cache.inserted == []
    assert run.finalized is True
    assert run.winner_branch_id == 0
    assert snapshot["branch_count"] == 1
    assert snapshot["branches"]["0"]["output_ids"] == [1, 2]
    assert parent.sampling_params.max_new_tokens == 32
    assert parent.sampling_params.ignore_eos is False


def test_fork_delimiter_with_one_branch_is_a_noop():
    runtime, scheduler, cache = start_runtime(
        fork_at_text="</plan>", branches=1, pieces={1: "</plan>"}
    )
    parent_base = scheduler.requests[0]
    parent = FakeReq(parent_base, first_output=[1])

    runtime.on_prefill_done(parent)

    assert parent_base.extra_key == "tenant"
    assert cache.inserted == []
    assert list(runtime.runs[parent.rid].branches) == ["0"]
    assert len(scheduler.requests) == 1


def test_default_fork_mode_still_forks_immediately_from_prompt():
    runtime, scheduler, cache = start_runtime(fork_at_text=None, branches=3)
    parent_base = scheduler.requests[0]
    parent = FakeReq(parent_base, first_output=[1])

    runtime.on_prefill_done(parent)

    assert parent_base.extra_key == "tenant"
    assert cache.inserted == []
    assert len(scheduler.requests) == 3
    assert all(list(child.input_ids) == [10, 11] for child in scheduler.requests[1:])
    assert parent.sampling_params.max_new_tokens == 96
    assert parent.sampling_params.ignore_eos is True


def test_identical_tree_prompts_and_plain_request_are_cache_isolated():
    cache = RadixCache.create_simulated(page_size=1)
    scheduler = FakeScheduler(PieceTokenizer({}), cache)
    runtime = SchedulerTreeRuntime(scheduler)

    for rid in ("tree-a", "tree-b"):
        base = FakeTokenized(
            rid=rid,
            input_ids=array("q", [5, 6]),
            sampling_params=FakeSampling(),
            extra_key="tenant",
        )
        runtime.handle_tree_request(
            TokenizedTreeGenerateReqInput(
                base,
                {
                    "policy": "beam",
                    "branches": 2,
                    "budget_tokens": 32,
                    "fork_at_text": "</plan>",
                },
            )
        )

    key_a = scheduler.requests[0].extra_key
    key_b = scheduler.requests[1].extra_key
    tokens = array("q", [5, 6, 7, 8])
    cache.insert(
        InsertParams(
            key=RadixKey(tokens, key_a), value=torch.tensor([101, 102, 103, 104])
        )
    )
    assert len(
        cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens, key_b))).device_indices
    ) == 0
    assert len(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens, "tenant"))
        ).device_indices
    ) == 0

    cache.insert(
        InsertParams(
            key=RadixKey(tokens, key_b), value=torch.tensor([201, 202, 203, 204])
        )
    )
    assert key_a != key_b
    assert cache.match_prefix(
        MatchPrefixParams(key=RadixKey(tokens, key_a))
    ).device_indices.tolist() == [101, 102, 103, 104]
    assert cache.match_prefix(
        MatchPrefixParams(key=RadixKey(tokens, key_b))
    ).device_indices.tolist() == [201, 202, 203, 204]


def test_cache_unfinished_req_exposes_generated_prefix_to_child_prefill():
    allocator = MagicMock()
    allocator.device = torch.device("cpu")
    req_to_token = torch.zeros((2, 16), dtype=torch.int64)
    req_pool = SimpleNamespace(req_to_token=req_to_token)
    req_pool.write = lambda index, values: req_to_token.__setitem__(index, values)
    cache = RadixCache.create_simulated(mock_allocator=allocator, page_size=1)
    cache.req_to_token_pool = req_pool
    runtime, scheduler, _ = start_runtime(
        fork_at_text="</plan>",
        cache=cache,
        pieces={1: "prefix</", 2: "plan", 3: ">"},
    )
    parent = FakeReq(scheduler.requests[0], first_output=[1])
    parent.req_pool_idx = 0
    parent.cache_protected_len = 0
    parent.last_node = cache.root_node
    parent.prefix_indices = torch.empty(0, dtype=torch.int64)
    parent.priority = 0
    req_to_token[0, :5] = torch.tensor([110, 111, 201, 202, 203])

    runtime.on_prefill_done(parent)
    parent.append_decode_token(2)
    runtime.on_token(parent, [2], -0.2)
    parent.append_decode_token(3)
    runtime.on_token(parent, [3], -0.3)

    child = scheduler.requests[1]
    match = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(child.input_ids, child.extra_key))
    )
    assert match.device_indices.tolist() == [110, 111, 201, 202]
    assert len(match.device_indices) > len(parent.origin_input_ids)


def test_unsafe_cache_refuses_delimiter_fork_without_fallback():
    runtime, scheduler, _ = start_runtime(
        fork_at_text="</plan>",
        cache=UnsafeCache(),
        pieces={1: "</plan>"},
    )
    parent_base = scheduler.requests[0]
    parent = FakeReq(parent_base, first_output=[1])

    runtime.on_prefill_done(parent)

    run = runtime.runs[parent.rid]
    assert parent_base.extra_key == "tenant"
    assert run.fork_attempted is True
    assert run.forked is False
    assert len(scheduler.requests) == 1
