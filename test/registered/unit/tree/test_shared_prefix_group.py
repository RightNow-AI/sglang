import ast
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import msgspec

from sglang.srt.tree.shared_prefix import SharedPrefixGroup
from sglang.srt.tree.tree_runtime import (
    SchedulerTreeRuntime,
    TokenizedTreeGenerateReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeSampling:
    def __init__(self):
        self.max_new_tokens = 32
        self.ignore_eos = False
        self.seed = 7


class FakeTokenized(msgspec.Struct):
    rid: str
    input_ids: Any
    sampling_params: Any
    extra_key: Optional[str] = None


class FakeTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.pieces.get(int(token_id), "") for token_id in token_ids)


class FakeCache:
    supports_tree_fork_namespaces = True
    disable = False

    def cache_unfinished_req(self, req):
        return None


class FakeScheduler:
    def __init__(self, pieces):
        self.tokenizer = FakeTokenizer(pieces)
        self.tree_cache = FakeCache()
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

    def _refresh_fill_ids(self):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def set_extend_range(self, start, end):
        self.extend_range = SimpleNamespace(start=start, end=end)

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_range.end]

    def append_decode_token(self, token_id):
        self.output_ids.append(token_id)
        self.kv_committed_len = len(self.origin_input_ids) + len(self.output_ids) - 1


def start_runtime(*, fork_at_text, pieces=None):
    scheduler = FakeScheduler(pieces or {})
    runtime = SchedulerTreeRuntime(scheduler)
    base = FakeTokenized(
        rid="parent",
        input_ids=array("q", [10, 11]),
        sampling_params=FakeSampling(),
        extra_key="tenant",
    )
    runtime.handle_tree_request(
        TokenizedTreeGenerateReqInput(
            base,
            {
                "policy": "beam",
                "branches": 3,
                "budget_tokens": 100,
                "scorer": None,
                "fork_at_text": fork_at_text,
            },
        )
    )
    return runtime, scheduler


def test_prefill_fork_records_prompt_shared_prefix_for_all_member_rids():
    runtime, scheduler = start_runtime(fork_at_text=None)
    parent = FakeReq(scheduler.requests[0], first_output=[1])

    runtime.on_prefill_done(parent)

    expected = SharedPrefixGroup(
        rids=["parent", "parent#tree1", "parent#tree2"],
        shared_len=2,
        branch_ids=[0, 1, 2],
    )
    group = runtime.get_shared_prefix_group("parent")
    assert group == expected
    assert runtime.get_shared_prefix_group("parent#tree1") is group
    assert runtime.get_shared_prefix_group("parent#tree2") is group
    assert runtime.get_shared_prefix_group("unrelated") is None


def test_delimiter_fork_records_prompt_plus_generated_shared_prefix():
    runtime, scheduler = start_runtime(
        fork_at_text="</plan>", pieces={1: "prefix</", 2: "plan", 3: ">"}
    )
    parent = FakeReq(scheduler.requests[0], first_output=[1])

    runtime.on_prefill_done(parent)
    parent.append_decode_token(2)
    runtime.on_token(parent, [2], -0.2)
    parent.append_decode_token(3)
    runtime.on_token(parent, [3], -0.3)

    assert runtime.get_shared_prefix_group("parent") == SharedPrefixGroup(
        rids=["parent", "parent#tree1", "parent#tree2"],
        shared_len=5,
        branch_ids=[0, 1, 2],
    )


def test_forward_batch_shared_prefix_groups_default_is_none():
    source_path = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "model_executor"
        / "forward_batch_info.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    forward_batch = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ForwardBatch"
    )
    field = next(
        node
        for node in forward_batch.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "shared_prefix_groups"
    )

    assert isinstance(field.value, ast.Constant)
    assert field.value.value is None
