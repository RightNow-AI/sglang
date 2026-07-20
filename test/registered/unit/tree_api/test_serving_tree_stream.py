import json
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.serving_tree import OpenAIServingTree
from sglang.srt.tree import (
    TreeBranchEvent,
    TreeCounters,
    TreeResult,
    TreeSummary,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _result(counters=None):
    return TreeResult(
        winner_text="AB",
        winner_token_ids=[1, 2],
        prompt_tokens=4,
        completion_tokens=3,
        finish_reason="stop",
        counters=counters,
        summary=TreeSummary(
            policy="beam",
            branch_count=2,
            pruned_count=1,
            merged_count=0,
            winner_branch_id="b0",
            tokens_spent_per_branch={"b0": 2, "b1": 1},
            final_scores={"b0": -0.1, "b1": -0.5},
            scorer=None,
            kv_reuse_ratio=2.0,
        ),
    )


def _parse_sse(chunk):
    lines = chunk.rstrip().splitlines()
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    return lines[0][7:], json.loads(lines[1][6:])


@pytest.mark.asyncio
async def test_tree_stream_emits_branch_events_done_and_sentinel():
    scripted = [
        TreeBranchEvent(event="forked", branch_id="b0"),
        TreeBranchEvent(event="token", branch_id="b0", token_id=1, text="A", score=-0.1),
        TreeBranchEvent(event="forked", branch_id="b1", parent_id="b0"),
        TreeBranchEvent(event="token", branch_id="b1", token_id=None, text="X", score=-0.4),
        TreeBranchEvent(event="token", branch_id="b0", token_id=2, text="B", score=-0.2),
        TreeBranchEvent(event="pruned", branch_id="b1", reason="beam_pruned"),
        TreeBranchEvent(event="finalized", branch_id="b0"),
        _result(
            TreeCounters(
                logical_tokens=8,
                physical_tokens=4,
                useful_tokens=3,
                elapsed_seconds=0.5,
                ttft_seconds=0.1,
                unique_tokens_per_step=[2, 1],
                branch_tokens_per_step=[2, 1],
            )
        ),
    ]

    class StubTokenizerManager:
        server_args = SimpleNamespace(tokenizer_metrics_allowed_custom_labels=None)

        async def generate_request(self, adapted_request, raw_request):
            for item in scripted:
                yield item

    serving = OpenAIServingTree(StubTokenizerManager(), SimpleNamespace())
    chunks = [
        chunk
        async for chunk in serving._generate_tree_stream(
            SimpleNamespace(), SimpleNamespace()
        )
    ]

    assert chunks[-1] == "data: [DONE]\n\n"
    events = [_parse_sse(chunk) for chunk in chunks[:-1]]
    assert [name for name, _ in events] == [
        "branch_started",
        "token",
        "branch_started",
        "token",
        "token",
        "branch_pruned",
        "done",
    ]
    assert events[1][1] == {
        "type": "token",
        "branch_id": "b0",
        "token_index": 0,
        "token": "A",
        "token_id": 1,
        "logprob": -0.1,
    }
    assert events[3][1]["branch_id"] == "b1"
    assert events[3][1]["token_index"] == 0
    assert events[4][1]["branch_id"] == "b0"
    assert events[4][1]["token_index"] == 1
    assert events[-1][1]["branch_id"] == "b0"
    assert events[-1][1]["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }
    assert events[-1][1]["counters"]["physical_tokens"] == 4
    assert events[-1][1]["tree"]["tokens_spent_per_branch"] == {"b0": 2, "b1": 1}


@pytest.mark.asyncio
async def test_tree_stream_zeroes_missing_scheduler_counters():
    class StubTokenizerManager:
        server_args = SimpleNamespace(tokenizer_metrics_allowed_custom_labels=None)

        async def generate_request(self, adapted_request, raw_request):
            yield _result()

    serving = OpenAIServingTree(StubTokenizerManager(), SimpleNamespace())
    chunks = [
        chunk
        async for chunk in serving._generate_tree_stream(
            SimpleNamespace(), SimpleNamespace()
        )
    ]
    _, done = _parse_sse(chunks[0])

    assert done["counters"] == {
        "logical_tokens": 0,
        "physical_tokens": 0,
        "useful_tokens": 0,
        "elapsed_seconds": 0.0,
        "ttft_seconds": 0.0,
        "unique_tokens_per_step": [],
        "branch_tokens_per_step": [],
    }
