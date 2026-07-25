import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.responses import StreamingResponse

from sglang.srt.entrypoints.openai.serving_tree import OpenAIServingTree
from sglang.srt.tree import TreeGenerateReqInput, TreeParams
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def parse_sse(chunk: str):
    lines = chunk.rstrip().splitlines()
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    return lines[0][7:], json.loads(lines[1][6:])


def runtime_snapshot():
    return {
        "policy": "beam",
        "branch_count": 2,
        "alive_count": 0,
        "pruned_count": 1,
        "spent_tokens": 3,
        "budget_tokens": 8,
        "winner_branch_id": "0",
        "winner_is_final": True,
        "branches": {
            "0": {
                "tokens": 2,
                "mean_logprob": -0.15,
                "state": "finalized",
                "output_ids": [1, 2],
            },
            "1": {
                "tokens": 1,
                "mean_logprob": -0.4,
                "state": "pruned",
                "output_ids": [3],
            },
        },
        "events": [
            {"event": "forked", "branch_id": "0"},
            {
                "event": "token",
                "branch_id": "0",
                "token_id": 1,
                "text": "A",
                "score": -0.1,
            },
            {"event": "forked", "branch_id": "1", "parent_id": "0"},
            {
                "event": "token",
                "branch_id": "1",
                "token_id": 3,
                "text": "X",
                "score": -0.4,
            },
            {
                "event": "token",
                "branch_id": "0",
                "token_id": 2,
                "text": "B",
                "score": -0.2,
            },
            {
                "event": "pruned",
                "branch_id": "1",
                "reason": "consensus",
            },
            {"event": "finalized", "branch_id": "0"},
        ],
    }


def final_plain_result():
    return {
        "text": "AB",
        "output_ids": [1, 2],
        "meta_info": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "finish_reason": {"type": "stop"},
            "output_ids": [1, 2],
            "autotree": [runtime_snapshot()],
        },
    }


class PlainRuntimeTokenizerManager:
    server_args = SimpleNamespace(tokenizer_metrics_allowed_custom_labels=None)
    tokenizer = None

    async def generate_request(self, adapted_request, raw_request):
        yield final_plain_result()

    def create_abort_task(self, request):
        return None


def make_serving(manager=None):
    serving = OpenAIServingTree.__new__(OpenAIServingTree)
    serving.tokenizer_manager = manager or PlainRuntimeTokenizerManager()
    serving.chat_serving = SimpleNamespace()
    serving._memo_store = None
    return serving


def make_adapted(rid="tree-rid"):
    return TreeGenerateReqInput(
        base=SimpleNamespace(rid=rid),
        tree=TreeParams(policy="beam", branches=2, budget_tokens=8),
    )


@pytest.mark.asyncio
async def test_real_runtime_plain_stream_emits_events_and_done_end_to_end():
    serving = make_serving()
    response = await serving._handle_streaming_request(
        make_adapted(),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert isinstance(response, StreamingResponse)
    chunks = [chunk async for chunk in response.body_iterator]
    assert chunks[-1] == "data: [DONE]\n\n"
    events = [parse_sse(chunk) for chunk in chunks[:-1]]
    assert [name for name, _ in events] == [
        "branch_started",
        "token",
        "branch_started",
        "token",
        "token",
        "branch_pruned",
        "done",
    ]
    assert events[-1][1]["text"] == "AB"
    assert events[-1][1]["tree"]["winner_branch_id"] == "0"
    assert events[-1][1]["tree"]["tokens_spent_per_branch"] == {"0": 2, "1": 1}
    assert events[-1][1]["usage"]["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_disconnect_aborts_parent_and_clears_live_run():
    class DisconnectingTokenizerManager:
        server_args = SimpleNamespace(tokenizer_metrics_allowed_custom_labels=None)
        tokenizer = None

        def __init__(self):
            self.live_runs = {"tree-rid"}
            self.aborted = []

        async def generate_request(self, adapted_request, raw_request):
            yield {
                "text": "A",
                "output_ids": [1],
                "meta_info": {
                    "finish_reason": None,
                    "autotree": [
                        {
                            **runtime_snapshot(),
                            "events": runtime_snapshot()["events"][:2],
                        }
                    ],
                },
            }
            await asyncio.Event().wait()

        def abort_request(self, rid):
            self.aborted.append(rid)
            self.live_runs.discard(rid)

    manager = DisconnectingTokenizerManager()
    stream = make_serving(manager)._generate_tree_stream(
        make_adapted(),
        SimpleNamespace(),
    )

    first = await stream.__anext__()
    assert parse_sse(first)[0] == "branch_started"
    await stream.aclose()

    assert manager.aborted == ["tree-rid"]
    assert manager.live_runs == set()
