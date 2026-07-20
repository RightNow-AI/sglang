from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.protocol_tree import TreeCompletionRequest
from sglang.srt.entrypoints.openai.serving_tree import OpenAIServingTree
from sglang.srt.tree import TreeResult, TreeSummary
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _request() -> TreeCompletionRequest:
    return TreeCompletionRequest.model_validate(
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "Solve"}],
            "tree": {
                "policy": "beam",
                "branches": 3,
                "budget_tokens": 30,
                "scorer": "mean-logprob",
            },
        }
    )


def _result() -> TreeResult:
    return TreeResult(
        winner_text="winning answer",
        winner_token_ids=[7, 8],
        prompt_tokens=5,
        completion_tokens=9,
        finish_reason="length",
        summary=TreeSummary(
            policy="beam",
            branch_count=3,
            pruned_count=2,
            merged_count=0,
            winner_branch_id="b0",
            tokens_spent_per_branch={"b0": 2, "b1": 3, "b2": 4},
            final_scores={"b0": -0.1, "b1": -0.4, "b2": -0.7},
            scorer="mean-logprob",
            kv_reuse_ratio=2.5,
        ),
    )


def _serving(tokenizer_manager=None) -> OpenAIServingTree:
    tokenizer_manager = tokenizer_manager or SimpleNamespace(
        server_args=SimpleNamespace(tokenizer_metrics_allowed_custom_labels=None)
    )
    return OpenAIServingTree(tokenizer_manager, SimpleNamespace())


def test_non_streaming_response_matches_wire_spec_field_by_field():
    response = _serving()._build_completion_response(
        _request(), _result(), created=123, response_id="chatcmpl-tree-test"
    )

    assert response.model_dump() == {
        "id": "chatcmpl-tree-test",
        "object": "chat.completion",
        "created": 123,
        "model": "served-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "winning answer"},
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 9,
            "total_tokens": 14,
        },
        "tree": {
            "policy": "beam",
            "branch_count": 3,
            "pruned_count": 2,
            "merged_count": 0,
            "winner_branch_id": "b0",
            "tokens_spent_per_branch": {"b0": 2, "b1": 3, "b2": 4},
            "final_scores": {"b0": -0.1, "b1": -0.4, "b2": -0.7},
            "scorer": "mean-logprob",
            "kv_reuse_ratio": 2.5,
        },
    }


@pytest.mark.asyncio
async def test_non_streaming_handler_collects_scripted_tree_result():
    result = _result()

    class StubTokenizerManager:
        server_args = SimpleNamespace(tokenizer_metrics_allowed_custom_labels=None)

        async def generate_request(self, adapted_request, raw_request):
            yield result

    response = await _serving(StubTokenizerManager())._handle_non_streaming_request(
        SimpleNamespace(), _request(), SimpleNamespace()
    )

    assert response.usage.completion_tokens == 9
    assert response.tree.tokens_spent_per_branch == {"b0": 2, "b1": 3, "b2": 4}


def test_non_streaming_response_rejects_missing_accounting():
    result = _result()
    result.summary.winner_branch_id = None

    with pytest.raises(ValueError, match="winner_branch_id"):
        _serving()._build_completion_response(_request(), result)
