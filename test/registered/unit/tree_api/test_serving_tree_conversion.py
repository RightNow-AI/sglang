from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol_tree import TreeCompletionRequest
from sglang.srt.entrypoints.openai.serving_tree import OpenAIServingTree
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.tree import TreeGenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class StubChatServing:
    def __init__(self) -> None:
        self.request = None
        self.raw_request = None

    def _convert_to_internal_request(self, request, raw_request=None):
        self.request = request
        self.raw_request = raw_request
        return (
            GenerateReqInput(
                input_ids=[11, 12, 13],
                sampling_params={
                    "max_new_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "stop": request.stop,
                    "sampling_seed": request.seed,
                    "n": request.n,
                },
                stream=request.stream,
            ),
            request,
        )


def _serving(chat_serving: StubChatServing) -> OpenAIServingTree:
    tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(tokenizer_metrics_allowed_custom_labels=None)
    )
    return OpenAIServingTree(tokenizer_manager, chat_serving)


def test_convert_tree_request_to_internal_envelope():
    request = TreeCompletionRequest.model_validate(
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "Choose carefully"}],
            "max_completion_tokens": 37,
            "max_tokens": 12,
            "temperature": 0.25,
            "top_p": 0.8,
            "stop": ["END"],
            "seed": 9,
            "stream": True,
            "tree": {
                "policy": "best_first",
                "branches": 5,
                "budget_tokens": 321,
                "scorer": "reward-v1",
            },
        }
    )
    chat_serving = StubChatServing()

    adapted, processed = _serving(chat_serving)._convert_to_internal_request(request)

    assert processed is request
    assert isinstance(adapted, TreeGenerateReqInput)
    assert adapted.tree.policy == "best_first"
    assert adapted.tree.branches == 5
    assert adapted.tree.budget_tokens == 321
    assert adapted.tree.scorer == "reward-v1"
    assert adapted.base.input_ids == [11, 12, 13]
    assert adapted.base.stream is True
    assert adapted.base.return_logprob is True
    assert adapted.base.return_text_in_logprobs is True
    assert adapted.base.sampling_params == {
        "max_new_tokens": 37,
        "temperature": 0.25,
        "top_p": 0.8,
        "stop": ["END"],
        "sampling_seed": 9,
        "n": 1,
    }
    assert chat_serving.request.messages[0].role == "user"
    assert chat_serving.request.messages[0].content == "Choose carefully"


def test_convert_resolves_wire_defaults():
    request = TreeCompletionRequest.model_validate(
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tree": {"policy": "beam", "branches": 1, "budget_tokens": 1},
        }
    )
    chat_serving = StubChatServing()

    adapted, _ = _serving(chat_serving)._convert_to_internal_request(request)

    assert adapted.base.sampling_params["max_new_tokens"] == 16
    assert adapted.base.sampling_params["sampling_seed"] == 0
