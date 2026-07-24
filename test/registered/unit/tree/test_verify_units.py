import asyncio
import math
import sys
import types
from types import SimpleNamespace


if "sglang.srt.managers.io_struct" not in sys.modules:
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class GenerateReqInput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    io_struct.GenerateReqInput = GenerateReqInput
    sys.modules[io_struct.__name__] = io_struct


from sglang.srt.tree.verify import (  # noqa: E402
    _build_batched_request,
    _score_result,
    verify_branches,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        token_ids = {
            "prompt": [10, 11, 12],
            "one": [20, 21],
            "two": [30],
            "three": [40, 41, 42],
        }.get(text, [])
        if add_special_tokens and text != "prompt":
            raise AssertionError("special tokens are only expected for the prompt")
        return token_ids


def test_continuation_span_scores_exact_tokens():
    manager = SimpleNamespace(tokenizer=FakeTokenizer())

    request, prepared, scores = _build_batched_request(manager, "prompt", ["one"])

    assert request is not None
    assert scores[0]["error"] == "verification did not complete"
    assert request.input_ids == [[10, 11, 12, 20, 21]]
    assert request.logprob_start_len == [2]
    assert prepared[0].response_offset == 1
    assert request.input_ids[0][
        request.logprob_start_len[0] + prepared[0].response_offset :
    ] == [20, 21]

    score = _score_result(
        {
            "meta_info": {
                "input_token_logprobs": [
                    (None, 12, None),
                    (-0.2, 20, None),
                    (-0.4, 21, None),
                ]
            }
        },
        prepared[0],
    )
    assert math.isclose(score["mean_logprob"], -0.3)
    assert math.isclose(score["sum_logprob"], -0.6)
    assert score["n_scored_tokens"] == 2
    assert score["error"] is None


def test_malformed_and_empty_continuations_return_errors_without_calling_model():
    class NoCallManager:
        tokenizer = FakeTokenizer()

        def generate_request(self, *_args, **_kwargs):
            raise AssertionError("no model request should be made")

    scores = asyncio.run(verify_branches(NoCallManager(), "prompt", ["", None]))

    assert len(scores) == 2
    assert all(score["mean_logprob"] is None for score in scores)
    assert all(score["sum_logprob"] is None for score in scores)
    assert all(score["n_scored_tokens"] == 0 for score in scores)
    assert all(score["error"] for score in scores)


def test_multiple_continuations_build_one_batched_request_object():
    manager = SimpleNamespace(tokenizer=FakeTokenizer())

    request, prepared, _scores = _build_batched_request(
        manager, "prompt", ["one", "two", "three"]
    )

    assert request is not None
    assert len(prepared) == 3
    assert request.input_ids == [
        [10, 11, 12, 20, 21],
        [10, 11, 12, 30],
        [10, 11, 12, 40, 41, 42],
    ]
    assert request.sampling_params == [{"max_new_tokens": 0}] * 3
    assert request.return_logprob == [True, True, True]
    assert request.logprob_start_len == [2, 2, 2]
