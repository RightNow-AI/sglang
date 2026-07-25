import json

import httpx
import pytest

from thoughtbench.arms import build_request_body, endpoint_url, execute_arm
from thoughtbench.bench_models import HarnessConfig


def _config(tmp_path, arm: str) -> HarnessConfig:
    payload = {
        "label": "test",
        "engine_label": "engine",
        "model": "model",
        "base_url": "http://endpoint.test/v1",
        "arm": arm,
        "task_file": tmp_path / "tasks.jsonl",
        "output_dir": tmp_path,
        "max_tokens": 64,
        "temperature": 0.25,
        "seeds": [7],
    }
    if arm == "best_of_n":
        payload["n"] = 3
    elif arm == "tree":
        payload.update({"branches": 4, "budget_tokens": 256, "policy": "beam"})
    return HarnessConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("arm", "extra"),
    [
        ("single", {}),
        ("best_of_n", {"n": 3}),
        (
            "tree",
            {"tree": {"policy": "beam", "branches": 4, "budget_tokens": 256}},
        ),
    ],
)
def test_request_payload_builders_are_exact(tmp_path, arm, extra) -> None:
    config = _config(tmp_path, arm)
    expected = {
        "model": "model",
        "messages": [{"role": "user", "content": "problem"}],
        "max_tokens": 64,
        "temperature": 0.25,
        "seed": 7,
        **extra,
    }

    assert build_request_body(config, "problem", 7) == expected
    expected_path = "/v1/tree/completions" if arm == "tree" else "/v1/chat/completions"
    assert endpoint_url(str(config.base_url), arm) == f"http://endpoint.test{expected_path}"


@pytest.mark.parametrize(
    ("arm", "response_payload", "expected_answer", "expected_tokens"),
    [
        (
            "single",
            {"choices": [{"message": {"content": "Answer: 9"}}], "usage": {"completion_tokens": 5}},
            "9",
            5,
        ),
        (
            "best_of_n",
            {
                "choices": [
                    {"message": {"content": "Answer: 9"}},
                    {"message": {"content": "Answer: 8"}},
                    {"message": {"content": "#### 9"}},
                ],
                "usage": {"completion_tokens": 15},
            },
            "9",
            15,
        ),
        (
            "tree",
            {
                "choices": [{"message": {"content": "Answer: 9"}}],
                "usage": {"completion_tokens": 99},
                "tree": {"tokens_spent_per_branch": {"a": 4, "b": 6}},
            },
            "9",
            10,
        ),
    ],
)
def test_execute_arm_uses_mocked_transport_and_measured_tokens(
    tmp_path, arm, response_payload, expected_answer, expected_tokens
) -> None:
    config = _config(tmp_path, arm)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json=response_payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = execute_arm(client, config, "problem", 7)

    assert seen == [(endpoint_url(str(config.base_url), arm), build_request_body(config, "problem", 7))]
    assert outcome.answer == expected_answer
    assert outcome.completion_tokens == expected_tokens


def test_tree_tokens_fall_back_to_usage_when_envelope_is_missing(tmp_path) -> None:
    config = _config(tmp_path, "tree")
    payload = {
        "choices": [{"message": {"content": "Answer: 9"}}],
        "usage": {"completion_tokens": 11},
        "tree": {"winner_branch_id": "a"},
    }
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
        assert execute_arm(client, config, "problem", 7).completion_tokens == 11
