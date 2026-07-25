from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI

from autotree_serve import create_app
from autotree_serve.engine import KVCapacityExceededError, ModelMetadata

from conftest import MODEL_ID


class CapacityErrorEngine:
    model_metadata = ModelMetadata(
        id=MODEL_ID,
        engine="deterministic",
        description="Test engine that rejects generation for capacity.",
        real_model_weights=False,
        tree_policies=("beam", "best_first", "mcts"),
    )

    async def generate(self, _request):
        if False:
            yield
        raise KVCapacityExceededError(
            phase="admission",
            required_pages=2,
            available_pages=1,
            capacity_pages=1,
        )


async def test_official_openai_client_non_stream_needs_only_base_url(openai_client):
    completion = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "explain tree reuse"}],
        max_tokens=6,
        seed=7,
    )

    assert completion.object == "chat.completion"
    assert completion.model == MODEL_ID
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content
    assert completion.usage is not None
    assert completion.usage.completion_tokens == 6
    assert completion.usage.total_tokens == (
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    )


async def test_official_openai_client_stream_chunks_and_usage(openai_client):
    stream = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "stream a deterministic answer"}],
        max_tokens=5,
        seed=11,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [chunk async for chunk in stream]

    assert chunks
    assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0].choices[0].delta.role == "assistant"
    content = "".join(
        choice.delta.content or ""
        for chunk in chunks
        for choice in chunk.choices
    )
    assert content
    assert any(
        choice.finish_reason == "length"
        for chunk in chunks
        for choice in chunk.choices
    )
    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage.completion_tokens == 5


@pytest.mark.parametrize(
    "extra_body",
    [
        None,
        {
            "tree": {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 2,
            }
        },
    ],
)
async def test_official_openai_client_surfaces_stream_capacity_error(extra_body):
    app = create_app(engine=CapacityErrorEngine())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as raw_client:
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="http://test/v1",
            http_client=raw_client,
        )
        stream = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "over capacity"}],
            stream=True,
            extra_body=extra_body,
        )
        chunks = [chunk async for chunk in stream]

    error_chunks = [
        (chunk, choice)
        for chunk in chunks
        for choice in chunk.choices
        if choice.delta.model_extra.get("error")
    ]
    assert len(error_chunks) == 1
    error_chunk, error_choice = error_chunks[0]
    assert error_chunk.object == "chat.completion.chunk"
    assert error_choice.finish_reason == "length"
    assert error_choice.delta.model_extra["error"]["code"] == "kv_capacity_exhausted"


async def test_stream_usage_chunk_requires_include_usage_true(http_client):
    for stream_options in (None, {"include_usage": False}):
        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "omit stream usage"}],
            "max_tokens": 2,
            "stream": True,
        }
        if stream_options is not None:
            payload["stream_options"] = stream_options

        response = await http_client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        frames = [frame for frame in response.text.strip().split("\n\n") if frame]
        chunks = [json.loads(frame[6:]) for frame in frames if frame != "data: [DONE]"]
        assert all("usage" not in chunk for chunk in chunks)


async def test_tree_extra_body_is_accepted_and_reflected(openai_client):
    completion = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "compare candidate paths"}],
        max_tokens=5,
        seed=3,
        extra_body={
            "tree": {
                "policy": "beam",
                "branches": 4,
                "budget_tokens": 13,
                "scorer": "self_consistency",
            }
        },
    )

    tree = completion.model_extra["tree"]
    assert tree["policy"] == "beam"
    assert tree["branch_count"] == 4
    assert tree["scorer"] == "self_consistency"
    assert sum(tree["tokens_spent_per_branch"].values()) == 13
    assert completion.usage.completion_tokens == 13


async def test_models_are_honest_about_deterministic_toy_engine(http_client):
    response = await http_client.get("/v1/models")

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["id"] == MODEL_ID
    assert model["metadata"]["engine"] == "deterministic"
    assert model["metadata"]["real_model_weights"] is False
    assert "toy generator" in model["metadata"]["description"]


async def test_current_chat_fields_are_accepted_and_max_completion_tokens_wins(http_client):
    response = await http_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "use current OpenAI fields"}],
            "max_tokens": 7,
            "max_completion_tokens": 3,
            "top_p": 0.8,
            "temperature": 0.4,
            "stop": ["never-produced-stop"],
            "n": 1,
            "seed": 23,
            "stream_options": {"include_usage": False},
            "user": "compat-test-user",
            "harmless_client_metadata": {"trace_id": "abc123"},
        },
    )

    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 3
