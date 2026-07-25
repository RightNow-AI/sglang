from __future__ import annotations

from collections.abc import AsyncIterator
import math
from dataclasses import replace
from math import ceil

import httpx
import pytest
from openai import AsyncOpenAI, AsyncStream, BaseModel
from pydantic import ConfigDict

from autotree_serve import create_app


class TreeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class TreeEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str


@pytest.fixture(scope="module")
def treekv_app():
    pytest.importorskip("autotree_scheduler")
    from autotree_core.engine import TreeKVEngine

    return create_app(TreeKVEngine(model_id="gpt2"))


@pytest.fixture
async def treekv_openai_client(treekv_app) -> AsyncIterator[AsyncOpenAI]:
    transport = httpx.ASGITransport(app=treekv_app)
    async with httpx.AsyncClient(transport=transport) as raw_client:
        yield AsyncOpenAI(
            api_key="test-key",
            base_url="http://test/v1",
            http_client=raw_client,
        )


@pytest.fixture
async def treekv_http_client(treekv_app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=treekv_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        yield client


def limited_treekv_app(treekv_app, capacity_pages: int):
    from autotree_core.engine import TreeKVEngine
    from autotree_core.modeling import ModelExecutor

    base_engine = treekv_app.state.engine
    executor = ModelExecutor(
        replace(base_engine.executor.config, capacity_pages=capacity_pages),
        model=base_engine.executor.model,
    )
    return create_app(
        TreeKVEngine(
            model_id="gpt2",
            executor=executor,
            tokenizer=base_engine.tokenizer,
            scheduler_factory=base_engine._scheduler_factory,
        )
    )


def payload(*, stream: bool) -> dict[str, object]:
    return {
        "model": "gpt2",
        "messages": [{"role": "user", "content": "Name one color."}],
        "max_tokens": 4,
        "seed": 2026,
        "temperature": 0.0,
        "stream": stream,
        "tree": {
            "policy": "beam",
            "branches": 2,
            "budget_tokens": 6,
            "scorer": None,
        },
    }


async def test_official_openai_client_tree_completion_non_stream(treekv_openai_client) -> None:
    completion = await treekv_openai_client.post(
        "/tree/completions",
        body=payload(stream=False),
        cast_to=TreeResponse,
    )
    body = completion.model_dump()

    assert body["choices"][0]["message"]["content"]
    tree = body["tree"]
    assert tree["branch_count"] > 1
    assert tree["winner_branch_id"] in tree["final_scores"]
    assert set(tree["final_scores"]) == set(tree["tokens_spent_per_branch"])
    assert tree["kv_reuse_ratio"] > 1.0
    assert body["usage"]["completion_tokens"] == sum(
        tree["tokens_spent_per_branch"].values()
    )


async def test_official_openai_client_tree_completion_stream_usage(treekv_openai_client) -> None:
    stream = await treekv_openai_client.post(
        "/tree/completions",
        body=payload(stream=True),
        cast_to=TreeEvent,
        stream=True,
        stream_cls=AsyncStream[TreeEvent],
    )
    events = [event async for event in stream]

    assert events[-1].type == "done"
    raw_events = [event.model_dump() for event in events]
    tokens = [event for event in raw_events if event["type"] == "token"]
    assert all(math.isfinite(event["logprob"]) for event in tokens)
    assert all(
        event["reason"]
        for event in raw_events
        if event["type"] == "branch_pruned"
    )
    done = raw_events[-1]
    assert done["usage"]["completion_tokens"] == len(tokens)
    assert sum(done["tree"]["tokens_spent_per_branch"].values()) == len(tokens)
    assert done["tree"]["kv_reuse_ratio"] > 1.0


async def test_official_openai_client_same_seed_is_deterministic(
    treekv_openai_client,
) -> None:
    first = await treekv_openai_client.post(
        "/tree/completions",
        body=payload(stream=False),
        cast_to=TreeResponse,
    )
    second = await treekv_openai_client.post(
        "/tree/completions",
        body=payload(stream=False),
        cast_to=TreeResponse,
    )

    first_body = first.model_dump()
    second_body = second.model_dump()
    assert first_body["choices"][0]["message"]["content"] == second_body["choices"][0][
        "message"
    ]["content"]
    assert first_body["tree"]["winner_branch_id"] == second_body["tree"][
        "winner_branch_id"
    ]
    assert first_body["tree"]["final_scores"] == second_body["tree"]["final_scores"]


@pytest.mark.parametrize("policy", ["beam", "best_first", "mcts"])
async def test_default_capacity_handles_live_policy_stress_reproduction(
    treekv_app,
    treekv_http_client: httpx.AsyncClient,
    policy: str,
) -> None:
    executor = treekv_app.state.engine.executor
    context_tokens = int(executor.model.config.n_positions)
    expected_pages = ceil(
        context_tokens / executor.config.page_size * 1.5,
    )
    assert executor.config.capacity_pages == expected_pages

    response = await treekv_http_client.post(
        "/v1/tree/completions",
        json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "Count: one two"}],
            "max_completion_tokens": 6,
            "tree": {
                "policy": policy,
                "branches": 3,
                "budget_tokens": 24,
            },
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["tree"]["policy"] == policy


async def test_default_capacity_handles_long_prompt_reproduction(
    treekv_http_client: httpx.AsyncClient,
) -> None:
    response = await treekv_http_client.post(
        "/v1/tree/completions",
        json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "word " * 400}],
            "max_completion_tokens": 8,
            "temperature": 0.0,
            "tree": {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 16,
            },
        },
    )

    assert response.status_code == 200, response.text


async def test_explicit_low_capacity_returns_typed_openai_error(treekv_app) -> None:
    app = limited_treekv_app(treekv_app, capacity_pages=64)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/tree/completions",
            json={
                "model": "gpt2",
                "messages": [{"role": "user", "content": "word " * 400}],
                "max_completion_tokens": 8,
                "tree": {
                    "policy": "beam",
                    "branches": 2,
                    "budget_tokens": 16,
                },
            },
        )

    assert response.status_code == 429
    error = response.json()["error"]
    assert error["type"] == "rate_limit_error"
    assert error["code"] == "kv_capacity_exhausted"
    assert error["param"] == "kv_pages"
    assert "102 page(s)" in error["message"]
    assert "--kv-pages" in error["message"]


async def test_mid_decode_exhaustion_emits_error_event_and_closes(treekv_app) -> None:
    base_engine = treekv_app.state.engine
    prompt_ids = base_engine.tokenizer.encode(
        "user: Count: one two\nassistant:",
        add_special_tokens=True,
    )
    prompt_pages = ceil(len(prompt_ids) / base_engine.executor.config.page_size)
    app = limited_treekv_app(treekv_app, capacity_pages=prompt_pages + 1)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/tree/completions",
            json={
                "model": "gpt2",
                "messages": [{"role": "user", "content": "Count: one two"}],
                "max_completion_tokens": 16,
                "temperature": 0.0,
                "stream": True,
                "tree": {
                    "policy": "beam",
                    "branches": 1,
                    "budget_tokens": 16,
                },
            },
        )

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: error" in response.text
    assert '"code":"kv_capacity_exhausted"' in response.text
    assert "--kv-pages" in response.text
