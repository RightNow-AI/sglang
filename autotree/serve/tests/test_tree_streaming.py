from __future__ import annotations

import json

import pytest
import math

import httpx

from autotree_serve import create_app
from autotree_serve.engine import KVCapacityExceededError, ModelMetadata

from conftest import MODEL_ID


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for frame in body.strip().split("\n\n"):
        if frame == "data: [DONE]":
            continue
        lines = frame.splitlines()
        event_type = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((event_type, json.loads(data)))
    return events


async def test_tree_stream_terminal_events_and_usage_accounting(http_client):
    response = await http_client.post(
        "/v1/tree/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "search the reasoning tree"}],
            "max_tokens": 6,
            "seed": 19,
            "stream": True,
            "tree": {
                "policy": "best_first",
                "branches": 4,
                "budget_tokens": 17,
                "scorer": None,
            },
        },
    )

    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    events = parse_sse(response.text)
    event_types = [event_type for event_type, _ in events]
    assert event_types[-1] == "done"
    assert {"branch_started", "token", "branch_pruned", "branch_merged", "done"} <= set(
        event_types
    )

    started = {
        payload["branch_id"]
        for event_type, payload in events
        if event_type == "branch_started"
    }
    terminal = {
        payload["branch_id"]
        for event_type, payload in events
        if event_type in {"branch_pruned", "branch_merged", "done"}
    }
    assert terminal == started

    token_events = [payload for event_type, payload in events if event_type == "token"]
    assert all(math.isfinite(payload["logprob"]) for payload in token_events)
    assert all(
        payload["reason"]
        for event_type, payload in events
        if event_type == "branch_pruned"
    )
    done = events[-1][1]
    assert done["usage"]["completion_tokens"] == len(token_events)
    assert done["usage"]["total_tokens"] == (
        done["usage"]["prompt_tokens"] + len(token_events)
    )
    assert sum(done["tree"]["tokens_spent_per_branch"].values()) == len(token_events)


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


async def test_tree_capacity_error_stream_ends_with_done_sentinel():
    app = create_app(engine=CapacityErrorEngine())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/tree/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "reject this tree"}],
                "stream": True,
                "tree": {
                    "policy": "beam",
                    "branches": 2,
                    "budget_tokens": 2,
                },
            },
        )

    assert response.status_code == 200
    assert "event: error\n" in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    events = parse_sse(response.text)
    assert events[0][1]["retry_after_seconds"] == 1


async def test_tree_non_stream_capacity_error_has_retry_after_header():
    app = create_app(engine=CapacityErrorEngine())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/tree/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "reject this tree"}],
                "tree": {
                    "policy": "beam",
                    "branches": 2,
                    "budget_tokens": 2,
                },
            },
        )
        scrape = await client.get("/metrics")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    assert "capacity_rejections_total 1.0" in scrape.text


async def test_tree_non_stream_returns_winner_and_summary(http_client):
    response = await http_client.post(
        "/v1/tree/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "pick a branch"}],
            "max_tokens": 4,
            "tree": {
                "policy": "mcts",
                "branches": 3,
                "budget_tokens": 10,
                "scorer": "self_consistency",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"]
    assert body["tree"]["branch_count"] == 3
    assert len(body["tree"]["final_scores"]) == 3
    assert body["tree"]["winner_branch_id"] in body["tree"]["final_scores"]
    assert body["tree"]["scorer"] == "self_consistency"


@pytest.mark.parametrize("scorer", [None, "logprob", "self_consistency"])
async def test_tree_accepts_supported_scorers(http_client, scorer):
    response = await http_client.post(
        "/v1/tree/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "pick a branch"}],
            "tree": {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 3,
                "scorer": scorer,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["tree"]["scorer"] == scorer


async def test_tree_rejects_unknown_scorer(http_client):
    response = await http_client.post(
        "/v1/tree/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "reject scorer"}],
            "tree": {
                "policy": "beam",
                "branches": 2,
                "budget_tokens": 3,
                "scorer": "external",
            },
        },
    )

    assert 400 <= response.status_code < 500


async def test_chat_tree_stream_exposes_branch_events_as_chunk_extensions(http_client):
    response = await http_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "stream branches"}],
            "max_tokens": 3,
            "stream": True,
            "stream_options": {"include_usage": True},
            "tree": {
                "policy": "beam",
                "branches": 3,
                "budget_tokens": 8,
                "scorer": None,
            },
        },
    )

    frames = [frame for frame in response.text.strip().split("\n\n") if frame]
    assert frames[-1] == "data: [DONE]"
    chunks = [json.loads(frame[6:]) for frame in frames[:-1]]
    tree_events = [chunk["tree_event"] for chunk in chunks if "tree_event" in chunk]
    assert {event["type"] for event in tree_events} >= {
        "branch_started",
        "token",
        "branch_pruned",
        "branch_merged",
        "done",
    }
    assert chunks[-1]["usage"]["completion_tokens"] == len(
        [event for event in tree_events if event["type"] == "token"]
    )
