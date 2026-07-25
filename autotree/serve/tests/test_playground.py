from __future__ import annotations

import json
import re
from dataclasses import fields

from autotree_serve.engine import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    GenerationDone,
    TokenGenerated,
)
from conftest import MODEL_ID


PLAYGROUND_EVENT_SCHEMA = {
    BranchStarted: {"type", "branch_id", "parent_id"},
    TokenGenerated: {"type", "branch_id", "token", "token_index", "token_id"},
    BranchPruned: {"type", "branch_id", "reason"},
    BranchMerged: {"type", "branch_id", "into_branch_id"},
    GenerationDone: {
        "type",
        "branch_id",
        "text",
        "finish_reason",
        "usage",
        "counters",
        "tree_summary",
    },
}


def assert_playground_event_schema() -> None:
    for event_class, required_fields in PLAYGROUND_EVENT_SCHEMA.items():
        actual_fields = {field.name for field in fields(event_class)}
        assert required_fields <= actual_fields, (
            f"{event_class.__name__} lost playground fields: "
            f"{sorted(required_fields - actual_fields)}"
        )


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


async def test_playground_route_is_self_contained_and_csp_scoped(http_client):
    for path in ("/playground", "/playground/"):
        response = await http_client.get(path)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["content-security-policy"] == (
            "default-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; base-uri 'none'; form-action 'self'"
        )
        html = response.text
        assert re.search(r"https?://", html, flags=re.IGNORECASE) is None
        assert "<svg" in html
        assert 'fetch("/v1/models")' in html
        assert 'fetch("/v1/tree/completions"' in html
        assert "branch_started" in html
        assert "branch_pruned" in html
        assert "branch_merged" in html
        assert "kv_reuse_ratio" in html


async def test_playground_sse_contract_matches_event_schema(http_client):
    assert_playground_event_schema()

    response = await http_client.post(
        "/v1/tree/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "show the live tree"}],
            "max_tokens": 3,
            "seed": 17,
            "stream": True,
            "tree": {
                "policy": "beam",
                "branches": 4,
                "budget_tokens": 12,
                "scorer": None,
            },
        },
    )

    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    events = parse_sse(response.text)
    by_type: dict[str, list[dict[str, object]]] = {}
    for event_type, payload in events:
        by_type.setdefault(event_type, []).append(payload)

    assert {"branch_started", "token", "branch_pruned", "branch_merged", "done"} <= set(
        by_type
    )
    assert all({"type", "branch_id", "parent_id"} <= event.keys() for event in by_type["branch_started"])
    assert all(
        {"type", "branch_id", "token", "token_index", "token_id"} <= event.keys()
        for event in by_type["token"]
    )
    assert all({"type", "branch_id", "reason"} <= event.keys() for event in by_type["branch_pruned"])
    assert all(
        {"type", "branch_id", "into_branch_id"} <= event.keys()
        for event in by_type["branch_merged"]
    )

    token_indexes: dict[str, list[int]] = {}
    for event in by_type["token"]:
        token_indexes.setdefault(str(event["branch_id"]), []).append(int(event["token_index"]))
    assert all(indexes == list(range(len(indexes))) for indexes in token_indexes.values())

    done = by_type["done"][0]
    assert {"branch_id", "text", "finish_reason", "usage", "counters", "tree"} <= done.keys()
    assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= done["usage"].keys()
    assert {
        "logical_tokens",
        "physical_tokens",
        "useful_tokens",
        "elapsed_seconds",
        "ttft_seconds",
    } <= done["counters"].keys()
    assert {
        "branch_count",
        "pruned_count",
        "merged_count",
        "winner_branch_id",
        "tokens_spent_per_branch",
        "final_scores",
        "kv_reuse_ratio",
    } <= done["tree"].keys()
    started_ids = {str(event["branch_id"]) for event in by_type["branch_started"]}
    assert set(done["tree"]["final_scores"]) == started_ids
