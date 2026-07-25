from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from autotree_serve import create_app
from autotree_serve.engine import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    EngineCounters,
    EngineEvent,
    EngineUsage,
    GenerationDone,
    GenerationRequest,
    ModelMetadata,
    TokenGenerated,
    TreeSummary,
)


MODEL_ID = "contract-test-model"


class ScriptedEngine:
    def __init__(self, events: list[EngineEvent]) -> None:
        self.events = events
        self.model_metadata = ModelMetadata(
            id=MODEL_ID,
            engine="scripted",
            description="Scripted engine for event-contract tests.",
            real_model_weights=False,
            tree_policies=("beam",),
        )

    async def generate(self, _request: GenerationRequest) -> AsyncIterator[EngineEvent]:
        for event in self.events:
            yield event


def done_event(
    *,
    branch_id: str,
    text: str,
    completion_tokens: int,
    tree_summary: TreeSummary | None = None,
) -> GenerationDone:
    return GenerationDone(
        branch_id=branch_id,
        text=text,
        finish_reason="length",
        usage=EngineUsage(prompt_tokens=1, completion_tokens=completion_tokens),
        counters=EngineCounters(
            logical_tokens=completion_tokens + 1,
            physical_tokens=completion_tokens + 1,
            useful_tokens=completion_tokens,
            elapsed_seconds=0.01,
            ttft_seconds=0.001,
        ),
        tree_summary=tree_summary,
    )


async def post_script(events: list[EngineEvent]) -> httpx.Response:
    app = create_app(engine=ScriptedEngine(events))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "validate events"}],
            },
        )


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param([0, 0], id="duplicate"),
        pytest.param([0, 2, 1], id="out-of-order"),
    ],
)
async def test_non_monotonic_token_indices_return_honest_500(indices: list[int]):
    tokens = ["a", "b", "c"][: len(indices)]
    events: list[EngineEvent] = [BranchStarted(branch_id="branch-0", parent_id=None)]
    events.extend(
        TokenGenerated(
            branch_id="branch-0",
            token=token,
            token_index=index,
            logprob=-0.1,
        )
        for token, index in zip(tokens, indices, strict=True)
    )
    events.append(
        done_event(
            branch_id="branch-0",
            text="".join(tokens),
            completion_tokens=len(tokens),
        )
    )

    response = await post_script(events)

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "engine_contract_error"
    assert "token_index" in error["message"]


async def test_tree_summary_branch_token_mismatch_returns_honest_500():
    summary = TreeSummary(
        policy="beam",
        branch_count=1,
        pruned_count=0,
        merged_count=0,
        winner_branch_id="branch-0",
        tokens_spent_per_branch={"branch-0": 2},
        final_scores={"branch-0": 1.0},
        scorer=None,
    )
    response = await post_script(
        [
            BranchStarted(branch_id="branch-0", parent_id=None),
            TokenGenerated(
                branch_id="branch-0", token="a", token_index=0, logprob=-0.1
            ),
            done_event(
                branch_id="branch-0",
                text="a",
                completion_tokens=1,
                tree_summary=summary,
            ),
        ]
    )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "engine_contract_error"
    assert "tokens_spent_per_branch" in error["message"]


async def test_tree_summary_final_scores_are_keyed_by_every_branch_id():
    summary = TreeSummary(
        policy="beam",
        branch_count=1,
        pruned_count=0,
        merged_count=0,
        winner_branch_id="branch-0",
        tokens_spent_per_branch={"branch-0": 1},
        final_scores={},
        scorer=None,
    )
    response = await post_script(
        [
            BranchStarted(branch_id="branch-0", parent_id=None),
            TokenGenerated(
                branch_id="branch-0", token="a", token_index=0, logprob=-0.1
            ),
            done_event(
                branch_id="branch-0",
                text="a",
                completion_tokens=1,
                tree_summary=summary,
            ),
        ]
    )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "engine_contract_error"
    assert "final_scores" in error["message"]


async def test_tree_summary_terminal_counts_reconcile_with_events():
    summary = TreeSummary(
        policy="beam",
        branch_count=2,
        pruned_count=0,
        merged_count=0,
        winner_branch_id="branch-0",
        tokens_spent_per_branch={"branch-0": 0, "branch-1": 0},
        final_scores={"branch-0": 1.0, "branch-1": 0.0},
        scorer=None,
    )
    response = await post_script(
        [
            BranchStarted(branch_id="branch-0", parent_id=None),
            BranchStarted(branch_id="branch-1", parent_id="branch-0"),
            BranchPruned(branch_id="branch-1", reason="lower_score"),
            done_event(
                branch_id="branch-0",
                text="",
                completion_tokens=0,
                tree_summary=summary,
            ),
        ]
    )

    assert response.status_code == 500
    assert "pruned_count" in response.json()["error"]["message"]


async def test_tree_summary_rejects_non_finite_final_scores():
    summary = TreeSummary(
        policy="beam",
        branch_count=1,
        pruned_count=0,
        merged_count=0,
        winner_branch_id="branch-0",
        tokens_spent_per_branch={"branch-0": 0},
        final_scores={"branch-0": float("nan")},
        scorer=None,
    )
    response = await post_script(
        [
            BranchStarted(branch_id="branch-0", parent_id=None),
            done_event(
                branch_id="branch-0",
                text="",
                completion_tokens=0,
                tree_summary=summary,
            ),
        ]
    )

    assert response.status_code == 500
    assert "final_scores" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "events, invalid_relationship",
    [
        pytest.param(
            [
                BranchStarted(branch_id="child", parent_id="missing"),
                TokenGenerated(
                    branch_id="child", token="a", token_index=0, logprob=-0.1
                ),
                done_event(branch_id="child", text="a", completion_tokens=1),
            ],
            "parent_id",
            id="unknown-parent",
        ),
        pytest.param(
            [
                BranchStarted(branch_id="branch-0", parent_id=None),
                BranchStarted(branch_id="branch-1", parent_id="branch-0"),
                BranchMerged(branch_id="branch-1", into_branch_id="missing"),
                done_event(branch_id="branch-0", text="", completion_tokens=0),
            ],
            "into_branch_id",
            id="unknown-merge-target",
        ),
        pytest.param(
            [
                BranchStarted(branch_id="branch-0", parent_id=None),
                BranchStarted(branch_id="branch-1", parent_id="branch-0"),
                BranchStarted(branch_id="branch-2", parent_id="branch-0"),
                BranchPruned(branch_id="branch-0", reason="lower_score"),
                BranchMerged(branch_id="branch-1", into_branch_id="branch-0"),
                done_event(branch_id="branch-2", text="", completion_tokens=0),
            ],
            "into_branch_id",
            id="terminated-merge-target",
        ),
    ],
)
async def test_invalid_branch_relationships_return_honest_500(
    events: list[EngineEvent],
    invalid_relationship: str,
):
    response = await post_script(events)

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "engine_contract_error"
    assert invalid_relationship in error["message"]
