from __future__ import annotations

import pytest

from autotree_sdk import (
    RolloutBatch,
    RolloutBranch,
    RolloutTree,
    TreeClient,
    TreeSummary,
    Usage,
    SSEParseError,
    rollout,
)

from .mock_asgi import MockAutoTreeASGI, make_http_client


def test_rollout_end_to_end_and_rl_exports() -> None:
    app = MockAutoTreeASGI()
    http_client = make_http_client(app)
    client = TreeClient("http://autotree.test", http_client=http_client)
    try:
        batch = rollout(
            ["first", [{"role": "user", "content": "second"}]],
            2,
            policy="best_first",
            budget_tokens=64,
            seed=7,
            model="test-model",
            client=client,
        )
    finally:
        http_client.close()

    assert len(batch.trees) == 2
    root = batch.trees[0].branch("root")
    alt = batch.trees[0].branch("alt")
    assert root.completion == "answer"
    assert root.token_ids == [101, 102]
    assert root.token_logprobs == [-0.1, -0.2]
    assert root.cumulative_logprob == -0.30000000000000004
    assert alt.branch_path == ["root", "alt"]
    assert alt.pruned is True

    grpo = batch.to_grpo_samples()
    assert len(grpo) == 4
    assert grpo[0]["prompt"] == "first"
    assert grpo[0]["completion"] == "answer"
    assert grpo[0]["branch_path"] == ["root"]
    assert grpo[0]["pruned"] is False
    assert any(sample["pruned"] for sample in grpo)

    filtered = batch.to_grpo_samples(include_pruned=False)
    assert len(filtered) == 2
    assert all(not sample["pruned"] for sample in filtered)

    pairs = batch.to_rlhf_pairs()
    assert len(pairs) == 2
    assert pairs[0]["chosen"]["branch_id"] == "root"
    assert pairs[0]["rejected"]["branch_id"] == "alt"
    assert pairs[0]["chosen_score"] == 0.9
    assert app.requests[0]["body"]["tree"] == {
        "policy": "best_first",
        "branches": 2,
        "budget_tokens": 64,
        "scorer": None,
    }
    assert app.requests[0]["body"]["seed"] == 7


def test_stream_parser_rejects_positional_final_scores() -> None:
    app = MockAutoTreeASGI()
    http_client = make_http_client(app)
    client = TreeClient("http://autotree.test", http_client=http_client)
    try:
        with pytest.raises(SSEParseError, match="invalid_tree_event"):
            rollout(
                ["prompt"],
                2,
                client=client,
                scenario="positional_scores",
            )
    finally:
        http_client.close()


def test_exports_reconstruct_forked_branch_full_path_completion() -> None:
    root = RolloutBranch(
        branch_id="root",
        parent_id=None,
        branch_path=["root"],
        tokens=["shared "],
        token_ids=[None],
        token_logprobs=[-0.1],
        token_indices=[0],
        status="completed",
    )
    leaf = RolloutBranch(
        branch_id="leaf",
        parent_id="root",
        branch_path=["root", "leaf"],
        tokens=["completion"],
        token_ids=[None],
        token_logprobs=[-0.2],
        token_indices=[0],
        status="completed",
    )
    batch = RolloutBatch(
        trees=[
            RolloutTree(
                prompt="prompt",
                branches=[root, leaf],
                usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                tree_summary=TreeSummary(
                    policy="beam",
                    branch_count=2,
                    pruned_count=0,
                    merged_count=0,
                    winner_branch_id="leaf",
                    tokens_spent_per_branch={"root": 1, "leaf": 1},
                    final_scores={"root": 0.1, "leaf": 0.9},
                    scorer=None,
                    kv_reuse_ratio=1.0,
                ),
            )
        ]
    )

    grpo_leaf = next(
        sample for sample in batch.to_grpo_samples() if sample["branch_id"] == "leaf"
    )
    pair = batch.to_rlhf_pairs()[0]

    assert grpo_leaf["completion"] == "shared completion"
    assert grpo_leaf["token_logprobs"] == [-0.1, -0.2]
    assert pair["chosen"]["completion"] == "shared completion"
