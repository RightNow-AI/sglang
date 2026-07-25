"""High-level RL rollout entry point."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .client import TreeClient
from .models import Prompt, RolloutBatch, TreeParameters, TreePolicy
from .trace import TraceAssembler


def rollout(
    prompts: Sequence[Prompt],
    k: int,
    policy: TreePolicy = "beam",
    budget_tokens: int = 4096,
    scorer: str | None = None,
    seed: int | None = None,
    base_url: str = "http://localhost:8000",
    *,
    model: str | None = None,
    client: TreeClient | None = None,
    **parameters: Any,
) -> RolloutBatch:
    """Generate and reconstruct one branching trace per prompt.

    Prompts may be plain strings or OpenAI-style chat-message lists. Requests
    are issued sequentially and are never silently retried, preserving seed and
    accounting semantics for post-training data collection.
    """

    tree_parameters = TreeParameters(
        policy=policy,
        branches=k,
        budget_tokens=budget_tokens,
        scorer=scorer,
    )
    owned_client = client is None
    active_client = client or TreeClient(base_url)
    trees = []
    try:
        for prompt in prompts:
            messages = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt
            )
            assembler = TraceAssembler(prompt)
            request_parameters = dict(parameters)
            if seed is not None:
                request_parameters["seed"] = seed
            for event in active_client.stream_tree_completions(
                messages=messages,
                model=model,
                tree=tree_parameters,
                **request_parameters,
            ):
                assembler.add(event)
            trees.append(assembler.finish())
    finally:
        if owned_client:
            active_client.close()
    return RolloutBatch(trees=trees)
