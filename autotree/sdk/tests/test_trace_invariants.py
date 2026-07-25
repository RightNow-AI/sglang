from __future__ import annotations

from hypothesis import given, strategies as st
import pytest

from autotree_sdk import (
    BranchMergedEvent,
    BranchPrunedEvent,
    BranchStartedEvent,
    DoneEvent,
    TokenEvent,
    TraceAssembler,
    TraceInvariantError,
    TreeSummary,
    Usage,
)
from autotree_sdk.models import EngineCounters


COUNTERS = EngineCounters(
    logical_tokens=1,
    physical_tokens=1,
    useful_tokens=1,
    elapsed_seconds=0.01,
    ttft_seconds=0.001,
)


def test_token_indices_must_start_at_zero_and_be_contiguous() -> None:
    assembler = TraceAssembler("prompt")
    assembler.add(BranchStartedEvent(branch_id="root"))
    assembler.add(TokenEvent(branch_id="root", token_index=0, token="a", logprob=-0.1))

    with pytest.raises(TraceInvariantError, match="invalid_token_index"):
        assembler.add(
            TokenEvent(branch_id="root", token_index=2, token="b", logprob=-0.2)
        )


def test_terminal_branch_rejects_later_tokens() -> None:
    assembler = TraceAssembler("prompt")
    assembler.add(BranchStartedEvent(branch_id="root"))
    assembler.add(BranchPrunedEvent(branch_id="root", reason="low_score"))

    with pytest.raises(TraceInvariantError, match="branch_not_live"):
        assembler.add(
            TokenEvent(branch_id="root", token_index=0, token="late", logprob=-1.0)
        )


def test_merged_branch_records_destination_and_terminal_state() -> None:
    assembler = TraceAssembler("prompt")
    assembler.add(BranchStartedEvent(branch_id="root"))
    assembler.add(BranchStartedEvent(branch_id="alt", parent_id="root"))
    assembler.add(BranchMergedEvent(branch_id="alt", into_branch_id="root"))
    assembler.add(
        DoneEvent(
            branch_id="root",
            text="",
            finish_reason="length",
            usage=Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1),
            counters=COUNTERS,
            tree=TreeSummary(
                policy="beam",
                branch_count=2,
                pruned_count=0,
                merged_count=1,
                winner_branch_id="root",
                tokens_spent_per_branch={"root": 0, "alt": 0},
                final_scores={"root": 1.0, "alt": 1.0},
                scorer=None,
                kv_reuse_ratio=1.0,
            ),
        )
    )

    tree = assembler.finish()

    assert tree.branch("root").status == "completed"
    assert tree.branch("alt").status == "merged"
    assert tree.branch("alt").merged_into == "root"


@st.composite
def valid_event_streams(draw):
    branch_count = draw(st.integers(min_value=1, max_value=6))
    token_counts = draw(
        st.lists(st.integers(min_value=0, max_value=5), min_size=branch_count, max_size=branch_count)
    )
    winner_index = draw(st.integers(min_value=0, max_value=branch_count - 1))
    pruned = [index != winner_index for index in range(branch_count)]
    states = []
    for index, token_count in enumerate(token_counts):
        branch_id = f"b{index}"
        actions = [BranchStartedEvent(branch_id=branch_id)]
        actions.extend(
            TokenEvent(
                branch_id=branch_id,
                token_index=token_index,
                token=f"{index}:{token_index}",
                logprob=-float(token_index + 1),
            )
            for token_index in range(token_count)
        )
        if pruned[index]:
            actions.append(BranchPrunedEvent(branch_id=branch_id, reason="property"))
        states.append(actions)

    ordered = []
    while any(states):
        available = [index for index, actions in enumerate(states) if actions]
        selected = draw(st.sampled_from(available))
        ordered.append(states[selected].pop(0))

    summary = TreeSummary(
        policy="beam",
        branch_count=branch_count,
        pruned_count=sum(pruned),
        merged_count=0,
        winner_branch_id=f"b{winner_index}",
        tokens_spent_per_branch={
            f"b{index}": token_count for index, token_count in enumerate(token_counts)
        },
        final_scores={f"b{index}": float(index) for index in range(branch_count)},
        scorer=None,
        kv_reuse_ratio=1.0,
    )
    ordered.append(
        DoneEvent(
            branch_id=f"b{winner_index}",
            text="".join(
                f"{winner_index}:{token_index}"
                for token_index in range(token_counts[winner_index])
            ),
            finish_reason="length",
            usage=Usage(
                prompt_tokens=3,
                completion_tokens=sum(token_counts),
                total_tokens=3 + sum(token_counts),
            ),
            counters=COUNTERS,
            tree=summary,
        )
    )
    return ordered, token_counts, pruned


@given(valid_event_streams())
def test_randomized_valid_event_orders_preserve_trace_invariants(case) -> None:
    events, token_counts, pruned = case
    assembler = TraceAssembler("property prompt")
    for event in events:
        assembler.add(event)

    tree = assembler.finish()

    by_id = {branch.branch_id: branch for branch in tree.branches}
    assert [len(by_id[f"b{index}"].tokens) for index in range(len(token_counts))] == token_counts
    assert [by_id[f"b{index}"].pruned for index in range(len(pruned))] == pruned
    assert tree.usage.completion_tokens == sum(token_counts)
