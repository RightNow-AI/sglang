"""Required RED-first pins for terminal and usage invariants."""

import pytest

from autotree_sdk import (
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


def test_started_branch_requires_terminal_done_event() -> None:
    assembler = TraceAssembler()
    assembler.add(BranchStartedEvent(branch_id="root", parent_id=None))
    assembler.add(
        TokenEvent(branch_id="root", token_index=0, token="answer", logprob=-0.1)
    )

    with pytest.raises(TraceInvariantError, match="missing_done"):
        assembler.finish()


def test_done_usage_must_match_streamed_token_count() -> None:
    assembler = TraceAssembler()
    assembler.add(BranchStartedEvent(branch_id="root", parent_id=None))
    assembler.add(
        TokenEvent(branch_id="root", token_index=0, token="answer", logprob=-0.1)
    )
    with pytest.raises(TraceInvariantError, match="usage_token_mismatch"):
        assembler.add(
            DoneEvent(
                branch_id="root",
                text="answer",
                finish_reason="length",
                usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                counters=COUNTERS,
                tree=TreeSummary(
                    policy="beam",
                    branch_count=1,
                    pruned_count=0,
                    merged_count=0,
                    winner_branch_id="root",
                    tokens_spent_per_branch={"root": 1},
                    final_scores={"root": 1.0},
                    scorer=None,
                    kv_reuse_ratio=1.0,
                ),
            )
        )
