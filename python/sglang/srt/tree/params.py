"""Shared contract between the tree API layer and the tree scheduler module.

This file is the fixed interface both sides build against. Field names follow
the AutoTree wire specification (autotree repo, core/docs/wire-spec.md) so the
existing SDK and conformance suite apply unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

TREE_POLICIES = ("beam", "best_first", "mcts")


@dataclasses.dataclass
class TreeParams:
    """User-facing tree execution parameters."""

    policy: str = "beam"
    branches: int = 4
    budget_tokens: int = 1024
    scorer: Optional[str] = None

    def validate(self) -> None:
        if self.policy not in TREE_POLICIES:
            raise ValueError(f"policy must be one of {TREE_POLICIES}")
        if not isinstance(self.branches, int) or self.branches < 1:
            raise ValueError("branches must be a positive integer")
        if not isinstance(self.budget_tokens, int) or self.budget_tokens < 1:
            raise ValueError("budget_tokens must be a positive integer")


@dataclasses.dataclass
class TreeGenerateReqInput:
    """Envelope for a tree request travelling tokenizer-manager -> scheduler.

    base is a plain sglang GenerateReqInput (kept untyped here to avoid an
    import cycle); tree carries the AutoTree parameters. The scheduler treats
    base as the parent request to prefill, then forks branches from it.
    """

    base: Any
    tree: TreeParams


@dataclasses.dataclass
class TreeBranchEvent:
    """Per-branch lifecycle event surfaced to the API layer for streaming."""

    event: str                      # forked | token | pruned | merged | finalized
    branch_id: str
    parent_id: Optional[str] = None
    token_id: Optional[int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None    # for pruned: budget | policy | error


@dataclasses.dataclass
class TreeSummary:
    """Final per-tree accounting, mirroring the AutoTree wire spec."""

    policy: str
    branch_count: int
    pruned_count: int
    merged_count: int
    winner_branch_id: Optional[str]
    tokens_spent_per_branch: Dict[str, int]
    final_scores: Dict[str, float]
    scorer: Optional[str]
    kv_reuse_ratio: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TreeCounters:
    """Engine counters carried by the wire spec's final done event."""

    logical_tokens: int = 0
    physical_tokens: int = 0
    useful_tokens: int = 0
    elapsed_seconds: float = 0.0
    ttft_seconds: float = 0.0
    unique_tokens_per_step: List[int] = dataclasses.field(default_factory=list)
    branch_tokens_per_step: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TreeResult:
    """Non-streaming result envelope handed back to the serving layer."""

    winner_text: str
    winner_token_ids: List[int]
    prompt_tokens: int
    completion_tokens: int
    summary: TreeSummary
    finish_reason: str = "stop"
    counters: Optional[TreeCounters] = None
    branch_events: List[TreeBranchEvent] = dataclasses.field(default_factory=list)
