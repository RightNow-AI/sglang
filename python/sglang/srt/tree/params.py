"""Shared contract between the tree API layer and the tree scheduler module.

This file is the fixed interface both sides build against. Field names follow
the AutoTree wire specification (autotree repo, core/docs/wire-spec.md) so the
existing SDK and conformance suite apply unchanged.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Optional

TREE_POLICIES = ("beam", "best_first", "mcts")
MAX_BRANCHES = int(os.environ.get("AUTOTREE_MAX_BRANCHES", "64"))


@dataclasses.dataclass
class TreeParams:
    """User-facing tree execution parameters."""

    policy: str = "beam"
    branches: int = 4
    budget_tokens: int = 1024
    scorer: Optional[str] = None
    fork_at_text: Optional[str] = None
    fork_at_entropy: Optional[float] = None
    adaptive_width: Optional[int] = None

    def validate(self) -> None:
        if self.policy not in TREE_POLICIES:
            raise ValueError(f"policy must be one of {TREE_POLICIES}")
        if not isinstance(self.branches, int) or self.branches < 1:
            raise ValueError("branches must be a positive integer")
        if not isinstance(self.budget_tokens, int) or self.budget_tokens < 1:
            raise ValueError("budget_tokens must be a positive integer")
        if self.fork_at_text is not None:
            if not isinstance(self.fork_at_text, str) or not self.fork_at_text:
                raise ValueError("fork_at_text must be a non-empty string")
            if len(self.fork_at_text) > 64:
                raise ValueError("fork_at_text must be at most 64 characters")
        if self.fork_at_entropy is not None:
            if not isinstance(self.fork_at_entropy, (int, float)) or isinstance(
                self.fork_at_entropy, bool
            ):
                raise ValueError("fork_at_entropy must be greater than 0")
            if not self.fork_at_entropy > 0:
                raise ValueError("fork_at_entropy must be greater than 0")
        if self.fork_at_text is not None and self.fork_at_entropy is not None:
            raise ValueError(
                "fork_at_text and fork_at_entropy are mutually exclusive"
            )
        if self.adaptive_width is not None:
            if not isinstance(self.adaptive_width, int) or isinstance(
                self.adaptive_width, bool
            ):
                raise ValueError("adaptive_width must be an integer")
            if self.adaptive_width <= self.branches:
                raise ValueError("adaptive_width must be greater than branches")
            if self.adaptive_width > MAX_BRANCHES:
                raise ValueError(
                    f"adaptive_width must be at most {MAX_BRANCHES}"
                )

    def to_runtime_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


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
    # Per-branch extracted final answers (branch_id -> canonical number string
    # or None). Lets clients compute vote counts and agreement margins - the
    # escalation signal for cascade routing. Empty dict when the runtime did
    # not surface branch outputs.
    branch_answers: Dict[str, Optional[str]] = dataclasses.field(
        default_factory=dict
    )

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
