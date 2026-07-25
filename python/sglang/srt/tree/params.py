"""Shared contract between the tree API layer and the tree scheduler module.

This file is the fixed interface both sides build against. Field names follow
the AutoTree wire specification (autotree repo, core/docs/wire-spec.md) so the
existing SDK and conformance suite apply unchanged.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from sglang.srt.tree.selection import env_int

TREE_POLICIES = ("beam", "best_first", "mcts")
MAX_BRANCHES = env_int("AUTOTREE_MAX_BRANCHES", 64)
MAX_BUDGET_TOKENS = 1_000_000
DEFAULT_CONSENSUS_WARMUP = 64
DEFAULT_CONSENSUS_INTERVAL = 32
DEFAULT_MIN_SURVIVORS = 2
CALLBACK_MAX_TIMEOUT_S = 10.0
TREE_PARAM_NAMES = frozenset(
    {
        "policy",
        "branches",
        "budget_tokens",
        "scorer",
        "fork_at_text",
        "fork_at_entropy",
        "adaptive_width",
        "consensus_warmup",
        "consensus_interval",
        "min_survivors",
        "verifier",
    }
)


def validate_verifier_block(verifier: Mapping[str, Any]) -> None:
    """Validate one verifier block before it reaches the scheduler."""
    if not isinstance(verifier, Mapping):
        raise ValueError("verifier must be a mapping")
    verifier_type = verifier.get("type")
    if verifier_type not in {"regex", "numeric", "callback"}:
        raise ValueError("verifier type must be regex, numeric, or callback")

    allowed = {
        "regex": {"type", "pattern", "flags"},
        "numeric": {"type", "equals", "tolerance"},
        "callback": {"type", "url", "timeout_s"},
    }[verifier_type]
    unknown = sorted(set(verifier) - allowed)
    if unknown:
        raise ValueError(f"unknown {verifier_type} verifier field: {unknown[0]}")

    if verifier_type == "regex":
        pattern = verifier.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("regex verifier pattern must be a non-empty string")
        flags = verifier.get("flags", "")
        if flags not in {"", "i"}:
            raise ValueError("regex verifier flags must be empty or 'i'")
        try:
            re.compile(pattern, re.IGNORECASE if flags == "i" else 0)
        except re.error as error:
            raise ValueError(f"regex verifier pattern is invalid: {error}") from error
        return

    if verifier_type == "numeric":
        equals = verifier.get("equals")
        if (
            not isinstance(equals, (int, float))
            or isinstance(equals, bool)
            or not math.isfinite(float(equals))
        ):
            raise ValueError("numeric verifier equals must be a finite number")
        tolerance = verifier.get("tolerance")
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            raise ValueError(
                "numeric verifier tolerance must be a finite non-negative number"
            )
        return

    url = verifier.get("url")
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ValueError("callback verifier url must be a non-empty URL string")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("callback verifier url must use http or https")
    timeout_s = verifier.get("timeout_s")
    if (
        not isinstance(timeout_s, (int, float))
        or isinstance(timeout_s, bool)
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
        or timeout_s > CALLBACK_MAX_TIMEOUT_S
    ):
        raise ValueError(
            "callback verifier timeout_s must be greater than 0 and at most "
            f"{CALLBACK_MAX_TIMEOUT_S:g}"
        )


def validate_tree_params(params: Mapping[str, Any]) -> None:
    """Validate the complete scheduler-facing AutoTree parameter mapping."""
    if not isinstance(params, Mapping):
        raise ValueError("tree parameters must be a mapping")
    unknown = sorted(set(params) - TREE_PARAM_NAMES)
    if unknown:
        raise ValueError(f"unknown tree parameter: {unknown[0]}")

    policy = params.get("policy", "beam")
    if not isinstance(policy, str) or policy not in TREE_POLICIES:
        raise ValueError(f"policy must be one of {TREE_POLICIES}")

    branches = params.get("branches", 4)
    if not isinstance(branches, int) or isinstance(branches, bool) or branches < 1:
        raise ValueError("branches must be a positive integer")
    if branches > MAX_BRANCHES:
        raise ValueError(f"branches must be at most {MAX_BRANCHES}")

    budget_tokens = params.get("budget_tokens", 1024)
    if (
        not isinstance(budget_tokens, int)
        or isinstance(budget_tokens, bool)
        or budget_tokens < 1
    ):
        raise ValueError("budget_tokens must be a positive integer")
    if budget_tokens > MAX_BUDGET_TOKENS:
        raise ValueError(f"budget_tokens must be at most {MAX_BUDGET_TOKENS}")

    scorer = params.get("scorer")
    if scorer is not None and not isinstance(scorer, str):
        raise ValueError("scorer must be a string or null")

    fork_at_text = params.get("fork_at_text")
    if fork_at_text is not None:
        if not isinstance(fork_at_text, str) or not fork_at_text:
            raise ValueError("fork_at_text must be a non-empty string")
        if len(fork_at_text) > 64:
            raise ValueError("fork_at_text must be at most 64 characters")

    fork_at_entropy = params.get("fork_at_entropy")
    if fork_at_entropy is not None:
        if (
            not isinstance(fork_at_entropy, (int, float))
            or isinstance(fork_at_entropy, bool)
            or not math.isfinite(float(fork_at_entropy))
            or fork_at_entropy <= 0
        ):
            raise ValueError("fork_at_entropy must be a finite number greater than 0")

    if fork_at_text is not None and fork_at_entropy is not None:
        raise ValueError("fork_at_text and fork_at_entropy are mutually exclusive")

    adaptive_width = params.get("adaptive_width")
    if adaptive_width is not None:
        if not isinstance(adaptive_width, int) or isinstance(adaptive_width, bool):
            raise ValueError("adaptive_width must be an integer")
        if adaptive_width <= branches:
            raise ValueError("adaptive_width must be greater than branches")
        if adaptive_width > MAX_BRANCHES:
            raise ValueError(f"adaptive_width must be at most {MAX_BRANCHES}")

    consensus_warmup = params.get("consensus_warmup", DEFAULT_CONSENSUS_WARMUP)
    if (
        not isinstance(consensus_warmup, int)
        or isinstance(consensus_warmup, bool)
        or consensus_warmup < 0
    ):
        raise ValueError("consensus_warmup must be a non-negative integer")

    consensus_interval = params.get("consensus_interval", DEFAULT_CONSENSUS_INTERVAL)
    if (
        not isinstance(consensus_interval, int)
        or isinstance(consensus_interval, bool)
        or consensus_interval <= 0
    ):
        raise ValueError("consensus_interval must be a positive integer")

    min_survivors = params.get("min_survivors", DEFAULT_MIN_SURVIVORS)
    if (
        not isinstance(min_survivors, int)
        or isinstance(min_survivors, bool)
        or min_survivors <= 0
    ):
        raise ValueError("min_survivors must be a positive integer")

    verifier = params.get("verifier")
    if verifier is not None:
        validate_verifier_block(verifier)


def normalize_tree_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply wire defaults and return a validated scheduler parameter dict."""
    if not isinstance(params, Mapping):
        raise ValueError("tree parameters must be a mapping")
    normalized = {
        "policy": "beam",
        "branches": 4,
        "budget_tokens": 1024,
        "scorer": None,
        "fork_at_text": None,
        "fork_at_entropy": None,
        "adaptive_width": None,
        "consensus_warmup": DEFAULT_CONSENSUS_WARMUP,
        "consensus_interval": DEFAULT_CONSENSUS_INTERVAL,
        "min_survivors": DEFAULT_MIN_SURVIVORS,
        "verifier": None,
    }
    normalized.update(dict(params))
    validate_tree_params(normalized)
    if normalized["verifier"] is None:
        normalized.pop("verifier")
    return normalized


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
    consensus_warmup: int = DEFAULT_CONSENSUS_WARMUP
    consensus_interval: int = DEFAULT_CONSENSUS_INTERVAL
    min_survivors: int = DEFAULT_MIN_SURVIVORS
    verifier: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        validate_tree_params(dataclasses.asdict(self))

    def to_runtime_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        validate_tree_params(result)
        if result["verifier"] is None:
            result.pop("verifier")
        return result


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
    served_from_memo: bool = False
    memo_key: Optional[str] = None
    verifier_used: bool = False
    verifier_approved_count: int = 0
    verifier_fell_back: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        if not self.verifier_used:
            result.pop("verifier_used")
            result.pop("verifier_approved_count")
            result.pop("verifier_fell_back")
        return result


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
