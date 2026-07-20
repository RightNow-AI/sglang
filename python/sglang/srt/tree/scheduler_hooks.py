"""Scheduler-boundary hooks for phase-1 tree execution.

These functions return plans or commands; scheduler wiring applies them only at
batch boundaries. Shared-prefix ownership always remains with the radix cache.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.tree.params import TreeGenerateReqInput


def on_tree_request(tree_input: TreeGenerateReqInput) -> Any:
    """Build the parent request submission plan for a new tree."""
    raise NotImplementedError


def on_parent_prefill_done(parent: Any, *, branch_count: int, tree_cache: Any) -> Any:
    """Lock the cached parent prefix and describe N child requests."""
    raise NotImplementedError


def on_branch_token(
    policy_scheduler: Any,
    branch_id: int,
    token: int,
    logprob: float,
    *,
    eos: bool = False,
) -> Any:
    """Feed a token event to the Rust policy and return its commands."""
    raise NotImplementedError


def apply_kill(branch: Any, *, tree_cache: Any, reason: str = "policy") -> None:
    """Mark a branch for finish and release only its radix-lock hold."""
    raise NotImplementedError
