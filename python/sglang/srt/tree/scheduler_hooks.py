"""Scheduler-boundary hooks for phase-1 tree execution.

These functions return plans or commands; scheduler wiring applies them only at
batch boundaries. Shared-prefix ownership always remains with the radix cache.
"""

from __future__ import annotations

import dataclasses
from array import array
from typing import Any, Optional

from sglang.srt.tree.params import TreeGenerateReqInput


@dataclasses.dataclass(frozen=True)
class BranchRequestDescriptor:
    """Scheduler-neutral data needed to construct one child ``Req``."""

    rid: str
    branch_id: int
    parent_rid: str
    input_ids: array[int]
    prefix_indices: Any
    last_node: Any


@dataclasses.dataclass(frozen=True)
class ForkPlan:
    """Children and shared-prefix ownership produced at a batch boundary."""

    parent_rid: str
    prefix_length: int
    prefix_node: Any
    children: tuple[BranchRequestDescriptor, ...]


def on_tree_request(tree_input: TreeGenerateReqInput) -> Any:
    """Build the parent request submission plan for a new tree."""
    raise NotImplementedError


def on_parent_prefill_done(
    parent: Any,
    *,
    branch_count: int,
    tree_cache: Any,
    tree_id: Optional[str] = None,
) -> ForkPlan:
    """Lock the cached parent prefix and describe N child requests."""
    if branch_count < 1:
        raise ValueError("branch_count must be positive")

    # cache_unfinished_req moves the parent's current hold to the complete
    # prefill sequence. Each child then receives its own hold on that node.
    tree_cache.cache_unfinished_req(parent)
    prefix_node = parent.last_node
    prefix_indices = parent.prefix_indices
    prefix_length = len(prefix_indices)
    child_input_ids = array(
        "q", list(parent.origin_input_ids) + list(parent.output_ids)
    )
    tree_id = tree_id or parent.rid

    children = []
    for branch_id in range(branch_count):
        tree_cache.inc_lock_ref(prefix_node)
        child_prefix_indices = (
            prefix_indices.clone()
            if hasattr(prefix_indices, "clone")
            else prefix_indices[:]
        )
        children.append(
            BranchRequestDescriptor(
                rid=f"{tree_id}:branch:{branch_id}",
                branch_id=branch_id,
                parent_rid=parent.rid,
                input_ids=array("q", child_input_ids),
                prefix_indices=child_prefix_indices,
                last_node=prefix_node,
            )
        )

    # Transfer ownership from the parent request to the child descriptors. The
    # parent now points at the root so a later generic cleanup is a no-op.
    tree_cache.dec_lock_ref(prefix_node)
    parent.last_node = tree_cache.root_node

    return ForkPlan(
        parent_rid=parent.rid,
        prefix_length=prefix_length,
        prefix_node=prefix_node,
        children=tuple(children),
    )


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
