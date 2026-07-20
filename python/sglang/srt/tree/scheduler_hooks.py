"""Scheduler-boundary hooks for phase-1 tree execution.

These functions return plans or commands; scheduler wiring applies them only at
batch boundaries. Shared-prefix ownership always remains with the radix cache.
"""

from __future__ import annotations

import dataclasses
from array import array
from typing import Any, Callable, Optional

from sglang.srt.tree.params import TreeGenerateReqInput, TreeParams


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


class RustSchedulerAdapter:
    """Narrow adapter over the optional ``autotree_scheduler`` PyO3 wheel."""

    def __init__(
        self,
        params: TreeParams,
        *,
        scheduler_cls: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> None:
        if scheduler_cls is None:
            try:
                from autotree_scheduler import Scheduler
            except ImportError as exc:
                raise RuntimeError(
                    "tree execution requires the autotree_scheduler wheel"
                ) from exc
            scheduler_cls = Scheduler

        config = {
            "policy": params.policy,
            "branches": params.branches,
            "budget_tokens": params.budget_tokens,
        }
        if params.scorer is not None:
            config["scorer"] = params.scorer
        self.scheduler = scheduler_cls(config)

    def feed_token(
        self,
        branch_id: int,
        token: int,
        logprob: float,
        *,
        eos: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Feed one sampled token and drain boundary-safe policy commands."""
        self.scheduler.feed_event(
            {
                "type": "token_sampled",
                "branch": branch_id,
                "token": token,
                "logprob": logprob,
                "eos": eos,
            }
        )
        poll = getattr(self.scheduler, "poll_commands", None)
        if poll is None:
            poll = getattr(self.scheduler, "next_commands", None)
        if poll is None:
            raise TypeError("policy scheduler exposes no command polling method")
        commands = tuple(dict(command) for command in poll())
        valid_types = {"continue", "kill", "finalize", "fork_at"}
        for command in commands:
            if command.get("type") not in valid_types:
                raise ValueError(f"unknown tree scheduler command: {command!r}")
        return commands

    def drain(self) -> tuple[dict[str, Any], ...]:
        """Ask the policy to finalize remaining work and return its commands."""
        self.scheduler.drain()
        poll = getattr(self.scheduler, "poll_commands", None)
        if poll is None:
            poll = getattr(self.scheduler, "next_commands", None)
        if poll is None:
            raise TypeError("policy scheduler exposes no command polling method")
        return tuple(dict(command) for command in poll())


def on_tree_request(tree_input: TreeGenerateReqInput) -> Any:
    """Build the parent request submission plan for a new tree."""
    tree_input.tree.validate()
    return tree_input.base


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
    if isinstance(policy_scheduler, RustSchedulerAdapter):
        return policy_scheduler.feed_token(branch_id, token, logprob, eos=eos)
    policy_scheduler.feed_event(
        {
            "type": "token_sampled",
            "branch": branch_id,
            "token": token,
            "logprob": logprob,
            "eos": eos,
        }
    )
    poll = getattr(policy_scheduler, "poll_commands", None)
    if poll is None:
        poll = getattr(policy_scheduler, "next_commands", None)
    if poll is None:
        raise TypeError("policy scheduler exposes no command polling method")
    return tuple(dict(command) for command in poll())


def apply_kill(
    branch: Any,
    *,
    tree_cache: Any,
    reason: str = "policy",
    release_fn: Optional[Callable[..., None]] = None,
    finish_reason_factory: Optional[Callable[[str], Any]] = None,
) -> bool:
    """Prune one branch and reclaim its private KV exactly once.

    ``release_kv_cache(..., is_insert=False)`` frees only indices after
    ``cache_protected_len`` and drops the request's ``last_node`` lock. The
    shared prefix is therefore never passed directly to an allocator.
    """
    if getattr(branch, "_tree_kv_released", False):
        return False

    if finish_reason_factory is None:
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        finish_reason_factory = lambda prune_reason: FINISH_ABORT(
            f"tree branch pruned: {prune_reason}"
        )
    if release_fn is None:
        from sglang.srt.mem_cache.common import release_kv_cache

        release_fn = release_kv_cache

    branch.to_finish = finish_reason_factory(reason)
    release_fn(branch, tree_cache, is_insert=False)
    branch._tree_kv_released = True
    return True
