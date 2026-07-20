"""Lifecycle and accounting for scheduler-owned tree generation runs.

The implementation deliberately avoids importing scheduler or CUDA modules at
module import time so its policy and accounting behavior remains CPU-testable.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from sglang.srt.tree.params import TreeGenerateReqInput


class TreeRunManager:
    """Own live tree runs and assemble their branch events and final results."""

    def __init__(
        self,
        scheduler_factory: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> None:
        """Create a manager, optionally injecting a CPU-test policy factory."""
        raise NotImplementedError

    def start_tree(self, tree_input: TreeGenerateReqInput) -> Any:
        """Register a tree request and return its parent submission plan."""
        raise NotImplementedError

    def on_parent_prefill_done(self, parent: Any) -> Any:
        """Record parent prefill completion and return a branch fork plan."""
        raise NotImplementedError

    def on_branch_token(
        self, branch: Any, token: int, logprob: float, *, eos: bool = False
    ) -> Any:
        """Account for one sampled token and return boundary-safe commands."""
        raise NotImplementedError

    def finalize_tree(self, parent_rid: str, winner_branch_id: str) -> Any:
        """Finalize a tree and assemble its immutable result envelope."""
        raise NotImplementedError
