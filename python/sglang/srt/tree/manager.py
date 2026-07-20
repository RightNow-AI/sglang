"""Lifecycle and accounting for scheduler-owned tree generation runs.

The implementation deliberately avoids importing scheduler or CUDA modules at
module import time so its policy and accounting behavior remains CPU-testable.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from sglang.srt.tree.params import TreeGenerateReqInput


@dataclasses.dataclass(frozen=True)
class BudgetUpdate:
    """Outcome of charging one generated token to a tree budget."""

    accepted: bool
    spent: int
    remaining: int
    exhausted: bool


class TreeTokenBudget:
    """Hard aggregate token budget shared by every branch in one tree."""

    def __init__(self, budget_tokens: int) -> None:
        if budget_tokens < 1:
            raise ValueError("budget_tokens must be positive")
        self.limit = budget_tokens
        self.spent = 0
        self.spent_per_branch: dict[str, int] = {}

    def consume(self, branch_id: str) -> BudgetUpdate:
        """Charge one token, never allowing aggregate spend above the limit."""
        if self.spent >= self.limit:
            return BudgetUpdate(
                accepted=False,
                spent=self.spent,
                remaining=0,
                exhausted=True,
            )

        self.spent += 1
        self.spent_per_branch[branch_id] = (
            self.spent_per_branch.get(branch_id, 0) + 1
        )
        remaining = self.limit - self.spent
        return BudgetUpdate(
            accepted=True,
            spent=self.spent,
            remaining=remaining,
            exhausted=remaining == 0,
        )


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
