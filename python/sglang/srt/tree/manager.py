"""Lifecycle and accounting for scheduler-owned tree generation runs.

The implementation deliberately avoids importing scheduler or CUDA modules at
module import time so its policy and accounting behavior remains CPU-testable.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Optional

from sglang.srt.tree.params import (
    TreeBranchEvent,
    TreeCounters,
    TreeGenerateReqInput,
    TreeResult,
    TreeSummary,
)
from sglang.srt.tree.scheduler_hooks import (
    RustSchedulerAdapter,
    on_parent_prefill_done,
    on_tree_request,
)


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
        self.spent_per_branch[branch_id] = self.spent_per_branch.get(branch_id, 0) + 1
        remaining = self.limit - self.spent
        return BudgetUpdate(
            accepted=True,
            spent=self.spent,
            remaining=remaining,
            exhausted=remaining == 0,
        )


@dataclasses.dataclass
class TreeBranchState:
    """Scheduler-neutral state for one live branch."""

    rid: str
    branch_id: str
    tokens: list[int] = dataclasses.field(default_factory=list)
    score: float = 0.0
    state: str = "active"
    request: Any = None


@dataclasses.dataclass
class _TreeRun:
    tree_input: TreeGenerateReqInput
    policy: RustSchedulerAdapter
    budget: TreeTokenBudget
    started_at: float
    branches: dict[str, TreeBranchState] = dataclasses.field(default_factory=dict)
    events: list[TreeBranchEvent] = dataclasses.field(default_factory=list)
    prompt_tokens: int = 0
    first_token_at: Optional[float] = None
    budget_exhausted: bool = False


class TreeRunManager:
    """Own live tree runs and assemble their branch events and final results."""

    def __init__(
        self,
        *,
        tree_cache: Any,
        scheduler_factory: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> None:
        """Create a manager, optionally injecting a CPU-test policy factory."""
        self.tree_cache = tree_cache
        self.scheduler_factory = scheduler_factory
        self.live_trees: dict[str, _TreeRun] = {}
        self.branch_registry: dict[str, TreeBranchState] = {}
        self.completed_results: dict[str, TreeResult] = {}

    def start_tree(self, tree_input: TreeGenerateReqInput) -> Any:
        """Register a tree request and return its parent submission plan."""
        parent = on_tree_request(tree_input)
        parent_rid = parent.rid
        if parent_rid in self.live_trees:
            raise ValueError(f"tree request already exists: {parent_rid}")

        self.live_trees[parent_rid] = _TreeRun(
            tree_input=tree_input,
            policy=RustSchedulerAdapter(
                tree_input.tree, scheduler_cls=self.scheduler_factory
            ),
            budget=TreeTokenBudget(tree_input.tree.budget_tokens),
            started_at=time.perf_counter(),
        )
        return parent

    def on_parent_prefill_done(self, parent: Any) -> Any:
        """Record parent prefill completion and return a branch fork plan."""
        run = self._get_run(parent.rid)
        if run.branches:
            raise ValueError(f"tree parent was already forked: {parent.rid}")

        plan = on_parent_prefill_done(
            parent,
            branch_count=run.tree_input.tree.branches,
            tree_cache=self.tree_cache,
            tree_id=parent.rid,
        )
        run.prompt_tokens = plan.prefix_length
        for child in plan.children:
            branch_id = str(child.branch_id)
            branch = TreeBranchState(rid=child.rid, branch_id=branch_id)
            run.branches[branch_id] = branch
            self.branch_registry[child.rid] = branch
            run.events.append(
                TreeBranchEvent(
                    event="forked",
                    branch_id=branch_id,
                    parent_id=parent.rid,
                )
            )
        return plan

    def on_branch_token(
        self, branch: Any, token: int, logprob: float, *, eos: bool = False
    ) -> Any:
        """Account for one sampled token and return boundary-safe commands."""
        state = self.branch_registry.get(branch.rid)
        if state is None:
            raise KeyError(f"unknown tree branch: {branch.rid}")
        if state.state != "active":
            return ()

        run = self._run_for_branch(state)
        update = run.budget.consume(state.branch_id)
        if not update.accepted:
            return self._apply_commands(
                run,
                ({"type": "kill", "branch": int(state.branch_id), "reason": "budget"},),
            )

        now = time.perf_counter()
        if run.first_token_at is None:
            run.first_token_at = now
        state.request = branch
        state.tokens.append(token)
        state.score += logprob
        run.events.append(
            TreeBranchEvent(
                event="token",
                branch_id=state.branch_id,
                token_id=token,
                score=state.score,
            )
        )

        commands = list(
            run.policy.feed_token(int(state.branch_id), token, logprob, eos=eos)
        )
        if update.exhausted:
            run.budget_exhausted = True
            commands.extend(run.policy.drain())
            finalized = {
                str(command["branch"])
                for command in commands
                if command.get("type") == "finalize"
            }
            for candidate in run.branches.values():
                if candidate.state == "active" and candidate.branch_id not in finalized:
                    commands.append(
                        {
                            "type": "kill",
                            "branch": int(candidate.branch_id),
                            "reason": "budget",
                        }
                    )
        return self._apply_commands(run, tuple(commands))

    def finalize_tree(self, parent_rid: str, winner_branch_id: str) -> Any:
        """Finalize a tree and assemble its immutable result envelope."""
        run = self._get_run(parent_rid)
        winner_branch_id = str(winner_branch_id)
        winner = run.branches.get(winner_branch_id)
        if winner is None:
            raise KeyError(f"unknown winner branch: {winner_branch_id}")
        if winner.state != "finalized":
            self._record_finalized(run, winner)

        for branch in run.branches.values():
            if branch.state == "active" and branch is not winner:
                self._record_pruned(run, branch, "finalize")

        completion_tokens = run.budget.spent
        logical_tokens = run.prompt_tokens * len(run.branches) + completion_tokens
        physical_tokens = run.prompt_tokens + len(winner.tokens)
        elapsed_seconds = max(time.perf_counter() - run.started_at, 0.0)
        ttft_seconds = (
            max(run.first_token_at - run.started_at, 0.0)
            if run.first_token_at is not None
            else 0.0
        )
        counters = TreeCounters(
            logical_tokens=logical_tokens,
            physical_tokens=physical_tokens,
            useful_tokens=physical_tokens,
            elapsed_seconds=elapsed_seconds,
            ttft_seconds=ttft_seconds,
            unique_tokens_per_step=[1] * completion_tokens,
            branch_tokens_per_step=[1] * completion_tokens,
        )
        summary = TreeSummary(
            policy=run.tree_input.tree.policy,
            branch_count=len(run.branches),
            pruned_count=sum(
                branch.state == "pruned" for branch in run.branches.values()
            ),
            merged_count=sum(
                branch.state == "merged" for branch in run.branches.values()
            ),
            winner_branch_id=winner_branch_id,
            tokens_spent_per_branch=dict(run.budget.spent_per_branch),
            final_scores={
                branch.branch_id: branch.score for branch in run.branches.values()
            },
            scorer=run.tree_input.tree.scorer,
            kv_reuse_ratio=(
                logical_tokens / physical_tokens if physical_tokens else None
            ),
        )
        result = TreeResult(
            winner_text=getattr(winner.request, "decoded_text", ""),
            winner_token_ids=list(winner.tokens),
            prompt_tokens=run.prompt_tokens,
            completion_tokens=completion_tokens,
            summary=summary,
            finish_reason="length" if run.budget_exhausted else "stop",
            counters=counters,
            branch_events=list(run.events),
        )
        self.completed_results[parent_rid] = result
        self.live_trees.pop(parent_rid)
        for branch in run.branches.values():
            self.branch_registry.pop(branch.rid, None)
        return result

    def _get_run(self, parent_rid: str) -> _TreeRun:
        try:
            return self.live_trees[parent_rid]
        except KeyError as exc:
            raise KeyError(f"unknown tree request: {parent_rid}") from exc

    def _run_for_branch(self, branch: TreeBranchState) -> _TreeRun:
        for run in self.live_trees.values():
            if run.branches.get(branch.branch_id) is branch:
                return run
        raise KeyError(f"orphaned tree branch: {branch.rid}")

    def _apply_commands(
        self, run: _TreeRun, commands: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        applied = []
        for command in commands:
            command = dict(command)
            command_type = command.get("type")
            if command_type not in {"continue", "kill", "finalize", "fork_at"}:
                raise ValueError(f"unknown tree scheduler command: {command!r}")
            branch_id = command.get("branch")
            state = run.branches.get(str(branch_id)) if branch_id is not None else None
            if command_type in {"kill", "finalize"} and state is None:
                raise KeyError(f"unknown policy branch: {branch_id}")
            if command_type == "kill":
                if state.state != "active":
                    continue
                self._record_pruned(run, state, command.get("reason", "policy"))
            elif command_type == "finalize":
                if state.state == "pruned":
                    continue
                self._record_finalized(run, state)
            applied.append(command)
        return tuple(applied)

    @staticmethod
    def _record_pruned(run: _TreeRun, branch: TreeBranchState, reason: str) -> None:
        branch.state = "pruned"
        run.events.append(
            TreeBranchEvent(
                event="pruned",
                branch_id=branch.branch_id,
                score=branch.score,
                reason=reason,
            )
        )

    @staticmethod
    def _record_finalized(run: _TreeRun, branch: TreeBranchState) -> None:
        if branch.state == "finalized":
            return
        branch.state = "finalized"
        run.events.append(
            TreeBranchEvent(
                event="finalized",
                branch_id=branch.branch_id,
                score=branch.score,
            )
        )
