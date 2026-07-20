"""Bridge between the SGLang scheduler and the tree run manager.

The bridge is scheduler-neutral on purpose: it talks to the scheduler through
three injected callables (enqueue, req_factory, mark_finish) so every decision
path is unit-testable on CPU with fakes. The scheduler applies bridge output
only at batch boundaries; shared-prefix ownership stays with the radix cache.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from sglang.srt.tree.manager import TreeRunManager
from sglang.srt.tree.params import TreeGenerateReqInput, TreeResult
from sglang.srt.tree.scheduler_hooks import BranchRequestDescriptor


class SchedulerTreeBridge:
    """Apply tree plans to a live scheduler at batch boundaries."""

    def __init__(
        self,
        *,
        tree_cache: Any,
        enqueue: Callable[[Any], None],
        req_factory: Callable[[BranchRequestDescriptor, Any], Any],
        mark_finish: Callable[[Any], None],
        scheduler_factory: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self.manager = TreeRunManager(
            tree_cache=tree_cache, scheduler_factory=scheduler_factory
        )
        self.enqueue = enqueue
        self.req_factory = req_factory
        self.mark_finish = mark_finish
        self.parent_reqs: Dict[str, Any] = {}
        self.branch_reqs: Dict[str, Any] = {}
        self.pending_results: Dict[str, TreeResult] = {}

    # -- request intake ----------------------------------------------------

    def handle_tree_request(self, tree_input: TreeGenerateReqInput) -> Any:
        """Register the tree and return the parent submission plan."""
        return self.manager.start_tree(tree_input)

    def register_parent(self, parent_req: Any) -> None:
        self.parent_reqs[parent_req.rid] = parent_req

    def is_tree_parent(self, rid: str) -> bool:
        return rid in self.parent_reqs

    def is_tree_branch(self, rid: str) -> bool:
        return rid in self.branch_reqs

    # -- batch-boundary transitions ---------------------------------------

    def on_parent_prefill_done(self, parent_req: Any) -> int:
        """Fork branches off the finished parent; returns branch count."""
        plan = self.manager.on_parent_prefill_done(parent_req)
        for child in plan.children:
            req = self.req_factory(child, parent_req)
            self.branch_reqs[child.rid] = req
            self.enqueue(req)
        return len(plan.children)

    def on_branch_token(
        self, branch_req: Any, token: int, logprob: float, *, eos: bool = False
    ) -> Optional[TreeResult]:
        """Feed one sampled token; apply kill/finalize commands.

        Returns the finished TreeResult when this token finalizes the tree.
        """
        commands = self.manager.on_branch_token(
            branch_req, token, logprob, eos=eos
        )
        result: Optional[TreeResult] = None
        for command in commands:
            kind = command.get("type")
            branch_id = str(command.get("branch"))
            target = self._req_for_branch_id(branch_req, branch_id)
            if target is None:
                continue
            if kind == "kill":
                self.mark_finish(target)
            elif kind == "finalize":
                parent_rid = self._parent_rid_for(target.rid)
                if parent_rid is not None:
                    result = self.manager.finalize_tree(parent_rid, branch_id)
                    self.pending_results[parent_rid] = result
                    for rid in list(self.branch_reqs):
                        if self._parent_rid_for(rid) == parent_rid:
                            self.mark_finish(self.branch_reqs[rid])
        return result

    def take_result(self, parent_rid: str) -> Optional[TreeResult]:
        return self.pending_results.pop(parent_rid, None)

    # -- internals ---------------------------------------------------------

    def _req_for_branch_id(self, hint_req: Any, branch_id: str) -> Optional[Any]:
        state = self.manager.branch_registry.get(hint_req.rid)
        if state is not None and state.branch_id == branch_id:
            return hint_req
        for rid, req in self.branch_reqs.items():
            other = self.manager.branch_registry.get(rid)
            if other is not None and other.branch_id == branch_id:
                return req
        return None

    def _parent_rid_for(self, branch_rid: str) -> Optional[str]:
        for parent_rid, run in self.manager.live_trees.items():
            for branch in run.branches.values():
                if branch.rid == branch_rid:
                    return parent_rid
        for parent_rid in self.pending_results:
            return parent_rid
        return None


def make_child_req(descriptor: BranchRequestDescriptor, parent_req: Any) -> Any:
    """Build a real scheduler Req for one branch (runtime path, Linux only).

    Imported lazily so the bridge stays importable without the full runtime.
    """
    from sglang.srt.managers.schedule_batch import Req  # local: CUDA-adjacent

    req = Req(
        rid=descriptor.rid,
        origin_input_text=None,
        origin_input_ids=list(descriptor.input_ids),
        sampling_params=parent_req.sampling_params,
    )
    req.prefix_indices = descriptor.prefix_indices
    req.last_node = descriptor.last_node
    return req
