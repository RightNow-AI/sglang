"""Phase-1 tree runtime for the SGLang 0.5.15 scheduler (container base).

Parent request doubles as branch 0. When its prefill completes, sibling
branches are spawned as ordinary requests whose prompts radix-share the
parent's prefix. Per-token hooks feed the tree manager; kills reuse the
scheduler's abort path; at finalize the winner's tokens are copied onto the
parent so the existing result channel returns the winning answer under the
parent rid. All hooks are defensive: a tree bug degrades to plain generation,
never a scheduler crash.
"""

from __future__ import annotations

import copy

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TokenizedTreeGenerateReqInput:
    """Envelope the tokenizer manager sends to the scheduler.

    Delegates every unknown attribute (read and write) to the wrapped
    TokenizedGenerateReqInput so the transport layer (time_stats, shm/pickle
    wrapping) treats it exactly like a plain tokenized request.
    """

    _OWN = ("base", "tree")

    def __init__(self, base: Any, tree: Dict[str, Any]) -> None:
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "tree", dict(tree))

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "base"), name)

    def __setattr__(self, name: str, value) -> None:
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "base"), name, value)


# Marginal-value scheduling (EMVPT) knobs. The value proxy is the running mean
# output-token logprob; a trained value head replaces it later. Prune fires only
# when the proxy is actually flowing (all-zero scores never prune).
# Env-overridable so ablations (prune off = large margin) need no code edit.
import os as _os

VALUE_CHECK_INTERVAL = int(_os.environ.get("AUTOTREE_VALUE_CHECK_INTERVAL", "16"))
VALUE_WARMUP_TOKENS = int(_os.environ.get("AUTOTREE_VALUE_WARMUP_TOKENS", "8"))
VALUE_MARGIN = float(_os.environ.get("AUTOTREE_VALUE_MARGIN", "0.35"))
VALUE_MIN_KEEP = int(_os.environ.get("AUTOTREE_VALUE_MIN_KEEP", "2"))


class _BranchState:
    __slots__ = ("rid", "branch_id", "req", "tokens", "score", "state", "lp_seen")

    def __init__(self, rid: str, branch_id: int, req: Any) -> None:
        self.rid = rid
        self.branch_id = branch_id
        self.req = req
        self.tokens = 0
        self.score = 0.0
        self.state = "active"
        self.lp_seen = 0

    def mean_logprob(self) -> float:
        return self.score / self.tokens if self.tokens else 0.0


class _TreeRun:
    __slots__ = (
        "parent_rid", "params", "branches", "spent", "finalized",
        "winner_branch_id", "pruned", "base_tokenized", "last_value_check",
    )

    def __init__(self, parent_rid: str, params: Dict[str, Any]) -> None:
        self.parent_rid = parent_rid
        self.params = params
        self.base_tokenized = None
        self.branches: Dict[str, _BranchState] = {}
        self.spent = 0
        self.finalized = False
        self.winner_branch_id: Optional[int] = None
        self.pruned = 0
        self.last_value_check = 0


class SchedulerTreeRuntime:
    """Owns live tree runs inside one scheduler process."""

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.runs: Dict[str, _TreeRun] = {}          # parent rid -> run
        self.branch_index: Dict[str, _TreeRun] = {}  # any member rid -> run

    # -- intake ------------------------------------------------------------

    def handle_tree_request(self, recv: TokenizedTreeGenerateReqInput):
        """Dispatcher target: route the parent through normal intake."""
        try:
            params = dict(recv.tree)
            run = _TreeRun(recv.rid, params)
            run.base_tokenized = recv.base
            self.runs[recv.rid] = run
            self.branch_index[recv.rid] = run
            logger.info(
                "[tree] request %s policy=%s branches=%s budget=%s",
                recv.rid, params.get("policy"), params.get("branches"),
                params.get("budget_tokens"),
            )
        except Exception:
            logger.exception("[tree] intake failed; degrading to plain request")
        return self.scheduler.handle_generate_request(recv.base)

    # -- hooks (called from batch_result_processor) ------------------------

    def on_prefill_done(self, req: Any) -> None:
        run = self.runs.get(req.rid)
        if run is None or run.branches:
            return
        try:
            self._fork_branches(run, req)
        except Exception:
            logger.exception("[tree] fork failed; parent continues alone")
            if not run.branches:
                run.branches["0"] = _BranchState(req.rid, 0, req)

    def on_token(self, req: Any, token_ids, logprob: Optional[float]) -> None:
        run = self.branch_index.get(req.rid)
        if run is None or run.finalized:
            return
        try:
            self._account_tokens(run, req, token_ids, logprob)
        except Exception:
            logger.exception("[tree] token hook failed for %s", req.rid)

    # -- internals ---------------------------------------------------------

    def _fork_branches(self, run: _TreeRun, parent_req: Any) -> None:
        n = max(1, int(run.params.get("branches", 1)))
        run.branches["0"] = _BranchState(parent_req.rid, 0, parent_req)

        base = run.base_tokenized
        if base is None:
            logger.warning("[tree] %s no tokenized base; parent runs alone",
                           run.parent_rid)
            return

        import msgspec

        for b in range(1, n):
            child_rid = f"{run.parent_rid}#tree{b}"
            sp = getattr(base, "sampling_params", None)
            child_sp = copy.copy(sp) if sp is not None else sp
            seed = getattr(child_sp, "seed", None)
            if child_sp is not None and seed is not None:
                try:
                    child_sp.seed = seed + b
                except Exception:
                    pass
            child_base = msgspec.structs.replace(
                base, rid=child_rid, sampling_params=child_sp
            )
            # Route through the real intake path: all Req invariants and radix
            # prefix-sharing are established exactly as for a normal request.
            self.scheduler.handle_generate_request(child_base)
            branch = _BranchState(child_rid, b, None)
            run.branches[str(b)] = branch
            self.branch_index[child_rid] = run
        logger.info(
            "[tree] %s forked %d sibling branches via intake path",
            run.parent_rid, n - 1,
        )

    def _account_tokens(
        self, run: _TreeRun, req: Any, token_ids, logprob: Optional[float]
    ) -> None:
        state = None
        for candidate in run.branches.values():
            if candidate.rid == req.rid:
                state = candidate
                break
        if state is None or state.state != "active":
            return
        if state.req is None:
            state.req = req  # capture the real Req the scheduler created

        count = len(token_ids) if hasattr(token_ids, "__len__") else 1
        state.tokens += count
        run.spent += count
        if logprob is not None:
            # scalar for normal decode, a per-token list under spec decoding
            if hasattr(logprob, "__len__"):
                state.score += float(sum(float(v) for v in logprob))
            else:
                state.score += float(logprob)
        else:
            # Harvest the value proxy from the request itself: the serving layer
            # sets return_logprob on the base request and forked children inherit
            # it, so the scheduler appends per-token logprobs as decode proceeds.
            vals = getattr(req, "output_token_logprobs_val", None)
            if vals:
                fresh = vals[state.lp_seen:]
                if fresh:
                    state.score += float(sum(fresh))
                    state.lp_seen = len(vals)

        if run.spent - run.last_value_check >= VALUE_CHECK_INTERVAL:
            run.last_value_check = run.spent
            self._maybe_value_prune(run)

        if state.branch_id == 0:
            self._attach_snapshot(run)

        budget = int(run.params.get("budget_tokens", 0) or 0)
        if budget and run.spent >= budget and not run.finalized:
            self._finalize(run, reason="budget")

    def _attach_snapshot(self, run: _TreeRun) -> None:
        """Publish the live tree trace on the parent request. The output
        streamer forwards ``customized_info`` to the tokenizer manager, which
        merges it into ``meta_info`` - for non-streaming requests exactly once,
        at finish, so the parent's final result carries the latest snapshot."""
        parent = run.branches.get("0")
        if parent is None or parent.req is None:
            return
        alive = [b for b in run.branches.values() if b.state == "active"]
        leading = max(
            (b for b in run.branches.values() if b.tokens),
            key=lambda b: b.mean_logprob(),
            default=None,
        )
        snapshot = {
            "policy": run.params.get("policy"),
            "branch_count": len(run.branches),
            "alive_count": len(alive),
            "pruned_count": run.pruned,
            "spent_tokens": run.spent,
            "budget_tokens": int(run.params.get("budget_tokens", 0) or 0),
            "winner_branch_id": (
                str(run.winner_branch_id) if run.winner_branch_id is not None
                else (str(leading.branch_id) if leading is not None else None)
            ),
            "winner_is_final": run.finalized,
            "value_margin": VALUE_MARGIN,
            "branches": {
                str(b.branch_id): {
                    "tokens": b.tokens,
                    "mean_logprob": round(b.mean_logprob(), 4),
                    "state": b.state,
                }
                for b in run.branches.values()
            },
        }
        parent.req.customized_info = {"autotree": [snapshot]}

    def _maybe_value_prune(self, run: _TreeRun) -> None:
        """EMVPT: prune branches whose mean-logprob value proxy trails the best
        sibling by more than VALUE_MARGIN nats/token. Branch 0 is never pruned
        here because its request object carries the wire response."""
        if run.finalized:
            return
        alive = [b for b in run.branches.values() if b.state == "active"]
        if len(alive) <= VALUE_MIN_KEEP:
            return
        scored = [b for b in alive if b.tokens >= VALUE_WARMUP_TOKENS]
        logger.info(
            "[tree] %s value check at %d: %s", run.parent_rid, run.spent,
            ", ".join(
                f"b{b.branch_id}={b.mean_logprob():.3f}/{b.tokens}t"
                for b in alive
            ),
        )
        if len(scored) < 2 or all(b.score == 0.0 for b in scored):
            return
        best = max(b.mean_logprob() for b in scored)

        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        n_alive = len(alive)
        for branch in sorted(scored, key=lambda b: b.mean_logprob()):
            if n_alive <= VALUE_MIN_KEEP:
                break
            if branch.branch_id == 0:
                continue
            gap = best - branch.mean_logprob()
            if gap <= VALUE_MARGIN:
                break
            branch.state = "pruned"
            run.pruned += 1
            n_alive -= 1
            logger.info(
                "[tree] %s pruned branch-%d at token %d: value gap %.3f "
                "nats/token (mean %.3f vs best %.3f); pages reclaim on finish",
                run.parent_rid, branch.branch_id, branch.tokens,
                gap, branch.mean_logprob(), best,
            )
            if branch.req is not None:
                branch.req.to_finish = FINISH_LENGTH(
                    length=len(branch.req.output_ids)
                )

    def _finalize(self, run: _TreeRun, reason: str) -> None:
        run.finalized = True
        active = [b for b in run.branches.values() if b.state == "active"]
        if not active:
            return
        winner = max(active, key=lambda b: (b.mean_logprob(), -b.branch_id))
        run.winner_branch_id = winner.branch_id
        logger.info(
            "[tree] %s finalize (%s): winner=branch-%d spent=%d pruned=%d",
            run.parent_rid, reason, winner.branch_id, run.spent, run.pruned,
        )

        parent = run.branches["0"].req
        if winner.branch_id != 0 and winner.req is not None:
            try:
                parent.output_ids[:] = list(winner.req.output_ids)
            except Exception:
                logger.exception("[tree] winner copy failed; parent keeps own text")

        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        for branch in run.branches.values():
            if branch.state != "active":
                continue
            branch.state = "finalized" if branch is winner else "pruned"
            if branch is not winner:
                run.pruned += 1
            target = branch.req
            if target is None:
                continue
            target.to_finish = FINISH_LENGTH(length=len(target.output_ids))

        self._attach_snapshot(run)


_ACTIVE: Optional[SchedulerTreeRuntime] = None


def get_active() -> Optional[SchedulerTreeRuntime]:
    return _ACTIVE


def install(scheduler: Any) -> SchedulerTreeRuntime:
    """Attach the tree runtime to a scheduler process (one per process)."""
    global _ACTIVE
    runtime = SchedulerTreeRuntime(scheduler)
    scheduler.tree_runtime = runtime
    _ACTIVE = runtime
    return runtime
