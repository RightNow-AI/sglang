"""Phase-1 tree runtime for the SGLang 0.5.15 scheduler (container base).

Parent request doubles as branch 0. By default, sibling branches are spawned
after prefill as ordinary requests whose prompts radix-share the parent. A
request may instead wait for a text delimiter and publish the parent's
generated KV under a fork-local cache namespace before spawning siblings.
Per-token hooks feed the tree manager; kills reuse the scheduler's abort path;
the parent remains the wire carrier for the final tree snapshot. All hooks are
defensive: a tree bug degrades to plain generation, never a scheduler crash.

KV ownership stays with the stock scheduler: children use ordinary intake, so
allocator exhaustion, retraction, finish, and release follow the non-tree
paths. Fan-out is rejected before intake above AUTOTREE_MAX_BRANCHES (64 by
default) to bound per-request pressure.

For multi-branch runs the parent is retained past natural EOS by setting
ignore_eos and adding 64 tokens to its limit. Normal tree finalization marks
every branch to finish. If the client aborts, or the retained parent otherwise
finishes first, cleanup marks every remaining child to finish and forgets all
runtime references; the scheduler then releases their KV through its normal
finish path.
"""

from __future__ import annotations

import copy
import logging
import os as _os
import secrets
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Intake and marginal-value scheduling knobs are environment-overridable so
# operators can bound tenant fan-out and run policy ablations without edits.
MAX_BRANCHES = int(_os.environ.get('AUTOTREE_MAX_BRANCHES', '64'))
VALUE_CHECK_INTERVAL = int(_os.environ.get('AUTOTREE_VALUE_CHECK_INTERVAL', '16'))
VALUE_WARMUP_TOKENS = int(_os.environ.get('AUTOTREE_VALUE_WARMUP_TOKENS', '8'))
# Default 0.8: measured on 12-task math at 1.5B, margins <= 0.5 prune
# minority-correct branches (accuracy loss); the naive logprob proxy
# cannot separate branches more finely. Lower this only with a scorer
# stronger than mean logprob (value head).
VALUE_MARGIN = float(_os.environ.get('AUTOTREE_VALUE_MARGIN', '0.8'))
VALUE_MIN_KEEP = int(_os.environ.get('AUTOTREE_VALUE_MIN_KEEP', '2'))


def _validate_branch_count(params: Dict[str, Any]) -> int:
    branches = int(params.get('branches', 1) or 1)
    if branches > MAX_BRANCHES:
        raise ValueError(
            f'tree branches={branches} exceeds configured maximum='
            f'{MAX_BRANCHES} (AUTOTREE_MAX_BRANCHES)'
        )
    return branches


class TokenizedTreeGenerateReqInput:
    """Envelope the tokenizer manager sends to the scheduler.

    Delegates every unknown attribute (read and write) to the wrapped
    TokenizedGenerateReqInput so the transport layer (time_stats, shm/pickle
    wrapping) treats it exactly like a plain tokenized request.
    """

    _OWN = ("base", "tree")

    def __init__(self, base: Any, tree: Dict[str, Any]) -> None:
        params = dict(tree)
        _validate_branch_count(params)
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

VALUE_CHECK_INTERVAL = int(_os.environ.get("AUTOTREE_VALUE_CHECK_INTERVAL", "16"))
VALUE_WARMUP_TOKENS = int(_os.environ.get("AUTOTREE_VALUE_WARMUP_TOKENS", "8"))
# Default 0.8: measured on 12-task math at 1.5B, margins <= 0.5 prune
# minority-correct branches (accuracy loss); the naive logprob proxy
# cannot separate branches more finely. Lower this only with a scorer
# stronger than mean logprob (value head).
VALUE_MARGIN = float(_os.environ.get("AUTOTREE_VALUE_MARGIN", "0.8"))
VALUE_MIN_KEEP = int(_os.environ.get("AUTOTREE_VALUE_MIN_KEEP", "2"))


class _BranchState:
    __slots__ = (
        "rid", "branch_id", "req", "tokens", "score", "state", "lp_seen",
        "final_answer",
    )

    def __init__(self, rid: str, branch_id: int, req: Any) -> None:
        self.rid = rid
        self.branch_id = branch_id
        self.req = req
        self.tokens = 0
        self.score = 0.0
        self.state = "active"
        self.lp_seen = 0
        self.final_answer = None  # extracted once the branch finishes

    def mean_logprob(self) -> float:
        return self.score / self.tokens if self.tokens else 0.0


class _TreeRun:
    __slots__ = (
        "parent_rid", "params", "branches", "spent", "finalized",
        "winner_branch_id", "pruned", "base_tokenized", "last_value_check",
        "orig_sampling", "forked", "fork_attempted", "cache_namespace",
        "fork_cache_supported",
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
        self.orig_sampling = None
        self.forked = False
        self.fork_attempted = False
        self.cache_namespace = None
        self.fork_cache_supported = False


class SchedulerTreeRuntime:
    """Owns live tree runs inside one scheduler process."""

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.runs: Dict[str, _TreeRun] = {}          # parent rid -> run
        self.branch_index: Dict[str, _TreeRun] = {}  # any member rid -> run

    # -- intake ------------------------------------------------------------

    def handle_tree_request(self, recv: TokenizedTreeGenerateReqInput):
        """Dispatcher target: route the parent through normal intake."""
        params = dict(recv.tree)
        _validate_branch_count(params)
        try:
            run = _TreeRun(recv.rid, params)
            branch_count = max(1, int(params.get("branches", 1) or 1))
            delayed_fork = params.get("fork_at_text") is not None and branch_count > 1
            if delayed_fork:
                tree_cache = getattr(self.scheduler, "tree_cache", None)
                run.fork_cache_supported = self._cache_supports_fork_namespaces(
                    tree_cache
                )
                if run.fork_cache_supported:
                    import msgspec

                    original_key = getattr(recv.base, "extra_key", None)
                    namespace = f"autotree-fork:{secrets.token_hex(16)}"
                    if original_key:
                        namespace = f"{original_key}|{namespace}"
                    recv.base = msgspec.structs.replace(
                        recv.base, extra_key=namespace
                    )
                    run.cache_namespace = namespace
                else:
                    logger.error(
                        "[tree] %s delayed fork disabled: cache %s does not "
                        "guarantee extra_key isolation",
                        recv.rid,
                        type(tree_cache).__name__ if tree_cache is not None else None,
                    )
            run.base_tokenized = recv.base
            self.runs[recv.rid] = run
            self.branch_index[recv.rid] = run
            # Hold the parent past its natural EOS: it is the wire carrier, and
            # winner selection needs every sibling finished before the parent's
            # final result (with the full tree snapshot) streams out. Siblings
            # keep the caller's original sampling; the budget stays the hard cap.
            sp = getattr(recv.base, "sampling_params", None)
            if sp is not None and branch_count > 1:
                try:
                    run.orig_sampling = (
                        getattr(sp, "max_new_tokens", None),
                        getattr(sp, "ignore_eos", False),
                    )
                    if not delayed_fork:
                        self._hold_parent(run, recv.base)
                except Exception:
                    run.orig_sampling = None
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
            if self._uses_delayed_fork(run):
                run.branches["0"] = _BranchState(req.rid, 0, req)
                self._maybe_trigger_delayed_fork(run, req)
                return
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

    def on_request_finished(self, req: Any) -> None:
        """Finish a delimiter-gated request that never reached its trigger."""
        run = self.branch_index.get(req.rid)
        if (
            run is None
            or run.finalized
            or req.rid != run.parent_rid
            or not self._uses_delayed_fork(run)
            or run.forked
        ):
            return
        parent = run.branches.get("0")
        if parent is None:
            parent = _BranchState(req.rid, 0, req)
            run.branches["0"] = parent
        parent.req = req
        parent.state = "finalized"
        run.finalized = True
        run.winner_branch_id = 0
        self._attach_snapshot(run, include_outputs=True)
        logger.info(
            "[tree] %s delimiter not found; returning parent only", run.parent_rid
        )

    # -- internals ---------------------------------------------------------

    def _cleanup_run(self, run: _TreeRun, reason: str) -> None:
        if not run.finalized:
            self._finalize(run, reason=reason)
        self.runs.pop(run.parent_rid, None)
        self.branch_index.pop(run.parent_rid, None)
        for branch in run.branches.values():
            self.branch_index.pop(branch.rid, None)

    @staticmethod
    def _uses_delayed_fork(run: _TreeRun) -> bool:
        return (
            run.params.get("fork_at_text") is not None
            and max(1, int(run.params.get("branches", 1) or 1)) > 1
        )

    @classmethod
    def _cache_supports_fork_namespaces(cls, tree_cache: Any) -> bool:
        """Allow only caches whose prefix lookup includes RadixKey.extra_key.

        The C++ radix cache currently passes raw token ids to its tree and is
        intentionally absent from this allowlist.
        """
        if tree_cache is None or bool(getattr(tree_cache, "disable", False)):
            return False
        marker = getattr(tree_cache, "supports_tree_fork_namespaces", None)
        if marker is not None:
            return bool(marker)
        inner = getattr(tree_cache, "inner", None)
        if inner is not None and inner is not tree_cache:
            return cls._cache_supports_fork_namespaces(inner)
        safe_bases = {
            ("sglang.srt.mem_cache.radix_cache", "RadixCache"),
            ("sglang.srt.mem_cache.swa_radix_cache", "SWARadixCache"),
            ("sglang.srt.mem_cache.mamba_radix_cache", "MambaRadixCache"),
            ("sglang.srt.mem_cache.unified_radix_cache", "UnifiedRadixCache"),
        }
        return any(
            (base.__module__, base.__name__) in safe_bases
            for base in type(tree_cache).__mro__
        )

    def _hold_parent(self, run: _TreeRun, parent: Any) -> None:
        sp = getattr(parent, "sampling_params", None)
        if sp is None or run.orig_sampling is None:
            return
        original_max, _ = run.orig_sampling
        if original_max is not None:
            sp.max_new_tokens = original_max + 64
        sp.ignore_eos = True

    def _restore_parent_sampling(self, run: _TreeRun, parent_req: Any) -> None:
        sp = getattr(parent_req, "sampling_params", None)
        if sp is None or run.orig_sampling is None:
            return
        sp.max_new_tokens, sp.ignore_eos = run.orig_sampling

    def _maybe_trigger_delayed_fork(self, run: _TreeRun, parent_req: Any) -> bool:
        if run.forked or run.fork_attempted:
            return False
        delimiter = run.params.get("fork_at_text")
        tokenizer = getattr(self.scheduler, "tokenizer", None)
        if not delimiter or tokenizer is None:
            return False
        output_ids = list(getattr(parent_req, "output_ids", ()))
        if not output_ids:
            return False
        tail_tokens = max(24, len(delimiter) + 4)
        try:
            tail = tokenizer.decode(
                output_ids[-tail_tokens:], skip_special_tokens=False
            )
        except Exception:
            logger.exception("[tree] delimiter decode failed for %s", run.parent_rid)
            return False
        if delimiter not in tail:
            return False

        run.fork_attempted = True
        if not run.fork_cache_supported:
            logger.error(
                "[tree] %s delimiter reached but fork was refused: no isolated "
                "radix namespace",
                run.parent_rid,
            )
            return True

        fork_output_len = len(output_ids)
        try:
            self._cache_parent_generated(parent_req, fork_output_len)
            self._hold_parent(run, parent_req)
            parent = run.branches["0"]
            parent.tokens = 0
            parent.score = 0.0
            vals = getattr(parent_req, "output_token_logprobs_val", None)
            parent.lp_seen = len(vals) if vals else 0
            run.spent = 0
            run.last_value_check = 0
            child_input_ids = (
                parent_req.origin_input_ids
                + parent_req.output_ids[:fork_output_len]
            )
            self._fork_branches(
                run, parent_req, child_input_ids=child_input_ids
            )
            logger.info(
                "[tree] forked %d at delimiter (k=%d)",
                max(0, int(run.params.get("branches", 1) or 1) - 1),
                fork_output_len,
            )
        except Exception:
            if len(run.branches) <= 1:
                self._restore_parent_sampling(run, parent_req)
            logger.exception("[tree] delimiter fork failed; parent continues alone")
        return True

    def _cache_parent_generated(
        self, parent_req: Any, fork_output_len: int
    ) -> None:
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        if not self._cache_supports_fork_namespaces(tree_cache):
            raise RuntimeError("tree cache does not guarantee fork namespace isolation")
        if getattr(parent_req, "skip_radix_cache_insert", False):
            raise RuntimeError("request is not eligible for radix cache insertion")

        fork_input_len = len(parent_req.origin_input_ids) + fork_output_len
        max_reusable_len = max(0, fork_input_len - 1)
        committed_len = min(
            int(getattr(parent_req, "kv_committed_len", max_reusable_len)),
            max_reusable_len,
        )
        if committed_len <= 0:
            raise RuntimeError("parent has no committed KV to publish")
        refresh = getattr(parent_req, "_refresh_fill_ids", None)
        set_range = getattr(parent_req, "set_extend_range", None)
        if refresh is None or set_range is None:
            raise RuntimeError("parent request cannot expose committed fill ids")

        old_range = parent_req.extend_range
        refresh()
        set_range(0, committed_len)
        try:
            tree_cache.cache_unfinished_req(parent_req)
        finally:
            parent_req.extend_range = old_range

    def _fork_branches(
        self, run: _TreeRun, parent_req: Any, child_input_ids: Any = None
    ) -> None:
        n = max(1, int(run.params.get("branches", 1)))
        parent = run.branches.get("0")
        if parent is None:
            run.branches["0"] = _BranchState(parent_req.rid, 0, parent_req)
        else:
            parent.req = parent_req

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
            if child_sp is not None and run.orig_sampling is not None:
                # children keep the caller's sampling; only the parent is held
                try:
                    child_sp.max_new_tokens, child_sp.ignore_eos = run.orig_sampling
                except Exception:
                    pass
            seed = getattr(child_sp, "seed", None)
            if child_sp is not None and seed is not None:
                try:
                    child_sp.seed = seed + b
                except Exception:
                    pass
            replacements = {"rid": child_rid, "sampling_params": child_sp}
            if child_input_ids is not None:
                replacements["input_ids"] = child_input_ids
            child_base = msgspec.structs.replace(base, **replacements)
            # Route through the real intake path: all Req invariants and radix
            # prefix-sharing are established exactly as for a normal request.
            self.scheduler.handle_generate_request(child_base)
            branch = _BranchState(child_rid, b, None)
            run.branches[str(b)] = branch
            self.branch_index[child_rid] = run
        run.forked = n > 1
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

        if state.branch_id == 0 and self._uses_delayed_fork(run) and not run.forked:
            self._maybe_trigger_delayed_fork(run, req)
            return

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
            self._maybe_majority_lock(run)

        if state.branch_id == 0:
            self._attach_snapshot(run)
            if not run.finalized and len(run.branches) > 1:
                siblings = [
                    b for b in run.branches.values() if b.branch_id != 0
                ]
                if siblings and all(
                    b.req is not None and b.req.finished() for b in siblings
                ):
                    self._finalize(run, reason="siblings_done")
                    return

        budget = int(run.params.get("budget_tokens", 0) or 0)
        if budget and run.spent >= budget and not run.finalized:
            self._finalize(run, reason="budget")

    def _extract_branch_answer(self, branch: _BranchState) -> Optional[str]:
        """Decode a finished branch and extract its final numeric answer
        ('#### N' preferred, else the last number). Cached per branch."""
        if branch.final_answer is not None:
            return branch.final_answer or None
        tokenizer = getattr(self.scheduler, "tokenizer", None)
        if tokenizer is None or branch.req is None:
            return None
        import re

        ids = list(branch.req.output_ids)
        eos = getattr(tokenizer, "eos_token_id", None)
        if branch.branch_id == 0:
            # The held parent votes only once its natural EOS has appeared:
            # everything before it is the parent's immutable final answer
            # (serving trims at the same point), everything after is
            # scaffolding from the hold.
            if eos is None or eos not in ids:
                return None
            ids = ids[: ids.index(eos)]
        elif eos is not None and eos in ids:
            ids = ids[: ids.index(eos)]
        try:
            text = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            branch.final_answer = ""
            return None
        marked = re.findall(r"####\s*([-+]?[\d.,]+)", text)
        raw = marked[-1] if marked else None
        if raw is None:
            nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
            raw = nums[-1] if nums else None
        if raw is None:
            branch.final_answer = ""
            return None
        raw = raw.replace(",", "").rstrip(".")
        try:
            value = float(raw)
            answer = str(int(value)) if value == int(value) else str(value)
        except ValueError:
            branch.final_answer = ""
            return None
        branch.final_answer = answer
        return answer

    def _maybe_majority_lock(self, run: _TreeRun) -> None:
        """Zero-accuracy-cost early termination: once finished branches agree
        on an answer that the still-running branches can no longer outvote,
        the tree's outcome is decided - finalize immediately and reclaim every
        remaining token. Safe by construction with respect to the final vote."""
        if run.finalized or len(run.branches) < 3:
            return
        total = len(run.branches)
        needed = total // 2 + 1
        counts: Dict[str, int] = {}
        for b in run.branches.values():
            if b.req is None:
                continue
            if b.branch_id != 0 and not b.req.finished():
                continue
            # branch 0 (the held parent) is eligible once its natural EOS has
            # appeared; _extract_branch_answer returns None before that.
            answer = self._extract_branch_answer(b)
            if answer:
                counts[answer] = counts.get(answer, 0) + 1
        if not counts:
            return
        top_answer, top_count = max(counts.items(), key=lambda kv: kv[1])
        if top_count >= needed:
            logger.info(
                "[tree] %s majority locked on '%s' (%d/%d finished votes); "
                "terminating remaining branches early",
                run.parent_rid, top_answer, top_count, total,
            )
            self._finalize(run, reason="majority_locked")

    def _attach_snapshot(self, run: _TreeRun, include_outputs: bool = False) -> None:
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
                    **(
                        {"output_ids": list(b.req.output_ids)}
                        if include_outputs and b.req is not None
                        else {}
                    ),
                }
                for b in run.branches.values()
            },
        }
        # The customized_info channel is token-aligned: the output streamer
        # slices the value list by the token range of each chunk. Place the
        # snapshot at the parent's newest token index (not yet streamed) so it
        # rides out on the next chunk; earlier indices are padding.
        n = len(parent.req.output_ids)
        values: list = [None] * max(n, 1)
        values[-1] = snapshot
        parent.req.customized_info = {"autotree": values}

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

        # NOTE: never mutate parent.output_ids here. Rewriting the token array
        # post-hoc desyncs the KV allocator's page accounting (measured: pool
        # leak abort from the invariant checker). The serving layer reconstructs
        # the winner's text from the final snapshot instead.

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

        self._attach_snapshot(run, include_outputs=True)


_ACTIVE: Optional[SchedulerTreeRuntime] = None


def get_active() -> Optional[SchedulerTreeRuntime]:
    runtime = _ACTIVE
    if runtime is None:
        return None
    for run in list(runtime.runs.values()):
        parent = run.branches.get("0")
        if (
            parent is not None
            and parent.req is not None
            and parent.req.finished()
        ):
            runtime._cleanup_run(run, reason="parent_left")
    return runtime if any(not run.finalized for run in runtime.runs.values()) else None


def install(scheduler: Any) -> SchedulerTreeRuntime:
    """Attach the tree runtime to a scheduler process (one per process)."""
    global _ACTIVE
    runtime = SchedulerTreeRuntime(scheduler)
    scheduler.tree_runtime = runtime
    original_abort = getattr(scheduler, "abort_request", None)
    if callable(original_abort):
        def abort_request(recv_req):
            if recv_req.abort_all:
                runs = list(runtime.runs.values())
            else:
                run = runtime.runs.get(recv_req.rid)
                runs = [run] if run is not None else []
            for run in runs:
                runtime._cleanup_run(run, reason="parent_abort")
            return original_abort(recv_req)

        scheduler.abort_request = abort_request
    _ACTIVE = runtime
    return runtime
