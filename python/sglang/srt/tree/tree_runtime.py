"""Phase-1 tree runtime for the SGLang 0.5.15 scheduler (container base).

Parent request doubles as branch 0. By default, sibling branches are spawned
after prefill as ordinary requests whose prompts radix-share the parent. A
request may instead wait for a text delimiter or chosen-logprob uncertainty
trigger, then publish the parent's generated KV under a fork-local cache
namespace before spawning siblings.
Per-token hooks feed the tree manager; kills reuse the scheduler's abort path;
the parent remains the wire carrier for the final tree snapshot. All hooks are
defensive: a tree bug degrades to plain generation, never a scheduler crash.

KV ownership stays with the stock scheduler: children use ordinary intake, so
allocator exhaustion, retraction, finish, and release follow the non-tree
paths. Fan-out is rejected before intake above AUTOTREE_MAX_BRANCHES (64 by
default) to bound per-request pressure.

For multi-branch runs the parent finishes under the caller's original sampling
limits. Its final wire output is deferred until sibling selection completes,
while the stock finish path releases its KV immediately. Normal finalization
marks every remaining branch to finish. Client abort cleanup resolves a deferred
parent response, finishes every child, and forgets all runtime references.
"""

from __future__ import annotations

import copy
import logging
import os as _os
import secrets
from collections import deque
from typing import Any, Dict, Optional

from sglang.srt.tree import selection
from sglang.srt.tree.consensus import ConsensusConfig, consensus_scores
from sglang.srt.tree.params import (
    DEFAULT_CONSENSUS_INTERVAL,
    DEFAULT_CONSENSUS_WARMUP,
    DEFAULT_MIN_SURVIVORS,
    normalize_tree_params,
    validate_tree_params,
)
from sglang.srt.tree.profile import ENABLED as _AUTOTREE_PROFILE_ENABLED
from sglang.srt.tree.profile import incr as _autotree_profile_incr
from sglang.srt.tree.profile import span as _autotree_profile_span
from sglang.srt.tree.shared_prefix import SharedPrefixGroup

logger = logging.getLogger(__name__)

# Intake and marginal-value scheduling knobs are environment-overridable so
# operators can bound tenant fan-out and run policy ablations without edits.
MAX_BRANCHES = selection.env_int("AUTOTREE_MAX_BRANCHES", 64)
ENTROPY_FORK_WINDOW_SIZE = 8
# Entropy-triggered runs must eventually fork even when the chosen-logprob
# proxy stays confident. The budget-derived cap may raise this floor.
ENTROPY_FORK_STARVATION_MIN_TOKENS = 64
def _early_parent_stop_enabled() -> bool:
    return _os.environ.get("AUTOTREE_EARLY_PARENT_STOP") == "1"


def _validate_branch_count(params: Dict[str, Any]) -> int:
    validate_tree_params(params)
    branches = params["branches"]
    if branches > MAX_BRANCHES:
        raise ValueError(
            f"tree branches={branches} exceeds configured maximum="
            f"{MAX_BRANCHES} (AUTOTREE_MAX_BRANCHES)"
        )
    adaptive_width = params.get("adaptive_width")
    if adaptive_width is not None:
        if not isinstance(adaptive_width, int) or isinstance(adaptive_width, bool):
            raise ValueError("adaptive_width must be an integer")
        if adaptive_width <= branches:
            raise ValueError("adaptive_width must be greater than branches")
        if adaptive_width > MAX_BRANCHES:
            raise ValueError(
                f"adaptive_width={adaptive_width} exceeds configured maximum="
                f"{MAX_BRANCHES} (AUTOTREE_MAX_BRANCHES)"
            )
    if params.get("scorer") == "self_consistency":
        consensus_warmup = params.get(
            "consensus_warmup", DEFAULT_CONSENSUS_WARMUP
        )
        consensus_interval = params.get(
            "consensus_interval", DEFAULT_CONSENSUS_INTERVAL
        )
        min_survivors = params.get("min_survivors", DEFAULT_MIN_SURVIVORS)
        if (
            not isinstance(consensus_warmup, int)
            or isinstance(consensus_warmup, bool)
            or consensus_warmup < 0
        ):
            raise ValueError("consensus_warmup must be a non-negative integer")
        if (
            not isinstance(consensus_interval, int)
            or isinstance(consensus_interval, bool)
            or consensus_interval <= 0
        ):
            raise ValueError("consensus_interval must be a positive integer")
        if (
            not isinstance(min_survivors, int)
            or isinstance(min_survivors, bool)
            or min_survivors <= 0
        ):
            raise ValueError("min_survivors must be a positive integer")
    return branches


class TokenizedTreeGenerateReqInput:
    """Envelope the tokenizer manager sends to the scheduler.

    Delegates every unknown attribute (read and write) to the wrapped
    TokenizedGenerateReqInput so the transport layer (time_stats, shm/pickle
    wrapping) treats it exactly like a plain tokenized request.
    """

    _OWN = ("base", "tree")

    def __init__(self, base: Any, tree: Dict[str, Any]) -> None:
        params = normalize_tree_params(tree)
        _validate_branch_count(params)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "tree", params)

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

VALUE_CHECK_INTERVAL = selection.env_int("AUTOTREE_VALUE_CHECK_INTERVAL", 16)
VALUE_WARMUP_TOKENS = selection.env_int("AUTOTREE_VALUE_WARMUP_TOKENS", 8)
# Tail-phase snapshot margin: when the parent is within this many tokens of
# its cap, periodic snapshots start carrying branch outputs (see the
# branch-0 attach site for the delivery-race rationale).
TAIL_SNAPSHOT_PARENT_MARGIN = selection.env_int("AUTOTREE_TAIL_SNAPSHOT_MARGIN", 32)
# Default 0.8: measured on 12-task math at 1.5B, margins <= 0.5 prune
# minority-correct branches (accuracy loss); the naive logprob proxy
# cannot separate branches more finely. Lower this only with a scorer
# stronger than mean logprob (value head).
VALUE_MARGIN = selection.env_float("AUTOTREE_VALUE_MARGIN", 0.8)
VALUE_MIN_KEEP = selection.env_int("AUTOTREE_VALUE_MIN_KEEP", 2)
ADAPT_MARGIN = selection.env_float("AUTOTREE_ADAPT_MARGIN", 2.0)


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
        "parent_rid", "params", "branches", "branches_by_rid", "spent",
        "finalized",
        "winner_branch_id", "pruned", "base_tokenized", "last_value_check",
        "orig_sampling", "forked", "fork_attempted", "cache_namespace",
        "fork_cache_supported", "shared_prefix_group", "entropy_logprobs",
        "entropy_lp_seen", "tail_snapshot_active", "finished_sibling_rids",
        "fork_input_ids", "adaptive_failed", "tail_tokens_saved",
        "parent_output_deferred",
        "last_consensus_check",
    )

    def __init__(self, parent_rid: str, params: Dict[str, Any]) -> None:
        self.parent_rid = parent_rid
        self.params = params
        self.base_tokenized = None
        self.branches: Dict[str, _BranchState] = {}
        self.branches_by_rid: Dict[str, _BranchState] = {}
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
        self.shared_prefix_group: Optional[SharedPrefixGroup] = None
        self.entropy_logprobs = deque(maxlen=ENTROPY_FORK_WINDOW_SIZE)
        self.entropy_lp_seen = 0
        self.tail_snapshot_active = False
        self.finished_sibling_rids = set()
        self.fork_input_ids = None
        self.adaptive_failed = False
        self.tail_tokens_saved = 0
        self.parent_output_deferred = False
        self.last_consensus_check: Optional[int] = None


class SchedulerTreeRuntime:
    """Owns live tree runs inside one scheduler process."""

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.runs: Dict[str, _TreeRun] = {}          # parent rid -> run
        self.branch_index: Dict[str, _TreeRun] = {}  # any member rid -> run

    def get_shared_prefix_group(self, rid: str) -> Optional[SharedPrefixGroup]:
        run = self.branch_index.get(rid)
        return run.shared_prefix_group if run is not None else None

    def _register_branch(self, run: _TreeRun, branch: _BranchState) -> None:
        run.branches[str(branch.branch_id)] = branch
        run.branches_by_rid[branch.rid] = branch
        self.branch_index[branch.rid] = run

    # -- intake ------------------------------------------------------------

    def handle_tree_request(self, recv: TokenizedTreeGenerateReqInput):
        """Dispatcher target: route the parent through normal intake."""
        params = dict(recv.tree)
        branch_count = _validate_branch_count(params)
        try:
            run = _TreeRun(recv.rid, params)
            delayed_fork = self._uses_delayed_fork(run)
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
                self._register_branch(run, _BranchState(req.rid, 0, req))
                if run.params.get("fork_at_entropy") is not None:
                    self._record_entropy_logprobs(run, req, None)
                self._maybe_trigger_delayed_fork(run, req)
                return
            self._fork_branches(run, req)
            self._park_parent_if_complete(run)
        except Exception:
            logger.exception("[tree] fork failed; parent continues alone")
            if not run.branches:
                self._register_branch(run, _BranchState(req.rid, 0, req))

    def on_token(self, req: Any, token_ids, logprob: Optional[float]) -> None:
        run = self.branch_index.get(req.rid)
        if run is None or run.finalized:
            return
        try:
            self._account_tokens(run, req, token_ids, logprob)
        except Exception:
            logger.exception("[tree] token hook failed for %s", req.rid)

    def on_request_finished(self, req: Any) -> None:
        """Track tail entry or finish a delayed fork that never triggered."""
        run = self.branch_index.get(req.rid)
        if run is None or run.finalized:
            return
        if req.rid != run.parent_rid:
            run.finished_sibling_rids.add(req.rid)
            run.tail_snapshot_active = True
            parent = run.branches.get("0")
            if (
                parent is not None
                and parent.req is not None
                and parent.req.finished()
                and self._all_siblings_finished(run)
            ):
                self._finalize(run, reason="siblings_done")
                return
            # If the parent's newest token already streamed, wait for its next
            # token so the final snapshot has a chunk to ride.
            if (
                _early_parent_stop_enabled()
                and self._parent_has_unstreamed_token(run)
            ):
                self._maybe_stop_parent_early(run)
            return
        if run.forked and self._all_siblings_finished(run):
            self._finalize(run, reason="siblings_done")
            return
        if not self._uses_delayed_fork(run) or run.forked:
            return
        parent = run.branches.get("0")
        if parent is None:
            parent = _BranchState(req.rid, 0, req)
            self._register_branch(run, parent)
        parent.req = req
        parent.state = "finalized"
        self._restore_parent_sampling(run, req)
        run.finalized = True
        run.winner_branch_id = 0
        self._attach_snapshot(run, include_outputs=True)
        logger.info(
            "[tree] %s delayed fork not triggered; returning parent only",
            run.parent_rid,
        )

    # -- internals ---------------------------------------------------------

    def _forget_run(self, run: _TreeRun) -> None:
        self.runs.pop(run.parent_rid, None)
        self.branch_index.pop(run.parent_rid, None)
        for branch in run.branches.values():
            self.branch_index.pop(branch.rid, None)

    def _cleanup_run(self, run: _TreeRun, reason: str) -> None:
        if reason == "parent_abort" and run.parent_output_deferred:
            self._abort_deferred_parent(run)
        if not run.finalized:
            self._finalize(run, reason=reason)
        self._forget_run(run)

    def should_defer_parent_output(self, req: Any) -> bool:
        """Retain a naturally finished parent response until tree finalization."""
        run = self.branch_index.get(getattr(req, "rid", None))
        if (
            run is None
            or run.finalized
            or not run.forked
            or req.rid != run.parent_rid
            or not req.finished()
        ):
            return False
        run.parent_output_deferred = True
        return True

    def _stream_deferred_parent(self, run: _TreeRun) -> bool:
        if not run.parent_output_deferred:
            return False
        parent = run.branches.get("0")
        req = parent.req if parent is not None else None
        streamer = getattr(getattr(self.scheduler, "output_streamer", None), "stream_output", None)
        if req is None or not callable(streamer):
            logger.error("[tree] %s cannot emit deferred parent output", run.parent_rid)
            return False
        try:
            streamer([req], bool(getattr(req, "return_logprob", False)))
        except Exception:
            logger.exception(
                "[tree] %s failed to emit deferred parent output", run.parent_rid
            )
            return False
        run.parent_output_deferred = False
        self._forget_run(run)
        return True

    def _abort_deferred_parent(self, run: _TreeRun) -> None:
        parent = run.branches.get("0")
        if parent is None or parent.req is None:
            run.parent_output_deferred = False
            return
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        parent.req.to_finish = FINISH_ABORT()
        parent.state = "finalized"
        self._stream_deferred_parent(run)

    @staticmethod
    def _uses_delayed_fork(run: _TreeRun) -> bool:
        return (
            (
                run.params.get("fork_at_text") is not None
                or run.params.get("fork_at_entropy") is not None
            )
            and max(1, int(run.params.get("branches", 1) or 1)) > 1
        )

    @staticmethod
    def _consensus_enabled(run: _TreeRun) -> bool:
        return run.params.get("scorer") == "self_consistency"

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
        """Keep the legacy carrier cap while runtime hooks park at the real limit."""
        sp = getattr(parent, "sampling_params", None)
        if sp is None or run.orig_sampling is None:
            return
        original_max, _ = run.orig_sampling
        if original_max is not None:
            sp.max_new_tokens = original_max + 64
        sp.ignore_eos = True

    def _park_parent_if_complete(self, run: _TreeRun) -> bool:
        """Finish the carrier at natural EOS or the caller's original token cap."""
        if not run.forked:
            return False
        parent = run.branches.get("0")
        if parent is None or parent.req is None or run.orig_sampling is None:
            return False
        original_max, _ = run.orig_sampling
        reached_original_cap = (
            original_max is not None
            and len(parent.req.output_ids) >= int(original_max)
        )
        if not self._parent_has_natural_eos(run) and not reached_original_cap:
            return False

        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        parent.req.to_finish = FINISH_LENGTH(length=len(parent.req.output_ids))
        return True

    def _parent_has_natural_eos(self, run: _TreeRun) -> bool:
        parent = run.branches.get("0")
        tokenizer = getattr(self.scheduler, "tokenizer", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        return bool(
            parent is not None
            and parent.req is not None
            and eos is not None
            and eos in parent.req.output_ids
        )

    @staticmethod
    def _parent_has_unstreamed_token(run: _TreeRun) -> bool:
        parent = run.branches.get("0")
        if parent is None or parent.req is None:
            return False
        send_offset = getattr(parent.req, "send_token_offset", 0)
        try:
            return int(send_offset) < len(parent.req.output_ids)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _all_siblings_finished(run: _TreeRun) -> bool:
        sibling_count = len(run.branches) - 1
        return (
            sibling_count > 0
            and len(run.finished_sibling_rids) >= sibling_count
        )

    def _maybe_stop_parent_early(self, run: _TreeRun) -> bool:
        """Stop the held parent only after its vote and the outcome are final."""
        if (
            not _early_parent_stop_enabled()
            or not self._parent_has_natural_eos(run)
        ):
            return False
        if run.finalized:
            return True

        self._maybe_majority_lock(run)
        if run.finalized:
            return True
        if not self._all_siblings_finished(run):
            return False

        self._finalize(run, reason="siblings_done")
        return True

    @staticmethod
    def _record_parent_tail_tokens_saved(run: _TreeRun) -> None:
        parent = run.branches.get("0")
        if parent is None or parent.req is None:
            return
        sampling_params = getattr(parent.req, "sampling_params", None)
        parent_cap = getattr(sampling_params, "max_new_tokens", None)
        if parent_cap is None:
            return
        generated = len(parent.req.output_ids)
        try:
            parent_cap = int(parent_cap)
        except (TypeError, ValueError):
            return
        run.tail_tokens_saved = max(0, parent_cap - generated)
        logger.info(
            "[tree] %s early parent stop saved %d tail tokens "
            "(cap=%d generated=%d)",
            run.parent_rid,
            run.tail_tokens_saved,
            parent_cap,
            generated,
        )

    def _restore_parent_sampling(self, run: _TreeRun, parent_req: Any) -> None:
        sp = getattr(parent_req, "sampling_params", None)
        if sp is None or run.orig_sampling is None:
            return
        sp.max_new_tokens, sp.ignore_eos = run.orig_sampling

    def _maybe_trigger_delayed_fork(self, run: _TreeRun, parent_req: Any) -> bool:
        if run.forked or run.fork_attempted:
            return False
        delimiter = run.params.get("fork_at_text")
        output_ids = list(getattr(parent_req, "output_ids", ()))
        if not output_ids:
            return False
        trigger = None
        if delimiter is not None:
            tokenizer = getattr(self.scheduler, "tokenizer", None)
            if not delimiter or tokenizer is None:
                return False
            tail_tokens = max(24, len(delimiter) + 4)
            try:
                tail = tokenizer.decode(
                    output_ids[-tail_tokens:], skip_special_tokens=False
                )
            except Exception:
                logger.exception(
                    "[tree] delimiter decode failed for %s", run.parent_rid
                )
                return False
            if delimiter not in tail:
                return False
            trigger = "delimiter"
        else:
            threshold = run.params.get("fork_at_entropy")
            if threshold is None:
                return False
            if len(run.entropy_logprobs) == ENTROPY_FORK_WINDOW_SIZE:
                # This is not distribution entropy. It is a windowed uncertainty
                # proxy using the mean negative chosen-token logprob in nats.
                mean_negative_logprob = -sum(run.entropy_logprobs) / len(
                    run.entropy_logprobs
                )
                if mean_negative_logprob > float(threshold):
                    trigger = (
                        "chosen-logprob uncertainty "
                        f"(mean_nll={mean_negative_logprob:.4f})"
                    )
            starvation_cap = self._entropy_fork_starvation_cap(run)
            if trigger is None and len(output_ids) >= starvation_cap:
                trigger = f"starvation fallback (cap={starvation_cap})"
            if trigger is None:
                return False

        return self._trigger_delayed_fork(run, parent_req, output_ids, trigger)

    @staticmethod
    def _entropy_fork_starvation_cap(run: _TreeRun) -> int:
        branches = max(1, int(run.params.get("branches", 1) or 1))
        budget = max(0, int(run.params.get("budget_tokens", 0) or 0))
        return max(
            ENTROPY_FORK_STARVATION_MIN_TOKENS,
            budget // branches // 4,
        )

    @staticmethod
    def _request_output_logprobs(req: Any):
        vals = getattr(req, "output_token_logprobs_val", None)
        if vals is not None:
            return vals
        logprob_state = getattr(req, "logprob", None)
        return getattr(logprob_state, "output_token_logprobs_val", None)

    @staticmethod
    def _as_logprob_values(logprob: Any):
        if logprob is None:
            return []
        try:
            return [float(value) for value in logprob]
        except TypeError:
            return [float(logprob)]

    def _record_entropy_logprobs(
        self, run: _TreeRun, parent_req: Any, logprob: Any
    ) -> None:
        stored = self._request_output_logprobs(parent_req)
        if stored is not None and len(stored) > run.entropy_lp_seen:
            fresh = stored[run.entropy_lp_seen:]
            run.entropy_logprobs.extend(float(value) for value in fresh)
            run.entropy_lp_seen = len(stored)

        current = self._as_logprob_values(logprob)
        if current:
            run.entropy_logprobs.extend(current)
            # The scheduler stores these values after the tree token hook runs.
            run.entropy_lp_seen += len(current)

    def _trigger_delayed_fork(
        self,
        run: _TreeRun,
        parent_req: Any,
        output_ids: Any,
        trigger: str,
    ) -> bool:
        run.fork_attempted = True
        if not run.fork_cache_supported:
            logger.error(
                "[tree] %s %s reached but fork was refused: no isolated "
                "radix namespace",
                run.parent_rid,
                trigger,
            )
            return True

        fork_output_len = len(output_ids)
        try:
            self._cache_parent_generated(parent_req, fork_output_len)
            self._hold_parent(run, parent_req)
            parent = run.branches["0"]
            parent.tokens = 0
            parent.score = 0.0
            vals = self._request_output_logprobs(parent_req)
            parent.lp_seen = max(
                len(vals) if vals else 0,
                run.entropy_lp_seen,
            )
            run.spent = 0
            run.last_value_check = 0
            run.last_consensus_check = None
            run.entropy_logprobs.clear()
            child_input_ids = (
                parent_req.origin_input_ids
                + parent_req.output_ids[:fork_output_len]
            )
            self._fork_branches(
                run, parent_req, child_input_ids=child_input_ids
            )
            self._park_parent_if_complete(run)
            logger.info(
                "[tree] forked %d at %s (k=%d)",
                max(0, int(run.params.get("branches", 1) or 1) - 1),
                trigger,
                fork_output_len,
            )
        except Exception:
            if len(run.branches) <= 1:
                self._restore_parent_sampling(run, parent_req)
            logger.exception(
                "[tree] %s fork failed; parent continues alone", trigger
            )
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
        self,
        run: _TreeRun,
        parent_req: Any,
        child_input_ids: Any = None,
        target_count: Optional[int] = None,
    ) -> None:
        n = max(
            1,
            int(
                target_count
                if target_count is not None
                else run.params.get("branches", 1)
            ),
        )
        if n > MAX_BRANCHES:
            raise ValueError(
                f"tree branches={n} exceeds configured maximum="
                f"{MAX_BRANCHES} (AUTOTREE_MAX_BRANCHES)"
            )
        parent = run.branches.get("0")
        if parent is None:
            self._register_branch(run, _BranchState(parent_req.rid, 0, parent_req))
        else:
            parent.req = parent_req
            run.branches_by_rid[parent.rid] = parent

        base = run.base_tokenized
        if base is None:
            logger.warning("[tree] %s no tokenized base; parent runs alone",
                           run.parent_rid)
            return

        import msgspec

        if child_input_ids is not None:
            run.fork_input_ids = child_input_ids
        elif run.fork_input_ids is not None:
            child_input_ids = run.fork_input_ids

        spawned = 0
        for b in range(1, n):
            if len(run.branches) >= n:
                break
            if str(b) in run.branches:
                continue
            child_rid = f"{run.parent_rid}#tree{b}"
            sp = getattr(base, "sampling_params", None)
            child_sp = copy.copy(sp) if sp is not None else sp
            if child_sp is not None and run.orig_sampling is not None:
                # children keep the caller's sampling; only the parent is held
                try:
                    child_sp.max_new_tokens, child_sp.ignore_eos = run.orig_sampling
                    # Benchmark mode: force every branch to run to max_new_tokens
                    # (ignore_eos) so all B branches generate an identical, fixed
                    # token count. This makes ms/token comparisons apples-to-apples
                    # by removing natural-EOS / siblings-done early termination.
                    if _os.environ.get("AUTOTREE_BENCH_FIXED_LEN") == "1":
                        child_sp.ignore_eos = True
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
            self._register_branch(run, branch)
            spawned += 1
        run.forked = len(run.branches) > 1
        if run.forked:
            try:
                shared_input_ids = (
                    child_input_ids
                    if child_input_ids is not None
                    else parent_req.origin_input_ids
                )
                branches = sorted(
                    run.branches.values(), key=lambda branch: branch.branch_id
                )
                run.shared_prefix_group = SharedPrefixGroup(
                    rids=[branch.rid for branch in branches],
                    shared_len=len(shared_input_ids),
                    branch_ids=[branch.branch_id for branch in branches],
                )
            except Exception:
                logger.exception(
                    "[tree] %s shared-prefix tagging failed", run.parent_rid
                )
        logger.info(
            "[tree] %s forked %d sibling branches via intake path",
            run.parent_rid, spawned,
        )

    def trim_tokens_to_budget(
        self,
        req: Any,
        token_ids: Any,
        logprobs: Any = None,
        *,
        reserved: int = 0,
    ):
        """Trim one accepted token run to the remaining shared tree budget."""
        run = self.branch_index.get(getattr(req, "rid", None))
        if run is None or run.finalized:
            return token_ids, logprobs

        tokens = list(token_ids) if hasattr(token_ids, "__len__") else [token_ids]
        allowed = len(tokens)

        budget = int(run.params.get("budget_tokens", 0) or 0)
        if budget > 0:
            remaining_budget = max(
                0,
                budget - run.spent - max(0, int(reserved)),
            )
            allowed = min(allowed, remaining_budget)

        if req.rid == run.parent_rid and run.orig_sampling is not None:
            original_max, _ = run.orig_sampling
            if original_max is not None:
                remaining_parent = max(
                    0,
                    int(original_max) - len(getattr(req, "output_ids", ())),
                )
                allowed = min(allowed, remaining_parent)
            tokenizer = getattr(self.scheduler, "tokenizer", None)
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and eos in tokens[:allowed]:
                allowed = tokens.index(eos) + 1

        if len(tokens) <= allowed:
            return token_ids, logprobs

        tokens = tokens[:allowed]
        if logprobs is None:
            trimmed_logprobs = None
        elif hasattr(logprobs, "__len__"):
            trimmed_logprobs = list(logprobs)[:allowed]
        else:
            trimmed_logprobs = logprobs if allowed else None
        return tokens, trimmed_logprobs

    def _account_tokens(
        self, run: _TreeRun, req: Any, token_ids, logprob: Optional[float]
    ) -> None:
        state = run.branches_by_rid.get(req.rid)
        if state is None:
            # Compatibility for runs assembled by older callers: pay the scan
            # once, then index every later token for this branch.
            state = next(
                (branch for branch in run.branches.values() if branch.rid == req.rid),
                None,
            )
            if state is not None:
                run.branches_by_rid[req.rid] = state
        if state is None or state.state != "active":
            return
        if state.req is None:
            state.req = req  # capture the real Req the scheduler created

        if state.branch_id == 0 and self._uses_delayed_fork(run) and not run.forked:
            if run.params.get("fork_at_entropy") is not None:
                self._record_entropy_logprobs(run, req, logprob)
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

        if state.branch_id == 0:
            self._park_parent_if_complete(run)

        consensus_enabled = self._consensus_enabled(run)
        if consensus_enabled:
            self._maybe_consensus_prune(run)

        if run.spent - run.last_value_check >= VALUE_CHECK_INTERVAL:
            run.last_value_check = run.spent
            if not consensus_enabled:
                self._maybe_value_prune(run)
            if _os.environ.get("AUTOTREE_BENCH_FIXED_LEN") != "1":
                self._maybe_majority_lock(run)

        if state.branch_id == 0:
            if self._maybe_stop_parent_early(run):
                return
            # The finalize-time snapshot rides the parent's stream; when the
            # parent finishes (token cap) before the last sibling, that
            # snapshot has no chunk left to ride and the response ships
            # without branch outputs (measured: 26% empty branch_answers).
            # Once the run enters its tail phase, periodic snapshots carry
            # outputs so the last snapshot to escape always has them. Before
            # then, building a snapshot has no delivery value and is skipped.
            if run.forked and not run.finalized:
                if not run.tail_snapshot_active:
                    parent_sp = getattr(req, "sampling_params", None)
                    parent_cap = getattr(parent_sp, "max_new_tokens", None)
                    if parent_cap:
                        remaining = int(parent_cap) - len(req.output_ids)
                        if remaining <= TAIL_SNAPSHOT_PARENT_MARGIN:
                            run.tail_snapshot_active = True
                if run.tail_snapshot_active:
                    self._attach_snapshot(run, include_outputs=True)
            if (
                not _early_parent_stop_enabled()
                and not run.finalized
                and len(run.branches) > 1
                and _os.environ.get("AUTOTREE_BENCH_FIXED_LEN") != "1"
            ):
                sibling_count = len(run.branches) - 1
                if len(run.finished_sibling_rids) >= sibling_count:
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
            if _AUTOTREE_PROFILE_ENABLED:
                _autotree_profile_incr(
                    "tree.extract_branch_answer.tokens_detokenized", len(ids)
                )
            text = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            branch.final_answer = ""
            return None
        from sglang.srt.tree.answers import extract_answer_text

        answer = extract_answer_text(text)
        branch.final_answer = answer or ""
        return answer

    def _maybe_majority_lock(self, run: _TreeRun) -> None:
        """Zero-accuracy-cost early termination: once finished branches agree
        on an answer that the still-running branches can no longer outvote,
        the tree's outcome is decided - finalize immediately and reclaim every
        remaining token. Safe by construction with respect to the final vote."""
        if run.finalized:
            return
        if (
            _early_parent_stop_enabled()
            and not self._parent_has_natural_eos(run)
        ):
            return
        adaptive_width = run.params.get("adaptive_width")
        if adaptive_width is None and len(run.branches) < 3:
            return
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
        if self._maybe_expand_adaptive_width(run, counts):
            return
        total = len(run.branches)
        if total < 3:
            return
        needed = total // 2 + 1
        top_answer, top_count = max(counts.items(), key=lambda kv: kv[1])
        if top_count >= needed:
            logger.info(
                "[tree] %s majority locked on '%s' (%d/%d finished votes); "
                "terminating remaining branches early",
                run.parent_rid, top_answer, top_count, total,
            )
            self._finalize(run, reason="majority_locked")

    def _maybe_expand_adaptive_width(
        self, run: _TreeRun, counts: Dict[str, int]
    ) -> bool:
        adaptive_width = run.params.get("adaptive_width")
        if adaptive_width is None or run.finalized or run.adaptive_failed:
            return False
        if len(counts) < 2:
            return False

        max_target = min(int(adaptive_width), MAX_BRANCHES)
        current = len(run.branches)
        if current >= max_target:
            return False
        if sum(counts.values()) < current:
            return False

        ordered_counts = sorted(counts.values(), reverse=True)
        margin = ordered_counts[0] - ordered_counts[1]
        if margin >= ADAPT_MARGIN:
            return False

        budget = int(run.params.get("budget_tokens", 0) or 0)
        if budget and run.spent >= budget:
            return False

        parent = run.branches.get("0")
        if parent is None or parent.req is None:
            return False

        before = len(run.branches)
        target = min(max_target, max(current + 1, current * 2))
        try:
            self._fork_branches(
                run,
                parent.req,
                child_input_ids=run.fork_input_ids,
                target_count=target,
            )
        except Exception:
            run.adaptive_failed = True
            logger.exception(
                "[tree] %s adaptive-width spawn failed; continuing at width %d",
                run.parent_rid,
                len(run.branches),
            )
        spawned = len(run.branches) - before
        if spawned <= 0:
            return False
        logger.info(
            "[tree] %s adaptive width %d -> %d: vote margin %d < %.3f",
            run.parent_rid,
            before,
            len(run.branches),
            margin,
            ADAPT_MARGIN,
        )
        return True

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
        if _early_parent_stop_enabled():
            snapshot["tail_tokens_saved"] = run.tail_tokens_saved
        # The customized_info channel is token-aligned: the output streamer
        # slices the value list by the token range of each chunk. Place the
        # snapshot at the parent's newest token index (not yet streamed) so it
        # rides out on the next chunk; earlier indices are padding.
        n = len(parent.req.output_ids)
        values: list = [None] * max(n, 1)
        values[-1] = snapshot
        parent.req.customized_info = {"autotree": values}

    def _consensus_branch_text(self, branch: _BranchState) -> str:
        tokenizer = getattr(self.scheduler, "tokenizer", None)
        if tokenizer is None or branch.req is None:
            return ""
        ids = list(branch.req.output_ids)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and eos in ids:
            ids = ids[: ids.index(eos)]
        if _AUTOTREE_PROFILE_ENABLED:
            _autotree_profile_incr(
                "tree.consensus.tokens_detokenized",
                len(ids),
            )
        return tokenizer.decode(ids, skip_special_tokens=True)

    @staticmethod
    def _branch_is_live_for_consensus(branch: _BranchState) -> bool:
        if branch.state != "active" or branch.req is None:
            return False
        finished = getattr(branch.req, "finished", None)
        return not callable(finished) or not finished()

    def _maybe_consensus_prune(self, run: _TreeRun) -> None:
        """Prune live branches that trail the best deterministic agreement score."""
        if run.finalized or not self._consensus_enabled(run):
            return
        alive = sorted(
            (
                branch
                for branch in run.branches.values()
                if self._branch_is_live_for_consensus(branch)
            ),
            key=lambda branch: branch.branch_id,
        )
        min_survivors = int(
            run.params.get("min_survivors", DEFAULT_MIN_SURVIVORS)
        )
        if len(alive) <= min_survivors:
            return

        minimum_tokens = min(branch.tokens for branch in alive)
        if minimum_tokens <= 0:
            return
        warmup = int(
            run.params.get("consensus_warmup", DEFAULT_CONSENSUS_WARMUP)
        )
        if minimum_tokens < warmup:
            return
        interval = int(
            run.params.get("consensus_interval", DEFAULT_CONSENSUS_INTERVAL)
        )
        checkpoint = warmup + ((minimum_tokens - warmup) // interval) * interval
        if (
            run.last_consensus_check is not None
            and checkpoint <= run.last_consensus_check
        ):
            return

        texts = {
            branch.branch_id: self._consensus_branch_text(branch)
            for branch in alive
        }
        scores = consensus_scores(
            texts,
            ConsensusConfig(min_survivors=min_survivors),
        )
        run.last_consensus_check = checkpoint
        best_score = max(scores.values())
        logger.info(
            "[tree] %s consensus check at branch token %d: %s",
            run.parent_rid,
            minimum_tokens,
            ", ".join(
                f"b{branch_id}={scores[branch_id]:.3f}"
                for branch_id in sorted(scores)
            ),
        )

        killed = 0
        n_alive = len(alive)
        victims = sorted(
            (
                branch
                for branch in alive
                if branch.branch_id != 0 and scores[branch.branch_id] < best_score
            ),
            key=lambda branch: (scores[branch.branch_id], branch.branch_id),
        )
        for branch in victims:
            if n_alive <= min_survivors:
                break
            branch_score = scores[branch.branch_id]
            if not self._prune_branch(run, branch):
                continue
            killed += 1
            n_alive -= 1
            logger.info(
                "[tree] %s consensus pruned branch-%d at token %d: "
                "agreement %.3f vs best %.3f; pages reclaim on finish",
                run.parent_rid,
                branch.branch_id,
                branch.tokens,
                branch_score,
                best_score,
            )
        if killed:
            _autotree_profile_incr("tree.consensus.branches_killed", killed)

    @staticmethod
    def _prune_branch(run: _TreeRun, branch: _BranchState) -> bool:
        """Mark one active branch for the scheduler's normal finish path."""
        if branch.state != "active":
            return False
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        branch.state = "pruned"
        run.pruned += 1
        if branch.req is not None:
            branch.req.to_finish = FINISH_LENGTH(length=len(branch.req.output_ids))
        return True

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

        n_alive = len(alive)
        for branch in sorted(scored, key=lambda b: b.mean_logprob()):
            if n_alive <= VALUE_MIN_KEEP:
                break
            if branch.branch_id == 0:
                continue
            gap = best - branch.mean_logprob()
            if gap <= VALUE_MARGIN:
                break
            self._prune_branch(run, branch)
            n_alive -= 1
            logger.info(
                "[tree] %s pruned branch-%d at token %d: value gap %.3f "
                "nats/token (mean %.3f vs best %.3f); pages reclaim on finish",
                run.parent_rid, branch.branch_id, branch.tokens,
                gap, branch.mean_logprob(), best,
            )

    def _finalize(self, run: _TreeRun, reason: str) -> None:
        run.finalized = True
        active = [b for b in run.branches.values() if b.state == "active"]
        if not active:
            return
        if (
            _early_parent_stop_enabled()
            and reason in {"majority_locked", "siblings_done"}
            and self._parent_has_natural_eos(run)
        ):
            self._record_parent_tail_tokens_saved(run)
        # Self-consistency winner selection: the plurality answer across
        # finished branches beats confidence-argmax on reasoning tasks, so
        # vote first and use mean-logprob only to choose among the branches
        # holding the winning answer (and as the fallback when no branch
        # yields an extractable answer).
        votes: Dict[str, int] = {}
        for b in active:
            answer = self._extract_branch_answer(b)
            if answer:
                votes[answer] = votes.get(answer, 0) + 1
        pool = active
        if votes:
            if selection.vote_mode() == "weighted":
                # Confidence-weighted class selection (opt-in). The plurality
                # branch below is the default and stays byte-identical.
                win_key = selection.weighted_winning_key(
                    (self._extract_branch_answer(b), b.mean_logprob())
                    for b in active
                )
                voted = [
                    b for b in active
                    if self._extract_branch_answer(b) == win_key
                ]
                if voted:
                    pool = voted
            else:
                top_count = max(votes.values())
                leaders = {a for a, c in votes.items() if c == top_count}
                voted = [
                    b for b in active
                    if (self._extract_branch_answer(b) or "") in leaders
                ]
                if voted:
                    pool = voted
        winner = max(pool, key=lambda b: (b.mean_logprob(), -b.branch_id))
        run.winner_branch_id = winner.branch_id
        logger.info(
            "[tree] %s finalize (%s): winner=branch-%d spent=%d pruned=%d",
            run.parent_rid, reason, winner.branch_id, run.spent, run.pruned,
        )

        # NOTE: never mutate parent.output_ids here. Rewriting the token array
        # post-hoc desyncs the KV allocator's page accounting (measured: pool
        # leak abort from the invariant checker). The serving layer reconstructs
        # the winner's text from the final snapshot instead.

        # Branch states must be final before the snapshot is published, but the
        # snapshot must land on the parent BEFORE any to_finish is set: once
        # the parent's stream can complete, a snapshot attached afterwards can
        # lose the race and the response ships without branch outputs
        # (observed on GPU: empty branch_answers on ~40% of items).
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        finish_targets = []
        for branch in run.branches.values():
            if branch.state != "active":
                continue
            branch.state = "finalized" if branch is winner else "pruned"
            if branch is not winner:
                run.pruned += 1
            target = branch.req
            if target is None:
                continue
            finish_targets.append(target)

        self._attach_snapshot(run, include_outputs=True)

        for target in finish_targets:
            target.to_finish = FINISH_LENGTH(length=len(target.output_ids))

        self._stream_deferred_parent(run)


if _AUTOTREE_PROFILE_ENABLED:
    from functools import wraps as _profile_wraps

    def _profile_method(name, method):
        @_profile_wraps(method)
        def profiled(*args, **kwargs):
            with _autotree_profile_span(name):
                return method(*args, **kwargs)

        return profiled

    for _method_name in (
        "_account_tokens",
        "_attach_snapshot",
        "_extract_branch_answer",
        "_maybe_consensus_prune",
        "_maybe_value_prune",
        "_maybe_majority_lock",
        "_fork_branches",
        "_finalize",
    ):
        setattr(
            SchedulerTreeRuntime,
            _method_name,
            _profile_method(
                f"tree.{_method_name.removeprefix('_')}",
                getattr(SchedulerTreeRuntime, _method_name),
            ),
        )


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
            if run.forked and not run.finalized:
                runtime.should_defer_parent_output(parent.req)
                continue
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
