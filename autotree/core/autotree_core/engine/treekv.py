"""Real HuggingFace + Tree-KV execution driven by the Rust scheduler binding."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Literal, Protocol

import torch
from transformers import AutoConfig, AutoTokenizer

from autotree_core.kv import KVCapacityError
from autotree_core.modeling import ModelExecution, ModelExecutor, ModelExecutorConfig

from .answers import extract_final_answer
from .consensus import ConsensusConfig, consensus_scores
from .protocol import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    EngineCounters,
    EngineUsage,
    GenerationDone,
    GenerationRequest,
    KVCapacityExceededError,
    ModelMetadata,
    TokenGenerated,
    TreeSummary,
)


class SchedulerBinding(Protocol):
    def feed_event(self, event: dict[str, object]) -> None: ...

    def poll_commands(self) -> list[dict[str, object]]: ...

    def branch_state(self, branch: int) -> str | None: ...


SchedulerFactory = Callable[[dict[str, object]], SchedulerBinding]


@dataclass(frozen=True, slots=True)
class _KVSnapshot:
    logical_tokens: int
    physical_tokens: int

    @property
    def ratio(self) -> float:
        return self.logical_tokens / max(self.physical_tokens, 1)


@dataclass(frozen=True, slots=True)
class _EMVPTConfig:
    check_interval: int
    margin: float
    min_keep: int
    warmup_tokens: int


@dataclass(frozen=True, slots=True)
class _ConsensusRuntimeConfig:
    interval: int
    warmup_tokens: int
    scoring: ConsensusConfig


_ENGINE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class TreeKVEngine:
    """Engine using real model weights and real Tree-KV pages (CPU or CUDA)."""

    def __init__(
        self,
        model_id: str = "gpt2",
        *,
        executor: ModelExecutor | None = None,
        tokenizer: Any | None = None,
        scheduler_factory: SchedulerFactory | None = None,
        kv_pages: int | None = None,
        kv_branch_headroom: float = 1.5,
        dedup_every_steps: int | None = 1,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        if dtype not in _ENGINE_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(_ENGINE_DTYPES)}, got {dtype!r}"
            )
        if kv_pages is not None and (
            isinstance(kv_pages, bool) or not isinstance(kv_pages, int) or kv_pages <= 0
        ):
            raise ValueError("kv_pages must be a positive integer or None")
        if not math.isfinite(kv_branch_headroom) or kv_branch_headroom < 1.0:
            raise ValueError("kv_branch_headroom must be finite and at least 1.0")
        if dedup_every_steps is not None and (
            isinstance(dedup_every_steps, bool)
            or not isinstance(dedup_every_steps, int)
            or dedup_every_steps <= 0
        ):
            raise ValueError("dedup_every_steps must be a positive integer or None")
        if executor is None:
            executor_config = ModelExecutorConfig(
                model_id=model_id, device=device, dtype=_ENGINE_DTYPES[dtype]
            )
            if kv_pages is None:
                model_config = AutoConfig.from_pretrained(model_id)
                context_tokens = self._model_context_tokens(model_config)
                kv_pages = math.ceil(
                    context_tokens / executor_config.page_size * kv_branch_headroom
                )
            executor_config = replace(executor_config, capacity_pages=kv_pages)
            executor = ModelExecutor(executor_config)
        self.executor = executor
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self._tokenizer_lock = RLock()
        if scheduler_factory is None:
            try:
                from autotree_scheduler import Scheduler
            except ImportError as error:
                raise RuntimeError(
                    "TreeKVEngine requires the autotree-scheduler PyO3 wheel; "
                    "build and install it with maturin before selecting --engine treekv"
                ) from error
            scheduler_factory = Scheduler
        self._scheduler_factory = scheduler_factory
        self._scheduler_factory_lock = RLock()
        self._dedup_every_steps = dedup_every_steps
        self._metadata = ModelMetadata(
            id=model_id,
            engine="treekv",
            description=(
                "CPU Tree-KV demo engine using real HuggingFace model weights and "
                "the Rust branch scheduler; no GPU throughput claim is implied."
            ),
            real_model_weights=True,
            tree_policies=("beam", "best_first", "mcts", "emvpt"),
        )

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(self, request: GenerationRequest):
        if request.model != self._metadata.id:
            raise ValueError(
                f"request model {request.model!r} does not match loaded model "
                f"{self._metadata.id!r}"
            )
        started_at = time.perf_counter()
        prompt_ids = self._encode_prompt(request)
        required_prompt_pages = math.ceil(
            len(prompt_ids) / self.executor.config.page_size
        )
        if required_prompt_pages > self.executor.config.capacity_pages:
            raise KVCapacityExceededError(
                phase="admission",
                required_pages=required_prompt_pages,
                available_pages=self.executor.config.capacity_pages,
                capacity_pages=self.executor.config.capacity_pages,
            )
        try:
            execution = self.executor.prefill(prompt_ids)
        except KVCapacityError as error:
            raise self._capacity_error("admission", error) from error
        emvpt_config = self._emvpt_config(request)
        consensus_config = self._consensus_config(request)
        with self._scheduler_factory_lock:
            scheduler = self._scheduler_factory(self._scheduler_config(request))
        generator = torch.Generator(device=self.executor.config.device).manual_seed(
            self._resolve_seed(request.seed)
        )

        parents: dict[int, int | None] = {execution.root_id: None}
        own_text: dict[int, list[str]] = {execution.root_id: []}
        path_text: dict[int, str] = {execution.root_id: ""}
        token_counts: dict[int, int] = {execution.root_id: 0}
        ranking_token_counts: dict[int, int] = {execution.root_id: 0}
        scores: dict[int, float] = {execution.root_id: 0.0}
        value_logprob_sums: dict[int, float] = {execution.root_id: 0.0}
        value_token_counts: dict[int, int] = {execution.root_id: 0}
        pruned_at_tokens: dict[int, int | None] = {execution.root_id: None}
        active = {execution.root_id}
        finalized: set[int] = set()
        merged: set[int] = set()
        value_pruned: set[int] = set()
        exhaustion_pending: set[int] = set()
        stopped: set[int] = set()
        commands: deque[dict[str, object]] = deque()
        completion_tokens = 0
        pruned_count = 0
        merged_count = 0
        decode_steps = 0
        unique_tokens_per_step: list[int] = []
        branch_tokens_per_step: list[int] = []
        first_token_at: float | None = None
        best_snapshot = self._snapshot(execution)

        yield BranchStarted(branch_id="branch-0", parent_id=None)

        def scheduler_branch_state(branch_id: int) -> str | None:
            checker = getattr(scheduler, "branch_state", None)
            return None if checker is None else checker(branch_id)

        def scheduler_branch_is_active(branch_id: int) -> bool:
            state = scheduler_branch_state(branch_id)
            return state is None or state == "active"

        def scheduler_branch_is_live(branch_id: int) -> bool:
            state = scheduler_branch_state(branch_id)
            return state is None or state in {"active", "expanded"}

        def value_estimate(branch_id: int) -> float | None:
            count = value_token_counts[branch_id]
            if count == 0:
                return None
            return value_logprob_sums[branch_id] / count

        def feed_consensus_values(sampled_branch_ids: tuple[int, ...]) -> None:
            if consensus_config is None:
                return

            pending = [
                branch_id
                for branch_id in sorted(sampled_branch_ids)
                if branch_id not in exhaustion_pending
                and scheduler_branch_is_live(branch_id)
            ]
            if not pending:
                return

            live = [
                branch_id
                for branch_id in sorted(active)
                if branch_id not in exhaustion_pending
                and scheduler_branch_is_live(branch_id)
            ]
            score_now = any(
                ranking_token_counts[branch_id] >= consensus_config.warmup_tokens
                and (
                    ranking_token_counts[branch_id] - consensus_config.warmup_tokens
                )
                % consensus_config.interval
                == 0
                for branch_id in pending
            )
            if score_now:
                scores_by_branch = consensus_scores(
                    {branch_id: path_text[branch_id] for branch_id in live},
                    consensus_config.scoring,
                )
            else:
                scores_by_branch = {branch_id: 1.0 for branch_id in live}

            for branch_id in pending:
                scheduler.feed_event(
                    {
                        "type": "value_scored",
                        "branch": branch_id,
                        "score": scores_by_branch[branch_id],
                    }
                )

        def prune_low_value_branches() -> tuple[BranchPruned, ...]:
            nonlocal pruned_count
            if emvpt_config is None:
                return ()

            live = [
                branch_id
                for branch_id in sorted(active)
                if branch_id not in exhaustion_pending
                and scheduler_branch_is_active(branch_id)
            ]
            available_prunes = len(live) - emvpt_config.min_keep
            if available_prunes <= 0:
                return ()

            sibling_groups: dict[int | None, list[int]] = {}
            for branch_id in live:
                sibling_groups.setdefault(parents[branch_id], []).append(branch_id)

            candidates: list[tuple[float, int]] = []
            for siblings in sibling_groups.values():
                estimates = {
                    branch_id: value_estimate(branch_id) for branch_id in siblings
                }
                finite_estimates = {
                    branch_id: estimate
                    for branch_id, estimate in estimates.items()
                    if estimate is not None
                }
                if len(finite_estimates) < 2:
                    continue
                best = max(finite_estimates.values())
                for branch_id, estimate in finite_estimates.items():
                    count = value_token_counts[branch_id]
                    if (
                        count >= emvpt_config.warmup_tokens
                        and count % emvpt_config.check_interval == 0
                        and best - estimate > emvpt_config.margin
                    ):
                        candidates.append((best - estimate, branch_id))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            events: list[BranchPruned] = []
            for _, branch_id in candidates[:available_prunes]:
                if branch_id not in active:
                    continue
                active.remove(branch_id)
                exhaustion_pending.discard(branch_id)
                self.executor.prune(execution, branch_id)
                value_pruned.add(branch_id)
                pruned_at_tokens[branch_id] = value_token_counts[branch_id]
                pruned_count += 1
                if scheduler_branch_is_active(branch_id):
                    scheduler.feed_event(
                        {"type": "branch_exhausted", "branch": branch_id}
                    )
                events.append(
                    BranchPruned(
                        branch_id=self._branch_name(branch_id),
                        reason="emvpt_below_margin",
                    )
                )
            return tuple(events)

        def merge_converged_branches(
            pending_commands: tuple[dict[str, object], ...],
        ) -> tuple[BranchMerged, ...]:
            nonlocal merged_count
            scheduler_terminal = {
                int(command["branch"])
                for command in pending_commands
                if str(command.get("type")) in {"kill", "finalize"}
            }
            scheduler_terminal.update(
                branch_id
                for branch_id in active
                if scheduler_branch_state(branch_id) in {"killed", "finalized"}
            )
            scheduler_expanding = {
                int(command["branch"])
                for command in pending_commands
                if str(command.get("type")) == "fork_at"
            }
            scheduler_expanding.update(
                branch_id
                for branch_id in active
                if scheduler_branch_state(branch_id) == "expanded"
            )
            groups: dict[tuple[tuple[int, ...], int], list[int]] = {}
            for branch_id in sorted(active):
                if branch_id in exhaustion_pending or branch_id in scheduler_expanding:
                    continue
                branch = execution.tree.get_branch(branch_id)
                if (
                    not branch.block_table
                    or branch.num_tokens % execution.pool.config.page_size
                ):
                    continue
                key = (execution.token_ids(branch_id), branch.block_table[-1])
                groups.setdefault(key, []).append(branch_id)

            events: list[BranchMerged] = []
            for branch_ids in groups.values():
                if len(branch_ids) < 2:
                    continue
                target = min(
                    branch_ids,
                    key=lambda branch_id: (
                        branch_id in scheduler_terminal,
                        branch_id,
                    ),
                )
                for branch_id in sorted(branch_ids):
                    if branch_id == target:
                        continue
                    if branch_id not in scheduler_terminal:
                        scheduler.feed_event(
                            {"type": "branch_exhausted", "branch": branch_id}
                        )
                    active.remove(branch_id)
                    exhaustion_pending.discard(branch_id)
                    self.executor.prune(execution, branch_id)
                    merged.add(branch_id)
                    merged_count += 1
                    events.append(
                        BranchMerged(
                            branch_id=self._branch_name(branch_id),
                            into_branch_id=self._branch_name(target),
                        )
                    )
            return tuple(events)

        async def advance_batch(
            branch_ids: tuple[int, ...],
        ) -> tuple[
            tuple[TokenGenerated | BranchPruned, ...], tuple[BranchMerged, ...]
        ]:
            nonlocal completion_tokens, decode_steps, first_token_at, best_snapshot
            if not branch_ids or len(set(branch_ids)) != len(branch_ids):
                raise RuntimeError(
                    "Continue commands must target distinct active branches"
                )
            inactive = [
                branch_id for branch_id in branch_ids if branch_id not in active
            ]
            if inactive:
                raise RuntimeError(f"Continue targeted inactive branch {inactive[0]}")
            budget = request.tree.budget_tokens if request.tree else request.max_tokens
            if completion_tokens + len(branch_ids) > budget:
                raise RuntimeError(
                    "scheduler Continue exceeded the request token budget"
                )

            sampled = tuple(
                self._sample(execution.next_logits(branch_id), request, generator)
                for branch_id in branch_ids
            )
            token_ids = tuple(token_id for token_id, _ in sampled)
            try:
                if len(branch_ids) == 1:
                    self.executor.decode(execution, branch_ids[0], token_ids[0])
                else:
                    self.executor.decode_batch(execution, branch_ids, token_ids)
            except KVCapacityError as error:
                raise self._capacity_error("decode", error) from error

            events: list[TokenGenerated] = []
            for branch_id, (token_id, logprob) in zip(branch_ids, sampled, strict=True):
                with self._tokenizer_lock:
                    token = self.tokenizer.decode(
                        [token_id],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                token_index = token_counts[branch_id]
                token_counts[branch_id] += 1
                ranking_token_counts[branch_id] += 1
                own_text[branch_id].append(token)
                path_text[branch_id] += token
                scores[branch_id] += logprob
                value_logprob_sums[branch_id] += logprob
                value_token_counts[branch_id] += 1
                completion_tokens += 1
                branch_exhausted = self._token_exhausts_branch(
                    token_id, path_text[branch_id], request
                )
                if branch_exhausted:
                    exhaustion_pending.add(branch_id)
                    stopped.add(branch_id)
                if first_token_at is None:
                    first_token_at = time.perf_counter()

                scheduler.feed_event(
                    {
                        "type": "token_sampled",
                        "branch": branch_id,
                        "token": token_id,
                        "logprob": logprob,
                        "eos": branch_exhausted,
                    }
                )
                events.append(
                    TokenGenerated(
                        branch_id=self._branch_name(branch_id),
                        token=token,
                        token_index=token_index,
                        logprob=logprob,
                        token_id=token_id,
                    )
                )

            decode_steps += 1
            branch_tokens_per_step.append(len(branch_ids))
            unique_tokens_per_step.append(
                len({execution.token_ids(branch_id) for branch_id in branch_ids})
            )
            feed_consensus_values(branch_ids)
            value_prune_events = prune_low_value_branches()
            scheduler_commands = tuple(scheduler.poll_commands())
            merge_events: tuple[BranchMerged, ...] = ()
            if (
                self._dedup_every_steps is not None
                and decode_steps % self._dedup_every_steps == 0
            ):
                self.executor.deduplicate(execution)
                best_snapshot = self._better_snapshot(
                    best_snapshot, self._snapshot(execution)
                )
                merge_events = merge_converged_branches(scheduler_commands)
            commands.extend(scheduler_commands)
            commands.extend(scheduler.poll_commands())
            best_snapshot = self._better_snapshot(
                best_snapshot, self._snapshot(execution)
            )
            await asyncio.sleep(0)
            return (*events, *value_prune_events), merge_events

        token_events, merge_events = await advance_batch((execution.root_id,))
        for event in (*token_events, *merge_events):
            yield event

        while commands:
            command = commands.popleft()
            command_type = str(command.get("type"))
            branch_id = int(command["branch"])
            if branch_id in merged or branch_id in value_pruned:
                continue
            if command_type == "continue":
                continue_ids = [branch_id]
                while commands and str(commands[0].get("type")) == "continue":
                    continue_ids.append(int(commands.popleft()["branch"]))
                ready: list[int] = []
                for continued_id in continue_ids:
                    if continued_id in merged or continued_id not in active:
                        continue
                    if not scheduler_branch_is_active(continued_id):
                        continue
                    if continued_id in exhaustion_pending:
                        exhaustion_pending.remove(continued_id)
                        scheduler.feed_event(
                            {"type": "branch_exhausted", "branch": continued_id}
                        )
                        commands.extend(scheduler.poll_commands())
                    else:
                        ready.append(continued_id)
                if ready:
                    token_events, merge_events = await advance_batch(tuple(ready))
                    for event in (*token_events, *merge_events):
                        yield event
                continue
            if command_type == "fork_at":
                if branch_id not in active:
                    raise RuntimeError(f"ForkAt targeted inactive branch {branch_id}")
                width = int(command["width"])
                children_are_exhausted = branch_id in exhaustion_pending
                exhaustion_pending.discard(branch_id)
                active.remove(branch_id)
                for _ in range(width):
                    expected_id = max(parents) + 1
                    child_id = self.executor.fork(execution, branch_id)
                    if child_id != expected_id:
                        raise RuntimeError(
                            "TreeState child allocation diverged from scheduler contract: "
                            f"expected {expected_id}, got {child_id}"
                        )
                    parents[child_id] = branch_id
                    own_text[child_id] = []
                    path_text[child_id] = path_text[branch_id]
                    token_counts[child_id] = 0
                    ranking_token_counts[child_id] = ranking_token_counts[branch_id]
                    scores[child_id] = scores[branch_id]
                    value_logprob_sums[child_id] = 0.0
                    value_token_counts[child_id] = 0
                    pruned_at_tokens[child_id] = None
                    active.add(child_id)
                    if children_are_exhausted:
                        exhaustion_pending.add(child_id)
                        stopped.add(child_id)
                    yield BranchStarted(
                        branch_id=self._branch_name(child_id),
                        parent_id=self._branch_name(branch_id),
                    )
                best_snapshot = self._better_snapshot(
                    best_snapshot, self._snapshot(execution)
                )
                continue
            if command_type == "kill":
                exhaustion_pending.discard(branch_id)
                if branch_id in active:
                    active.remove(branch_id)
                self.executor.prune(execution, branch_id)
                pruned_count += 1
                yield BranchPruned(
                    branch_id=self._branch_name(branch_id),
                    reason=str(command.get("reason") or "scheduler_pruned"),
                )
                continue
            if command_type == "finalize":
                exhaustion_pending.discard(branch_id)
                if branch_id in active:
                    active.remove(branch_id)
                finalized.add(branch_id)
                self.executor.prune(execution, branch_id)
                continue
            raise RuntimeError(f"unsupported scheduler command type {command_type!r}")

        if active:
            raise RuntimeError(
                "scheduler stopped issuing commands with active branches remaining: "
                f"{sorted(active)}"
            )
        if not finalized:
            raise RuntimeError("scheduler terminated without a finalized branch")

        ranking_scores = {
            branch_id: scores[branch_id]
            / max(ranking_token_counts[branch_id], 1)
            for branch_id in parents
        }
        winner = self._select_winner(
            finalized=finalized,
            scorer=request.tree.scorer if request.tree is not None else None,
            path_text=path_text,
            cumulative_logprobs=scores,
            ranking_scores=ranking_scores,
        )
        for branch_id in sorted(finalized - {winner}):
            pruned_count += 1
            yield BranchPruned(
                branch_id=self._branch_name(branch_id),
                reason="not_selected",
            )

        ended_at = time.perf_counter()
        winner_path = self._path(winner, parents)
        winner_text = "".join(
            token for branch_id in winner_path for token in own_text[branch_id]
        )
        useful_tokens = sum(token_counts[branch_id] for branch_id in winner_path)
        summary = None
        if request.tree is not None:
            summary = TreeSummary(
                policy=request.tree.policy,
                branch_count=len(parents),
                pruned_count=pruned_count,
                merged_count=merged_count,
                winner_branch_id=self._branch_name(winner),
                tokens_spent_per_branch={
                    self._branch_name(branch_id): token_counts[branch_id]
                    for branch_id in sorted(parents)
                },
                final_scores={
                    self._branch_name(branch_id): ranking_scores[branch_id]
                    for branch_id in sorted(parents)
                },
                scorer=request.tree.scorer,
                kv_reuse_ratio=best_snapshot.ratio,
                value_estimates=(
                    {
                        self._branch_name(branch_id): value_estimate(branch_id)
                        for branch_id in sorted(parents)
                    }
                    if emvpt_config is not None
                    else None
                ),
                pruned_at_tokens=(
                    {
                        self._branch_name(branch_id): pruned_at_tokens[branch_id]
                        for branch_id in sorted(parents)
                    }
                    if emvpt_config is not None
                    else None
                ),
            )
        yield GenerationDone(
            branch_id=self._branch_name(winner),
            text=winner_text,
            finish_reason="stop" if winner in stopped else "length",
            usage=EngineUsage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_tokens,
            ),
            counters=EngineCounters(
                logical_tokens=best_snapshot.logical_tokens,
                physical_tokens=best_snapshot.physical_tokens,
                useful_tokens=useful_tokens,
                elapsed_seconds=max(ended_at - started_at, 1e-9),
                ttft_seconds=max((first_token_at or ended_at) - started_at, 0.0),
                unique_tokens_per_step=tuple(unique_tokens_per_step),
                branch_tokens_per_step=tuple(branch_tokens_per_step),
            ),
            tree_summary=summary,
        )

    def _encode_prompt(self, request: GenerationRequest) -> list[int]:
        prompt = "\n".join(
            f"{message.role}: {message.content}" for message in request.messages
        )
        prompt = f"{prompt}\nassistant:"
        with self._tokenizer_lock:
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not token_ids:
            raise ValueError("tokenizer produced an empty prompt")
        return list(token_ids)

    def _capacity_error(
        self,
        phase: Literal["admission", "decode"],
        error: KVCapacityError,
    ) -> KVCapacityExceededError:
        return KVCapacityExceededError(
            phase=phase,
            required_pages=error.required_pages,
            available_pages=error.available_pages,
            capacity_pages=self.executor.config.capacity_pages,
        )

    @staticmethod
    def _model_context_tokens(model_config: Any) -> int:
        for field_name in ("max_position_embeddings", "n_positions", "n_ctx"):
            value = getattr(model_config, field_name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        raise ValueError(
            "model config does not expose a positive context length through "
            "max_position_embeddings, n_positions, or n_ctx; pass --kv-pages"
        )

    @staticmethod
    def _branch_name(branch_id: int) -> str:
        return f"branch-{branch_id}"

    @staticmethod
    def _path(branch_id: int, parents: dict[int, int | None]) -> tuple[int, ...]:
        reversed_path: list[int] = []
        current: int | None = branch_id
        while current is not None:
            reversed_path.append(current)
            current = parents[current]
        return tuple(reversed(reversed_path))

    @staticmethod
    def _snapshot(execution: ModelExecution) -> _KVSnapshot:
        stats = execution.stats
        return _KVSnapshot(
            logical_tokens=stats.logical_tokens,
            physical_tokens=stats.physical_tokens,
        )

    @staticmethod
    def _better_snapshot(current: _KVSnapshot, candidate: _KVSnapshot) -> _KVSnapshot:
        return candidate if candidate.ratio > current.ratio else current

    @staticmethod
    def _select_winner(
        *,
        finalized: set[int],
        scorer: str | None,
        path_text: dict[int, str],
        cumulative_logprobs: dict[int, float],
        ranking_scores: dict[int, float],
    ) -> int:
        if scorer != "self_consistency":
            return max(
                finalized, key=lambda branch: (ranking_scores[branch], -branch)
            )

        groups: dict[tuple[str, object], list[int]] = {}
        for branch_id in sorted(finalized):
            answer = extract_final_answer(path_text[branch_id])
            key = ("answer", answer) if answer is not None else ("branch", branch_id)
            groups.setdefault(key, []).append(branch_id)

        winning_group = max(
            groups.values(),
            key=lambda branches: (
                len(branches),
                sum(cumulative_logprobs[branch] for branch in branches),
                max(cumulative_logprobs[branch] for branch in branches),
                -min(branches),
            ),
        )
        return max(
            winning_group,
            key=lambda branch: (cumulative_logprobs[branch], -branch),
        )

    def _token_exhausts_branch(
        self,
        token_id: int,
        text: str,
        request: GenerationRequest,
    ) -> bool:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            is_eos = token_id == eos_token_id
        elif isinstance(eos_token_id, (list, tuple, set, frozenset)):
            is_eos = token_id in eos_token_id
        else:
            is_eos = False
        return is_eos or any(stop and stop in text for stop in request.stop)

    @staticmethod
    def _emvpt_config(request: GenerationRequest) -> _EMVPTConfig | None:
        tree = request.tree
        if tree is None or tree.policy != "emvpt":
            return None
        if (
            isinstance(tree.value_check_interval, bool)
            or not isinstance(tree.value_check_interval, int)
            or tree.value_check_interval <= 0
        ):
            raise ValueError("value_check_interval must be a positive integer")
        if (
            isinstance(tree.value_margin, bool)
            or not isinstance(tree.value_margin, (int, float))
            or not math.isfinite(tree.value_margin)
            or tree.value_margin < 0
        ):
            raise ValueError("value_margin must be finite and non-negative")
        if (
            isinstance(tree.value_min_keep, bool)
            or not isinstance(tree.value_min_keep, int)
            or tree.value_min_keep <= 0
        ):
            raise ValueError("value_min_keep must be a positive integer")
        if (
            isinstance(tree.value_warmup_tokens, bool)
            or not isinstance(tree.value_warmup_tokens, int)
            or tree.value_warmup_tokens < 0
        ):
            raise ValueError("value_warmup_tokens must be a non-negative integer")
        return _EMVPTConfig(
            check_interval=tree.value_check_interval,
            margin=float(tree.value_margin),
            min_keep=tree.value_min_keep,
            warmup_tokens=tree.value_warmup_tokens,
        )

    @staticmethod
    def _consensus_config(
        request: GenerationRequest,
    ) -> _ConsensusRuntimeConfig | None:
        tree = request.tree
        if tree is None or tree.scorer != "self_consistency":
            return None
        if (
            isinstance(tree.consensus_interval, bool)
            or not isinstance(tree.consensus_interval, int)
            or tree.consensus_interval <= 0
        ):
            raise ValueError("consensus_interval must be a positive integer")
        if (
            isinstance(tree.consensus_warmup, bool)
            or not isinstance(tree.consensus_warmup, int)
            or tree.consensus_warmup < 0
        ):
            raise ValueError("consensus_warmup must be a non-negative integer")
        if (
            isinstance(tree.min_survivors, bool)
            or not isinstance(tree.min_survivors, int)
            or tree.min_survivors <= 0
        ):
            raise ValueError("min_survivors must be a positive integer")
        return _ConsensusRuntimeConfig(
            interval=tree.consensus_interval,
            warmup_tokens=tree.consensus_warmup,
            scoring=ConsensusConfig(min_survivors=tree.min_survivors),
        )

    @classmethod
    def _scheduler_config(cls, request: GenerationRequest) -> dict[str, object]:
        tree = request.tree
        if tree is None or tree.policy == "emvpt":
            policy = "beam"
        else:
            policy = tree.policy.replace("_", "-")
        requested_scorer = tree.scorer if tree else None
        if requested_scorer not in {None, "logprob", "self_consistency"}:
            raise ValueError(
                "TreeKVEngine scorer must be None, 'logprob', or 'self_consistency'"
            )
        branches = tree.branches if tree else 1
        config: dict[str, object] = {
            "policy": policy,
            "branches": branches,
            "fork_width": branches,
            "fork_at_tokens": [1] if tree and branches > 1 else [],
            "max_depth": max(1, request.max_tokens),
            "budget_tokens": tree.budget_tokens if tree else request.max_tokens,
            "per_branch_token_budget": request.max_tokens,
            "seed": cls._resolve_seed(request.seed),
            "scorer": (
                "external" if requested_scorer == "self_consistency" else "logprob"
            ),
        }
        if requested_scorer == "self_consistency":
            config["speculative_kill_margin"] = 0.0
        return config

    @staticmethod
    def _resolve_seed(seed: int | None) -> int:
        """Resolve an omitted/null request seed to the documented default."""

        return 0 if seed is None else seed

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        request: GenerationRequest,
        generator: torch.Generator,
    ) -> tuple[int, float]:
        model_scores = logits.float()
        sampling_scores = model_scores
        if request.temperature == 0:
            token_id = int(torch.argmax(sampling_scores).item())
        else:
            sampling_scores = sampling_scores / request.temperature
            probabilities = torch.softmax(sampling_scores, dim=-1)
            if request.top_p < 1.0:
                sorted_probabilities, sorted_indices = torch.sort(
                    probabilities, descending=True
                )
                cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                remove = cumulative > request.top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
                probabilities = torch.zeros_like(probabilities).scatter(
                    0, sorted_indices, sorted_probabilities
                )
                probabilities = probabilities / probabilities.sum()
            token_id = int(
                torch.multinomial(probabilities, 1, generator=generator).item()
            )
        logprob = float(torch.log_softmax(model_scores, dim=-1)[token_id].item())
        if not math.isfinite(logprob):
            raise RuntimeError("model produced a non-finite sampled-token logprob")
        return token_id, logprob


__all__ = ["TreeKVEngine"]
