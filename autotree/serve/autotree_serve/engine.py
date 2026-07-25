"""Engine protocol and the deterministic in-tree reference engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable

try:
    from autotree_core.engine.protocol import (
        BranchMerged,
        BranchPruned,
        BranchStarted,
        EngineCounters,
        EngineEvent,
        EngineProtocol,
        EngineUsage,
        GenerationDone,
        GenerationRequest,
        KVCapacityExceededError,
        Message,
        ModelMetadata,
        TokenGenerated,
        TreeExecution,
        TreeSummary,
    )
except ModuleNotFoundError as error:
    if not (error.name or "").startswith("autotree_core"):
        raise

    @dataclass(frozen=True, slots=True)
    class ModelMetadata:
        id: str
        engine: str
        description: str
        real_model_weights: bool
        tree_policies: tuple[str, ...]

    @dataclass(frozen=True, slots=True)
    class Message:
        role: str
        content: str

    @dataclass(frozen=True, slots=True)
    class TreeExecution:
        policy: Literal["beam", "best_first", "mcts"]
        branches: int
        budget_tokens: int
        scorer: str | None
        consensus_interval: int = 32
        consensus_warmup: int = 64
        min_survivors: int = 2

    @dataclass(frozen=True, slots=True)
    class GenerationRequest:
        model: str
        messages: tuple[Message, ...]
        max_tokens: int
        temperature: float
        top_p: float
        stop: tuple[str, ...]
        seed: int | None
        user: str | None
        tree: TreeExecution | None

    class KVCapacityExceededError(RuntimeError):
        """Fallback copy of the core engine capacity error contract."""

        def __init__(
            self,
            *,
            phase: Literal["admission", "decode"],
            required_pages: int,
            available_pages: int,
            capacity_pages: int,
        ) -> None:
            self.phase = phase
            self.required_pages = required_pages
            self.available_pages = available_pages
            self.capacity_pages = capacity_pages
            super().__init__(
                f"Tree-KV {phase} requires {required_pages} page(s), but only "
                f"{available_pages} are available within the {capacity_pages}-page "
                "limit. Increase --kv-pages or reduce prompt/tree size."
            )

    @dataclass(frozen=True, slots=True)
    class BranchStarted:
        branch_id: str
        parent_id: str | None
        type: Literal["branch_started"] = field(default="branch_started", init=False)

    @dataclass(frozen=True, slots=True)
    class TokenGenerated:
        branch_id: str
        token: str
        token_index: int
        logprob: float
        token_id: int | None = None
        type: Literal["token"] = field(default="token", init=False)

    @dataclass(frozen=True, slots=True)
    class BranchPruned:
        branch_id: str
        reason: str
        type: Literal["branch_pruned"] = field(default="branch_pruned", init=False)

    @dataclass(frozen=True, slots=True)
    class BranchMerged:
        branch_id: str
        into_branch_id: str
        type: Literal["branch_merged"] = field(default="branch_merged", init=False)

    @dataclass(frozen=True, slots=True)
    class EngineUsage:
        prompt_tokens: int
        completion_tokens: int

        @property
        def total_tokens(self) -> int:
            return self.prompt_tokens + self.completion_tokens

        def to_dict(self) -> dict[str, int]:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    @dataclass(frozen=True, slots=True)
    class TreeSummary:
        policy: str
        branch_count: int
        pruned_count: int
        merged_count: int
        winner_branch_id: str
        tokens_spent_per_branch: dict[str, int]
        final_scores: dict[str, float]
        scorer: str | None
        kv_reuse_ratio: float = 1.0

        def to_dict(self) -> dict[str, object]:
            return asdict(self)

    @dataclass(frozen=True, slots=True)
    class EngineCounters:
        logical_tokens: int
        physical_tokens: int
        useful_tokens: int
        elapsed_seconds: float
        ttft_seconds: float

    @dataclass(frozen=True, slots=True)
    class GenerationDone:
        branch_id: str
        text: str
        finish_reason: Literal["stop", "length"]
        usage: EngineUsage
        counters: EngineCounters
        tree_summary: TreeSummary | None
        type: Literal["done"] = field(default="done", init=False)

    EngineEvent: TypeAlias = (
        BranchStarted | TokenGenerated | BranchPruned | BranchMerged | GenerationDone
    )

    @runtime_checkable
    class EngineProtocol(Protocol):
        @property
        def model_metadata(self) -> ModelMetadata: ...

        def generate(self, request: GenerationRequest) -> AsyncIterator[EngineEvent]: ...


DETERMINISTIC_MODEL_ID = "deterministic-demo"


class DeterministicEngine:
    """Seeded toy generator used for contract tests and CPU-only local serving."""

    _VOCABULARY = (
        "branch",
        "checks",
        "clear",
        "evidence",
        "fork",
        "merge",
        "path",
        "prune",
        "reason",
        "reuse",
        "score",
        "token",
        "tree",
        "verify",
    )

    def __init__(self, model_id: str | None = None) -> None:
        # Keep accepting the historical argument for callers, but never expose it:
        # this engine does not load the named weights and must not impersonate them.
        _ = model_id
        self._metadata = ModelMetadata(
            id=DETERMINISTIC_MODEL_ID,
            engine="deterministic",
            description=(
                "Seeded deterministic toy generator for API development; "
                "it does not load or serve real model weights."
            ),
            real_model_weights=False,
            tree_policies=("beam", "best_first", "mcts"),
        )

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(self, request: GenerationRequest) -> AsyncIterator[EngineEvent]:
        started_at = time.perf_counter()
        branch_count = request.tree.branches if request.tree else 1
        branch_ids = [f"branch-{index}" for index in range(branch_count)]
        prompt_tokens = sum(max(1, len(message.content.split())) for message in request.messages)

        if request.tree:
            total_budget = min(request.tree.budget_tokens, request.max_tokens * branch_count)
        else:
            total_budget = request.max_tokens

        allocations = self._allocate_tokens(total_budget, branch_count)
        branch_samples = {
            branch_id: self._apply_stop_sequences(
                self._sample_tokens(request, branch_id, allocations[index]),
                request.stop,
            )
            for index, branch_id in enumerate(branch_ids)
        }
        branch_tokens = {
            branch_id: [token for token, _logprob in branch_samples[branch_id]]
            for branch_id in branch_ids
        }
        allocations = [len(branch_tokens[branch_id]) for branch_id in branch_ids]
        scores = {
            branch_id: self._score_branch(request, branch_id, branch_tokens[branch_id])
            for branch_id in branch_ids
        }
        winner = max(branch_ids, key=lambda branch_id: (scores[branch_id], branch_id))

        for index, branch_id in enumerate(branch_ids):
            yield BranchStarted(
                branch_id=branch_id,
                parent_id=None,
            )

        first_token_at: float | None = None
        for token_index in range(max(allocations, default=0)):
            for branch_id in branch_ids:
                tokens = branch_tokens[branch_id]
                if token_index >= len(tokens):
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield TokenGenerated(
                    branch_id=branch_id,
                    token=tokens[token_index],
                    token_index=token_index,
                    logprob=branch_samples[branch_id][token_index][1],
                    token_id=None,
                )
                await asyncio.sleep(0)

        pruned_count = 0
        merged_count = 0
        losing_branches = [branch_id for branch_id in branch_ids if branch_id != winner]
        for loser_index, branch_id in enumerate(losing_branches):
            if branch_count >= 3 and loser_index % 2 == 1:
                merged_count += 1
                yield BranchMerged(
                    branch_id=branch_id,
                    into_branch_id=winner,
                )
            else:
                pruned_count += 1
                yield BranchPruned(branch_id=branch_id, reason="lower_score")

        completion_tokens = sum(allocations)
        winner_text = "".join(branch_tokens[winner])
        ended_at = time.perf_counter()
        elapsed = max(ended_at - started_at, 1e-9)
        ttft = max((first_token_at or ended_at) - started_at, 0.0)
        logical_tokens = prompt_tokens * branch_count + completion_tokens
        physical_tokens = prompt_tokens + completion_tokens
        summary = None
        if request.tree:
            summary = TreeSummary(
                policy=request.tree.policy,
                branch_count=branch_count,
                pruned_count=pruned_count,
                merged_count=merged_count,
                winner_branch_id=winner,
                tokens_spent_per_branch={
                    branch_id: allocations[index]
                    for index, branch_id in enumerate(branch_ids)
                },
                final_scores=scores,
                scorer=request.tree.scorer,
                kv_reuse_ratio=logical_tokens / physical_tokens,
            )

        yield GenerationDone(
            branch_id=winner,
            text=winner_text,
            finish_reason="length",
            usage=EngineUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            counters=EngineCounters(
                logical_tokens=logical_tokens,
                physical_tokens=physical_tokens,
                useful_tokens=len(branch_tokens[winner]),
                elapsed_seconds=elapsed,
                ttft_seconds=ttft,
            ),
            tree_summary=summary,
        )

    @staticmethod
    def _allocate_tokens(total: int, branches: int) -> list[int]:
        quotient, remainder = divmod(total, branches)
        return [quotient + (1 if index < remainder else 0) for index in range(branches)]

    def _sample_tokens(
        self,
        request: GenerationRequest,
        branch_id: str,
        count: int,
    ) -> list[tuple[str, float]]:
        rng = random.Random(self._stable_seed(request, branch_id, "tokens"))
        candidate_count = max(1, math.ceil(len(self._VOCABULARY) * request.top_p))
        vocabulary = self._VOCABULARY[:candidate_count]
        words = [rng.choice(vocabulary) for _ in range(count)]
        logprob = -math.log(candidate_count)
        return [
            (word if index == 0 else f" {word}", logprob)
            for index, word in enumerate(words)
        ]

    @staticmethod
    def _apply_stop_sequences(
        samples: list[tuple[str, float]], stop: tuple[str, ...]
    ) -> list[tuple[str, float]]:
        if not stop:
            return samples
        text = "".join(token for token, _logprob in samples)
        positions = [position for item in stop if (position := text.find(item)) >= 0]
        if not positions:
            return samples
        limit = min(positions)
        if limit == 0:
            return []
        truncated: list[tuple[str, float]] = []
        offset = 0
        for token, logprob in samples:
            if offset >= limit:
                break
            piece = token[: limit - offset]
            if piece:
                truncated.append((piece, logprob))
            offset += len(token)
        return truncated

    def _score_branch(
        self,
        request: GenerationRequest,
        branch_id: str,
        tokens: list[str],
    ) -> float:
        if not tokens:
            return -1.0
        rng = random.Random(self._stable_seed(request, branch_id, "score"))
        length_signal = min(len(tokens), 1000) / 10_000
        return round(rng.random() + length_signal, 6)

    @staticmethod
    def _stable_seed(request: GenerationRequest, branch_id: str, purpose: str) -> int:
        material = json.dumps(
            {
                "seed": request.seed if request.seed is not None else 0,
                "messages": [(message.role, message.content) for message in request.messages],
                "branch_id": branch_id,
                "purpose": purpose,
                "policy": request.tree.policy if request.tree else None,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
            sort_keys=True,
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
