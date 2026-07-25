"""Shared engine request/event contract consumed by autotree-serve."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable


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
    policy: Literal["beam", "best_first", "mcts", "emvpt"]
    branches: int
    budget_tokens: int
    scorer: str | None
    value_check_interval: int = 16
    value_margin: float = 0.35
    value_min_keep: int = 2
    value_warmup_tokens: int = 8
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
    """A foreseeable Tree-KV capacity limit at the engine boundary."""

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
            f"{available_pages} are available within the {capacity_pages}-page limit. "
            "Increase --kv-pages or reduce prompt/tree size."
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
    value_estimates: dict[str, float | None] | None = None
    pruned_at_tokens: dict[str, int | None] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.value_estimates is None:
            payload.pop("value_estimates")
        if self.pruned_at_tokens is None:
            payload.pop("pruned_at_tokens")
        return payload


@dataclass(frozen=True, slots=True)
class EngineCounters:
    logical_tokens: int
    physical_tokens: int
    useful_tokens: int
    elapsed_seconds: float
    ttft_seconds: float
    unique_tokens_per_step: tuple[int, ...] = ()
    branch_tokens_per_step: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        unique = tuple(self.unique_tokens_per_step)
        branch = tuple(self.branch_tokens_per_step)
        object.__setattr__(self, "unique_tokens_per_step", unique)
        object.__setattr__(self, "branch_tokens_per_step", branch)
        if len(unique) != len(branch):
            raise ValueError("step token counters must have matching lengths")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*unique, *branch)
        ):
            raise ValueError("step token counters must contain non-negative integers")
        if any(
            unique_count > branch_count
            for unique_count, branch_count in zip(unique, branch, strict=True)
        ):
            raise ValueError("unique step tokens cannot exceed branch step tokens")


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


__all__ = [
    "BranchMerged",
    "BranchPruned",
    "BranchStarted",
    "EngineCounters",
    "EngineEvent",
    "EngineProtocol",
    "EngineUsage",
    "GenerationDone",
    "GenerationRequest",
    "KVCapacityExceededError",
    "Message",
    "ModelMetadata",
    "TokenGenerated",
    "TreeExecution",
    "TreeSummary",
]
