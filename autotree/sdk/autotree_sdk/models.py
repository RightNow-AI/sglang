"""Typed request, response, event, and rollout models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Annotated, Any, Literal, Sequence, TypeAlias
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_serializer,
    model_validator,
)

from .errors import ExportError, SSEParseError

TreePolicy: TypeAlias = Literal["beam", "best_first", "mcts"]
Prompt: TypeAlias = str | list[dict[str, Any]]
NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]


class RegexVerifierParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["regex"]
    pattern: str = Field(min_length=1)
    flags: Literal["", "i"] = ""

    @model_validator(mode="after")
    def validate_pattern(self) -> "RegexVerifierParameters":
        try:
            re.compile(
                self.pattern,
                re.IGNORECASE if self.flags == "i" else 0,
            )
        except re.error as error:
            raise ValueError(f"invalid regex verifier pattern: {error}") from error
        return self


class NumericVerifierParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["numeric"]
    equals: float
    tolerance: float = Field(ge=0)


class CallbackVerifierParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["callback"]
    url: str = Field(min_length=1, max_length=2048)
    timeout_s: float = Field(gt=0, le=10)

    @model_validator(mode="after")
    def validate_url(self) -> "CallbackVerifierParameters":
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("callback verifier url must use http or https")
        return self


VerifierParameters: TypeAlias = Annotated[
    RegexVerifierParameters
    | NumericVerifierParameters
    | CallbackVerifierParameters,
    Field(discriminator="type"),
]


# matches tree-engine serving_tree.py envelope
class TreeParameters(BaseModel):
    """Tree-search controls accepted by AutoTree serving endpoints."""

    model_config = ConfigDict(extra="forbid")

    policy: TreePolicy = "beam"
    branches: int = Field(gt=0)
    budget_tokens: int = Field(gt=0)
    scorer: str | None = None
    # Fork triggers and adaptive width. The engine accepts all of these; when
    # they are not declared here extra="forbid" rejects a valid request before
    # it is ever sent.
    fork_at_text: str | None = None
    fork_at_entropy: float | None = Field(default=None, gt=0)
    adaptive_width: int | None = Field(default=None, gt=0)
    # Consensus pruning knobs. Engine defaults are 64 / 32 / 2; None means
    # "let the engine decide" so the SDK does not pin defaults that drift.
    consensus_warmup: NonNegativeInt | None = None
    consensus_interval: int | None = Field(default=None, gt=0)
    min_survivors: int | None = Field(default=None, gt=0)
    verifier: VerifierParameters | None = None

    @model_serializer(mode="wrap")
    def serialize_optional_verifier(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.verifier is None:
            data.pop("verifier", None)
        return data


class Usage(BaseModel):
    """OpenAI-compatible token usage returned by the server."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "Usage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


# matches tree-engine serving_tree.py envelope
class TreeSummary(BaseModel):
    """Server summary for a completed tree execution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    policy: TreePolicy
    branch_count: int = Field(ge=1)
    pruned_count: int = Field(ge=0)
    merged_count: int = Field(ge=0)
    winner_branch_id: str
    tokens_spent_per_branch: dict[str, NonNegativeInt]
    final_scores: dict[str, float]
    scorer: str | None
    kv_reuse_ratio: float | None = Field(ge=1)
    # The engine ALWAYS sends these three. Without them declared,
    # extra="forbid" rejected every production response.
    branch_answers: dict[str, str | None] = Field(default_factory=dict)
    served_from_memo: bool = False
    memo_key: str | None = None
    verifier_used: bool = False
    verifier_approved_count: NonNegativeInt = 0
    verifier_fell_back: bool = False

    @model_validator(mode="after")
    def validate_branches(self) -> "TreeSummary":
        branch_ids = set(self.tokens_spent_per_branch)
        if set(self.final_scores) != branch_ids:
            raise ValueError(
                "final_scores and tokens_spent_per_branch must use the same branch IDs"
            )
        if len(branch_ids) != self.branch_count:
            raise ValueError("branch_count must match the per-branch maps")
        if self.winner_branch_id not in branch_ids:
            raise ValueError("winner_branch_id must identify a reported branch")
        return self


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Typed subset of a non-stream OpenAI chat completion response."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    object: str | None = None
    created: int | None = None
    model: str | None = None
    choices: list[CompletionChoice]
    usage: Usage | None = None
    tree: TreeSummary | dict[str, Any] | None = None


class TreeCompletionResponse(ChatCompletionResponse):
    """Exact non-stream tree endpoint wire response."""

    choices: list[CompletionChoice] = Field(min_length=1)
    usage: Usage
    tree: TreeSummary


class BranchStats(BaseModel):
    """Token spend and terminal score for one explored branch."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    tokens_spent: NonNegativeInt
    final_score: float


class TreeCompletion(BaseModel):
    """Convenient typed result returned by ``TreeClient.tree_completions``."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    text: str
    winner_branch_id: str
    branches: dict[str, BranchStats]
    pruned_count: NonNegativeInt
    scorer: str | None
    usage: Usage

    @classmethod
    def from_response(cls, response: TreeCompletionResponse) -> "TreeCompletion":
        summary = response.tree
        return cls(
            text=response.choices[0].message.content,
            winner_branch_id=summary.winner_branch_id,
            branches={
                branch_id: BranchStats(
                    tokens_spent=tokens_spent,
                    final_score=summary.final_scores[branch_id],
                )
                for branch_id, tokens_spent in summary.tokens_spent_per_branch.items()
            },
            pruned_count=summary.pruned_count,
            scorer=summary.scorer,
            usage=response.usage,
        )


class TreeEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BranchStartedEvent(TreeEventModel):
    type: Literal["branch_started"] = "branch_started"
    branch_id: str
    parent_id: str | None = None


class TokenEvent(TreeEventModel):
    type: Literal["token"] = "token"
    branch_id: str
    token_index: int = Field(ge=0)
    token: str
    token_id: NonNegativeInt | None = None
    logprob: float


class BranchPrunedEvent(TreeEventModel):
    type: Literal["branch_pruned"] = "branch_pruned"
    branch_id: str
    reason: str = Field(min_length=1)


class BranchMergedEvent(TreeEventModel):
    type: Literal["branch_merged"] = "branch_merged"
    branch_id: str
    into_branch_id: str


class EngineCounters(TreeEventModel):
    logical_tokens: int = Field(ge=0)
    physical_tokens: int = Field(ge=0)
    useful_tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(gt=0)
    ttft_seconds: float = Field(ge=0)
    unique_tokens_per_step: list[int] = Field(default_factory=list)
    branch_tokens_per_step: list[int] = Field(default_factory=list)


class DoneEvent(TreeEventModel):
    type: Literal["done"] = "done"
    branch_id: str
    text: str
    finish_reason: Literal["stop", "length"]
    usage: Usage
    counters: EngineCounters
    tree: TreeSummary


class StreamErrorDetails(TreeEventModel):
    message: str = Field(min_length=1)
    type: str = Field(min_length=1)
    param: str | None = None
    code: str = Field(min_length=1)


class ErrorEvent(TreeEventModel):
    type: Literal["error"] = "error"
    error: StreamErrorDetails
    retry_after_seconds: NonNegativeInt | None = None


TreeEvent: TypeAlias = Annotated[
    BranchStartedEvent
    | TokenEvent
    | BranchPrunedEvent
    | BranchMergedEvent
    | ErrorEvent
    | DoneEvent,
    Field(discriminator="type"),
]
_TREE_EVENT_ADAPTER = TypeAdapter(TreeEvent)


def parse_tree_event(payload: Any) -> TreeEvent:
    """Parse one SSE JSON object into its concrete typed event."""

    try:
        return _TREE_EVENT_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise SSEParseError("invalid_tree_event", str(exc)) from exc


@dataclass(slots=True)
class RolloutBranch:
    """One branch reconstructed from streamed token events."""

    branch_id: str
    parent_id: str | None
    branch_path: list[str]
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int | None] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    token_indices: list[int] = field(default_factory=list)
    status: Literal["live", "completed", "pruned", "merged"] = "live"
    prune_reason: str | None = None
    merged_into: str | None = None

    @property
    def completion(self) -> str:
        return "".join(self.tokens)

    @property
    def pruned(self) -> bool:
        return self.status == "pruned"

    @property
    def cumulative_logprob(self) -> float:
        return sum(self.token_logprobs)

    def as_sample(
        self,
        prompt: Prompt,
        prompt_index: int,
        *,
        lineage: Sequence["RolloutBranch"] | None = None,
    ) -> dict[str, Any]:
        """Return a flat, JSON-friendly sample shared by rollout exporters."""

        path = tuple(lineage) if lineage is not None else (self,)
        tokens = [token for branch in path for token in branch.tokens]
        token_ids = [token_id for branch in path for token_id in branch.token_ids]
        token_indices = [index for branch in path for index in branch.token_indices]
        token_logprobs = [
            logprob for branch in path for logprob in branch.token_logprobs
        ]
        return {
            "prompt": prompt,
            "completion": "".join(tokens),
            "token_ids": token_ids,
            "token_indices": token_indices,
            "token_logprobs": token_logprobs,
            "cumulative_logprob": sum(token_logprobs),
            "branch_path": list(self.branch_path),
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "prompt_index": prompt_index,
            "status": self.status,
            "pruned": self.pruned,
            "prune_reason": self.prune_reason,
            "merged_into": self.merged_into,
        }


@dataclass(slots=True)
class RolloutTree:
    """A complete branching trace for one input prompt."""

    prompt: Prompt
    branches: list[RolloutBranch]
    usage: Usage
    tree_summary: TreeSummary

    def branch(self, branch_id: str) -> RolloutBranch:
        for branch in self.branches:
            if branch.branch_id == branch_id:
                return branch
        raise KeyError(branch_id)

    def lineage(self, branch: RolloutBranch) -> tuple[RolloutBranch, ...]:
        """Resolve and validate a branch's root-to-leaf provenance path."""

        if not branch.branch_path or branch.branch_path[-1] != branch.branch_id:
            raise ExportError(
                "invalid_branch_path",
                f"branch {branch.branch_id!r} has an invalid branch_path",
            )
        try:
            lineage = tuple(self.branch(branch_id) for branch_id in branch.branch_path)
        except KeyError as exc:
            raise ExportError(
                "invalid_branch_path",
                f"branch_path references unknown branch {exc.args[0]!r}",
            ) from exc
        for parent, child in zip(lineage, lineage[1:], strict=False):
            if child.parent_id != parent.branch_id:
                raise ExportError(
                    "invalid_branch_path",
                    f"branch {child.branch_id!r} does not descend from {parent.branch_id!r}",
                )
        return lineage

    def as_sample(self, branch: RolloutBranch, prompt_index: int) -> dict[str, Any]:
        return branch.as_sample(
            self.prompt,
            prompt_index,
            lineage=self.lineage(branch),
        )


@dataclass(slots=True)
class RolloutBatch:
    """Rollout trees plus common RL post-training export adapters."""

    trees: list[RolloutTree]

    def to_grpo_samples(
        self, *, include_pruned: bool = True, include_merged: bool = False
    ) -> list[dict[str, Any]]:
        """Export flat prompt/completion samples for grouped-policy training.

        Each record retains the original prompt (text or chat messages), token
        logprobs, cumulative logprob, root-to-branch path, and pruning metadata.
        ``prompt_index`` is the group key. Pruned branches are included by
        default so group-relative exports retain rejected alternatives; merged
        traces remain opt-in because they do not own an independent completion.
        """

        samples: list[dict[str, Any]] = []
        for prompt_index, tree in enumerate(self.trees):
            for branch in tree.branches:
                if branch.status == "pruned" and not include_pruned:
                    continue
                if branch.status == "merged" and not include_merged:
                    continue
                samples.append(tree.as_sample(branch, prompt_index))
        return samples

    def to_rlhf_pairs(self, *, include_pruned: bool = True) -> list[dict[str, Any]]:
        """Export score-ordered chosen/rejected preference pairs.

        ``final_scores`` must map branch IDs to numeric scores. Each unequal
        within-prompt pair becomes one record with full chosen/rejected samples
        and their scores. Merged branches are excluded because they do not own
        an independent terminal completion.
        """

        pairs: list[dict[str, Any]] = []
        for prompt_index, tree in enumerate(self.trees):
            scores = tree.tree_summary.final_scores
            candidates = [
                branch
                for branch in tree.branches
                if branch.status != "merged" and (include_pruned or not branch.pruned)
            ]
            missing = [
                branch.branch_id
                for branch in candidates
                if branch.branch_id not in scores
            ]
            if missing:
                raise ExportError(
                    "missing_branch_scores",
                    f"final_scores lacks branch IDs: {', '.join(sorted(missing))}",
                )
            for left, right in combinations(candidates, 2):
                left_score = scores[left.branch_id]
                right_score = scores[right.branch_id]
                if left_score == right_score:
                    continue
                chosen, rejected = (
                    (left, right) if left_score > right_score else (right, left)
                )
                pairs.append(
                    {
                        "prompt": tree.prompt,
                        "prompt_index": prompt_index,
                        "chosen": tree.as_sample(chosen, prompt_index),
                        "rejected": tree.as_sample(rejected, prompt_index),
                        "chosen_score": scores[chosen.branch_id],
                        "rejected_score": scores[rejected.branch_id],
                    }
                )
        return pairs
