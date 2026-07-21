"""Pydantic models for the AutoTree-compatible tree completions API."""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


TreePolicy = Literal["beam", "best_first", "mcts"]
FinishReason = Literal["stop", "length"]


class TreeChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = Field(min_length=1)
    content: str


class TreeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: TreePolicy
    branches: int = Field(ge=1, le=64)
    budget_tokens: int = Field(ge=1, le=1_000_000)
    scorer: Optional[str] = None
    fork_at_text: Optional[str] = Field(default=None, min_length=1, max_length=64)


class TreeStreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class TreeCompletionRequest(BaseModel):
    """Normative request body for ``POST /v1/tree/completions``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: List[TreeChatMessage] = Field(min_length=1)
    tree: TreeParameters
    stream: bool = False
    stream_options: Optional[TreeStreamOptions] = None
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1
    seed: Optional[int] = None
    user: Optional[str] = None

    @model_validator(mode="after")
    def validate_single_completion(self) -> "TreeCompletionRequest":
        if self.n != 1:
            raise ValueError("n must be 1 for tree completions")
        return self

    @property
    def resolved_max_tokens(self) -> int:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return 16

    @property
    def resolved_seed(self) -> int:
        return 0 if self.seed is None else self.seed


class TreeUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class TreeSummaryResponse(BaseModel):
    policy: str
    branch_count: int = Field(ge=1)
    pruned_count: int = Field(ge=0)
    merged_count: int = Field(ge=0)
    winner_branch_id: str
    tokens_spent_per_branch: Dict[str, int]
    final_scores: Dict[str, float]
    scorer: Optional[str]
    kv_reuse_ratio: Optional[float] = None


class TreeResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class TreeCompletionChoice(BaseModel):
    index: Literal[0] = 0
    message: TreeResponseMessage
    logprobs: None = None
    finish_reason: FinishReason


class TreeCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[TreeCompletionChoice]
    usage: TreeUsage
    tree: TreeSummaryResponse


class TreeBranchStartedEvent(BaseModel):
    type: Literal["branch_started"] = "branch_started"
    branch_id: str
    parent_id: Optional[str] = None


class TreeTokenEvent(BaseModel):
    type: Literal["token"] = "token"
    branch_id: str
    token_index: int = Field(ge=0)
    token: str
    token_id: Optional[int]
    logprob: float


class TreeBranchPrunedEvent(BaseModel):
    type: Literal["branch_pruned"] = "branch_pruned"
    branch_id: str
    reason: str = Field(min_length=1)


class TreeBranchMergedEvent(BaseModel):
    type: Literal["branch_merged"] = "branch_merged"
    branch_id: str
    into_branch_id: str


class TreeCountersResponse(BaseModel):
    logical_tokens: int = Field(ge=0)
    physical_tokens: int = Field(ge=0)
    useful_tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    ttft_seconds: float = Field(ge=0)
    unique_tokens_per_step: List[int] = Field(default_factory=list)
    branch_tokens_per_step: List[int] = Field(default_factory=list)


class TreeDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    branch_id: str
    text: str
    finish_reason: FinishReason
    usage: TreeUsage
    counters: TreeCountersResponse
    tree: TreeSummaryResponse


TreeStreamEvent = Union[
    TreeBranchStartedEvent,
    TreeTokenEvent,
    TreeBranchPrunedEvent,
    TreeBranchMergedEvent,
    TreeDoneEvent,
]
