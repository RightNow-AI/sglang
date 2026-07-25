"""HTTP request validation models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = Field(min_length=1)
    content: str


class TreeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["beam", "best_first", "mcts"]
    branches: int = Field(ge=1, le=64)
    budget_tokens: int = Field(ge=1, le=1_000_000)
    scorer: Literal["logprob", "self_consistency"] | None = None
    consensus_interval: int = Field(default=32, ge=1)
    consensus_warmup: int = Field(default=64, ge=0)
    min_survivors: int = Field(default=2, ge=1, le=64)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1, le=4096)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1)
    seed: int | None = None
    user: str | None = None
    tree: TreeParameters | None = None

    @property
    def resolved_max_tokens(self) -> int:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return 16


class TreeCompletionRequest(ChatCompletionRequest):
    tree: TreeParameters
