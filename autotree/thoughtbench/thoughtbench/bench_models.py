"""Models for the endpoint-agnostic ThoughtBench harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchTask(StrictModel):
    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    gold: str = Field(min_length=1)


class HarnessConfig(StrictModel):
    label: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    engine_label: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: AnyHttpUrl
    arm: Literal["single", "best_of_n", "tree"]
    task_file: Path
    output_dir: Path = Path("results")
    n: int | None = Field(default=None, ge=1)
    branches: int | None = Field(default=None, ge=1)
    budget_tokens: int | None = Field(default=None, ge=1)
    policy: str = Field(default="beam", min_length=1)
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0)
    seeds: list[int] = Field(min_length=1)
    concurrency: int = Field(default=1, ge=1)
    timeout: float = Field(default=60, gt=0)
    api_key_env: str | None = Field(default="OPENAI_API_KEY", min_length=1)
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_arm_parameters(self) -> "HarnessConfig":
        parsed_url = urlsplit(str(self.base_url))
        if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise ValueError(
                "base_url must not contain credentials, query parameters, or a fragment"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.arm == "best_of_n":
            if self.n is None:
                raise ValueError("best_of_n requires n")
            if self.branches is not None or self.budget_tokens is not None:
                raise ValueError("best_of_n does not accept tree parameters")
        elif self.arm == "tree":
            if self.branches is None or self.budget_tokens is None:
                raise ValueError("tree requires branches and budget_tokens")
            if self.n is not None:
                raise ValueError("tree does not accept n")
        elif any(value is not None for value in (self.n, self.branches, self.budget_tokens)):
            raise ValueError("single does not accept n or tree parameters")
        return self


class TaskResult(StrictModel):
    id: str
    seed: int
    correct: bool
    answer: str | None
    gold: str
    tokens: int | None = Field(default=None, ge=0)
    wall_s: float = Field(ge=0)


class ResultMeta(StrictModel):
    engine_label: str
    model: str
    base_url_redacted: str
    arm: Literal["single", "best_of_n", "tree"]
    params: dict[str, Any]
    seeds: list[int]
    git_sha: str | None
    started_at: str


class ResultSummary(StrictModel):
    accuracy: float = Field(ge=0, le=1)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)
    mean_tokens: float | None = Field(default=None, ge=0)
    median_tokens: float | None = Field(default=None, ge=0)
    tokens_per_correct: float | None = Field(default=None, ge=0)
    n_tasks: int = Field(ge=1)


class BenchmarkResults(StrictModel):
    meta: ResultMeta
    tasks: list[TaskResult] = Field(min_length=1)
    summary: ResultSummary


class LeaderboardEntry(StrictModel):
    source: str
    meta: ResultMeta
    summary: ResultSummary


class Leaderboard(StrictModel):
    generated_at: str
    entries: list[LeaderboardEntry]
