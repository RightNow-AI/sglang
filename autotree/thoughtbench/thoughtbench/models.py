"""Validated task, configuration, sample, metric, and results models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

FIXTURE_NOTICE = "FIXTURE TASKS ONLY - NOT A REAL BENCHMARK RESULT."
REAL_NOTICE = (
    "REAL TASKS - MEASURED RESULT. Claims are limited to the stated protocol scope."
)
RESULTS_SCHEMA_VERSION = "thoughtbench.results.v2"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Task(StrictModel):
    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    grader: Literal["exact-match", "numeric"]
    tags: list[str] = Field(default_factory=list)


class FixtureProvenance(StrictModel):
    kind: Literal["fixture"] = "fixture"
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    notice: Literal[FIXTURE_NOTICE] = FIXTURE_NOTICE


class RealProvenance(StrictModel):
    kind: Literal["real"] = "real"
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    notice: Literal[REAL_NOTICE] = REAL_NOTICE


class TaskSetConfig(StrictModel):
    name: str = Field(min_length=1)
    path: Path
    provenance: FixtureProvenance | RealProvenance = Field(discriminator="kind")


class PricingConfig(StrictModel):
    input_per_million_usd: float = Field(default=0, ge=0)
    output_per_million_usd: float = Field(default=0, ge=0)


class TreeConfig(StrictModel):
    policy: Literal["beam", "best_first", "mcts"] = "beam"
    branches: int = Field(ge=1)
    scorer: str | None = None


class BudgetConfig(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    max_tokens: int = Field(ge=1)
    tree_budget_tokens: int | None = Field(default=None, ge=1)


class RunConfig(StrictModel):
    model: str = Field(min_length=1)
    base_url: AnyHttpUrl
    mode: Literal["sequential", "tree"]
    task_set: TaskSetConfig
    output_path: Path
    budgets: list[BudgetConfig] = Field(min_length=1)
    k_samples: int = Field(ge=1, le=64)
    seeds: list[int]
    concurrency: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=60, gt=0)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    stop: list[str] = Field(default_factory=list)
    tree: TreeConfig | None = None
    pricing: PricingConfig = Field(default_factory=PricingConfig)

    @model_validator(mode="after")
    def validate_protocol(self) -> "RunConfig":
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("ThoughtBench v1 requires exactly three unique seeds")
        if len({budget.name for budget in self.budgets}) != len(self.budgets):
            raise ValueError("budget names must be unique")
        if self.mode == "tree":
            if self.tree is None:
                raise ValueError("tree mode requires tree parameters")
            if any(budget.tree_budget_tokens is None for budget in self.budgets):
                raise ValueError("tree mode requires tree_budget_tokens for every budget")
        elif self.tree is not None:
            raise ValueError("sequential mode must not include tree parameters")
        return self

    @property
    def metric_ks(self) -> tuple[int, ...]:
        return tuple(k for k in (1, 4, 16, 64) if k <= self.k_samples)


class TreeStats(StrictModel):
    policy: str | None = None
    branch_count: int = Field(ge=0)
    pruned_count: int = Field(ge=0)
    merged_count: int | None = Field(default=None, ge=0)
    winner_branch_id: str | None = None
    tokens_spent_per_branch: dict[str, int]
    final_scores: dict[str, float] | list[float]
    scorer: str | None = None


class SampleResult(StrictModel):
    sample_key: str
    task_id: str
    protocol_seed: int
    request_seed: int
    budget_name: str
    sample_index: int = Field(ge=0)
    response_text: str
    expected_answer: str
    grader: Literal["exact-match", "numeric"]
    tags: list[str]
    correct: bool
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    ttft_seconds: float | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)
    rollout_throughput_per_hour: float | None = Field(default=None, ge=0)
    kv_reuse_ratio: float | None = Field(default=None, ge=1)
    useful_token_ratio: float | None = Field(default=None, ge=0, le=1)
    tree: TreeStats | None = None


class NumericStats(StrictModel):
    count: int = Field(ge=0)
    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    spread: float | None = None
    p50: float | None = None
    p95: float | None = None


class MetricSet(StrictModel):
    task_count: int = Field(ge=1)
    sample_count: int = Field(ge=1)
    correct_sample_count: int = Field(ge=0)
    accuracy_at_k: dict[str, float]
    pass_power_k: dict[str, float]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    tokens_per_correct: float | None
    total_cost_usd: float = Field(ge=0)
    cost_per_correct_usd: float | None
    latency_seconds: NumericStats
    ttft_seconds: NumericStats
    tokens_per_second: NumericStats
    rollout_throughput_per_hour: NumericStats
    kv_reuse_ratio: NumericStats
    useful_token_ratio: NumericStats


class SeedMetrics(StrictModel):
    protocol_seed: int
    budget_name: str
    metrics: MetricSet


class AggregateValue(StrictModel):
    count: int = Field(ge=0)
    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    spread: float | None = None


class AggregateMetrics(StrictModel):
    budget_name: str
    seed_count: int = Field(ge=1)
    accuracy_at_k: dict[str, AggregateValue]
    pass_power_k: dict[str, AggregateValue]
    tokens_per_correct: AggregateValue
    cost_per_correct_usd: AggregateValue
    latency_mean_seconds: AggregateValue
    ttft_mean_seconds: AggregateValue
    tokens_per_second_mean: AggregateValue
    rollout_throughput_per_hour_mean: AggregateValue
    kv_reuse_ratio_mean: AggregateValue
    useful_token_ratio_mean: AggregateValue


class EngineConfigStamp(StrictModel):
    model: str
    base_url: str
    mode: Literal["sequential", "tree"]
    budgets: list[BudgetConfig]
    k_samples: int
    seeds: list[int]
    concurrency: int
    temperature: float
    top_p: float
    stop: list[str]
    tree: TreeConfig | None
    pricing: PricingConfig


class TaskSetStamp(StrictModel):
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    provenance: FixtureProvenance | RealProvenance = Field(discriminator="kind")


class EnvironmentStamp(StrictModel):
    generated_at_utc: str
    python: str
    platform: str
    thoughtbench_version: str
    autotree_sdk_version: str
    autotree_serve_version: str | None


class ResultsDocument(StrictModel):
    schema_version: Literal[RESULTS_SCHEMA_VERSION] = RESULTS_SCHEMA_VERSION
    artifact_notice: Literal[FIXTURE_NOTICE, REAL_NOTICE] = FIXTURE_NOTICE
    benchmark_claims_allowed: bool = False
    @model_validator(mode="after")
    def validate_notice_matches_provenance(self) -> "ResultsDocument":
        is_real = self.task_set.provenance.kind == "real"
        expected_notice = REAL_NOTICE if is_real else FIXTURE_NOTICE
        if self.artifact_notice != expected_notice:
            raise ValueError("artifact_notice must match task-set provenance kind")
        if self.benchmark_claims_allowed is not is_real:
            raise ValueError(
                "benchmark_claims_allowed must be True exactly when provenance is real"
            )
        return self

    run_id: str
    run_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine_config: EngineConfigStamp
    task_set: TaskSetStamp
    per_seed_metrics: list[SeedMetrics]
    aggregate_metrics: list[AggregateMetrics]
    samples: list[SampleResult]
    environment: EnvironmentStamp
