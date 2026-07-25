"""Resumable benchmark execution through the public AutoTree SDK."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable
from uuid import uuid4

from autotree_sdk import TreeClient, TreeParameters
from pydantic import ValidationError

from . import __version__
from .graders import grade_task
from .metrics import aggregate_values, compute_metric_set
from .models import (
    FIXTURE_NOTICE,
    REAL_NOTICE,
    AggregateMetrics,
    BudgetConfig,
    EngineConfigStamp,
    EnvironmentStamp,
    ResultsDocument,
    RunConfig,
    SampleResult,
    SeedMetrics,
    Task,
    TaskSetStamp,
    TreeStats,
)
from .schema import validate_results_payload
from .tasks import load_tasks


class ResumeError(ValueError):
    """Raised when a partial run cannot safely be resumed."""


def load_run_config(path: Path) -> RunConfig:
    """Load JSON configuration and resolve artifact paths beside that file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    task_path = Path(payload["task_set"]["path"])
    output_path = Path(payload["output_path"])
    if not task_path.is_absolute():
        payload["task_set"]["path"] = str(base / task_path)
    if not output_path.is_absolute():
        payload["output_path"] = str(base / output_path)
    return RunConfig.model_validate(payload)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def run_fingerprint(config: RunConfig, task_sha256: str) -> str:
    """Bind resumable samples to the complete config and exact task bytes."""

    material = {
        "config": config.model_dump(mode="json"),
        "task_sha256": task_sha256,
    }
    return hashlib.sha256(_canonical_json(material)).hexdigest()


def sample_key(task_id: str, seed: int, budget_name: str, sample_index: int) -> str:
    material = [task_id, seed, budget_name, sample_index]
    return hashlib.sha256(_canonical_json(material)).hexdigest()


def request_seed(protocol_seed: int, sample_index: int) -> int:
    digest = hashlib.sha256(f"{protocol_seed}:{sample_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def partial_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.partial.jsonl")


class PartialStore:
    """Append-only sample journal used for crash-safe resume."""

    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint

    def load(self) -> dict[str, SampleResult]:
        if not self.path.exists():
            return {}
        raw = self.path.read_bytes()
        lines = raw.splitlines()
        completed: dict[str, SampleResult] = {}
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                envelope = json.loads(line)
                if envelope["run_fingerprint"] != self.fingerprint:
                    raise ResumeError(
                        "partial run fingerprint does not match this config/task set"
                    )
                sample = SampleResult.model_validate(envelope["sample"])
            except ResumeError:
                raise
            except (json.JSONDecodeError, KeyError, ValidationError) as exc:
                is_truncated_tail = index == len(lines) and not raw.endswith(b"\n")
                if is_truncated_tail:
                    break
                raise ResumeError(f"invalid partial record on line {index}: {exc}") from exc
            previous = completed.get(sample.sample_key)
            if previous is not None and previous != sample:
                raise ResumeError(f"conflicting partial records for {sample.sample_key}")
            completed[sample.sample_key] = sample
        return completed

    def append(self, sample: SampleResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "run_fingerprint": self.fingerprint,
            "sample": sample.model_dump(mode="json"),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _content_text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("ThoughtBench requires text completion content")
    return value


def _tree_stats(tree: Any) -> TreeStats:
    extras = tree.model_extra or {}
    return TreeStats(
        policy=extras.get("policy"),
        branch_count=tree.branch_count,
        pruned_count=tree.pruned_count,
        merged_count=extras.get("merged_count"),
        winner_branch_id=extras.get("winner_branch_id"),
        tokens_spent_per_branch=tree.tokens_spent_per_branch,
        final_scores=tree.final_scores,
        scorer=extras.get("scorer"),
    )


def _reported_ratio(source: Any, name: str) -> float | None:
    value = getattr(source, name, None)
    if value is None:
        extras = getattr(source, "model_extra", None) or {}
        value = extras.get(name)
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    if name == "kv_reuse_ratio":
        return parsed if parsed >= 1 else None
    return parsed if 0 <= parsed <= 1 else None


def _sequential_completion(
    config: RunConfig,
    task: Task,
    budget: BudgetConfig,
    seed: int,
) -> tuple[str, int, int, float, float | None, TreeStats | None, float | None, float | None]:
    started = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    with TreeClient(str(config.base_url), timeout=config.timeout_seconds) as client:
        chunks = client.completions(
            model=config.model,
            messages=[{"role": "user", "content": task.prompt}],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=budget.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop or None,
            seed=seed,
        )
        for chunk in chunks:
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(content)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
    ended = time.perf_counter()
    if usage is None:
        raise RuntimeError("endpoint did not report token usage for streamed completion")
    latency = ended - started
    ttft = first_token_at - started if first_token_at is not None else None
    return (
        "".join(text_parts),
        int(usage.get("prompt_tokens", 0)),
        int(usage["completion_tokens"]),
        latency,
        ttft,
        None,
        None,
        None,
    )


def _tree_completion(
    config: RunConfig,
    task: Task,
    budget: BudgetConfig,
    seed: int,
) -> tuple[str, int, int, float, float | None, TreeStats | None, float | None, float | None]:
    assert config.tree is not None
    assert budget.tree_budget_tokens is not None
    tree_parameters = TreeParameters(
        policy=config.tree.policy,
        branches=config.tree.branches,
        budget_tokens=budget.tree_budget_tokens,
        scorer=config.tree.scorer,
    )
    started = time.perf_counter()
    with TreeClient(str(config.base_url), timeout=config.timeout_seconds) as client:
        response = client.tree_completions(
            model=config.model,
            messages=[{"role": "user", "content": task.prompt}],
            tree=tree_parameters,
            max_tokens=budget.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop or None,
            seed=seed,
        )
    ended = time.perf_counter()
    if response.usage is None:
        raise RuntimeError("tree endpoint did not report token usage")
    return (
        _content_text(response.choices[0].message.content),
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        ended - started,
        None,
        _tree_stats(response.tree),
        _reported_ratio(response.tree, "kv_reuse_ratio"),
        _reported_ratio(response, "useful_token_ratio"),
    )


def _execute_sample(
    config: RunConfig,
    task: Task,
    budget: BudgetConfig,
    protocol_seed: int,
    sample_index: int,
) -> SampleResult:
    derived_seed = request_seed(protocol_seed, sample_index)
    completion = (
        _sequential_completion(config, task, budget, derived_seed)
        if config.mode == "sequential"
        else _tree_completion(config, task, budget, derived_seed)
    )
    response, prompt_tokens, completion_tokens, latency, ttft, tree, kv, useful = completion
    total_tokens = prompt_tokens + completion_tokens
    return SampleResult(
        sample_key=sample_key(task.id, protocol_seed, budget.name, sample_index),
        task_id=task.id,
        protocol_seed=protocol_seed,
        request_seed=derived_seed,
        budget_name=budget.name,
        sample_index=sample_index,
        response_text=response,
        expected_answer=task.answer,
        grader=task.grader,
        tags=task.tags,
        correct=grade_task(task, response),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency,
        ttft_seconds=ttft,
        tokens_per_second=(completion_tokens / latency if latency > 0 else None),
        rollout_throughput_per_hour=(3600 / latency if latency > 0 else None),
        kv_reuse_ratio=kv,
        useful_token_ratio=useful,
        tree=tree,
    )


def _specifications(
    config: RunConfig, tasks: Iterable[Task]
) -> list[tuple[Task, BudgetConfig, int, int]]:
    return [
        (task, budget, seed, sample_index)
        for budget in config.budgets
        for seed in config.seeds
        for task in tasks
        for sample_index in range(config.k_samples)
    ]


def _aggregate(per_seed: list[SeedMetrics], config: RunConfig) -> list[AggregateMetrics]:
    aggregates: list[AggregateMetrics] = []
    for budget in config.budgets:
        cells = [cell.metrics for cell in per_seed if cell.budget_name == budget.name]
        aggregates.append(
            AggregateMetrics(
                budget_name=budget.name,
                seed_count=len(cells),
                accuracy_at_k={
                    str(k): aggregate_values([cell.accuracy_at_k[str(k)] for cell in cells])
                    for k in config.metric_ks
                },
                pass_power_k={
                    str(k): aggregate_values([cell.pass_power_k[str(k)] for cell in cells])
                    for k in config.metric_ks
                },
                tokens_per_correct=aggregate_values(
                    [cell.tokens_per_correct for cell in cells]
                ),
                cost_per_correct_usd=aggregate_values(
                    [cell.cost_per_correct_usd for cell in cells]
                ),
                latency_mean_seconds=aggregate_values(
                    [cell.latency_seconds.mean for cell in cells]
                ),
                ttft_mean_seconds=aggregate_values(
                    [cell.ttft_seconds.mean for cell in cells]
                ),
                tokens_per_second_mean=aggregate_values(
                    [cell.tokens_per_second.mean for cell in cells]
                ),
                rollout_throughput_per_hour_mean=aggregate_values(
                    [cell.rollout_throughput_per_hour.mean for cell in cells]
                ),
                kv_reuse_ratio_mean=aggregate_values(
                    [cell.kv_reuse_ratio.mean for cell in cells]
                ),
                useful_token_ratio_mean=aggregate_values(
                    [cell.useful_token_ratio.mean for cell in cells]
                ),
            )
        )
    return aggregates


def _write_results(document: ResultsDocument, path: Path) -> None:
    payload = document.model_dump(mode="json")
    validate_results_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_benchmark(config: RunConfig, *, output_path: Path | None = None) -> ResultsDocument:
    """Execute or resume a complete fixture benchmark run."""

    tasks, task_sha256 = load_tasks(config.task_set.path)
    fingerprint = run_fingerprint(config, task_sha256)
    destination = (output_path or config.output_path).resolve()
    store = PartialStore(partial_path_for(destination), fingerprint)
    completed = store.load()
    specifications = _specifications(config, tasks)
    missing = [
        spec
        for spec in specifications
        if sample_key(spec[0].id, spec[2], spec[1].name, spec[3]) not in completed
    ]
    if missing:
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {
                executor.submit(_execute_sample, config, *spec): spec for spec in missing
            }
            for future in as_completed(futures):
                sample = future.result()
                store.append(sample)
                completed[sample.sample_key] = sample

    expected_keys = {
        sample_key(task.id, seed, budget.name, sample_index)
        for task, budget, seed, sample_index in specifications
    }
    if completed.keys() != expected_keys:
        unexpected = sorted(completed.keys() - expected_keys)
        missing_keys = sorted(expected_keys - completed.keys())
        raise ResumeError(
            f"partial sample set mismatch; unexpected={unexpected}, missing={missing_keys}"
        )
    samples = sorted(
        completed.values(),
        key=lambda item: (
            item.budget_name,
            item.protocol_seed,
            item.task_id,
            item.sample_index,
        ),
    )
    per_seed: list[SeedMetrics] = []
    for budget in config.budgets:
        for seed in config.seeds:
            cell_samples = [
                sample
                for sample in samples
                if sample.budget_name == budget.name and sample.protocol_seed == seed
            ]
            per_seed.append(
                SeedMetrics(
                    protocol_seed=seed,
                    budget_name=budget.name,
                    metrics=compute_metric_set(
                        cell_samples,
                        metric_ks=config.metric_ks,
                        input_cost_per_million=config.pricing.input_per_million_usd,
                        output_cost_per_million=config.pricing.output_per_million_usd,
                    ),
                )
            )
    sdk_version = _package_version("autotree-sdk") or "unknown"
    is_real = config.task_set.provenance.kind == "real"
    document = ResultsDocument(
        artifact_notice=REAL_NOTICE if is_real else FIXTURE_NOTICE,
        benchmark_claims_allowed=is_real,
        run_id=str(uuid4()),
        run_fingerprint=fingerprint,
        engine_config=EngineConfigStamp(
            model=config.model,
            base_url=str(config.base_url),
            mode=config.mode,
            budgets=config.budgets,
            k_samples=config.k_samples,
            seeds=config.seeds,
            concurrency=config.concurrency,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop,
            tree=config.tree,
            pricing=config.pricing,
        ),
        task_set=TaskSetStamp(
            name=config.task_set.name,
            sha256=task_sha256,
            task_count=len(tasks),
            provenance=config.task_set.provenance,
        ),
        per_seed_metrics=per_seed,
        aggregate_metrics=_aggregate(per_seed, config),
        samples=samples,
        environment=EnvironmentStamp(
            generated_at_utc=datetime.now(UTC).isoformat(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            thoughtbench_version=__version__,
            autotree_sdk_version=sdk_version,
            autotree_serve_version=_package_version("autotree-serve"),
        ),
    )
    _write_results(document, destination)
    store.clear()
    return document
