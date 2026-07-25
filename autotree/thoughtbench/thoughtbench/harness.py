"""Endpoint-agnostic benchmark execution and artifact writing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from pydantic import ValidationError
import yaml

from .answers import extract_answer, normalize_answer
from .arms import build_request_body, endpoint_url, execute_arm
from .bench_models import BenchmarkResults, BenchTask, HarnessConfig, ResultMeta, TaskResult
from .harness_schema import validate_benchmark_results
from .statistics import summarize


class HarnessFileError(ValueError):
    """Raised when a harness configuration or task file is malformed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise HarnessFileError(f"could not load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HarnessFileError(f"{path}: top-level value must be an object")
    return payload


def load_harness_config(path: Path) -> HarnessConfig:
    """Load JSON or YAML and resolve artifact paths beside the config file."""

    payload = _load_mapping(path)
    base = path.resolve().parent
    for field in ("task_file", "output_dir"):
        if field not in payload:
            continue
        value = Path(payload[field])
        if not value.is_absolute():
            payload[field] = str(base / value)
    try:
        return HarnessConfig.model_validate(payload)
    except ValidationError as exc:
        raise HarnessFileError(f"invalid harness config {path}: {exc}") from exc


def load_bench_tasks(path: Path) -> tuple[list[BenchTask], str]:
    """Load strict JSONL tasks and return the exact-file SHA-256 digest."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HarnessFileError(f"could not read task file {path}: {exc}") from exc
    tasks: list[BenchTask] = []
    seen: set[str] = set()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise HarnessFileError(f"{path}: task file must be UTF-8: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            task = BenchTask.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HarnessFileError(f"{path}:{line_number}: {exc}") from exc
        if task.id in seen:
            raise HarnessFileError(f"{path}:{line_number}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise HarnessFileError(f"{path}: task file is empty")
    return tasks, hashlib.sha256(raw).hexdigest()


def _redact_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    hostname = parsed.hostname or "redacted-host"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{parsed.port}" if parsed.port is not None else hostname
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _request_specs(tasks: list[BenchTask], seeds: list[int]) -> list[tuple[int, BenchTask, int]]:
    return [
        (task_index * len(seeds) + seed_index, task, seed)
        for task_index, task in enumerate(tasks)
        for seed_index, seed in enumerate(seeds)
    ]


def _limit_tasks(tasks: list[BenchTask], limit: int | None) -> list[BenchTask]:
    if limit is None:
        return tasks
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return tasks[:limit]


def dry_run_requests(
    config: HarnessConfig,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return every exact request body without opening a network connection."""

    tasks, _digest = load_bench_tasks(config.task_file)
    tasks = _limit_tasks(tasks, limit)
    return [
        {
            "task_id": task.id,
            "seed": seed,
            "url": endpoint_url(str(config.base_url), config.arm),
            "body": build_request_body(config, task.prompt, seed),
        }
        for _index, task, seed in _request_specs(tasks, config.seeds)
    ]


def _headers(config: HarnessConfig) -> dict[str, str]:
    headers = dict(config.extra_headers)
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env)
        if api_key:
            headers.setdefault("Authorization", f"Bearer {api_key}")
    return headers


def _execute_task(
    index: int,
    task: BenchTask,
    seed: int,
    config: HarnessConfig,
    client: httpx.Client,
) -> tuple[int, TaskResult]:
    started = time.perf_counter()
    outcome = execute_arm(client, config, task.prompt, seed)
    wall_s = time.perf_counter() - started
    normalized_gold = extract_answer(task.gold) or normalize_answer(task.gold)
    return index, TaskResult(
        id=task.id,
        seed=seed,
        correct=outcome.answer is not None and outcome.answer == normalized_gold,
        answer=outcome.answer,
        gold=task.gold,
        tokens=outcome.completion_tokens,
        wall_s=wall_s,
    )


def _params(
    config: HarnessConfig,
    task_sha256: str,
    task_count: int,
    limit: int | None,
) -> dict[str, Any]:
    return {
        "label": config.label,
        "task_file": config.task_file.name,
        "task_sha256": task_sha256,
        "task_count": task_count,
        "limit": limit,
        "n": config.n,
        "branches": config.branches,
        "budget_tokens": config.budget_tokens,
        "policy": config.policy if config.arm == "tree" else None,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "concurrency": config.concurrency,
        "timeout": config.timeout,
    }


def _write_results(results: BenchmarkResults, output_dir: Path, label: str) -> Path:
    payload = results.model_dump(mode="json")
    validate_benchmark_results(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_dir / f"{timestamp}-{label}.json"
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def run_harness(
    config: HarnessConfig,
    *,
    transport: httpx.BaseTransport | None = None,
    limit: int | None = None,
) -> tuple[BenchmarkResults, Path]:
    """Run every task for every configured seed and write one validated artifact."""

    tasks, task_sha256 = load_bench_tasks(config.task_file)
    tasks = _limit_tasks(tasks, limit)
    started_at = datetime.now(UTC).isoformat()
    ordered: list[TaskResult | None] = [None] * (len(tasks) * len(config.seeds))
    with httpx.Client(
        timeout=config.timeout,
        headers=_headers(config),
        transport=transport,
    ) as client:
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = [
                executor.submit(_execute_task, index, task, seed, config, client)
                for index, task, seed in _request_specs(tasks, config.seeds)
            ]
            for future in as_completed(futures):
                index, result = future.result()
                ordered[index] = result
    task_results = [result for result in ordered if result is not None]
    if len(task_results) != len(ordered):
        raise RuntimeError("benchmark execution completed with missing task results")
    results = BenchmarkResults(
        meta=ResultMeta(
            engine_label=config.engine_label,
            model=config.model,
            base_url_redacted=_redact_base_url(str(config.base_url)),
            arm=config.arm,
            params=_params(config, task_sha256, len(tasks), limit),
            seeds=config.seeds,
            git_sha=_git_sha(),
            started_at=started_at,
        ),
        tasks=task_results,
        summary=summarize(task_results),
    )
    return results, _write_results(results, config.output_dir, config.label)
