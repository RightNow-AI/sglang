"""Merge validated ThoughtBench result artifacts into a leaderboard."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from uuid import uuid4

from .bench_models import BenchmarkResults, Leaderboard, LeaderboardEntry
from .harness_schema import validate_benchmark_results


def _candidate_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f"aggregate source does not exist: {source}")
    return [path for path in sorted(source.glob("*.json")) if path.name != "leaderboard.json"]


def aggregate_results(source: Path) -> Leaderboard:
    """Load current-schema artifacts, ignoring preserved legacy result shapes."""

    entries: list[LeaderboardEntry] = []
    for path in _candidate_files(source):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not {"meta", "tasks", "summary"}.issubset(payload):
            continue
        validate_benchmark_results(payload)
        results = BenchmarkResults.model_validate(payload)
        entries.append(
            LeaderboardEntry(source=path.name, meta=results.meta, summary=results.summary)
        )
    if not entries:
        raise ValueError(f"no current ThoughtBench result files found in {source}")
    entries.sort(
        key=lambda entry: (
            -entry.summary.accuracy,
            entry.summary.tokens_per_correct is None,
            entry.summary.tokens_per_correct or 0,
            entry.meta.engine_label,
            entry.source,
        )
    )
    return Leaderboard(generated_at=datetime.now(UTC).isoformat(), entries=entries)


def write_leaderboard(source: Path, output: Path | None = None) -> tuple[Leaderboard, Path]:
    leaderboard = aggregate_results(source)
    destination = output or (source / "leaderboard.json" if source.is_dir() else source.with_name("leaderboard.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(leaderboard.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return leaderboard, destination
