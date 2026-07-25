"""Terse, dependency-free results table rendering."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ResultsDocument
from .schema import validate_results_payload


def _number(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_results_payload(payload)
    results = ResultsDocument.model_validate(payload)
    ks = sorted(
        {int(k) for aggregate in results.aggregate_metrics for k in aggregate.accuracy_at_k}
    )
    headers = ["mode", "budget", "seeds", "samples"]
    headers.extend(f"acc@{k}" for k in ks)
    headers.extend(["$/correct", "ttft_s", "kv_reuse"])
    rows: list[list[str]] = []
    for aggregate in results.aggregate_metrics:
        sample_count = sum(
            cell.metrics.sample_count
            for cell in results.per_seed_metrics
            if cell.budget_name == aggregate.budget_name
        )
        row = [
            results.engine_config.mode,
            aggregate.budget_name,
            str(aggregate.seed_count),
            str(sample_count),
        ]
        row.extend(_number(aggregate.accuracy_at_k[str(k)].mean) for k in ks)
        row.extend(
            [
                _number(aggregate.cost_per_correct_usd.mean, 6),
                _number(aggregate.ttft_mean_seconds.mean, 6),
                _number(aggregate.kv_reuse_ratio_mean.mean),
            ]
        )
        rows.append(row)
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [results.artifact_notice]
    lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)
