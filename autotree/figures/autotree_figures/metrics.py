"""Metric extraction from versioned ThoughtBench result documents."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

from .loaders import LoadedRun


@dataclass(frozen=True)
class MetricPoint:
    budget_name: str
    accuracy: float
    accuracy_error: float
    total_tokens: float
    total_cost_usd: float
    cost_per_correct_usd: float | None
    throughput: float | None
    throughput_error: float | None


def _accuracy(metrics: dict[str, Any]) -> float:
    values = metrics["accuracy_at_k"]
    key = max(values, key=lambda item: int(item))
    return float(values[key])


def _optional_mean(stats: dict[str, Any]) -> float | None:
    value = stats.get("mean")
    return None if value is None else float(value)


def _spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_points(run: LoadedRun) -> list[MetricPoint]:
    """Return budget points with one-standard-deviation seed error bars."""

    rows_by_budget: dict[str, list[dict[str, Any]]] = {}
    for row in run.payload["per_seed_metrics"]:
        rows_by_budget.setdefault(row["budget_name"], []).append(row["metrics"])
    ordered_budgets = [item["name"] for item in run.payload["engine_config"]["budgets"]]
    points: list[MetricPoint] = []
    for budget_name in ordered_budgets:
        rows = rows_by_budget.get(budget_name, [])
        if not rows:
            continue
        if len(rows) != 3:
            raise ValueError(
                f"{run.path} budget {budget_name!r} must contain exactly three protocol seeds"
            )
        accuracies = [_accuracy(row) for row in rows]
        tokens = [float(row["input_tokens"] + row["output_tokens"]) for row in rows]
        costs = [float(row["total_cost_usd"]) for row in rows]
        throughputs = [
            value
            for row in rows
            if (value := _optional_mean(row["rollout_throughput_per_hour"])) is not None
        ]
        points.append(
            MetricPoint(
                budget_name=budget_name,
                accuracy=mean(accuracies),
                accuracy_error=_spread(accuracies),
                total_tokens=mean(tokens),
                total_cost_usd=mean(costs),
                cost_per_correct_usd=(
                    mean(
                        float(row["cost_per_correct_usd"])
                        for row in rows
                        if row.get("cost_per_correct_usd") is not None
                    )
                    if any(row.get("cost_per_correct_usd") is not None for row in rows)
                    else None
                ),
                throughput=mean(throughputs) if throughputs else None,
                throughput_error=_spread(throughputs) if throughputs else None,
            )
        )
    return points


def branch_count(run: LoadedRun) -> int:
    tree = run.payload["engine_config"].get("tree")
    if not isinstance(tree, dict) or not isinstance(tree.get("branches"), int):
        raise ValueError(f"{run.path} has no tree branching factor")
    return int(tree["branches"])
