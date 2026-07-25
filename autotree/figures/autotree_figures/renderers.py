"""Publication renderers for the seven Phase-4 paper figures."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

from .theme import PALETTE, save_figure

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .loaders import LoadedBundle, LoadedRun
from .metrics import branch_count, metric_points


@dataclass(frozen=True)
class RenderedFigure:
    figure_id: str
    title: str
    files: tuple[Path, Path]
    sources: tuple[str, ...]
    provenance: dict[str, Any]


def _runs(bundle: LoadedBundle, panel: str) -> list[LoadedRun]:
    selected = [run for run in bundle.runs if panel in run.reference.panels]
    if not selected:
        raise ValueError(f"figure bundle has no ThoughtBench runs for panel {panel!r}")
    return selected


def _run_provenance(runs: list[LoadedRun]) -> dict[str, Any]:
    kinds = {run.provenance["kind"].strip().lower() for run in runs}
    if len(kinds) != 1:
        raise ValueError("a figure cannot mix fixture and non-fixture result provenance")
    sources = sorted({str(run.provenance["source"]) for run in runs})
    return {"kind": next(iter(kinds)), "source": "; ".join(sources)}


def _source_paths(runs: list[LoadedRun]) -> tuple[str, ...]:
    return tuple(run.reference.path for run in runs)


def _group_by_model(runs: list[LoadedRun]) -> dict[str, list[LoadedRun]]:
    grouped: dict[str, list[LoadedRun]] = {}
    for run in runs:
        model = run.reference.model or str(run.payload["engine_config"]["model"])
        grouped.setdefault(model, []).append(run)
    return grouped


def _system(run: LoadedRun) -> str:
    return run.reference.system or run.reference.label


def render_scaling(bundle: LoadedBundle, stem: Path) -> RenderedFigure:
    runs = _runs(bundle, "scaling")
    grouped = _group_by_model(runs)
    if len(grouped) != 4:
        raise ValueError("accuracy-vs-tokens requires exactly four model groups")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True, sharey=True)
    styles = {"Sequential": "--", "AutoTree": "-"}
    for axis, (model, model_runs) in zip(axes.flat, grouped.items(), strict=True):
        systems = {_system(run) for run in model_runs}
        if systems != set(styles):
            raise ValueError(f"scaling model {model!r} requires Sequential and AutoTree series")
        for index, run in enumerate(model_runs):
            points = metric_points(run)
            system = _system(run)
            axis.errorbar(
                [point.total_tokens for point in points],
                [point.accuracy for point in points],
                yerr=[point.accuracy_error for point in points],
                marker="o",
                capsize=2.5,
                color=PALETTE[index % len(PALETTE)],
                linestyle=styles.get(system, "-"),
                label=system,
            )
        axis.set(title=model, xlabel="Mean tokens across tasks", ylabel="Accuracy@k")
        axis.set_ylim(-0.03, 1.03)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=2,
    )
    figure.suptitle("Accuracy vs. token budget", y=1.08)
    provenance = _run_provenance(runs)
    files = save_figure(figure, stem, provenance=provenance)
    plt.close(figure)
    return RenderedFigure("accuracy_vs_tokens", "Accuracy vs. tokens", files, _source_paths(runs), provenance)


def _pareto_indices(costs: list[float], accuracies: list[float]) -> list[int]:
    frontier: list[int] = []
    best = -np.inf
    for index in sorted(range(len(costs)), key=lambda item: (costs[item], -accuracies[item])):
        if accuracies[index] > best:
            frontier.append(index)
            best = accuracies[index]
    return frontier


def render_pareto(bundle: LoadedBundle, stem: Path) -> RenderedFigure:
    runs = _runs(bundle, "pareto")
    grouped = _group_by_model(runs)
    if len(grouped) != 4:
        raise ValueError("cost-accuracy Pareto requires exactly four model groups")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True, sharey=True)
    markers = {"vLLM": "s", "SGLang": "^", "AutoTree": "o"}
    for axis, (model, model_runs) in zip(axes.flat, grouped.items(), strict=True):
        systems = {_system(run) for run in model_runs}
        if systems != set(markers):
            raise ValueError(f"Pareto model {model!r} requires vLLM, SGLang, and AutoTree series")
        for index, run in enumerate(model_runs):
            points = metric_points(run)
            if any(point.cost_per_correct_usd is None for point in points):
                raise ValueError(f"{run.path} has no cost-per-correct measurements")
            costs = [float(point.cost_per_correct_usd) for point in points]
            accuracies = [point.accuracy for point in points]
            color = PALETTE[index % len(PALETTE)]
            system = _system(run)
            axis.errorbar(
                costs,
                accuracies,
                yerr=[point.accuracy_error for point in points],
                fmt=markers.get(system, "o"),
                capsize=2.5,
                color=color,
                alpha=0.75,
                label=system,
            )
            frontier = _pareto_indices(costs, accuracies)
            axis.plot([costs[item] for item in frontier], [accuracies[item] for item in frontier], color=color)
        axis.set(title=model, xlabel="Cost per correct answer (USD)", ylabel="Accuracy@k")
        axis.set_ylim(-0.03, 1.03)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=3,
    )
    figure.suptitle("Cost-accuracy Pareto frontier", y=1.08)
    provenance = _run_provenance(runs)
    files = save_figure(figure, stem, provenance=provenance)
    plt.close(figure)
    return RenderedFigure("cost_accuracy_pareto", "Cost vs. accuracy Pareto", files, _source_paths(runs), provenance)


def render_kv_reuse(bundle: LoadedBundle, stem: Path) -> RenderedFigure:
    spec = bundle.spec.supplemental.kv_reuse_heatmap
    values = np.asarray(spec.values, dtype=float)
    figure, axis = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    image = axis.imshow(values, vmin=1, vmax=float(values.max()), cmap="viridis", aspect="auto", origin="lower")
    axis.set(
        title="KV reuse by tree shape",
        xlabel="Branching factor",
        ylabel="Tree depth",
        xticks=np.arange(len(spec.branching_factors)),
        yticks=np.arange(len(spec.depths)),
        xticklabels=spec.branching_factors,
        yticklabels=spec.depths,
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            midpoint = 1 + (float(values.max()) - 1) / 2
            color = "white" if values[row, column] < midpoint else "black"
            axis.text(column, row, f"{values[row, column]:.1f}×", ha="center", va="center", color=color, fontsize=7)
    figure.colorbar(image, ax=axis, label="KV reuse ratio (logical / physical tokens)")
    provenance = spec.provenance.model_dump(exclude_none=True)
    files = save_figure(figure, stem, provenance=provenance)
    plt.close(figure)
    return RenderedFigure("kv_reuse_heatmap", "KV reuse heatmap", files, ("supplemental.kv_reuse_heatmap",), provenance)


def _topology_positions(nodes: list[Any]) -> dict[str, tuple[float, float]]:
    children: dict[str | None, list[str]] = defaultdict(list)
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        children[node.parent].append(node.id)
    for group in children.values():
        group.sort()
    root = children[None][0]
    depths: dict[str, int] = {root: 0}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for child in children[current]:
            depths[child] = depths[current] + 1
            queue.append(child)
    levels: dict[int, list[str]] = defaultdict(list)
    for node_id in by_id:
        levels[depths[node_id]].append(node_id)
    positions: dict[str, tuple[float, float]] = {}
    for depth, ids in levels.items():
        ids.sort()
        xs = np.linspace(0, 1, len(ids) + 2)[1:-1]
        for x, node_id in zip(xs, ids, strict=True):
            positions[node_id] = (float(x), float(-depth))
    return positions


def render_topology(bundle: LoadedBundle, stem: Path) -> RenderedFigure:
    spec = bundle.spec.supplemental.tree_topology
    positions = _topology_positions(spec.nodes)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for node in spec.nodes:
        if node.parent is not None:
            x0, y0 = positions[node.parent]
            x1, y1 = positions[node.id]
            axis.plot([x0, x1], [y0, y1], color="#9A9A9A", linewidth=1.1, zorder=1)
    active = [node for node in spec.nodes if not node.pruned]
    pruned = [node for node in spec.nodes if node.pruned]
    active_plot = axis.scatter(
        [positions[node.id][0] for node in active],
        [positions[node.id][1] for node in active],
        c=[node.value for node in active],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=180,
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )
    if pruned:
        axis.scatter(
            [positions[node.id][0] for node in pruned],
            [positions[node.id][1] for node in pruned],
            c="#D73027",
            marker="X",
            s=180,
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )
    for node in spec.nodes:
        if node.label:
            x, y = positions[node.id]
            axis.text(x, y - 0.22, node.label, ha="center", va="top", fontsize=6.5)
    figure.colorbar(active_plot, ax=axis, label="Value estimate", shrink=0.8)
    axis.legend(handles=[Line2D([0], [0], marker="X", color="none", markerfacecolor="#D73027", markeredgecolor="white", markersize=9, label="Pruned")], loc="lower left")
    axis.set_title(spec.title)
    axis.set_axis_off()
    provenance = spec.provenance.model_dump(exclude_none=True)
    files = save_figure(figure, stem, provenance=provenance)
    plt.close(figure)
    return RenderedFigure("tree_topology", "Tree topology", files, ("supplemental.tree_topology",), provenance)


def render_branching(bundle: LoadedBundle, stem: Path) -> RenderedFigure:
    runs = sorted(_runs(bundle, "branching"), key=branch_count)
    branches: list[int] = []
    accuracy: list[float] = []
    errors: list[float] = []
    for run in runs:
        point = metric_points(run)[0]
        branches.append(branch_count(run))
        accuracy.append(point.accuracy)
        errors.append(point.accuracy_error)
    if branches != [1, 2, 4, 8, 16, 32]:
        raise ValueError("branching ablation requires k in {1, 2, 4, 8, 16, 32}")
    figure, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    axis.errorbar(branches, accuracy, yerr=errors, marker="o", capsize=3, color=PALETTE[0])
    axis.set_xscale("log", base=2)
    axis.set_xticks(branches, labels=branches)
    axis.set(title="Branching-factor ablation", xlabel="Branching factor k", ylabel="Accuracy@k")
    axis.set_ylim(-0.03, 1.03)
    axis.text(0.02, 0.97, "Error bars: ±1 SD over seeds", transform=axis.transAxes, va="top", fontsize=7)
    provenance = _run_provenance(runs)
    files = save_figure(figure, stem, provenance=provenance)
    plt.close(figure)
    return RenderedFigure("branching_factor_ablation", "Branching-factor ablation", files, _source_paths(runs), provenance)


def render_throughput(bundle: LoadedBundle, stem: Path) -> RenderedFigure:
    runs = _runs(bundle, "throughput")
    labels: list[str] = []
    values: list[float] = []
    errors: list[float] = []
    for run in runs:
        available = [point for point in metric_points(run) if point.throughput is not None]
        if not available:
            raise ValueError(f"{run.path} has no rollout throughput metrics")
        labels.append(_system(run))
        values.append(mean(point.throughput for point in available if point.throughput is not None))
        errors.append(mean(point.throughput_error or 0.0 for point in available))
    if set(labels) != {"Sequential", "vLLM", "SGLang", "AutoTree"}:
        raise ValueError("throughput panel requires Sequential, vLLM, SGLang, and AutoTree")
    figure, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    positions = np.arange(len(labels))
    axis.bar(positions, values, yerr=errors, capsize=4, color=PALETTE[: len(labels)])
    axis.set_xticks(positions, labels=labels)
    axis.set(title="Rollout throughput", ylabel="Rollouts per hour per GPU", xlabel="Execution mode")
    axis.text(0.02, 0.97, "Bars: budget mean; error: mean seed SD", transform=axis.transAxes, va="top", fontsize=7)
    provenance = _run_provenance(runs)
    files = save_figure(figure, stem, provenance=provenance)
    plt.close(figure)
    return RenderedFigure("rollout_throughput", "Rollout throughput", files, _source_paths(runs), provenance)


def render_thinking_effort(bundle: LoadedBundle, stem: Path) -> RenderedFigure:
    spec = bundle.spec.supplemental.thinking_effort
    figure, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for index, series in enumerate(spec.series):
        efforts = [point.effort for point in series.points]
        values = [mean(point.seed_values) for point in series.points]
        errors = [stdev(point.seed_values) if len(point.seed_values) > 1 else 0.0 for point in series.points]
        axis.errorbar(efforts, values, yerr=errors, marker="o", capsize=3, color=PALETTE[index], label=series.label)
    axis.set(title=f"Thinking-effort sweep - {spec.model}", xlabel="thinking_effort", ylabel="Accuracy@k")
    axis.set_xlim(0.18, 1.01)
    axis.set_ylim(-0.03, 1.03)
    axis.legend(title="Mean ± 1 SD over seeds")
    provenance = spec.provenance.model_dump(exclude_none=True)
    files = save_figure(figure, stem, provenance=provenance)
    plt.close(figure)
    return RenderedFigure("thinking_effort_sweep", "Thinking-effort sweep", files, ("supplemental.thinking_effort",), provenance)


Renderer = Callable[[LoadedBundle, Path], RenderedFigure]

RENDERERS: tuple[tuple[str, str, Renderer], ...] = (
    ("01_accuracy_vs_tokens", "Accuracy vs. tokens", render_scaling),
    ("02_cost_accuracy_pareto", "Cost vs. accuracy Pareto", render_pareto),
    ("03_kv_reuse_heatmap", "KV reuse heatmap", render_kv_reuse),
    ("04_tree_topology", "Tree topology", render_topology),
    ("05_branching_factor_ablation", "Branching-factor ablation", render_branching),
    ("06_rollout_throughput", "Rollout throughput", render_throughput),
    ("07_thinking_effort_sweep", "Thinking-effort sweep", render_thinking_effort),
)
