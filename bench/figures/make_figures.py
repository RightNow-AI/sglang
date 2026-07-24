#!/usr/bin/env python3
"""Render publication figures from cascade and rollout summary JSON files."""

import argparse
import glob
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


FORMATS = ("pdf", "svg", "png")
PERFORMANCE_FIELDS = (
    "mode",
    "seed",
    "accuracy",
    "correct_count",
    "items",
    "total_cost_units",
)
ROLLOUT_FIELDS = ("generated_tokens_per_rollout", "effective_diversity")
COLORS = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
)
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "h")


class FigureUnavailable(ValueError):
    """Raised when the input JSON does not contain a figure's prerequisites."""


@dataclass(frozen=True)
class SummaryDocument:
    path: Path
    label: str
    data: dict


@dataclass(frozen=True)
class PerformanceSeries:
    path: Path
    label: str
    mode: str
    total_items: int
    total_correct: int
    total_cost_units: float
    seed_accuracies: tuple
    seed_costs_per_problem: tuple

    @property
    def accuracy(self):
        return self.total_correct / self.total_items

    @property
    def cost_per_problem(self):
        return self.total_cost_units / self.total_items

    @property
    def cost_per_correct(self):
        if self.total_correct == 0:
            return None
        return self.total_cost_units / self.total_correct


@dataclass(frozen=True)
class RolloutArm:
    path: Path
    label: str
    mode: str
    raw_tokens_per_rollout: float
    adjusted_tokens_per_effective_sample: float
    effective_diversity: float


def require_mapping(value, source, field):
    if not isinstance(value, dict):
        raise ValueError("{}: field '{}' must be an object".format(source, field))
    return value


def require_list(value, source, field):
    if not isinstance(value, list) or not value:
        raise ValueError(
            "{}: field '{}' must be a nonempty list".format(source, field)
        )
    return value


def require_field(mapping, field, source, location):
    if field not in mapping:
        raise ValueError(
            "{}: missing field '{}.{}'".format(source, location, field)
        )
    return mapping[field]


def require_nonempty_string(value, source, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "{}: field '{}' must be a nonempty string".format(source, field)
        )
    return value.strip()


def require_integer(value, source, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            "{}: field '{}' must be an integer >= {}".format(
                source, field, minimum
            )
        )
    return value


def require_number(value, source, field, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{}: field '{}' must be numeric".format(source, field))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{}: field '{}' must be finite".format(source, field))
    if minimum is not None and value < minimum:
        raise ValueError(
            "{}: field '{}' must be >= {}".format(source, field, minimum)
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            "{}: field '{}' must be <= {}".format(source, field, maximum)
        )
    return value


def normalized_path(path):
    return os.path.normcase(str(Path(path).resolve()))


def expand_summary_paths(patterns):
    expanded = []
    seen = set()
    for pattern in patterns:
        literal = Path(pattern)
        if literal.is_file():
            matches = [str(literal)]
        else:
            matches = sorted(glob.glob(pattern, recursive=True), key=os.path.normcase)
        file_matches = [Path(match) for match in matches if Path(match).is_file()]
        if not file_matches:
            raise ValueError(
                "--summaries pattern matched no files: {}".format(pattern)
            )
        for match in file_matches:
            resolved = match.resolve()
            key = normalized_path(resolved)
            if key not in seen:
                seen.add(key)
                expanded.append(resolved)
    return expanded


def parse_labels(raw_labels, paths):
    known = {normalized_path(path): path for path in paths}
    labels = {}
    for raw in raw_labels:
        if "=" not in raw:
            raise ValueError(
                "--label must use <path>=<display name>: {}".format(raw)
            )
        raw_path, display_name = raw.split("=", 1)
        if not raw_path.strip() or not display_name.strip():
            raise ValueError(
                "--label must use nonempty path and display name: {}".format(raw)
            )
        key = normalized_path(raw_path.strip())
        if key not in known:
            raise ValueError(
                "--label path is not among expanded summaries: {}".format(raw_path)
            )
        if key in labels:
            raise ValueError("duplicate --label for path: {}".format(raw_path))
        labels[key] = display_name.strip()
    return {path: labels.get(normalized_path(path), path.stem) for path in paths}


def read_documents(paths, labels):
    documents = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError("{}: invalid JSON: {}".format(path, exc)) from exc
        data = require_mapping(data, str(path), "root")
        documents.append(SummaryDocument(path, labels[path], data))
    return documents


def parse_performance_series(document):
    source = str(document.path)
    config = require_mapping(
        require_field(document.data, "config", source, "root"), source, "config"
    )
    config_mode = require_nonempty_string(
        require_field(config, "mode", source, "config"), source, "config.mode"
    )
    rows = require_list(
        require_field(document.data, "summaries", source, "root"),
        source,
        "summaries",
    )
    total_items = 0
    total_correct = 0
    total_cost_units = 0.0
    seed_accuracies = []
    seed_costs_per_problem = []
    seeds = set()
    modes = set()

    for index, value in enumerate(rows):
        location = "summaries[{}]".format(index)
        row = require_mapping(value, source, location)
        for field in PERFORMANCE_FIELDS:
            require_field(row, field, source, location)
        mode = require_nonempty_string(row["mode"], source, location + ".mode")
        seed = row["seed"]
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise ValueError(
                "{}: field '{}.seed' must be an integer or string".format(
                    source, location
                )
            )
        if seed in seeds:
            raise ValueError(
                "{}: duplicate seed {!r} in summaries".format(source, seed)
            )
        seeds.add(seed)
        modes.add(mode)
        items = require_integer(row["items"], source, location + ".items", minimum=1)
        correct = require_integer(
            row["correct_count"], source, location + ".correct_count"
        )
        if correct > items:
            raise ValueError(
                "{}: field '{}.correct_count' cannot exceed items".format(
                    source, location
                )
            )
        accuracy = require_number(
            row["accuracy"], source, location + ".accuracy", minimum=0.0, maximum=1.0
        )
        expected_accuracy = correct / items
        if not math.isclose(
            accuracy, expected_accuracy, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                "{}: field '{}.accuracy' is {}, expected {} from counts".format(
                    source, location, accuracy, expected_accuracy
                )
            )
        cost = require_number(
            row["total_cost_units"],
            source,
            location + ".total_cost_units",
            minimum=0.0,
        )
        total_items += items
        total_correct += correct
        total_cost_units += cost
        seed_accuracies.append(accuracy)
        seed_costs_per_problem.append(cost / items)

    if modes != {config_mode}:
        raise ValueError(
            "{}: config.mode {!r} does not match summary modes {}".format(
                source, config_mode, sorted(modes)
            )
        )
    return PerformanceSeries(
        path=document.path,
        label=document.label,
        mode=config_mode,
        total_items=total_items,
        total_correct=total_correct,
        total_cost_units=total_cost_units,
        seed_accuracies=tuple(seed_accuracies),
        seed_costs_per_problem=tuple(seed_costs_per_problem),
    )


def parse_rollout_arm(document):
    source = str(document.path)
    config = require_mapping(
        require_field(document.data, "config", source, "root"), source, "config"
    )
    mode = require_nonempty_string(
        require_field(config, "mode", source, "config"), source, "config.mode"
    )
    summary = require_mapping(
        require_field(document.data, "summary", source, "root"), source, "summary"
    )
    for field in ROLLOUT_FIELDS:
        require_field(summary, field, source, "summary")
    raw = require_number(
        summary["generated_tokens_per_rollout"],
        source,
        "summary.generated_tokens_per_rollout",
        minimum=0.0,
    )
    diversity = require_number(
        summary["effective_diversity"],
        source,
        "summary.effective_diversity",
        minimum=0.0,
        maximum=1.0,
    )
    if diversity == 0.0:
        raise ValueError(
            "{}: summary.effective_diversity is zero, so adjusted tokens per "
            "effective sample is undefined".format(source)
        )
    return RolloutArm(
        path=document.path,
        label=document.label,
        mode=mode,
        raw_tokens_per_rollout=raw,
        adjusted_tokens_per_effective_sample=raw / diversity,
        effective_diversity=diversity,
    )


def collect_inputs(documents):
    performance = []
    rollout = []
    performance_errors = []
    rollout_errors = []
    for document in documents:
        recognized = False
        if "summaries" in document.data:
            recognized = True
            try:
                performance.append(parse_performance_series(document))
            except ValueError as exc:
                performance_errors.append(str(exc))
        if "summary" in document.data:
            recognized = True
            try:
                rollout.append(parse_rollout_arm(document))
            except ValueError as exc:
                rollout_errors.append(str(exc))
        if not recognized:
            print(
                "IGNORE {}: root contains neither 'summaries' nor 'summary'".format(
                    document.path
                )
            )
    return performance, rollout, performance_errors, rollout_errors


def normalized_mode(mode):
    return "".join(character for character in mode.lower() if character.isalnum())


def baseline_role(series):
    mode = normalized_mode(series.mode)
    if mode in {
        "largebo8",
        "bo8",
        "vote8",
        "largevote8",
        "bestof8",
        "largebestof8",
    }:
        return "vote8"
    if mode in {"largegreedy", "greedy"}:
        return "greedy"
    return "candidate"


def unique_baseline(series, role, display_name):
    matches = [result for result in series if baseline_role(result) == role]
    if not matches:
        raise FigureUnavailable("no {} summary is present".format(display_name))
    if len(matches) > 1:
        raise FigureUnavailable(
            "multiple {} summaries are present; pass one comparable experiment set"
            .format(display_name)
        )
    return matches[0]


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render publication figures. Install "
            "matplotlib and rerun this command. Original import error: {}".format(exc)
        ) from exc
    return matplotlib, plt, PercentFormatter


def save_figure(fig, out_dir, stem):
    written = []
    for extension in FORMATS:
        path = out_dir / "{}.{}".format(stem, extension)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path.resolve())
    return written


def lighten(color, amount):
    color = color.lstrip("#")
    channels = tuple(int(color[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    return tuple(channel + (1.0 - channel) * amount for channel in channels)


def format_value(value):
    absolute = abs(value)
    if absolute >= 1000.0:
        return "{:,.0f}".format(value)
    if absolute >= 10.0:
        return "{:,.1f}".format(value)
    return "{:,.2f}".format(value)


def plot_cost_vs_accuracy(series, plt, PercentFormatter):
    if not series:
        raise FigureUnavailable("no config + summaries performance JSONs are present")
    vote8 = unique_baseline(series, "vote8", "vote@8")
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    bounds = []
    for index, result in enumerate(series):
        x_value = result.cost_per_problem
        y_value = 100.0 * result.accuracy
        x_min = min(result.seed_costs_per_problem)
        x_max = max(result.seed_costs_per_problem)
        y_min = 100.0 * min(result.seed_accuracies)
        y_max = 100.0 * max(result.seed_accuracies)
        bounds.extend((y_min, y_max))
        ax.errorbar(
            x_value,
            y_value,
            xerr=[[x_value - x_min], [x_max - x_value]],
            yerr=[[y_value - y_min], [y_max - y_value]],
            fmt=MARKERS[index % len(MARKERS)],
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color=COLORS[index % len(COLORS)],
            ecolor=COLORS[index % len(COLORS)],
            elinewidth=0.9,
            capsize=2.5,
            zorder=3,
            label=result.label,
        )
    guide_accuracy = 100.0 * vote8.accuracy
    ax.axhline(
        guide_accuracy,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
        label="vote@8 accuracy guide",
    )
    lower = max(0.0, min(bounds) - 3.0)
    upper = min(100.0, max(bounds) + 3.0)
    if math.isclose(lower, upper):
        lower = max(0.0, lower - 1.0)
        upper = min(100.0, upper + 1.0)
    ax.set_ylim(lower, upper)
    ax.set_xlim(left=0.0)
    ax.margins(x=0.08)
    ax.set_title("Cost vs accuracy")
    ax.set_xlabel("Cost units per problem")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.grid(True, color="0.5", alpha=0.3, linewidth=0.65)
    ax.legend(loc="best", fontsize=6.8, frameon=False, handlelength=1.5)
    fig.tight_layout()
    return fig


def plot_cost_per_correct_bars(series, plt):
    if not series:
        raise FigureUnavailable("no config + summaries performance JSONs are present")
    missing = [result.label for result in series if result.cost_per_correct is None]
    if missing:
        raise FigureUnavailable(
            "cost per correct is undefined because correct_count is zero for: {}"
            .format(", ".join(missing))
        )
    ordered = sorted(series, key=lambda result: result.cost_per_correct)
    height = max(2.35, 0.34 * len(ordered) + 1.15)
    fig, ax = plt.subplots(figsize=(3.45, height))
    best_value = ordered[0].cost_per_correct
    dark_color = COLORS[0]
    light_color = lighten(dark_color, 0.5)
    colors = [
        dark_color
        if math.isclose(result.cost_per_correct, best_value)
        else light_color
        for result in ordered
    ]
    positions = list(range(len(ordered)))
    values = [result.cost_per_correct for result in ordered]
    bars = ax.barh(positions, values, color=colors, edgecolor="none", height=0.65)
    ax.set_yticks(positions)
    ax.set_yticklabels([result.label for result in ordered])
    ax.invert_yaxis()
    maximum = max(values)
    text_offset = max(maximum * 0.018, 0.02)
    for bar, value in zip(bars, values):
        ax.text(
            value + text_offset,
            bar.get_y() + bar.get_height() / 2.0,
            format_value(value),
            va="center",
            ha="left",
            fontsize=7.2,
        )
    ax.set_xlim(0.0, maximum * 1.23 if maximum > 0.0 else 1.0)
    ax.set_title("Cost per correct answer")
    ax.set_xlabel("Cost units per correct answer")
    ax.grid(True, axis="x", color="0.5", alpha=0.3, linewidth=0.65)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_capture_vs_cost(series, plt, PercentFormatter):
    if not series:
        raise FigureUnavailable("no config + summaries performance JSONs are present")
    vote8 = unique_baseline(series, "vote8", "vote@8")
    greedy = unique_baseline(series, "greedy", "greedy")
    candidates = [result for result in series if baseline_role(result) == "candidate"]
    if not candidates:
        raise FigureUnavailable("no non-baseline system summaries are present")
    if vote8.cost_per_correct in (None, 0.0):
        raise FigureUnavailable("vote@8 cost per correct is undefined or zero")
    search_lift = vote8.accuracy - greedy.accuracy
    if search_lift <= 0.0:
        raise FigureUnavailable(
            "vote@8 accuracy does not exceed greedy accuracy, so search-lift "
            "capture is undefined"
        )

    points = []
    for result in candidates:
        if result.cost_per_correct is None:
            raise FigureUnavailable(
                "cost per correct is undefined because correct_count is zero for: {}"
                .format(result.label)
            )
        cost_ratio = result.cost_per_correct / vote8.cost_per_correct
        capture = 100.0 * (result.accuracy - greedy.accuracy) / search_lift
        points.append((result, cost_ratio, capture))

    x_values = [point[1] for point in points]
    y_values = [point[2] for point in points]
    x_upper = max(1.05, 0.42, max(x_values) * 1.12)
    y_lower = min(0.0, min(y_values) - 10.0)
    y_upper = max(105.0, max(y_values) + 10.0)

    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    ax.fill_between(
        [0.0, 0.35],
        [50.0, 50.0],
        [y_upper, y_upper],
        color="#D9EAF3",
        alpha=0.8,
        linewidth=0.0,
        zorder=0,
        label="Target: K <= 0.35, capture >= 50%",
    )
    ax.axvline(0.35, color="0.45", linestyle=":", linewidth=0.9, zorder=1)
    ax.axhline(50.0, color="0.45", linestyle=":", linewidth=0.9, zorder=1)
    for index, (result, cost_ratio, capture) in enumerate(points):
        ax.scatter(
            [cost_ratio],
            [capture],
            s=36,
            marker=MARKERS[index % len(MARKERS)],
            color=COLORS[index % len(COLORS)],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=result.label,
        )
    ax.set_xlim(0.0, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_title("Search-lift capture vs cost")
    ax.set_xlabel("Cost ratio K vs vote@8")
    ax.set_ylabel("Search-lift capture")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.grid(True, color="0.5", alpha=0.3, linewidth=0.65)
    ax.legend(loc="best", fontsize=6.6, frameon=False, handlelength=1.4)
    fig.tight_layout()
    return fig


def plot_rollout_ess(arms, plt):
    if not arms:
        raise FigureUnavailable("no config + summary rollout JSONs are present")
    width = 0.36
    positions = list(range(len(arms)))
    raw = [arm.raw_tokens_per_rollout for arm in arms]
    adjusted = [arm.adjusted_tokens_per_effective_sample for arm in arms]
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    ax.bar(
        [position - width / 2.0 for position in positions],
        raw,
        width=width,
        color=COLORS[4],
        edgecolor="none",
        label="Raw tokens / rollout",
    )
    ax.bar(
        [position + width / 2.0 for position in positions],
        adjusted,
        width=width,
        color=COLORS[0],
        edgecolor="none",
        label="Adjusted tokens / effective sample",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([arm.label for arm in arms])
    if len(arms) > 3:
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_horizontalalignment("right")
    ax.set_title("Rollout cost with ESS adjustment")
    ax.set_ylabel("Generated tokens")
    ax.grid(True, axis="y", color="0.5", alpha=0.3, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=6.8, frameon=False)
    fig.tight_layout()
    return fig


def render_figures(documents, out_dir):
    performance, rollout, performance_errors, rollout_errors = collect_inputs(
        documents
    )
    matplotlib, plt, PercentFormatter = load_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    style = {
        "font.size": 8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    written = []
    builders = (
        (
            "cost_vs_accuracy",
            lambda: plot_cost_vs_accuracy(performance, plt, PercentFormatter),
            performance_errors,
        ),
        (
            "cost_per_correct_bars",
            lambda: plot_cost_per_correct_bars(performance, plt),
            performance_errors,
        ),
        (
            "capture_vs_cost",
            lambda: plot_capture_vs_cost(performance, plt, PercentFormatter),
            performance_errors,
        ),
        (
            "rollout_ess",
            lambda: plot_rollout_ess(rollout, plt),
            rollout_errors,
        ),
    )
    with matplotlib.rc_context(style):
        for stem, builder, parse_errors in builders:
            if parse_errors:
                print("SKIP {}: {}".format(stem, " | ".join(parse_errors)))
                continue
            try:
                figure = builder()
            except FigureUnavailable as exc:
                print("SKIP {}: {}".format(stem, exc))
                continue
            try:
                written.extend(save_figure(figure, out_dir, stem))
            finally:
                plt.close(figure)
    for path in written:
        print(path)
    return written


def performance_row(mode, seed, items, correct, total_cost_units):
    return {
        "mode": mode,
        "seed": seed,
        "items": items,
        "correct_count": correct,
        "accuracy": correct / items,
        "total_generated_tokens": int(total_cost_units),
        "total_cost_units": float(total_cost_units),
    }


def performance_document(mode, rows):
    return {
        "config": {
            "mode": mode,
            "data": "synthetic-demo",
            "seeds": [row["seed"] for row in rows],
            "small_cost": 1.0,
            "large_cost": 10.0,
        },
        "summaries": rows,
    }


def rollout_document(mode, raw_tokens, diversity, valid=12, n=32):
    return {
        "config": {"mode": mode, "n": n, "data": "synthetic-demo"},
        "summary": {
            "items": valid,
            "valid": valid,
            "total_generated_tokens": raw_tokens * n * valid,
            "generated_tokens_per_rollout": raw_tokens,
            "effective_diversity": diversity,
        },
    }


def write_demo_files(directory):
    fixtures = (
        (
            "autotree_cascade.json",
            "AutoTree cascade",
            performance_document(
                "cascade",
                [
                    performance_row("cascade", 0, 100, 70, 1600),
                    performance_row("cascade", 1, 100, 73, 1660),
                    performance_row("cascade", 2, 100, 71, 1580),
                ],
            ),
        ),
        (
            "autotree_large_tree.json",
            "AutoTree large tree",
            performance_document(
                "large_tree",
                [
                    performance_row("large_tree", 0, 100, 76, 2380),
                    performance_row("large_tree", 1, 100, 79, 2460),
                    performance_row("large_tree", 2, 100, 78, 2410),
                ],
            ),
        ),
        (
            "vote8.json",
            "vote@8",
            performance_document(
                "large_bo8",
                [
                    performance_row("large_bo8", 0, 100, 78, 8000),
                    performance_row("large_bo8", 1, 100, 80, 8160),
                    performance_row("large_bo8", 2, 100, 79, 7920),
                ],
            ),
        ),
        (
            "greedy.json",
            "greedy",
            performance_document(
                "large_greedy",
                [
                    performance_row("large_greedy", 0, 100, 60, 990),
                    performance_row("large_greedy", 1, 100, 61, 1010),
                    performance_row("large_greedy", 2, 100, 59, 980),
                ],
            ),
        ),
        (
            "rollout_tree.json",
            "Tree rollouts",
            rollout_document("tree", 42.0, 0.64),
        ),
        (
            "rollout_independent.json",
            "Independent",
            rollout_document("independent", 130.0, 0.94),
        ),
    )
    paths = []
    labels = {}
    for filename, label, data in fixtures:
        path = directory / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        paths.append(path)
        labels[path] = label
    return paths, labels


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render publication figures from measured summary JSON files."
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        metavar="PATH_OR_GLOB",
        help="one or more cascade, large-tree, or rollout summary paths/globs",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="PATH=DISPLAY_NAME",
        help="override a summary label; may be repeated",
    )
    parser.add_argument(
        "--out-dir",
        default="bench/figures/out",
        help="output directory (default: bench/figures/out)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the full pipeline with explicitly synthetic fixture JSONs",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.demo:
            if args.summaries or args.label:
                parser.error("--demo cannot be combined with --summaries or --label")
            print("DEMO: synthetic fixtures only")
            with tempfile.TemporaryDirectory(prefix="publication-figures-demo-") as temp:
                paths, labels = write_demo_files(Path(temp))
                documents = read_documents(paths, labels)
                render_figures(documents, Path(args.out_dir))
            return 0

        if not args.summaries:
            parser.error("--summaries is required unless --demo is used")
        paths = expand_summary_paths(args.summaries)
        labels = parse_labels(args.label, paths)
        documents = read_documents(paths, labels)
        render_figures(documents, Path(args.out_dir))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
