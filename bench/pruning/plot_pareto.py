#!/usr/bin/env python3
"""Render pruning economics figures from measurement summary JSON files."""

import argparse
import glob
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


SUMMARY_FIELDS = (
    "mode",
    "seed",
    "items",
    "correct_count",
    "accuracy",
    "total_generated_tokens",
    "mean_tokens_per_item",
    "tokens_per_correct",
    "mean_wall_s",
    "error_count",
)


@dataclass(frozen=True)
class ResultSeries:
    path: Path
    label: str
    mode: str
    items: int
    correct_count: int
    total_generated_tokens: int
    accuracy: float
    mean_tokens_per_item: float
    tokens_per_correct: float
    seed_accuracies: tuple


def require_field(mapping, field, source, location):
    if field not in mapping:
        raise ValueError(
            "{}: missing field '{}.{}'".format(source, location, field)
        )
    return mapping[field]


def require_mapping(value, source, field):
    if not isinstance(value, dict):
        raise ValueError("{}: field '{}' must be an object".format(source, field))
    return value


def require_list(value, source, field):
    if not isinstance(value, list):
        raise ValueError("{}: field '{}' must be a list".format(source, field))
    return value


def require_nonempty_string(value, source, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "{}: field '{}' must be a nonempty string".format(source, field)
        )
    return value


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


def check_close(actual, expected, source, field):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "{}: field '{}' is {}, expected {} from count totals".format(
                source, field, actual, expected
            )
        )


def read_summary(path, label):
    source = str(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError("{}: invalid JSON: {}".format(source, exc)) from exc

    document = require_mapping(document, source, "root")
    config = require_mapping(
        require_field(document, "config", source, "root"), source, "config"
    )
    mode = require_nonempty_string(
        require_field(config, "mode", source, "config"), source, "config.mode"
    )
    require_integer(
        require_field(config, "branches", source, "config"),
        source,
        "config.branches",
        minimum=1,
    )
    require_integer(
        require_field(config, "max_tokens", source, "config"),
        source,
        "config.max_tokens",
        minimum=1,
    )

    rows = require_list(
        require_field(document, "summaries", source, "root"), source, "summaries"
    )
    if not rows:
        raise ValueError("{}: field 'summaries' must not be empty".format(source))

    total_items = 0
    total_correct = 0
    total_tokens = 0
    seed_accuracies = []
    seen_seeds = set()

    for index, raw_row in enumerate(rows):
        location = "summaries[{}]".format(index)
        row = require_mapping(raw_row, source, location)
        for field in SUMMARY_FIELDS:
            require_field(row, field, source, location)

        row_mode = require_nonempty_string(
            row["mode"], source, location + ".mode"
        )
        if row_mode != mode:
            raise ValueError(
                "{}: field '{}.mode' is {!r}, but config.mode is {!r}".format(
                    source, location, row_mode, mode
                )
            )

        seed = require_integer(row["seed"], source, location + ".seed")
        if seed in seen_seeds:
            raise ValueError(
                "{}: field '{}.seed' duplicates seed {}".format(
                    source, location, seed
                )
            )
        seen_seeds.add(seed)

        items = require_integer(
            row["items"], source, location + ".items", minimum=1
        )
        correct = require_integer(
            row["correct_count"], source, location + ".correct_count"
        )
        if correct > items:
            raise ValueError(
                "{}: field '{}.correct_count' exceeds items".format(
                    source, location
                )
            )
        tokens = require_integer(
            row["total_generated_tokens"],
            source,
            location + ".total_generated_tokens",
        )
        accuracy = require_number(
            row["accuracy"], source, location + ".accuracy", 0.0, 1.0
        )
        mean_tokens = require_number(
            row["mean_tokens_per_item"],
            source,
            location + ".mean_tokens_per_item",
            0.0,
        )
        tokens_per_correct = require_number(
            row["tokens_per_correct"],
            source,
            location + ".tokens_per_correct",
            0.0,
        )
        require_number(
            row["mean_wall_s"], source, location + ".mean_wall_s", 0.0
        )
        error_count = require_integer(
            row["error_count"], source, location + ".error_count"
        )
        if error_count > items:
            raise ValueError(
                "{}: field '{}.error_count' exceeds items".format(
                    source, location
                )
            )

        check_close(accuracy, correct / items, source, location + ".accuracy")
        check_close(
            mean_tokens,
            tokens / items,
            source,
            location + ".mean_tokens_per_item",
        )
        check_close(
            tokens_per_correct,
            tokens / max(correct, 1),
            source,
            location + ".tokens_per_correct",
        )

        total_items += items
        total_correct += correct
        total_tokens += tokens
        seed_accuracies.append(accuracy)

    return ResultSeries(
        path=path,
        label=label,
        mode=mode,
        items=total_items,
        correct_count=total_correct,
        total_generated_tokens=total_tokens,
        accuracy=total_correct / total_items,
        mean_tokens_per_item=total_tokens / total_items,
        tokens_per_correct=total_tokens / max(total_correct, 1),
        seed_accuracies=tuple(seed_accuracies),
    )


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
    return {
        path: labels.get(normalized_path(path), path.stem) for path in paths
    }


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render figures. Install matplotlib and "
            "rerun this command. Original import error: {}".format(exc)
        ) from exc
    return matplotlib, plt, PercentFormatter


def annotation_offset(index):
    offsets = ((6, 6), (6, -12), (-6, 6), (-6, -12))
    x_offset, y_offset = offsets[index % len(offsets)]
    horizontal_alignment = "left" if x_offset > 0 else "right"
    return (x_offset, y_offset), horizontal_alignment


def save_figure(fig, out_dir, stem):
    written = []
    for extension in ("pdf", "svg", "png"):
        path = out_dir / "{}.{}".format(stem, extension)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path.resolve())
    return written


def plot_pareto(series, plt, PercentFormatter):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.get_cmap("tab10").colors

    bon_series = sorted(
        (result for result in series if result.mode == "bon"),
        key=lambda result: result.mean_tokens_per_item,
    )
    if bon_series:
        ax.plot(
            [result.mean_tokens_per_item for result in bon_series],
            [100.0 * result.accuracy for result in bon_series],
            linestyle="--",
            linewidth=1.1,
            color="0.55",
            alpha=0.8,
            zorder=1,
        )

    accuracy_bounds = []
    for index, result in enumerate(series):
        color = colors[index % len(colors)]
        pooled_accuracy = 100.0 * result.accuracy
        seed_min = 100.0 * min(result.seed_accuracies)
        seed_max = 100.0 * max(result.seed_accuracies)
        accuracy_bounds.extend((seed_min, seed_max))
        ax.errorbar(
            result.mean_tokens_per_item,
            pooled_accuracy,
            yerr=[
                [pooled_accuracy - seed_min],
                [seed_max - pooled_accuracy],
            ],
            fmt="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.7,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            zorder=3,
            label=result.label,
        )
        # Legend-keyed series instead of floating point annotations: with
        # clustered points an offset annotation lands beside a NEIGHBORING
        # point and silently mislabels the figure (observed on real data).

    lower = max(0.0, min(accuracy_bounds) - 5.0)
    upper = min(100.0, max(accuracy_bounds) + 5.0)
    if math.isclose(lower, upper):
        lower = max(0.0, lower - 1.0)
        upper = min(100.0, upper + 1.0)
    ax.set_ylim(lower, upper)
    ax.set_xlim(left=0.0)
    ax.margins(x=0.1)
    ax.set_title("Accuracy vs generated tokens per problem")
    ax.set_xlabel("Mean generated tokens per problem (tokens/problem)")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.grid(True, color="0.5", alpha=0.3, linewidth=0.7)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    return fig


def lighten(color, amount):
    return tuple(channel + (1.0 - channel) * amount for channel in color)


def format_bar_value(value):
    return "{:,.1f}".format(value)


def plot_tokens_per_correct(series, plt):
    ordered = sorted(series, key=lambda result: result.tokens_per_correct)
    fig, ax = plt.subplots(figsize=(6, 4))
    base_color = plt.get_cmap("tab10").colors[0]
    light_color = lighten(base_color, 0.48)
    best_value = ordered[0].tokens_per_correct
    colors = [
        base_color
        if math.isclose(result.tokens_per_correct, best_value)
        else light_color
        for result in ordered
    ]
    positions = list(range(len(ordered)))
    values = [result.tokens_per_correct for result in ordered]
    bars = ax.barh(positions, values, color=colors, edgecolor="none", height=0.68)
    ax.set_yticks(positions, [result.label for result in ordered])
    ax.invert_yaxis()

    maximum = max(values)
    text_offset = max(maximum * 0.015, 0.05)
    for bar, value in zip(bars, values):
        ax.text(
            value + text_offset,
            bar.get_y() + bar.get_height() / 2.0,
            format_bar_value(value),
            va="center",
            ha="left",
            fontsize=8.5,
        )
    ax.set_xlim(0.0, maximum * 1.18 if maximum > 0 else 1.0)
    ax.set_title("Generated tokens per correct answer")
    ax.set_xlabel("Generated tokens per correct answer (tokens/correct)")
    ax.grid(True, axis="x", color="0.5", alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def render_figures(series, out_dir):
    matplotlib, plt, PercentFormatter = load_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    style = {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    written = []
    with matplotlib.rc_context(style):
        pareto_figure = plot_pareto(series, plt, PercentFormatter)
        try:
            written.extend(save_figure(pareto_figure, out_dir, "pareto"))
        finally:
            plt.close(pareto_figure)

        efficiency_figure = plot_tokens_per_correct(series, plt)
        try:
            written.extend(
                save_figure(efficiency_figure, out_dir, "tokens_per_correct")
            )
        finally:
            plt.close(efficiency_figure)

    for path in written:
        print(path)
    return written


def demo_summary(mode, seed, items, correct, tokens, wall_s, errors, pruned=None):
    row = {
        "mode": mode,
        "seed": seed,
        "items": items,
        "correct_count": correct,
        "accuracy": correct / items,
        "total_generated_tokens": tokens,
        "mean_tokens_per_item": tokens / items,
        "tokens_per_correct": tokens / max(correct, 1),
        "mean_wall_s": wall_s,
        "error_count": errors,
    }
    if pruned is not None:
        row["total_pruned"] = pruned
    return row


def write_demo_files(directory):
    fixtures = (
        (
            "tree_summary.json",
            {
                "config": {"mode": "tree", "branches": 8, "max_tokens": 512},
                "summaries": [
                    demo_summary("tree", 0, 100, 78, 24800, 1.18, 0, 511),
                    demo_summary("tree", 1, 100, 80, 25200, 1.21, 0, 506),
                    demo_summary("tree", 2, 100, 79, 24600, 1.16, 0, 519),
                ],
            },
        ),
        (
            "bon_summary.json",
            {
                "config": {"mode": "bon", "branches": 8, "max_tokens": 512},
                "summaries": [
                    demo_summary("bon", 0, 100, 79, 41800, 2.87, 0),
                    demo_summary("bon", 1, 100, 81, 42500, 2.93, 0),
                    demo_summary("bon", 2, 100, 80, 41700, 2.85, 0),
                ],
            },
        ),
    )
    paths = []
    for filename, document in fixtures:
        path = directory / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        paths.append(path)
    return paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot pooled pruning economics summaries for publication."
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        metavar="PATH_OR_GLOB",
        help="one or more summary JSON paths or glob patterns",
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
        default="bench/pruning/figures",
        help="output directory (default: bench/pruning/figures)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="render two synthetic three-seed summaries through the full pipeline",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.demo:
            if args.summaries or args.label:
                parser.error("--demo cannot be combined with --summaries or --label")
            load_matplotlib()
            with tempfile.TemporaryDirectory(prefix="pruning-pareto-demo-") as temp:
                paths = write_demo_files(Path(temp))
                labels = {
                    paths[0]: "AutoTree pruning",
                    paths[1]: "Sequential best-of-n",
                }
                series = [read_summary(path, labels[path]) for path in paths]
                render_figures(series, Path(args.out_dir))
            return 0

        if not args.summaries:
            parser.error("--summaries is required unless --demo is used")
        paths = expand_summary_paths(args.summaries)
        labels = parse_labels(args.label, paths)
        series = [read_summary(path, labels[path]) for path in paths]
        render_figures(series, Path(args.out_dir))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
