#!/usr/bin/env python3
"""Compute matched-accuracy economics from a k sweep without extrapolation."""

import argparse
import glob
import json
import math
from pathlib import Path
import random
import statistics


MODE_TO_ARM = {
    "large_bo8": "baseline",
    "large_tree": "tree",
}
COMPARABLE_CONFIG_FIELDS = (
    "data",
    "large_model",
    "large_url",
    "seeds",
    "answer_mode",
    "max_tokens",
    "large_cost",
    "small_cost",
)
ABS_TOL = 1e-12


def require_mapping(value, source, field):
    if not isinstance(value, dict):
        raise ValueError("{}: '{}' must be an object".format(source, field))
    return value


def require_list(value, source, field):
    if not isinstance(value, list):
        raise ValueError("{}: '{}' must be a list".format(source, field))
    return value


def require_integer(value, source, field, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{}: '{}' must be an integer".format(source, field))
    if minimum is not None and value < minimum:
        raise ValueError(
            "{}: '{}' must be at least {}".format(source, field, minimum)
        )
    return value


def require_number(value, source, field, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{}: '{}' must be numeric".format(source, field))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{}: '{}' must be finite".format(source, field))
    if minimum is not None and value < minimum:
        raise ValueError(
            "{}: '{}' must be at least {}".format(source, field, minimum)
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            "{}: '{}' must be at most {}".format(source, field, maximum)
        )
    return value


def check_close(actual, expected, source, field):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=ABS_TOL):
        raise ValueError(
            "{}: '{}' is {}, expected {}".format(source, field, actual, expected)
        )


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError("{}: invalid JSON: {}".format(path, exc)) from exc


def load_outcomes(path, mode, seeds, seed_rows):
    source = str(path)
    outcomes_by_seed = {seed: {} for seed in seeds}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{}:{}: invalid JSON: {}".format(source, line_number, exc)
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    "{}:{}: record must be an object".format(source, line_number)
                )
            if record.get("mode") != mode:
                raise ValueError(
                    "{}:{}: unexpected mode {!r}".format(
                        source, line_number, record.get("mode")
                    )
                )
            seed = record.get("seed")
            if seed not in outcomes_by_seed:
                raise ValueError(
                    "{}:{}: unexpected seed {!r}".format(source, line_number, seed)
                )
            problem_id = record.get("id")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError(
                    "{}:{}: 'id' must be a nonempty string".format(
                        source, line_number
                    )
                )
            correct = record.get("correct")
            if not isinstance(correct, bool):
                raise ValueError(
                    "{}:{}: 'correct' must be boolean".format(source, line_number)
                )
            if problem_id in outcomes_by_seed[seed]:
                raise ValueError(
                    "{}:{}: duplicate seed/id pair ({}, {!r})".format(
                        source, line_number, seed, problem_id
                    )
                )
            outcomes_by_seed[seed][problem_id] = correct

    problem_ids = None
    for seed in seeds:
        observed = outcomes_by_seed[seed]
        row = seed_rows[seed]
        if len(observed) != row["items"]:
            raise ValueError(
                "{}: seed {} has {} item records, summary reports {}".format(
                    source, seed, len(observed), row["items"]
                )
            )
        correct_count = sum(value is True for value in observed.values())
        if correct_count != row["correct_count"]:
            raise ValueError(
                "{}: seed {} has {} correct records, summary reports {}".format(
                    source, seed, correct_count, row["correct_count"]
                )
            )
        seed_problem_ids = set(observed)
        if problem_ids is None:
            problem_ids = seed_problem_ids
        elif seed_problem_ids != problem_ids:
            raise ValueError(
                "{}: problem ids differ across seeds".format(source)
            )

    nested = {
        problem_id: {
            seed: outcomes_by_seed[seed][problem_id] for seed in seeds
        }
        for problem_id in sorted(problem_ids or ())
    }
    return nested


def read_summary(path):
    source = str(path)
    document = require_mapping(load_json(path), source, "root")
    config = require_mapping(document.get("config"), source, "config")
    mode = config.get("mode")
    if mode not in MODE_TO_ARM:
        raise ValueError(
            "{}: config.mode must be 'large_bo8' or 'large_tree'".format(source)
        )
    k = require_integer(config.get("branches"), source, "config.branches", 1)
    if k > 64:
        raise ValueError("{}: config.branches must be at most 64".format(source))
    for field in COMPARABLE_CONFIG_FIELDS:
        if field not in config:
            raise ValueError("{}: config is missing '{}'".format(source, field))
    seeds = require_list(config.get("seeds"), source, "config.seeds")
    if not seeds:
        raise ValueError("{}: config.seeds must not be empty".format(source))
    seeds = tuple(
        require_integer(seed, source, "config.seeds[]") for seed in seeds
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError("{}: config.seeds contains duplicates".format(source))

    rows = require_list(document.get("summaries"), source, "summaries")
    if not rows:
        raise ValueError("{}: summaries must not be empty".format(source))
    seed_rows = {}
    total_items = 0
    total_correct = 0
    total_cost = 0.0
    for index, raw_row in enumerate(rows):
        location = "summaries[{}]".format(index)
        row = require_mapping(raw_row, source, location)
        if row.get("mode") != mode:
            raise ValueError(
                "{}: {}.mode does not match config.mode".format(source, location)
            )
        seed = require_integer(row.get("seed"), source, location + ".seed")
        if seed in seed_rows:
            raise ValueError("{}: duplicate summary seed {}".format(source, seed))
        items = require_integer(row.get("items"), source, location + ".items", 1)
        correct = require_integer(
            row.get("correct_count"), source, location + ".correct_count", 0
        )
        if correct > items:
            raise ValueError(
                "{}: {}.correct_count exceeds items".format(source, location)
            )
        accuracy = require_number(
            row.get("accuracy"), source, location + ".accuracy", 0.0, 1.0
        )
        cost = require_number(
            row.get("total_cost_units"),
            source,
            location + ".total_cost_units",
            0.0,
        )
        cost_per_correct = require_number(
            row.get("cost_per_correct"),
            source,
            location + ".cost_per_correct",
            0.0,
        )
        check_close(accuracy, correct / items, source, location + ".accuracy")
        check_close(
            cost_per_correct,
            cost / max(correct, 1),
            source,
            location + ".cost_per_correct",
        )
        seed_rows[seed] = {
            "items": items,
            "correct_count": correct,
        }
        total_items += items
        total_correct += correct
        total_cost += cost

    if set(seed_rows) != set(seeds):
        raise ValueError(
            "{}: summary seeds do not match config.seeds".format(source)
        )
    out_jsonl = config.get("out_jsonl")
    if not isinstance(out_jsonl, str) or not out_jsonl:
        raise ValueError("{}: config.out_jsonl must be a path".format(source))
    out_jsonl_path = Path(out_jsonl)
    if not out_jsonl_path.is_absolute():
        out_jsonl_path = path.parent / out_jsonl_path
    if not out_jsonl_path.is_file():
        raise ValueError(
            "{}: per-item JSONL does not exist: {}".format(source, out_jsonl_path)
        )
    outcomes = load_outcomes(out_jsonl_path, mode, seeds, seed_rows)
    return {
        "path": path.resolve(),
        "arm": MODE_TO_ARM[mode],
        "mode": mode,
        "k": k,
        "accuracy": total_correct / total_items,
        "cost_per_problem": total_cost / total_items,
        "cost_per_correct": total_cost / max(total_correct, 1),
        "config": config,
        "seeds": tuple(sorted(seeds)),
        "outcomes": outcomes,
        "problem_ids": tuple(outcomes),
    }


def expand_summary_paths(patterns):
    expanded = []
    seen = set()
    for pattern in patterns:
        literal = Path(pattern)
        if literal.is_file():
            matches = [literal]
        else:
            matches = [Path(match) for match in glob.glob(pattern, recursive=True)]
        files = sorted((path for path in matches if path.is_file()), key=str)
        if not files:
            raise ValueError(
                "--summaries pattern matched no files: {}".format(pattern)
            )
        for path in files:
            resolved = path.resolve()
            key = str(resolved).lower()
            if key not in seen:
                seen.add(key)
                expanded.append(resolved)
    return expanded


def validate_sweep(points):
    by_key = {}
    for point in points:
        key = (point["arm"], point["k"])
        if key in by_key:
            raise ValueError(
                "duplicate {} k={} summaries: {} and {}".format(
                    key[0], key[1], by_key[key]["path"], point["path"]
                )
            )
        by_key[key] = point
    arms = {point["arm"] for point in points}
    if arms != {"baseline", "tree"}:
        raise ValueError("summaries must contain both baseline and tree arms")
    baseline_ks = {point["k"] for point in points if point["arm"] == "baseline"}
    tree_ks = {point["k"] for point in points if point["arm"] == "tree"}
    if baseline_ks != tree_ks:
        raise ValueError(
            "baseline and tree k ladders differ: {} vs {}".format(
                sorted(baseline_ks), sorted(tree_ks)
            )
        )

    reference = points[0]
    for point in points[1:]:
        for field in COMPARABLE_CONFIG_FIELDS:
            if point["config"].get(field) != reference["config"].get(field):
                raise ValueError(
                    "summary configs differ for '{}': {} vs {}".format(
                        field, reference["path"], point["path"]
                    )
                )
        if point["seeds"] != reference["seeds"]:
            raise ValueError("summary seed sets differ")
        if point["problem_ids"] != reference["problem_ids"]:
            raise ValueError("per-item problem ids differ across sweep points")


def pareto_frontier(points):
    best_at_cost = {}
    for point in points:
        cost = point["cost_per_problem"]
        current = best_at_cost.get(cost)
        if current is None or point["accuracy"] > current["accuracy"] + ABS_TOL:
            best_at_cost[cost] = point
        elif math.isclose(
            point["accuracy"], current["accuracy"], rel_tol=0.0, abs_tol=ABS_TOL
        ) and point["k"] < current["k"]:
            best_at_cost[cost] = point

    frontier = []
    best_accuracy = -1.0
    for cost in sorted(best_at_cost):
        point = best_at_cost[cost]
        if point["accuracy"] > best_accuracy + ABS_TOL:
            frontier.append(point)
            best_accuracy = point["accuracy"]
    return frontier


def interpolate_cost_at_accuracy(points, target):
    frontier = pareto_frontier(points)
    if not frontier:
        return None
    for point in frontier:
        if math.isclose(point["accuracy"], target, rel_tol=0.0, abs_tol=ABS_TOL):
            return {"value": point["cost_per_problem"], "low": point, "high": point}
    if target < frontier[0]["accuracy"] or target > frontier[-1]["accuracy"]:
        return None
    for low, high in zip(frontier, frontier[1:]):
        if low["accuracy"] < target < high["accuracy"]:
            fraction = (target - low["accuracy"]) / (
                high["accuracy"] - low["accuracy"]
            )
            value = low["cost_per_problem"] + fraction * (
                high["cost_per_problem"] - low["cost_per_problem"]
            )
            return {"value": value, "low": low, "high": high}
    return None


def interpolate_accuracy_at_cost(points, target):
    frontier = pareto_frontier(points)
    if not frontier:
        return None
    for point in frontier:
        if math.isclose(
            point["cost_per_problem"], target, rel_tol=0.0, abs_tol=ABS_TOL
        ):
            return {"value": point["accuracy"], "low": point, "high": point}
    if (
        target < frontier[0]["cost_per_problem"]
        or target > frontier[-1]["cost_per_problem"]
    ):
        return None
    for low, high in zip(frontier, frontier[1:]):
        if low["cost_per_problem"] < target < high["cost_per_problem"]:
            fraction = (target - low["cost_per_problem"]) / (
                high["cost_per_problem"] - low["cost_per_problem"]
            )
            value = low["accuracy"] + fraction * (
                high["accuracy"] - low["accuracy"]
            )
            return {"value": value, "low": low, "high": high}
    return None


def ratio(numerator, denominator):
    if denominator > 0.0:
        return numerator / denominator
    if numerator > 0.0:
        return math.inf
    return None


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low_index = int(math.floor(position))
    high_index = int(math.ceil(position))
    if low_index == high_index:
        return ordered[low_index]
    low = ordered[low_index]
    high = ordered[high_index]
    if math.isinf(low) or math.isinf(high):
        return high
    fraction = position - low_index
    return low + fraction * (high - low)


def bootstrap_ratios(points, baseline_top, sample_count, random_seed):
    if sample_count <= 0:
        return []
    rng = random.Random(random_seed)
    problem_ids = points[0]["problem_ids"]
    seeds = points[0]["seeds"]
    baseline_index = points.index(baseline_top)
    vectors = {
        (problem_id, seed): tuple(
            int(point["outcomes"][problem_id][seed]) for point in points
        )
        for problem_id in problem_ids
        for seed in seeds
    }
    denominator = len(problem_ids) * len(seeds)
    ratios = []
    for _unused in range(sample_count):
        totals = [0] * len(points)
        for _problem_draw in problem_ids:
            problem_id = rng.choice(problem_ids)
            for _seed_draw in seeds:
                seed = rng.choice(seeds)
                vector = vectors[(problem_id, seed)]
                for index, value in enumerate(vector):
                    totals[index] += value
        replicate = []
        for point, correct in zip(points, totals):
            copy = dict(point)
            copy["accuracy"] = correct / denominator
            replicate.append(copy)
        target = replicate[baseline_index]["accuracy"]
        tree_points = [point for point in replicate if point["arm"] == "tree"]
        interpolation = interpolate_cost_at_accuracy(tree_points, target)
        if interpolation is None:
            continue
        value = ratio(baseline_top["cost_per_problem"], interpolation["value"])
        if value is not None and not math.isnan(value):
            ratios.append(value)
    return ratios


def format_ratio(value):
    if value is None:
        return "unavailable"
    if math.isinf(value):
        return "infinite"
    return "{:.6f}x".format(value)


def bracket_text(interpolation):
    low = interpolation["low"]
    high = interpolation["high"]
    if low["k"] == high["k"]:
        return "exact measured tree k={}".format(low["k"])
    return "linear interpolation between tree k={} and k={}".format(
        low["k"], high["k"]
    )


def print_table(points):
    print("ARM       K    ACCURACY    COST/PROBLEM    COST/CORRECT")
    for point in sorted(points, key=lambda item: (item["arm"] != "baseline", item["k"])):
        print(
            "{arm:<9} {k:>2}   {accuracy:>9.4%}   {cpp:>14.6f}   {cpc:>14.6f}".format(
                arm=point["arm"],
                k=point["k"],
                accuracy=point["accuracy"],
                cpp=point["cost_per_problem"],
                cpc=point["cost_per_correct"],
            )
        )


def analyze(points, bootstrap_samples, bootstrap_seed):
    baseline_points = [point for point in points if point["arm"] == "baseline"]
    tree_points = [point for point in points if point["arm"] == "tree"]
    baseline_top = max(baseline_points, key=lambda point: point["k"])
    tree_frontier = pareto_frontier(tree_points)

    print_table(points)
    print()
    print("tree_pareto_frontier_ks: {}".format(
        ",".join(str(point["k"]) for point in tree_frontier)
    ))
    print("baseline_top_k: {}".format(baseline_top["k"]))
    print("baseline_top_accuracy: {:.6%}".format(baseline_top["accuracy"]))
    print(
        "baseline_top_cost_per_problem: {:.6f}".format(
            baseline_top["cost_per_problem"]
        )
    )

    iso = interpolate_cost_at_accuracy(tree_points, baseline_top["accuracy"])
    reverse = interpolate_accuracy_at_cost(
        tree_points, baseline_top["cost_per_problem"]
    )
    if reverse is None:
        min_cost = tree_frontier[0]["cost_per_problem"]
        max_cost = tree_frontier[-1]["cost_per_problem"]
        gap = (
            min_cost - baseline_top["cost_per_problem"]
            if baseline_top["cost_per_problem"] < min_cost
            else baseline_top["cost_per_problem"] - max_cost
        )
        print(
            "tree_accuracy_at_baseline_cost: unavailable "
            "(NO MATCHED-COST POINT EXISTS; cost-range gap {:.6f})".format(gap)
        )
    else:
        print(
            "tree_accuracy_at_baseline_cost: {:.6%} ({})".format(
                reverse["value"], bracket_text(reverse)
            )
        )

    if iso is None:
        min_accuracy = tree_frontier[0]["accuracy"]
        max_accuracy = tree_frontier[-1]["accuracy"]
        if baseline_top["accuracy"] > max_accuracy:
            gap = baseline_top["accuracy"] - max_accuracy
            detail = "tree maximum is {:.6%}".format(max_accuracy)
        else:
            gap = min_accuracy - baseline_top["accuracy"]
            detail = "tree minimum frontier accuracy is {:.6%}".format(min_accuracy)
        print("NO ISO-ACCURACY POINT EXISTS")
        print("iso_accuracy_cost_ratio: unavailable")
        print("iso_accuracy_gap_percentage_points: {:.6f}".format(100.0 * gap))
        print("bootstrap_ratio_ci: unavailable because no measured iso point exists")
        print(
            "VERDICT: NO ISO-ACCURACY POINT EXISTS; baseline top-rung accuracy "
            "is outside the measured tree Pareto accuracy range, {}. No "
            "extrapolation was used.".format(detail)
        )
        return

    headline_ratio = ratio(baseline_top["cost_per_problem"], iso["value"])
    print(
        "tree_cost_at_baseline_accuracy: {:.6f} ({})".format(
            iso["value"], bracket_text(iso)
        )
    )
    print("iso_accuracy_cost_ratio: {}".format(format_ratio(headline_ratio)))

    ratios = bootstrap_ratios(points, baseline_top, bootstrap_samples, bootstrap_seed)
    print(
        "bootstrap_iso_point_replicates: {}/{}".format(
            len(ratios), bootstrap_samples
        )
    )
    if ratios:
        lower = percentile(ratios, 0.025)
        upper = percentile(ratios, 0.975)
        print(
            "iso_accuracy_cost_ratio_bootstrap_median: {}".format(
                format_ratio(statistics.median(ratios))
            )
        )
        print(
            "iso_accuracy_cost_ratio_95pct_ci_conditional_on_iso: "
            "[{}, {}]".format(format_ratio(lower), format_ratio(upper))
        )
    else:
        print("bootstrap_ratio_ci: unavailable; no replicate had an iso point")
    print(
        "VERDICT: At baseline k={}, baseline/tree cost at the same measured-range "
        "accuracy is {} using {}. No extrapolation was used.".format(
            baseline_top["k"], format_ratio(headline_ratio), bracket_text(iso)
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the baseline/tree cost ratio at matched accuracy from sweep "
            "summary JSON files."
        )
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        metavar="PATH_OR_GLOB",
        help="sweep summary JSON paths or glob patterns",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="hierarchical problem/seed bootstrap replicates (default: 2000)",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=0, help="bootstrap RNG seed"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    try:
        paths = expand_summary_paths(args.summaries)
        points = [read_summary(path) for path in paths]
        validate_sweep(points)
        analyze(points, args.bootstrap_samples, args.bootstrap_seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
