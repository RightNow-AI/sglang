#!/usr/bin/env python3
"""Analyze k-curve summaries with a paired problem-cluster bootstrap."""

import argparse
import json
import math
import os
import random
import statistics
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUMMARY_DIR = os.path.join(SCRIPT_DIR, "results")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def finite_nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(label))
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("{} must be finite and nonnegative".format(label))
    return number


def nonnegative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a nonnegative integer".format(label))
    return value


def close(left, right, tolerance=1e-9):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def prepare_summary(path):
    document = load_json(path)
    if document.get("schema_version") != 1 or document.get("kind") != "kcurve_k_summary":
        raise ValueError("{} is not a schema_version 1 kcurve summary".format(path))
    k = document.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("{} has invalid k".format(path))
    config = document.get("config")
    metrics = document.get("metrics")
    problems = document.get("problems")
    if not isinstance(config, dict):
        raise ValueError("{} has no config object".format(path))
    if config.get("k") != k or config.get("mode") != "large_bo8":
        raise ValueError("{} has inconsistent mode or k config".format(path))
    if not isinstance(metrics, dict):
        raise ValueError("{} has no metrics object".format(path))
    if not isinstance(problems, list) or not problems:
        raise ValueError("{} has no problem rows".format(path))
    expected_seeds = config.get("seeds")
    if not isinstance(expected_seeds, list) or not expected_seeds:
        raise ValueError("{} config.seeds must be a nonempty list".format(path))
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("{} config.seeds contains duplicates".format(path))
    clusters = {}
    item_indices = {}
    observation_rows = []
    for problem_number, problem in enumerate(problems, 1):
        if not isinstance(problem, dict):
            raise ValueError("{} problem {} is not an object".format(path, problem_number))
        item_id = problem.get("id")
        item_index = problem.get("item_index")
        seed_rows = problem.get("seeds")
        if not isinstance(item_id, str) or item_id in clusters:
            raise ValueError("{} problem {} has invalid or duplicate id".format(path, problem_number))
        item_indices[item_id] = nonnegative_integer(
            item_index, "{} problem {} item_index".format(path, problem_number)
        )
        if not isinstance(seed_rows, list):
            raise ValueError("{} problem {} has no seeds list".format(path, problem_number))
        seen_seeds = []
        rows = []
        for row_number, row in enumerate(seed_rows, 1):
            if not isinstance(row, dict):
                raise ValueError(
                    "{} problem {} seed row {} is not an object".format(
                        path, problem_number, row_number
                    )
                )
            seed = row.get("seed")
            if seed in seen_seeds:
                raise ValueError("{} problem {!r} repeats seed {}".format(path, item_id, seed))
            seen_seeds.append(seed)
            if not isinstance(row.get("correct"), bool):
                raise ValueError("{} problem {!r} has invalid correct".format(path, item_id))
            wall_s = finite_nonnegative(
                row.get("wall_s"), "{} problem {!r} wall_s".format(path, item_id)
            )
            generated_tokens = nonnegative_integer(
                row.get("generated_tokens"),
                "{} problem {!r} generated_tokens".format(path, item_id),
            )
            error = row.get("error")
            if error is not None and not isinstance(error, str):
                raise ValueError("{} problem {!r} has invalid error".format(path, item_id))
            clean = {
                "seed": seed,
                "correct": row["correct"],
                "wall_s": wall_s,
                "generated_tokens": generated_tokens,
                "error": error,
            }
            rows.append(clean)
            observation_rows.append(clean)
        if seen_seeds != expected_seeds:
            raise ValueError(
                "{} problem {!r} seeds do not match config order".format(path, item_id)
            )
        clusters[item_id] = {
            "correct": sum(row["correct"] for row in rows),
            "observations": len(rows),
            "wall_s": sum(row["wall_s"] for row in rows),
            "tokens": sum(row["generated_tokens"] for row in rows),
            "errors": sum(row["error"] is not None for row in rows),
        }
    observations = len(observation_rows)
    correct_count = sum(row["correct"] for row in observation_rows)
    total_wall = sum(row["wall_s"] for row in observation_rows)
    total_tokens = sum(row["generated_tokens"] for row in observation_rows)
    error_n = sum(row["error"] is not None for row in observation_rows)
    seeds_nominal = nonnegative_integer(
        metrics.get("seeds_nominal"), "{} metrics.seeds_nominal".format(path)
    )
    seeds_effective = nonnegative_integer(
        metrics.get("seeds_effective"), "{} metrics.seeds_effective".format(path)
    )
    if seeds_nominal != len(expected_seeds) or seeds_effective > seeds_nominal:
        raise ValueError("{} has inconsistent seed evidence".format(path))
    accuracy = correct_count / observations
    mean_wall = total_wall / observations
    gpu_cost = mean_wall / accuracy if accuracy > 0.0 else math.inf
    expected_metrics = {
        "problems": len(problems),
        "seeds": len(expected_seeds),
        "observations": observations,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "mean_wall_s_per_item": mean_wall,
        "total_generated_tokens": total_tokens,
        "tokens_per_item": total_tokens / observations,
        "error_n": error_n,
    }
    for key, expected in expected_metrics.items():
        actual = metrics.get(key)
        if isinstance(expected, float):
            if actual is None or not close(actual, expected):
                raise ValueError("{} metrics.{} does not match problem rows".format(path, key))
        elif actual != expected:
            raise ValueError("{} metrics.{} does not match problem rows".format(path, key))
    stored_gpu = metrics.get("gpu_seconds_per_correct")
    if math.isinf(gpu_cost):
        if stored_gpu is not None:
            raise ValueError("{} zero-accuracy GPU cost must be null".format(path))
    elif stored_gpu is None or not close(stored_gpu, gpu_cost):
        raise ValueError("{} GPU cost does not match mean_wall/accuracy".format(path))
    problem_ids = sorted(clusters, key=lambda item_id: (item_indices[item_id], item_id))
    return {
        "path": os.path.abspath(path),
        "k": k,
        "config": config,
        "problem_ids": problem_ids,
        "clusters": [clusters[item_id] for item_id in problem_ids],
        "metrics": {
            "problems": len(problems),
            "seeds": len(expected_seeds),
            "observations": observations,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "mean_wall_s_per_item": mean_wall,
            "gpu_seconds_per_correct": gpu_cost,
            "total_generated_tokens": total_tokens,
            "tokens_per_item": total_tokens / observations,
            "error_n": error_n,
            "seeds_nominal": seeds_nominal,
            "seeds_effective": seeds_effective,
        },
    }


def require_common_config(rows):
    keys = (
        "data",
        "data_sha256",
        "offset",
        "limit",
        "seeds",
        "large_model",
        "large_url",
        "max_tokens",
        "temperature",
        "timeout",
        "concurrency",
        "answer_mode",
        "cascade_script",
        "cascade_sha256",
    )
    baseline = rows[0]
    for row in rows[1:]:
        mismatches = [key for key in keys if row["config"].get(key) != baseline["config"].get(key)]
        if mismatches:
            raise ValueError(
                "{} differs from {} for: {}".format(
                    row["path"], baseline["path"], ", ".join(mismatches)
                )
            )
        if row["problem_ids"] != baseline["problem_ids"]:
            raise ValueError("{} does not contain the same locked problem set".format(row["path"]))
    return baseline["config"]


def percentile(values, probability):
    if not values:
        raise ValueError("cannot take percentile of an empty list")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def bootstrap(rows, samples, seed):
    rng = random.Random(seed)
    problem_count = rows[0]["metrics"]["problems"]
    gpu_values = {row["k"]: [] for row in rows}
    accuracy_values = {row["k"]: [] for row in rows}
    kstar_counts = {row["k"]: 0 for row in rows}
    largest_falling_count = 0
    ordered_ks = [row["k"] for row in rows]
    for _replicate in range(samples):
        sampled_indices = [rng.randrange(problem_count) for _unused in range(problem_count)]
        replicate_costs = {}
        for row in rows:
            correct_count = 0
            observations = 0
            total_wall = 0.0
            for index in sampled_indices:
                cluster = row["clusters"][index]
                correct_count += cluster["correct"]
                observations += cluster["observations"]
                total_wall += cluster["wall_s"]
            accuracy = correct_count / observations
            mean_wall = total_wall / observations
            gpu_cost = mean_wall / accuracy if accuracy > 0.0 else math.inf
            accuracy_values[row["k"]].append(accuracy)
            gpu_values[row["k"]].append(gpu_cost)
            replicate_costs[row["k"]] = gpu_cost
        replicate_kstar = min(ordered_ks, key=lambda k: (replicate_costs[k], k))
        kstar_counts[replicate_kstar] += 1
        if len(ordered_ks) >= 2 and replicate_costs[ordered_ks[-1]] < replicate_costs[ordered_ks[-2]]:
            largest_falling_count += 1
    intervals = {}
    for row in rows:
        k = row["k"]
        intervals[k] = {
            "accuracy_low": percentile(accuracy_values[k], 0.025),
            "accuracy_high": percentile(accuracy_values[k], 0.975),
            "gpu_low": percentile(gpu_values[k], 0.025),
            "gpu_high": percentile(gpu_values[k], 0.975),
        }
    selected_ks = []
    for k in ordered_ks:
        selected_ks.extend([k] * kstar_counts[k])
    return {
        "intervals": intervals,
        "kstar_counts": kstar_counts,
        "kstar_low": percentile(selected_ks, 0.025),
        "kstar_high": percentile(selected_ks, 0.975),
        "largest_falling_probability": largest_falling_count / samples,
    }


def regression_slope(rows):
    xs = [math.log(row["k"], 2.0) for row in rows]
    ys = [row["metrics"]["accuracy"] * 100.0 for row in rows]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0.0:
        raise ValueError("at least two distinct k values are required for a slope")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def format_number(value, digits=4):
    if math.isinf(value):
        return "INF"
    return ("{:." + str(digits) + "f}").format(value)


def analyze_paths(paths, bootstrap_samples, bootstrap_seed):
    rows = [prepare_summary(path) for path in paths]
    rows.sort(key=lambda row: row["k"])
    if len(rows) < 2:
        raise ValueError("at least two k summaries are required")
    ks = [row["k"] for row in rows]
    if len(ks) != len(set(ks)):
        raise ValueError("k summaries contain duplicate k values")
    config = require_common_config(rows)
    boot = bootstrap(rows, bootstrap_samples, bootstrap_seed)
    slope = regression_slope(rows)
    kstar_row = min(
        rows,
        key=lambda row: (row["metrics"]["gpu_seconds_per_correct"], row["k"]),
    )
    kstar = kstar_row["k"]
    previous = rows[-2]
    largest = rows[-1]
    previous_cost = previous["metrics"]["gpu_seconds_per_correct"]
    largest_cost = largest["metrics"]["gpu_seconds_per_correct"]
    largest_falling = largest_cost < previous_cost
    problem_count = rows[0]["metrics"]["problems"]
    seed_count = rows[0]["metrics"]["seeds"]
    print("K-CURVE ANALYSIS")
    print(
        "Problem set: {} | sha256={}".format(
            config.get("data"), config.get("data_sha256")
        )
    )
    print(
        "Problems: {} | seeds nested per problem: {} | observations per k: {}".format(
            problem_count, seed_count, rows[0]["metrics"]["observations"]
        )
    )
    print(
        "Effective seed result sets by k: {}.".format(
            ", ".join(
                "k={} {}/{}".format(
                    row["k"],
                    row["metrics"]["seeds_effective"],
                    row["metrics"]["seeds_nominal"],
                )
                for row in rows
            )
        )
    )
    print(
        "Concurrency: {} item workers. K_STAR is valid only at this concurrency, "
        "on this hardware and server configuration, and while the server remains below saturation.".format(
            config.get("concurrency")
        )
    )
    print(
        "GPU-s/correct uses the required wall proxy: mean wall seconds per item / accuracy. "
        "Tokens are reported separately with no token price."
    )
    print(
        "Bootstrap: {} paired problem-cluster resamples; every sampled problem keeps all {} seed observations nested.".format(
            bootstrap_samples, seed_count
        )
    )
    print()
    print("| k | accuracy | wall/item (s) | GPU-s/correct (95% CI) | tokens/item |")
    print("|---:|---:|---:|---:|---:|")
    for row in rows:
        k = row["k"]
        metrics = row["metrics"]
        interval = boot["intervals"][k]
        gpu_cell = "{} [{}, {}]".format(
            format_number(metrics["gpu_seconds_per_correct"]),
            format_number(interval["gpu_low"]),
            format_number(interval["gpu_high"]),
        )
        print(
            "| {} | {:.2%} | {:.4f} | {} | {:.1f} |".format(
                k,
                metrics["accuracy"],
                metrics["mean_wall_s_per_item"],
                gpu_cell,
                metrics["tokens_per_item"],
            )
        )
    print()
    print(
        "Accuracy-vs-log2(k) slope: {:+.3f} percentage points per doubling.".format(
            slope
        )
    )
    print(
        "K_STAR: k={} with point-estimate GPU-s/correct={}.".format(
            kstar, format_number(kstar_row["metrics"]["gpu_seconds_per_correct"])
        )
    )
    print(
        "K_STAR rule: minimize point-estimate mean_wall_s_per_item / accuracy; ties choose the smaller k."
    )
    frequency_parts = [
        "k={} {:.1%}".format(k, boot["kstar_counts"][k] / bootstrap_samples)
        for k in ks
        if boot["kstar_counts"][k]
    ]
    print("K_STAR bootstrap selection: {}.".format(", ".join(frequency_parts)))
    print(
        "K_STAR bootstrap 95% interval: [{}, {}].".format(
            boot["kstar_low"], boot["kstar_high"]
        )
    )
    by_k = {row["k"]: row for row in rows}
    regressions = []
    for row in rows:
        half = row["k"] / 2
        if half.is_integer() and int(half) in by_k:
            prior = by_k[int(half)]
            if row["metrics"]["accuracy"] < prior["metrics"]["accuracy"]:
                regressions.append((prior, row))
    if regressions:
        for prior, row in regressions:
            print(
                "FLAG: ACCURACY REGRESSION: k={} accuracy={:.2%} is below k={} accuracy={:.2%}. "
                "Treat this as noise or a vote-implementation bug. No smoothing was applied.".format(
                    row["k"],
                    row["metrics"]["accuracy"],
                    prior["k"],
                    prior["metrics"]["accuracy"],
                )
            )
    else:
        print("Accuracy regressions versus k/2: none. No smoothing was applied.")
    status_word = "FALLING" if largest_falling else "NOT FALLING"
    print(
        "Largest-k segment: {} from k={} ({}) to k={} ({}). Bootstrap P(falling)={:.1%}.".format(
            status_word,
            previous["k"],
            format_number(previous_cost),
            largest["k"],
            format_number(largest_cost),
            boot["largest_falling_probability"],
        )
    )
    if largest_falling and kstar == largest["k"]:
        print(
            "SWEEP_STATUS: EXTEND. The point-estimate minimum is at the largest measured k and the final segment is falling, so the optimum lies beyond this sweep."
        )
    elif largest_falling:
        print(
            "SWEEP_STATUS: NON-MONOTONIC. The final segment is falling, but the measured global minimum is k={}; do not claim the optimum is beyond the sweep.".format(
                kstar
            )
        )
    else:
        print(
            "SWEEP_STATUS: TURNAROUND OBSERVED within the measured sweep at point-estimate K_STAR=k={}.".format(
                kstar
            )
        )
    if 1 in by_k:
        print("k=1 note: majority vote is a no-op; the single sampled answer is the result.")
    total_errors = sum(row["metrics"]["error_n"] for row in rows)
    collapsed_seed_ks = [
        row["k"]
        for row in rows
        if row["metrics"]["seeds_effective"] < row["metrics"]["seeds_nominal"]
    ]
    if collapsed_seed_ks:
        print(
            "FLAG: EFFECTIVE SEED COUNT collapsed below three at k={}. Nominal seeds are not independent evidence there.".format(
                ",".join(str(k) for k in collapsed_seed_ks)
            )
        )
    if total_errors:
        print(
            "FLAG: {} errored problem-seed observations are included as incorrect. Inspect source JSONL before trusting the curve.".format(
                total_errors
            )
        )


def summary_paths_from_directory(directory):
    absolute = os.path.abspath(directory)
    if not os.path.isdir(absolute):
        raise ValueError("summary directory does not exist: {}".format(absolute))
    paths = [
        os.path.join(absolute, name)
        for name in os.listdir(absolute)
        if name.startswith("k_") and name.endswith("_summary.json")
    ]
    if not paths:
        raise ValueError("no k_*_summary.json files found in {}".format(absolute))
    return paths


def make_synthetic_summary(directory, k, correct_target, wall_base, label):
    seeds = [0, 1, 2]
    problems = []
    flat_index = 0
    all_rows = []
    for problem_index in range(12):
        seed_rows = []
        for seed_index, seed in enumerate(seeds):
            wall_s = wall_base * (0.94 + 0.012 * (problem_index % 6)) + 0.001 * seed_index
            row = {
                "seed": seed,
                "correct": flat_index < correct_target,
                "wall_s": round(wall_s, 6),
                "generated_tokens": k * 100 + problem_index * 2 + seed_index,
                "error": None,
            }
            flat_index += 1
            seed_rows.append(row)
            all_rows.append(row)
        problems.append(
            {
                "id": "problem-{:02d}".format(problem_index),
                "item_index": problem_index,
                "seeds": seed_rows,
            }
        )
    observations = len(all_rows)
    correct_count = sum(row["correct"] for row in all_rows)
    accuracy = correct_count / observations
    mean_wall = statistics.mean(row["wall_s"] for row in all_rows)
    total_tokens = sum(row["generated_tokens"] for row in all_rows)
    document = {
        "schema_version": 1,
        "kind": "kcurve_k_summary",
        "k": k,
        "config": {
            "mode": "large_bo8",
            "k": k,
            "data": os.path.join(directory, label + "_locked.jsonl"),
            "data_sha256": label + "-synthetic-sha256",
            "offset": 0,
            "limit": 12,
            "seeds": seeds,
            "large_model": "SYNTHETIC_MODEL",
            "large_url": "http://127.0.0.1:30001",
            "max_tokens": 512,
            "temperature": 0.7,
            "timeout": 300.0,
            "concurrency": 4,
            "answer_mode": "math",
            "cascade_script": "synthetic/measure_cascade.py",
            "cascade_sha256": "synthetic-cascade-sha256",
        },
        "metrics": {
            "problems": len(problems),
            "seeds": len(seeds),
            "observations": observations,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "mean_wall_s_per_item": mean_wall,
            "total_generated_tokens": total_tokens,
            "tokens_per_item": total_tokens / observations,
            "gpu_seconds_per_correct": mean_wall / accuracy,
            "error_n": 0,
            "seeds_nominal": len(seeds),
            "seeds_effective": len(seeds),
        },
        "problems": problems,
    }
    path = os.path.join(directory, "k_{:03d}_summary.json".format(k))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def run_self_check(bootstrap_samples, bootstrap_seed):
    root = None
    with tempfile.TemporaryDirectory(prefix="kcurve-selfcheck-") as root:
        turnaround_dir = os.path.join(root, "turnaround")
        falling_dir = os.path.join(root, "falling")
        os.makedirs(turnaround_dir)
        os.makedirs(falling_dir)
        turnaround_paths = [
            make_synthetic_summary(turnaround_dir, 1, 12, 1.00, "turnaround"),
            make_synthetic_summary(turnaround_dir, 2, 18, 1.05, "turnaround"),
            make_synthetic_summary(turnaround_dir, 4, 24, 1.10, "turnaround"),
            make_synthetic_summary(turnaround_dir, 8, 27, 1.50, "turnaround"),
        ]
        falling_paths = [
            make_synthetic_summary(falling_dir, 1, 12, 1.00, "falling"),
            make_synthetic_summary(falling_dir, 2, 18, 1.05, "falling"),
            make_synthetic_summary(falling_dir, 4, 24, 1.10, "falling"),
            make_synthetic_summary(falling_dir, 8, 30, 1.25, "falling"),
        ]
        print("=== SYNTHETIC TURNAROUND ===")
        analyze_paths(turnaround_paths, bootstrap_samples, bootstrap_seed)
        print()
        print("=== SYNTHETIC STILL FALLING ===")
        analyze_paths(falling_paths, bootstrap_samples, bootstrap_seed)
    print()
    print("Synthetic temp files deleted: {}".format("yes" if root and not os.path.exists(root) else "no"))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze accuracy-versus-k summaries and estimate K_STAR uncertainty."
    )
    parser.add_argument("summaries", nargs="*", help="kcurve k summary JSON files")
    parser.add_argument(
        "--summary-dir",
        help="directory containing k_*_summary.json files",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="construct, analyze, print, and delete two synthetic sweeps",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.self_check:
        if args.summaries or args.summary_dir:
            parser.error("--self-check cannot be combined with input summaries")
        run_self_check(args.bootstrap_samples, args.bootstrap_seed)
        return 0
    if args.summaries and args.summary_dir:
        parser.error("provide summary files or --summary-dir, not both")
    paths = args.summaries or summary_paths_from_directory(
        args.summary_dir or DEFAULT_SUMMARY_DIR
    )
    analyze_paths(paths, args.bootstrap_samples, args.bootstrap_seed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
