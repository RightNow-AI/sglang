#!/usr/bin/env python3
"""Run a fixed-problem accuracy-versus-k sweep through measure_cascade.py."""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CASCADE_SCRIPT = os.path.join(REPO_ROOT, "bench", "cascade", "measure_cascade.py")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "results")


def parse_integer_list(raw, label, minimum, maximum):
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("{} must be a comma-separated integer list".format(label))
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("{} must be a comma-separated integer list".format(label)) from exc
    if len(values) != len(set(values)):
        raise ValueError("{} must not contain duplicates".format(label))
    if any(value < minimum or value > maximum for value in values):
        raise ValueError(
            "{} values must be between {} and {}".format(label, minimum, maximum)
        )
    return values


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def same_value(left, right):
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


def require_matching_config(document, expected, path, include_runner_hash):
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError("{} has no config object".format(path))
    keys = list(expected)
    if not include_runner_hash:
        keys.remove("cascade_sha256")
    mismatches = [
        key for key in keys if key not in config or not same_value(config[key], expected[key])
    ]
    if mismatches:
        raise ValueError(
            "{} does not match this sweep for: {}".format(path, ", ".join(mismatches))
        )


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


def load_item_records(path, seeds):
    records = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{} line {} is invalid JSON: {}".format(path, line_number, exc)
                ) from exc
            if not isinstance(record, dict):
                raise ValueError("{} line {} is not an object".format(path, line_number))
            if record.get("mode") != "large_bo8" or record.get("seed") not in seeds:
                continue
            item_id = record.get("id")
            if not isinstance(item_id, str):
                raise ValueError("{} line {} has invalid id".format(path, line_number))
            key = (record["seed"], item_id)
            if key in records:
                continue
            if not isinstance(record.get("correct"), bool):
                raise ValueError("{} line {} has invalid correct".format(path, line_number))
            item_index = nonnegative_integer(
                record.get("item_index"), "{} line {} item_index".format(path, line_number)
            )
            wall_s = finite_nonnegative(
                record.get("wall_s"), "{} line {} wall_s".format(path, line_number)
            )
            tokens = nonnegative_integer(
                record.get("large_tokens"),
                "{} line {} large_tokens".format(path, line_number),
            )
            error = record.get("error")
            if error is not None and not isinstance(error, str):
                raise ValueError("{} line {} has invalid error".format(path, line_number))
            records[key] = {
                "id": item_id,
                "item_index": item_index,
                "seed": record["seed"],
                "correct": record["correct"],
                "wall_s": wall_s,
                "generated_tokens": tokens,
                "error": error,
            }
    if not records:
        raise ValueError("{} contains no large_bo8 records for the requested seeds".format(path))
    return records


def clean_seed_summaries(measure_document, seeds):
    rows = measure_document.get("summaries")
    if not isinstance(rows, list):
        raise ValueError("measure summary has no summaries list")
    by_seed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("mode") != "large_bo8":
            continue
        seed = row.get("seed")
        if seed in by_seed:
            raise ValueError("measure summary repeats seed {}".format(seed))
        if seed in seeds:
            by_seed[seed] = row
    if set(by_seed) != set(seeds):
        raise ValueError("measure summary does not contain exactly the requested seeds")
    cleaned = []
    for seed in seeds:
        row = by_seed[seed]
        cleaned.append(
            {
                "seed": seed,
                "items": nonnegative_integer(row.get("items"), "summary items"),
                "correct_count": nonnegative_integer(
                    row.get("correct_count"), "summary correct_count"
                ),
                "accuracy": finite_nonnegative(row.get("accuracy"), "summary accuracy"),
                "mean_wall_s": finite_nonnegative(
                    row.get("mean_wall_s"), "summary mean_wall_s"
                ),
                "total_generated_tokens": nonnegative_integer(
                    row.get("total_large_tokens"), "summary total_large_tokens"
                ),
                "error_n": nonnegative_integer(row.get("error_n"), "summary error_n"),
                "seeds_nominal": nonnegative_integer(
                    row.get("seeds_nominal"), "summary seeds_nominal"
                ),
                "seeds_effective": nonnegative_integer(
                    row.get("seeds_effective"), "summary seeds_effective"
                ),
            }
        )
    return cleaned


def build_problem_rows(records, seeds):
    ids_by_seed = {
        seed: {item_id for record_seed, item_id in records if record_seed == seed}
        for seed in seeds
    }
    first_ids = ids_by_seed[seeds[0]]
    for seed in seeds[1:]:
        if ids_by_seed[seed] != first_ids:
            raise ValueError("problem ids differ across seeds")
    rows = []
    for item_id in first_ids:
        observations = [records[(seed, item_id)] for seed in seeds]
        item_indices = {row["item_index"] for row in observations}
        if len(item_indices) != 1:
            raise ValueError("item_index differs across seeds for {!r}".format(item_id))
        rows.append(
            {
                "id": item_id,
                "item_index": observations[0]["item_index"],
                "seeds": [
                    {
                        "seed": row["seed"],
                        "correct": row["correct"],
                        "wall_s": row["wall_s"],
                        "generated_tokens": row["generated_tokens"],
                        "error": row["error"],
                    }
                    for row in observations
                ],
            }
        )
    rows.sort(key=lambda row: (row["item_index"], row["id"]))
    return rows


def compare_seed_metrics(seed_metrics, records):
    for row in seed_metrics:
        seed_records = [record for (seed, _item_id), record in records.items() if seed == row["seed"]]
        correct_count = sum(record["correct"] for record in seed_records)
        total_tokens = sum(record["generated_tokens"] for record in seed_records)
        mean_wall = statistics.mean(record["wall_s"] for record in seed_records)
        if len(seed_records) != row["items"]:
            raise ValueError("raw record count differs from measure summary for seed {}".format(row["seed"]))
        if correct_count != row["correct_count"]:
            raise ValueError("raw correct count differs from measure summary for seed {}".format(row["seed"]))
        if total_tokens != row["total_generated_tokens"]:
            raise ValueError("raw token count differs from measure summary for seed {}".format(row["seed"]))
        if not math.isclose(mean_wall, row["mean_wall_s"], rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("raw mean wall differs from measure summary for seed {}".format(row["seed"]))


def build_summary(k, expected, measure_path, items_path):
    measure = load_json(measure_path)
    measure_config = measure.get("config")
    if not isinstance(measure_config, dict):
        raise ValueError("{} has no config object".format(measure_path))
    required_measure = {
        "mode": "large_bo8",
        "data": expected["data"],
        "offset": expected["offset"],
        "limit": expected["limit"],
        "seeds": expected["seeds"],
        "branches": k,
        "large_model": expected["large_model"],
        "large_url": expected["large_url"],
        "max_tokens": expected["max_tokens"],
        "temperature": expected["temperature"],
        "timeout": expected["timeout"],
        "concurrency": expected["concurrency"],
        "answer_mode": expected["answer_mode"],
        "small_cost": 0.0,
        "large_cost": 0.0,
    }
    mismatches = [
        key
        for key, value in required_measure.items()
        if key not in measure_config or not same_value(measure_config[key], value)
    ]
    if mismatches:
        raise ValueError(
            "{} has unexpected cascade config for: {}".format(
                measure_path, ", ".join(mismatches)
            )
        )
    records = load_item_records(items_path, expected["seeds"])
    seed_metrics = clean_seed_summaries(measure, expected["seeds"])
    compare_seed_metrics(seed_metrics, records)
    problem_rows = build_problem_rows(records, expected["seeds"])
    observations = list(records.values())
    correct_count = sum(record["correct"] for record in observations)
    accuracy = correct_count / len(observations)
    mean_wall = statistics.mean(record["wall_s"] for record in observations)
    total_tokens = sum(record["generated_tokens"] for record in observations)
    gpu_seconds_per_correct = mean_wall / accuracy if accuracy > 0.0 else None
    nominal_seed_counts = {row["seeds_nominal"] for row in seed_metrics}
    effective_seed_counts = {row["seeds_effective"] for row in seed_metrics}
    if len(nominal_seed_counts) != 1 or len(effective_seed_counts) != 1:
        raise ValueError("measure summary seed evidence differs across seed rows")
    return {
        "schema_version": 1,
        "kind": "kcurve_k_summary",
        "k": k,
        "config": dict(expected),
        "metrics": {
            "problems": len(problem_rows),
            "seeds": len(expected["seeds"]),
            "observations": len(observations),
            "correct_count": correct_count,
            "accuracy": accuracy,
            "mean_wall_s_per_item": mean_wall,
            "total_generated_tokens": total_tokens,
            "tokens_per_item": total_tokens / len(observations),
            "gpu_seconds_per_correct": gpu_seconds_per_correct,
            "error_n": sum(record["error"] is not None for record in observations),
            "seeds_nominal": nominal_seed_counts.pop(),
            "seeds_effective": effective_seed_counts.pop(),
        },
        "seed_metrics": seed_metrics,
        "problems": problem_rows,
        "notes": [
            (
                "At k=1, majority vote is a no-op; the single sampled answer is the result. "
                "It uses the configured sampling temperature, so greedy-equivalent refers to branch count, not temperature-0 decoding."
                if k == 1
                else "Majority vote uses the unchanged measure_cascade.py implementation."
            ),
            "GPU-seconds per correct is mean wall seconds per item divided by accuracy.",
            "Token totals are reported separately and are not converted through a token price.",
        ],
        "source": {
            "measure_summary": os.path.abspath(measure_path),
            "items_jsonl": os.path.abspath(items_path),
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep best-of-k accuracy and wall cost through measure_cascade.py."
    )
    parser.add_argument("--data", required=True, help="locked JSONL problem set")
    parser.add_argument("--large-model", required=True, help="served large model name")
    parser.add_argument("--large-url", default="http://127.0.0.1:30001")
    parser.add_argument("--k-ladder", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", default="0,1,2", help="exactly three seeds")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--answer-mode", choices=("numeric", "math"), default="math")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        ladder = parse_integer_list(args.k_ladder, "--k-ladder", 1, 64)
        seeds = parse_integer_list(args.seeds, "--seeds", -(2**31), 2**31 - 1)
    except ValueError as exc:
        parser.error(str(exc))
    if len(seeds) != 3:
        parser.error("--seeds must contain exactly three distinct seeds")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.max_tokens <= 0 or args.max_tokens > 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0 or args.temperature > 2.0:
        parser.error("--temperature must be finite, greater than 0, and at most 2")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    if args.offset < 0:
        parser.error("--offset must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        parser.error("--data does not exist: {}".format(data_path))
    if not os.path.isfile(CASCADE_SCRIPT):
        parser.error("cascade runner does not exist: {}".format(CASCADE_SCRIPT))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    expected = {
        "mode": "large_bo8",
        "data": data_path,
        "data_sha256": sha256_file(data_path),
        "offset": args.offset,
        "limit": args.limit,
        "seeds": seeds,
        "large_model": args.large_model,
        "large_url": args.large_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "concurrency": args.concurrency,
        "answer_mode": args.answer_mode,
        "cascade_script": os.path.abspath(CASCADE_SCRIPT),
        "cascade_sha256": sha256_file(CASCADE_SCRIPT),
    }
    completed = []
    for k in ladder:
        stem = "k_{:03d}".format(k)
        manifest_path = os.path.join(out_dir, stem + "_run.json")
        items_path = os.path.join(out_dir, stem + "_items.jsonl")
        measure_path = os.path.join(out_dir, stem + "_measure.json")
        summary_path = os.path.join(out_dir, stem + "_summary.json")
        per_k_config = dict(expected)
        per_k_config["k"] = k
        if os.path.exists(summary_path):
            existing = load_json(summary_path)
            if existing.get("kind") != "kcurve_k_summary" or existing.get("k") != k:
                raise ValueError("{} is not a k={} kcurve summary".format(summary_path, k))
            require_matching_config(
                existing, per_k_config, summary_path, include_runner_hash=False
            )
            print("SKIP k={}: summary exists: {}".format(k, summary_path))
            completed.append(summary_path)
            continue
        if os.path.exists(manifest_path):
            manifest = load_json(manifest_path)
            require_matching_config(
                manifest, per_k_config, manifest_path, include_runner_hash=True
            )
        else:
            write_json_atomic(
                manifest_path,
                {
                    "schema_version": 1,
                    "kind": "kcurve_run_manifest",
                    "config": per_k_config,
                },
            )
        command = [
            sys.executable,
            CASCADE_SCRIPT,
            "--mode",
            "large_bo8",
            "--data",
            data_path,
            "--large-model",
            args.large_model,
            "--large-url",
            args.large_url,
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--branches",
            str(k),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--timeout",
            str(args.timeout),
            "--concurrency",
            str(args.concurrency),
            "--answer-mode",
            args.answer_mode,
            "--small-cost",
            "0",
            "--large-cost",
            "0",
            "--offset",
            str(args.offset),
            "--out-jsonl",
            items_path,
            "--out",
            measure_path,
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        print("RUN k={}: {}".format(k, subprocess.list2cmdline(command)))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "measure_cascade.py failed for k={} with exit code {}".format(
                    k, result.returncode
                )
            )
        summary = build_summary(k, per_k_config, measure_path, items_path)
        write_json_atomic(summary_path, summary)
        metrics = summary["metrics"]
        gpu_cost = metrics["gpu_seconds_per_correct"]
        gpu_text = "undefined" if gpu_cost is None else "{:.6f}".format(gpu_cost)
        print(
            "DONE k={} accuracy={:.4%} wall/item={:.6f}s gpu-s/correct={} "
            "tokens={}".format(
                k,
                metrics["accuracy"],
                metrics["mean_wall_s_per_item"],
                gpu_text,
                metrics["total_generated_tokens"],
            )
        )
        completed.append(summary_path)
    print("K-curve summaries:")
    for path in completed:
        print("  {}".format(path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
