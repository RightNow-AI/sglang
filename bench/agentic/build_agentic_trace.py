#!/usr/bin/env python3
"""Build deterministic replay traces with measured exact prompt repetition."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random


TRACE_FORMAT = "agentic-repetition-trace-v1"
ANSWER_SUFFIX = (
    "\n\nSolve this subproblem carefully. End with exactly one line in the form "
    "FINAL: <answer>."
)
DEFAULT_GSM8K = Path(
    "C:/Users/jaber/RightNow-Full/AutoTree/thoughtbench/tasks/gsm8k_test.jsonl"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(explicit, repo_root, relative_path):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError("data file does not exist: {}".format(path))
        return path, "explicit"

    local_path = (repo_root / relative_path).resolve()
    if local_path.is_file():
        return local_path, "current_worktree"

    candidates = []
    for sibling in sorted(repo_root.parent.iterdir(), key=lambda item: item.name):
        if sibling == repo_root or not sibling.is_dir():
            continue
        candidate = sibling / relative_path
        if candidate.is_file():
            candidates.append(candidate.resolve())
    if candidates:
        return candidates[0], "sibling_worktree_fallback"
    raise ValueError(
        "could not find {}; generate it in this worktree or pass its data flag".format(
            relative_path.as_posix()
        )
    )


def load_source(label, path, resolution):
    rows = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON in {} at line {}: {}".format(path, line_number, exc)
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    "{} line {} is not a JSON object".format(path, line_number)
                )
            for field in ("id", "prompt", "gold"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(
                        "{} line {} field {!r} must be a nonempty string".format(
                            path, line_number, field
                        )
                    )
            source_id = row["id"]
            if source_id in seen_ids:
                raise ValueError("duplicate id {!r} in {}".format(source_id, path))
            seen_ids.add(source_id)
            rows.append(
                {
                    "source": label,
                    "source_id": source_id,
                    "task_id": "{}:{}".format(label, source_id),
                    "prompt": row["prompt"],
                    "gold": row["gold"],
                }
            )
    if not rows:
        raise ValueError("{} contains no records".format(path))
    metadata = {
        "label": label,
        "path": str(path),
        "resolution": resolution,
        "sha256": sha256_file(path),
        "rows": len(rows),
    }
    return rows, metadata


def solve_prompt(problem):
    return "SUBPROBLEM SOLVE\n\n" + problem + ANSWER_SUFFIX


def plan_prompt(problem):
    return (
        "SUBPROBLEM PLAN AND SOLVE\n\n"
        + problem
        + "\n\nFirst identify the needed method."
        + ANSWER_SUFFIX
    )


def verify_prompt(problem):
    return (
        "SUBPROBLEM INDEPENDENT CHECK\n\n"
        + problem
        + "\n\nRecompute independently before answering."
        + ANSWER_SUFFIX
    )


def unique_solve_items(items):
    unique = []
    seen_prompts = set()
    for item in items:
        prompt = solve_prompt(item["prompt"])
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        prepared = dict(item)
        prepared["solve_prompt"] = prompt
        unique.append(prepared)
    return unique


def controlled_calls(items, total_calls, repeat_rate, rng):
    repeat_count = int(math.floor(total_calls * repeat_rate + 0.5))
    repeat_count = min(repeat_count, total_calls - 1)
    fresh_count = total_calls - repeat_count
    if len(items) < fresh_count:
        raise ValueError(
            "controlled trace needs {} unique prompts but only {} are available".format(
                fresh_count, len(items)
            )
        )

    selected = rng.sample(items, fresh_count)
    plan = ["fresh"] * fresh_count + ["repeat"] * repeat_count
    rng.shuffle(plan)
    first_fresh = plan.index("fresh")
    plan[0], plan[first_fresh] = plan[first_fresh], plan[0]

    calls = []
    next_fresh = 0
    prior = []
    for kind in plan:
        if kind == "fresh":
            item = selected[next_fresh]
            next_fresh += 1
            call = {
                "task_id": item["task_id"],
                "step": "solve",
                "prompt": item["solve_prompt"],
                "gold": item["gold"],
            }
            prior.append(call)
        else:
            original = rng.choice(prior)
            call = {
                "task_id": original["task_id"],
                "step": original["step"],
                "prompt": original["prompt"],
                "gold": original["gold"],
            }
        calls.append(call)
    return calls


def natural_calls(items, total_calls, natural_reuses):
    if total_calls < 3:
        raise ValueError("natural mode requires --calls of at least 3")
    calls = []
    solved = []
    item_index = 0
    while len(calls) < total_calls:
        if item_index >= len(items):
            raise ValueError(
                "natural trace exhausted unique source tasks at {} calls".format(
                    len(calls)
                )
            )
        item = items[item_index]
        workflow_id = "workflow:{}".format(item["task_id"])
        steps = [
            {
                "task_id": workflow_id,
                "step": "plan",
                "prompt": plan_prompt(item["prompt"]),
                "gold": item["gold"],
            },
            {
                "task_id": workflow_id,
                "step": "solve",
                "prompt": item["solve_prompt"],
                "gold": item["gold"],
            },
        ]
        if solved:
            reuse_count = min(natural_reuses, len(solved))
            for reuse_index in range(reuse_count):
                reused = solved[(item_index + reuse_index) % len(solved)]
                steps.append(
                    {
                        "task_id": workflow_id,
                        "step": "reuse_{}".format(reuse_index + 1),
                        "prompt": reused["prompt"],
                        "gold": reused["gold"],
                    }
                )
        steps.append(
            {
                "task_id": workflow_id,
                "step": "verify",
                "prompt": verify_prompt(item["prompt"]),
                "gold": item["gold"],
            }
        )
        calls.extend(steps[: total_calls - len(calls)])
        solved.append(steps[1])
        item_index += 1
    return calls


def repetition_stats(calls):
    seen_prompts = set()
    seen_strict = set()
    repeat_calls = 0
    strict_repeat_calls = 0
    for call in calls:
        prompt = call["prompt"]
        strict_key = (call["task_id"], call["step"], prompt)
        if prompt in seen_prompts:
            repeat_calls += 1
        if strict_key in seen_strict:
            strict_repeat_calls += 1
        seen_prompts.add(prompt)
        seen_strict.add(strict_key)
    total = len(calls)
    return {
        "total_calls": total,
        "unique_prompts": len(seen_prompts),
        "repeat_calls": repeat_calls,
        "realized_repeat_rate": repeat_calls / total if total else 0.0,
        "strict_repeat_calls": strict_repeat_calls,
        "strict_repeat_rate": strict_repeat_calls / total if total else 0.0,
    }


def write_trace(path, header, calls):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        for index, call in enumerate(calls, 1):
            record = {
                "call_id": "call-{:06d}".format(index),
                "task_id": call["task_id"],
                "step": call["step"],
                "prompt": call["prompt"],
                "gold": call["gold"],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build an exact-repetition agentic replay trace."
    )
    parser.add_argument("--out", required=True, help="output trace JSONL")
    parser.add_argument(
        "--mode", choices=("controlled", "natural"), default="controlled"
    )
    parser.add_argument(
        "--repeat-rate",
        type=float,
        default=0.8,
        help="target repeat fraction for controlled mode, from 0.0 through 0.95",
    )
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--natural-reuses",
        type=int,
        default=2,
        help="prior solve prompts reused by each later workflow in natural mode",
    )
    parser.add_argument("--math-data")
    parser.add_argument("--aime-data")
    parser.add_argument("--gsm8k-data", default=str(DEFAULT_GSM8K))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.repeat_rate) or not 0.0 <= args.repeat_rate <= 0.95:
        parser.error("--repeat-rate must be finite and between 0.0 and 0.95")
    if args.calls <= 0:
        parser.error("--calls must be positive")
    if args.natural_reuses < 0:
        parser.error("--natural-reuses must be nonnegative")

    repo_root = Path(__file__).resolve().parents[2]
    try:
        math_path, math_resolution = resolve_data_path(
            args.math_data,
            repo_root,
            Path("bench/tasks/data/math_hard.jsonl"),
        )
        aime_path, aime_resolution = resolve_data_path(
            args.aime_data,
            repo_root,
            Path("bench/tasks/data/aime.jsonl"),
        )
        gsm8k_path = Path(args.gsm8k_data).expanduser().resolve()
        if not gsm8k_path.is_file():
            raise ValueError("data file does not exist: {}".format(gsm8k_path))

        sources = []
        metadata = []
        for label, path, resolution in (
            ("math_hard", math_path, math_resolution),
            ("aime", aime_path, aime_resolution),
            ("gsm8k", gsm8k_path, "explicit_or_default_external"),
        ):
            rows, source_metadata = load_source(label, path, resolution)
            sources.extend(rows)
            metadata.append(source_metadata)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    rng = random.Random(args.seed)
    items = unique_solve_items(sources)
    rng.shuffle(items)
    try:
        if args.mode == "controlled":
            calls = controlled_calls(items, args.calls, args.repeat_rate, rng)
            requested_repeat_rate = args.repeat_rate
        else:
            calls = natural_calls(items, args.calls, args.natural_reuses)
            requested_repeat_rate = None
    except ValueError as exc:
        parser.error(str(exc))

    stats = repetition_stats(calls)
    header = {
        "type": "agentic_trace_header",
        "format": TRACE_FORMAT,
        "mode": args.mode,
        "seed": args.seed,
        "requested_repeat_rate": requested_repeat_rate,
        "repeat_definition": "call prompt is byte-identical to an earlier prompt",
        "strict_repeat_definition": (
            "task_id, step, and prompt are all identical to an earlier call"
        ),
        "natural_reuses": args.natural_reuses if args.mode == "natural" else None,
        "sources": metadata,
    }
    header.update(stats)
    write_trace(args.out, header, calls)

    print("trace: {}".format(Path(args.out).resolve()))
    print("mode: {}".format(args.mode))
    if args.mode == "natural":
        print("note: --repeat-rate is ignored in natural mode")
    print("total calls: {}".format(stats["total_calls"]))
    print("repeat calls: {}".format(stats["repeat_calls"]))
    print("realized repetition rate: {:.2%}".format(stats["realized_repeat_rate"]))
    print("strict repetition rate: {:.2%}".format(stats["strict_repeat_rate"]))
    for source in metadata:
        print(
            "source {}: {} rows, {}, {}".format(
                source["label"], source["rows"], source["resolution"], source["path"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
