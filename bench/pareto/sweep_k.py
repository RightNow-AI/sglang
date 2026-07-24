#!/usr/bin/env python3
"""Run matched large-model baseline and tree sweeps over a branch ladder."""

import argparse
import math
from pathlib import Path
import subprocess
import sys


ARMS = (
    ("baseline", "large_bo8"),
    ("tree", "large_tree"),
)


def parse_ks(raw):
    values = []
    seen = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise ValueError("--ks must be a comma-separated list of integers")
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError("invalid k value: {!r}".format(part)) from exc
        if value < 1 or value > 64:
            raise ValueError("k values must be between 1 and 64")
        if value in seen:
            raise ValueError("duplicate k value: {}".format(value))
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("--ks must contain at least one value")
    return values


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run large_bo8 and large_tree through measure_cascade.py over the "
            "same branch-count ladder."
        )
    )
    parser.add_argument("--data", required=True, help="evaluation JSONL input")
    parser.add_argument("--large-model", required=True, help="served model name")
    parser.add_argument(
        "--large-url", default="http://127.0.0.1:30001", help="large-model URL"
    )
    parser.add_argument("--seeds", default="0", help="seed list passed through")
    parser.add_argument(
        "--answer-mode", choices=("numeric", "math"), default="numeric"
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--large-cost", type=float, default=10.0)
    parser.add_argument("--small-cost", type=float, default=1.0)
    parser.add_argument(
        "--ks",
        default="1,2,4,8,16",
        help="comma-separated branch ladder (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--out-dir",
        default="bench/pareto/results",
        help="directory for per-arm, per-k summary and item files",
    )
    return parser


def validate_args(parser, args):
    if args.max_tokens < 1 or args.max_tokens > 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    for name in ("large_cost", "small_cost"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error("--{} must be finite and nonnegative".format(name.replace("_", "-")))
    try:
        return parse_ks(args.ks)
    except ValueError as exc:
        parser.error(str(exc))


def run_sweep(args, ks):
    repo_root = Path(__file__).resolve().parents[2]
    harness = repo_root / "bench" / "cascade" / "measure_cascade.py"
    if not harness.is_file():
        raise FileNotFoundError("measurement harness not found: {}".format(harness))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in ks:
        for arm, mode in ARMS:
            summary_path = out_dir / "{}_k{}_summary.json".format(arm, k)
            items_path = out_dir / "{}_k{}_items.jsonl".format(arm, k)
            if summary_path.exists():
                print("SKIP {} k={}: {} exists".format(arm, k, summary_path))
                continue

            command = [
                sys.executable,
                str(harness),
                "--data",
                args.data,
                "--mode",
                mode,
                "--branches",
                str(k),
                "--large-model",
                args.large_model,
                "--large-url",
                args.large_url,
                "--seeds",
                args.seeds,
                "--answer-mode",
                args.answer_mode,
                "--max-tokens",
                str(args.max_tokens),
                "--timeout",
                str(args.timeout),
                "--large-cost",
                str(args.large_cost),
                "--small-cost",
                str(args.small_cost),
                "--out",
                str(summary_path),
                "--out-jsonl",
                str(items_path),
            ]
            print("RUN {} k={}: {}".format(arm, k, subprocess.list2cmdline(command)))
            subprocess.run(command, cwd=str(repo_root), check=True)
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    ks = validate_args(parser, args)
    try:
        return run_sweep(args, ks)
    except subprocess.CalledProcessError as exc:
        print(
            "measurement failed with exit code {}".format(exc.returncode),
            file=sys.stderr,
        )
        return exc.returncode or 1
    except OSError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
