"""ThoughtBench command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import write_leaderboard
from .harness import dry_run_requests, load_harness_config, run_harness
from .report import render_report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thoughtbench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run an OpenAI-compatible benchmark arm")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print exact request URLs and JSON bodies without network access",
    )
    run.add_argument(
        "--limit",
        type=_positive_int,
        help="run only the first N tasks (useful for smoke tests)",
    )
    aggregate = commands.add_parser("aggregate", help="merge result files into a leaderboard")
    aggregate.add_argument("source", type=Path)
    aggregate.add_argument("--output", type=Path)
    report = commands.add_parser("report", help="render a results JSON table")
    report.add_argument("results", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        config = load_harness_config(args.config)
        if args.dry_run:
            for request in dry_run_requests(config, limit=args.limit):
                print(json.dumps(request, sort_keys=True))
            return 0
        results, path = run_harness(config, limit=args.limit)
        print(f"wrote {path} ({len(results.tasks)} task executions)")
        return 0
    if args.command == "aggregate":
        leaderboard, path = write_leaderboard(args.source, args.output)
        print(f"wrote {path} ({len(leaderboard.entries)} entries)")
        return 0
    print(render_report(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
