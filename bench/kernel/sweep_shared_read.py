#!/usr/bin/env python3
"""Measure AutoTree shared-prefix reads across context and branch counts."""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request


DEFAULT_CONTEXTS = "8000,32000,100000"
DEFAULT_BRANCHES = "4,8,16"
TREE_PATH = "/v1/tree/completions"
REQUEST_TIMEOUT_S = 3600
MODEL_NAME = "default"

PAD_SENTENCE = (
    "A patient observer walks through a quiet landscape and records each subtle "
    "change in light, weather, texture, and sound for a thoughtful travel journal. "
)
CONTINUATION = (
    "\n\nContinue this passage with calm, original prose. Do not use digits, "
    "calculations, lists, or a final answer."
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def comma_separated_ints(value, option_name):
    values = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            raise ValueError(f"{option_name} contains an empty value")
        number = int(raw)
        if number < 1:
            raise ValueError(f"{option_name} values must be positive: {number}")
        if number not in values:
            values.append(number)
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return values


def padded_prompt(target_context):
    target_chars = target_context * 4
    shared_chars = max(0, target_chars - len(CONTINUATION))
    repeats = max(1, (shared_chars + len(PAD_SENTENCE) - 1) // len(PAD_SENTENCE))
    prompt = PAD_SENTENCE * repeats + CONTINUATION
    return prompt, round(len(prompt) / 4)


def request_body(context, branches, gen_tokens, model=MODEL_NAME):
    prompt, approx_tokens = padded_prompt(context)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1,
        "n": 1,
        "stream": False,
        "ignore_eos": True,
        "tree": {
            "policy": "beam",
            "branches": branches,
            "budget_tokens": branches * gen_tokens,
            "scorer": None,
        },
    }
    return body, approx_tokens


def post_json(url, body):
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail[:1000]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request failed: {error.reason}") from error
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("server returned a non-JSON response") from error


def response_decode_tokens(response):
    tree = response.get("tree") if isinstance(response, dict) else None
    counts = tree.get("tokens_spent_per_branch") if isinstance(tree, dict) else None
    if not isinstance(counts, dict) or not counts:
        return None
    values = []
    for count in counts.values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        values.append(count)
    return sum(values)


def empty_result(context, branches, approx_tokens, error):
    return {
        "context": context,
        "branches": branches,
        "approx_ctx_tokens": approx_tokens,
        "median_wall_s": None,
        "total_decode_tokens": None,
        "ms_per_token": None,
        "valid": False,
        "error": error,
    }


def measure_config(base_url, context, branches, gen_tokens, reps):
    body, approx_tokens = request_body(context, branches, gen_tokens)
    expected_tokens = branches * gen_tokens
    url = base_url.rstrip("/") + TREE_PATH
    measured_walls = []
    measured_tokens = []

    for rep in range(reps + 1):
        label = "warmup" if rep == 0 else f"rep {rep}/{reps}"
        try:
            started = time.perf_counter()
            response = post_json(url, body)
            wall_s = time.perf_counter() - started
            total_tokens = response_decode_tokens(response)
        except Exception as error:
            message = f"{label} failed: {error}"
            print(
                f"ERROR context={context} branches={branches}: {message}",
                file=sys.stderr,
            )
            return empty_result(context, branches, approx_tokens, message)

        if total_tokens != expected_tokens:
            print(
                "WARN "
                f"context={context} branches={branches} {label}: "
                f"total decode tokens={total_tokens!r}, expected={expected_tokens}; "
                "early termination or missing accounting detected",
                file=sys.stderr,
            )

        if rep > 0:
            measured_walls.append(wall_s)
            measured_tokens.append(total_tokens)

    median_wall_s = statistics.median(measured_walls)
    fixed_length = all(value == expected_tokens for value in measured_tokens)
    total_decode_tokens = measured_tokens[0] if fixed_length else None
    ms_per_token = (
        median_wall_s * 1000.0 / total_decode_tokens
        if total_decode_tokens is not None
        else None
    )
    error = None
    if not fixed_length:
        error = (
            f"fixed-length invariant failed: expected {expected_tokens} decode "
            f"tokens in every measured repetition, observed {measured_tokens}"
        )

    return {
        "context": context,
        "branches": branches,
        "approx_ctx_tokens": approx_tokens,
        "median_wall_s": median_wall_s,
        "total_decode_tokens": total_decode_tokens,
        "ms_per_token": ms_per_token,
        "valid": fixed_length,
        "error": error,
    }


def write_results(path, mode, results):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as output:
        json.dump({"mode": mode, "results": results}, output, indent=2)
        output.write("\n")


def load_results(path):
    with open(path, "r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"{path} is not a sweep results JSON")
    return payload


def result_index(payload):
    indexed = {}
    for row in payload["results"]:
        if not isinstance(row, dict):
            continue
        key = (row.get("context"), row.get("branches"))
        if all(isinstance(value, int) and not isinstance(value, bool) for value in key):
            indexed[key] = row
    return indexed


def numeric_metric(row, name):
    if not isinstance(row, dict) or not row.get("valid"):
        return None
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def compare_results(on_path, off_path):
    on_payload = load_results(on_path)
    off_payload = load_results(off_path)
    on_rows = result_index(on_payload)
    off_rows = result_index(off_payload)
    keys = sorted(set(on_rows) | set(off_rows))

    headers = (
        "context",
        "branches",
        "on ms/tok",
        "off ms/tok",
        "speedup(off/on)",
        "verdict",
    )
    rows = []
    best = None
    for context, branches in keys:
        on_value = numeric_metric(on_rows.get((context, branches)), "ms_per_token")
        off_value = numeric_metric(off_rows.get((context, branches)), "ms_per_token")
        speedup = off_value / on_value if on_value and off_value is not None else None
        if speedup is None:
            verdict = "unavailable"
        elif speedup > 1.15:
            verdict = "shared-read win"
        elif speedup >= 0.9:
            verdict = "parity"
        else:
            verdict = "slower"
        if speedup is not None and (best is None or speedup > best[0]):
            best = (speedup, context, branches)
        rows.append(
            (
                str(context),
                str(branches),
                f"{on_value:.4f}" if on_value is not None else "-",
                f"{off_value:.4f}" if off_value is not None else "-",
                f"{speedup:.3f}x" if speedup is not None else "-",
                verdict,
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def format_row(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))
    if best is None:
        print("Best speedup: unavailable")
    else:
        print(
            f"Best speedup: {best[0]:.3f}x at context={best[1]}, "
            f"branches={best[2]}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep AutoTree shared-read performance or compare on/off JSON files."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--branches", default=DEFAULT_BRANCHES)
    parser.add_argument("--gen-tokens", type=positive_int, default=128)
    parser.add_argument("--reps", type=positive_int, default=3)
    parser.add_argument("--mode", choices=("on", "off"))
    parser.add_argument("--out", help="Output JSON path for a sweep run.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("ON_JSON", "OFF_JSON"),
        help="Compare completed on/off sweep JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first config's exact request body without network access.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        contexts = comma_separated_ints(args.contexts, "--contexts")
        branches_list = comma_separated_ints(args.branches, "--branches")
    except ValueError as error:
        parser.error(str(error))

    if args.compare:
        if args.dry_run:
            parser.error("--compare and --dry-run cannot be used together")
        try:
            compare_results(*args.compare)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        return 0

    if args.dry_run:
        body, _ = request_body(contexts[0], branches_list[0], args.gen_tokens)
        print(json.dumps(body, indent=2))
        return 0

    if args.mode is None or args.out is None:
        parser.error("a sweep run requires both --mode and --out")

    results = []
    for context in contexts:
        for branches in branches_list:
            print(f"MEASURE mode={args.mode} context={context} branches={branches}")
            results.append(
                measure_config(
                    args.base_url,
                    context,
                    branches,
                    args.gen_tokens,
                    args.reps,
                )
            )
    write_results(args.out, args.mode, results)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())