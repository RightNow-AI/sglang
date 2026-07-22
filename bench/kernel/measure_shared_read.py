#!/usr/bin/env python3
'''Measure AutoTree shared-read decode throughput against a running server.'''

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


DEFAULT_BASE_URL = 'http://127.0.0.1:30000'
MODEL_NAME = 'default'
REQUEST_TIMEOUT_S = 3600
APPROX_CHARS_PER_TOKEN = 4
MIN_DECODE_TOKENS_PER_BRANCH = 100

PADDING_SENTENCE = (
    'Along the harbor, patient crews tended weathered boats while gulls crossed '
    'the pale sky and small waves folded quietly against the stone quay. '
)
CONTINUATION_INSTRUCTION = (
    'Continue this passage in flowing prose for about one hundred twenty words, '
    'describing the harbor, its sounds, weather, and movement. Use no digits, '
    'equations, measurements, dates, or numbered lists, and do not give a short '
    'answer.'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Measure decode wall time per accounted AutoTree branch token with '
            'AUTOTREE_SHARED_READ already selected at server boot.'
        )
    )
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--branches', type=int, default=8)
    parser.add_argument('--context-tokens', type=int, default=12000)
    parser.add_argument('--gen-tokens', type=int, default=140)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--mode', choices=('on', 'off'))
    parser.add_argument('--out', help='JSON result path')
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('ON_JSON', 'OFF_JSON'),
        help='compare valid ON and OFF result files',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print the exact request body without contacting the server',
    )
    return parser.parse_args()


def fail_usage(message):
    print('error: ' + message, file=sys.stderr)
    return 2


def validate_measurement_args(args):
    if args.mode is None:
        return '--mode {on,off} is required when measuring or using --dry-run'
    if not args.dry_run and not args.out:
        return '--out is required when measuring'
    if not 2 <= args.branches <= 64:
        return '--branches must be between 2 and 64'
    if args.context_tokens < 1:
        return '--context-tokens must be positive'
    if not MIN_DECODE_TOKENS_PER_BRANCH <= args.gen_tokens <= 4096:
        return '--gen-tokens must be between 100 and 4096'
    if args.reps < 1:
        return '--reps must be positive'
    return None


def build_prompt(context_tokens):
    target_bytes = context_tokens * APPROX_CHARS_PER_TOKEN
    instruction_bytes = len(CONTINUATION_INSTRUCTION.encode('utf-8')) + 2
    padding_target = max(1, target_bytes - instruction_bytes)
    sentence_bytes = len(PADDING_SENTENCE.encode('utf-8'))
    repeats = max(1, (padding_target + sentence_bytes - 1) // sentence_bytes)
    prompt = (
        (PADDING_SENTENCE * repeats).rstrip()
        + '\n\n'
        + CONTINUATION_INSTRUCTION
    )
    approximate_tokens = round(
        len(prompt.encode('utf-8')) / APPROX_CHARS_PER_TOKEN
    )
    return prompt, approximate_tokens


def build_request_body(args, prompt):
    return {
        'model': MODEL_NAME,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_completion_tokens': args.gen_tokens,
        'temperature': 0.8,
        'top_p': 1.0,
        'seed': 1729,
        'n': 1,
        'stream': False,
        'tree': {
            'policy': 'beam',
            'branches': args.branches,
            'budget_tokens': args.branches * args.gen_tokens,
            'scorer': None,
        },
    }


def empty_result(args, approximate_tokens):
    return {
        'mode': args.mode,
        'branches': args.branches,
        'context_tokens_approx': approximate_tokens,
        'reps': args.reps,
        'median_wall_s': None,
        'total_decode_tokens': None,
        'ms_per_token': None,
        'tokens_per_sec': None,
        'valid': False,
    }


def write_result(path, result):
    rendered = json.dumps(result, indent=2, sort_keys=False)
    with open(path, 'w', encoding='utf-8') as output_file:
        output_file.write(rendered + '\n')
    print(rendered)


def post_tree_request(base_url, body, branches):
    url = base_url.rstrip('/') + '/v1/tree/completions'
    payload = json.dumps(body, separators=(',', ':')).encode('utf-8')
    request = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        raise RuntimeError('HTTP {}: {}'.format(error.code, detail)) from error
    except urllib.error.URLError as error:
        raise RuntimeError('request failed: {}'.format(error.reason)) from error
    elapsed = time.perf_counter() - started

    try:
        response_data = json.loads(response_bytes.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError('server returned invalid JSON: {}'.format(error)) from error

    if not isinstance(response_data, dict):
        raise RuntimeError('response JSON is not an object')
    tree = response_data.get('tree')
    if not isinstance(tree, dict):
        raise RuntimeError('response is missing the tree summary')
    if tree.get('branch_count') != branches:
        raise RuntimeError(
            'tree.branch_count was {!r}, expected {}'.format(
                tree.get('branch_count'), branches
            )
        )

    per_branch = tree.get('tokens_spent_per_branch')
    if not isinstance(per_branch, dict) or len(per_branch) != branches:
        count = len(per_branch) if isinstance(per_branch, dict) else None
        raise RuntimeError(
            'tree.tokens_spent_per_branch represented {!r} branches, expected {}'.format(
                count, branches
            )
        )

    token_counts = list(per_branch.values())
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in token_counts
    ):
        raise RuntimeError('tree.tokens_spent_per_branch contains an invalid count')

    total_decode_tokens = sum(token_counts)
    minimum_total = MIN_DECODE_TOKENS_PER_BRANCH * branches
    if total_decode_tokens < minimum_total:
        raise RuntimeError(
            'only {} decode tokens were accounted; sustained decode requires at least {}'.format(
                total_decode_tokens, minimum_total
            )
        )
    return elapsed, total_decode_tokens


def measure(args):
    prompt, approximate_tokens = build_prompt(args.context_tokens)
    body = build_request_body(args, prompt)

    if args.dry_run:
        print(json.dumps(body, indent=2, sort_keys=False))
        return 0

    result = empty_result(args, approximate_tokens)
    total_attempts = args.reps + 1
    measured_walls = []
    measured_tokens = []

    for attempt in range(total_attempts):
        label = 'warmup' if attempt == 0 else 'rep {}/{}'.format(attempt, args.reps)
        try:
            wall_s, total_tokens = post_tree_request(
                args.base_url, body, args.branches
            )
        except Exception as error:
            print('INVALID: {} failed: {}'.format(label, error), file=sys.stderr)
            write_result(args.out, result)
            return 1

        print(
            '{}: wall_s={:.6f} total_decode_tokens={}'.format(
                label, wall_s, total_tokens
            ),
            file=sys.stderr,
        )
        if attempt > 0:
            measured_walls.append(wall_s)
            measured_tokens.append(total_tokens)

    median_wall_s = statistics.median(measured_walls)
    representative_tokens = statistics.median_low(measured_tokens)
    result.update(
        {
            'median_wall_s': median_wall_s,
            'total_decode_tokens': representative_tokens,
            'ms_per_token': median_wall_s * 1000.0 / representative_tokens,
            'tokens_per_sec': representative_tokens / median_wall_s,
            'valid': True,
        }
    )
    write_result(args.out, result)
    return 0


def load_comparison_result(path, expected_mode):
    try:
        with open(path, 'r', encoding='utf-8') as input_file:
            result = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError('could not read {}: {}'.format(path, error)) from error
    if not isinstance(result, dict):
        raise RuntimeError('{} does not contain a JSON object'.format(path))
    if result.get('mode') != expected_mode:
        raise RuntimeError(
            '{} has mode {!r}, expected {!r}'.format(
                path, result.get('mode'), expected_mode
            )
        )
    if result.get('valid') is not True:
        raise RuntimeError('{} is not a valid measurement'.format(path))
    ms_per_token = result.get('ms_per_token')
    if (
        isinstance(ms_per_token, bool)
        or not isinstance(ms_per_token, (int, float))
        or ms_per_token <= 0
    ):
        raise RuntimeError('{} has invalid ms_per_token'.format(path))
    return result


def compare(on_path, off_path):
    try:
        on_result = load_comparison_result(on_path, 'on')
        off_result = load_comparison_result(off_path, 'off')
        for field in ('branches', 'context_tokens_approx', 'reps'):
            if on_result.get(field) != off_result.get(field):
                raise RuntimeError(
                    'ON and OFF results differ in {} ({!r} versus {!r})'.format(
                        field, on_result.get(field), off_result.get(field)
                    )
                )
    except RuntimeError as error:
        print('INVALID: {}'.format(error), file=sys.stderr)
        return 1

    ratio = float(off_result['ms_per_token']) / float(on_result['ms_per_token'])
    print('off/on ms_per_token ratio: {:.6f}'.format(ratio))
    if ratio > 1.0:
        print(
            'VERDICT: shared-read ON was faster in these measurements '
            '({:.6f}x off/on).'.format(ratio)
        )
    elif ratio < 1.0:
        print(
            'VERDICT: shared-read ON was slower in these measurements '
            '({:.6f}x off/on).'.format(ratio)
        )
    else:
        print('VERDICT: shared-read ON and OFF measured the same ms/token.')
    return 0


def main():
    args = parse_args()
    if args.compare:
        if args.dry_run or args.mode is not None or args.out:
            return fail_usage(
                '--compare cannot be combined with --dry-run, --mode, or --out'
            )
        return compare(args.compare[0], args.compare[1])

    error = validate_measurement_args(args)
    if error:
        return fail_usage(error)
    return measure(args)


if __name__ == '__main__':
    sys.exit(main())
