#!/usr/bin/env python3
'''Live HTTP verifier for AutoTree shared-prefix single-read decode.

This client imports no SGLang modules. --mode is an evidence label: the server
must already have AUTOTREE_SHARED_READ=1 (on) or 0 (off).
'''

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
DEFAULT_BASE_URL = 'http://127.0.0.1:30000'
TREE_PATH = '/v1/tree/completions'
FORK_MARKER = 'FORK:'
CORRECTNESS_BRANCHES = 4
CORRECTNESS_CONTEXT_TOKENS = 256
MEASURE_BRANCHES = (4, 8, 16)

CORRECTNESS_TASKS = (
    ('arithmetic', 'Return only the integer value of 137 + 248 after the required prefix.'),
    ('reverse', 'Reverse the ASCII string KERNEL42 and return only the reversed string after the required prefix.'),
    ('sort', 'Sort 19, 3, 11, 7, 2 in ascending order and return only a comma-separated list after the required prefix.'),
    ('uppercase', 'Convert shared-read parity gate to uppercase and return only that text after the required prefix.'),
    ('hex', 'Convert decimal 255 to lowercase hexadecimal without 0x and return only the result after the required prefix.'),
    ('count', 'Count the letter r in deterministic shared prefix verification and return only the integer after the required prefix.'),
    ('product', 'Return only the integer product of 12 and 17 after the required prefix.'),
    ('sequence', 'Return the first six positive even integers as a single comma-separated list after the required prefix.'),
)

FILLER_WORDS = (
    'amber', 'birch', 'cedar', 'delta', 'ember', 'frost', 'granite', 'harbor',
    'indigo', 'juniper', 'kernel', 'lantern', 'meadow', 'nickel', 'orbit', 'prairie',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Verify correctness and measure effect of AutoTree shared-read decode.'
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        '--compare', nargs=2, metavar=('ON_JSON', 'OFF_JSON'),
        help='Compare two correctness captures and exit nonzero on mismatch.',
    )
    action.add_argument(
        '--measure', action='store_true',
        help='Measure B=4,8,16 instead of running the eight correctness tasks.',
    )
    parser.add_argument('--mode', choices=('on', 'off'), help='Server configuration label.')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--model', default='default', help='Served model name and regime stamp.')
    parser.add_argument('--out', help='Output JSON path for a live capture.')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--context-tokens', type=int, default=4096,
                        help='Approximate measurement filler-token target.')
    parser.add_argument('--repeats', type=int, default=1, help='Requests per B in measure mode.')
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--api-key', default='EMPTY')
    parser.add_argument('--server-log', help='Optional live server log sampled after each B.')
    parser.add_argument('--log-settle-seconds', type=float, default=0.5)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print exact request bodies and perform no network I/O.')
    args = parser.parse_args()

    if args.compare:
        if args.dry_run:
            parser.error('--dry-run does not apply to --compare')
        return args
    if args.mode is None:
        parser.error('--mode {on,off} is required unless --compare is used')
    if not args.dry_run and not args.out:
        parser.error('--out is required for a live correctness or measurement run')
    if args.max_tokens < 1 or args.max_tokens > 4096:
        parser.error('--max-tokens must be in [1, 4096]')
    if args.context_tokens < 1 or args.repeats < 1 or args.timeout <= 0:
        parser.error('context-tokens, repeats, and timeout must be positive')
    if args.log_settle_seconds < 0:
        parser.error('--log-settle-seconds cannot be negative')
    return args


def endpoint_url(base_url):
    value = base_url.rstrip('/')
    path = urllib.parse.urlparse(value).path.rstrip('/')
    if path.endswith(TREE_PATH):
        return value
    if path.endswith('/v1'):
        return value + '/tree/completions'
    return value + TREE_PATH


def make_filler(target_approx_tokens):
    # The server usage field, not this target, is the authoritative token count.
    return ' '.join(
        FILLER_WORDS[index % len(FILLER_WORDS)]
        for index in range(target_approx_tokens)
    )


def shared_context(target_approx_tokens, label):
    return (
        'Shared-prefix verification context (' + label + '). The following stable filler '
        'exists only to create a repeatable KV prefix; do not quote it in the answer.\n'
        + make_filler(target_approx_tokens)
    )


def tree_body(model, prompt, branches, seed, max_tokens):
    return {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You are a deterministic verification assistant. Follow the exact response '
                    'prefix and answer-format instruction. Do not emit any character before the '
                    'required prefix.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ],
        'max_completion_tokens': max_tokens,
        'temperature': 0.0,
        'top_p': 1.0,
        'seed': seed,
        'n': 1,
        'stream': False,
        'tree': {
            'policy': 'beam',
            'branches': branches,
            'budget_tokens': branches * max_tokens,
            'scorer': None,
            'fork_at_text': FORK_MARKER,
        },
    }


def correctness_requests(args):
    context = shared_context(CORRECTNESS_CONTEXT_TOKENS, 'medium')
    requests = []
    for index, (name, instruction) in enumerate(CORRECTNESS_TASKS):
        seed = args.seed + index
        prompt = (
            context + '\n\nTask: ' + instruction
            + '\nResponse format: begin with exactly ' + FORK_MARKER
            + ' on the first line, then put the requested answer on the next line.'
        )
        requests.append({
            'task_id': '%02d-%s' % (index + 1, name),
            'name': name,
            'seed': seed,
            'body': tree_body(
                args.model, prompt, CORRECTNESS_BRANCHES, seed, args.max_tokens
            ),
        })
    return requests


def measurement_requests(args):
    context = shared_context(args.context_tokens, 'long')
    requests = []
    for branches in MEASURE_BRANCHES:
        bodies = []
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            prompt = (
                context
                + '\n\nTask: exercise deterministic decode. Begin with exactly '
                + FORK_MARKER
                + ' on the first line. On the next line, emit the positive integers from 1 '
                'upward, separated by single spaces, until generation stops. Emit no commentary.'
            )
            bodies.append(tree_body(args.model, prompt, branches, seed, args.max_tokens))
        requests.append({'B': branches, 'bodies': bodies})
    return requests


def expected_environment(mode):
    return {'AUTOTREE_SHARED_READ': 1 if mode == 'on' else 0}


def print_dry_run(args):
    payload = {
        'dry_run': True,
        'job': 'effect' if args.measure else 'correctness',
        'mode': args.mode,
        'server_environment_expected': expected_environment(args.mode),
        'endpoint': endpoint_url(args.base_url),
        'requests': measurement_requests(args) if args.measure else correctness_requests(args),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return EXIT_PASS


def post_json(url, body, api_key, timeout):
    encoded = json.dumps(
        body, ensure_ascii=False, separators=(',', ':')
    ).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + api_key,
        'User-Agent': 'autotree-shared-read-verifier/1',
    }
    request = urllib.request.Request(url, data=encoded, headers=headers, method='POST')
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.getcode()
        elapsed = time.perf_counter() - started
        return json.loads(raw.decode('utf-8')), status, elapsed, None
    except urllib.error.HTTPError as error:
        elapsed = time.perf_counter() - started
        raw = error.read()
        try:
            detail = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            detail = raw.decode('utf-8', errors='replace')
        return None, error.code, elapsed, {'kind': 'http', 'detail': detail}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        elapsed = time.perf_counter() - started
        return None, None, elapsed, {'kind': 'transport', 'detail': str(error)}


def normalize_branch_id(value):
    text = str(value)
    if len(text) > 1 and text[0].lower() == 'b' and text[1:].isdigit():
        return text[1:]
    return text


def normalize_token_ids(value):
    if value is None or not isinstance(value, list):
        return None
    output = []
    for token_id in value:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            return None
        output.append(token_id)
    return output


def normalize_branch_map(mapping):
    if not isinstance(mapping, dict):
        return None
    output = {}
    for branch_id, value in mapping.items():
        if isinstance(value, dict):
            ids = value.get('output_ids')
            if ids is None:
                ids = value.get('token_ids')
        else:
            ids = value
        output[normalize_branch_id(branch_id)] = normalize_token_ids(ids)
    return output or None


def last_snapshot(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict):
                return item
    return None


def extract_branch_token_ids(response):
    if not isinstance(response, dict):
        return None, None
    tree = response.get('tree')
    if isinstance(tree, dict):
        for key in ('branch_token_ids', 'per_branch_token_ids', 'output_ids_per_branch'):
            mapping = normalize_branch_map(tree.get(key))
            if mapping is not None:
                return mapping, 'tree.' + key
        mapping = normalize_branch_map(tree.get('branches'))
        if mapping is not None:
            return mapping, 'tree.branches'
    for key in ('branch_token_ids', 'per_branch_token_ids', 'output_ids_per_branch'):
        mapping = normalize_branch_map(response.get(key))
        if mapping is not None:
            return mapping, key
    containers = [('', response)]
    meta = response.get('meta_info')
    if isinstance(meta, dict):
        containers.append(('meta_info.', meta))
    for prefix, container in containers:
        for key in ('autotree', 'tree_trace', 'tree_snapshot'):
            snapshot = last_snapshot(container.get(key))
            if isinstance(snapshot, dict):
                mapping = normalize_branch_map(snapshot.get('branches'))
                if mapping is not None:
                    return mapping, prefix + key + '.branches'
    return None, None


def extract_winner_text(response):
    try:
        value = response['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        value = response.get('winner_text') if isinstance(response, dict) else None
    return value if isinstance(value, str) else None


def extract_tree(response):
    value = response.get('tree') if isinstance(response, dict) else None
    return value if isinstance(value, dict) else None


def extract_usage(response):
    value = response.get('usage') if isinstance(response, dict) else None
    return value if isinstance(value, dict) else None


def branch_ids_complete(mapping, branches):
    if not isinstance(mapping, dict):
        return False
    expected = {str(index) for index in range(branches)}
    return set(mapping) == expected and all(isinstance(mapping[key], list) for key in expected)


def write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def run_correctness(args):
    url = endpoint_url(args.base_url)
    tasks = []
    all_passed = True
    for request_item in correctness_requests(args):
        response, status, elapsed, error = post_json(
            url, request_item['body'], args.api_key, args.timeout
        )
        winner_text = extract_winner_text(response)
        branch_token_ids, token_source = extract_branch_token_ids(response)
        tree = extract_tree(response)
        usage = extract_usage(response)
        observed_branch_count = tree.get('branch_count') if tree else None
        complete = branch_ids_complete(branch_token_ids, CORRECTNESS_BRANCHES)
        passed = (
            error is None
            and winner_text is not None
            and complete
            and observed_branch_count == CORRECTNESS_BRANCHES
        )
        reasons = []
        if error is not None:
            reasons.append('request_error')
        if winner_text is None:
            reasons.append('winner_text_missing')
        if not complete:
            reasons.append('per_branch_token_ids_missing_or_incomplete')
        if observed_branch_count != CORRECTNESS_BRANCHES:
            reasons.append('observed_branch_count_not_4')
        actual_prompt_tokens = usage.get('prompt_tokens') if usage else None
        task = {
            'task_id': request_item['task_id'],
            'name': request_item['name'],
            'regime': {
                'model': args.model,
                'context': {
                    'kind': 'medium_shared_prefix',
                    'target_approx_tokens': CORRECTNESS_CONTEXT_TOKENS,
                    'actual_prompt_tokens': actual_prompt_tokens,
                },
                'B': CORRECTNESS_BRANCHES,
                'seed': request_item['seed'],
                'temperature': 0.0,
            },
            'request_body': request_item['body'],
            'http_status': status,
            'wall_seconds': elapsed,
            'winner_text': winner_text,
            'branch_token_ids': branch_token_ids,
            'branch_token_ids_source': token_source,
            'tree': tree,
            'usage': usage,
            'error': error,
            'capture_status': 'PASS' if passed else 'FAIL',
            'capture_reasons': reasons,
        }
        tasks.append(task)
        all_passed = all_passed and passed
        print(
            '[%s] %s %s (%.3fs)%s'
            % (
                args.mode,
                task['capture_status'],
                request_item['task_id'],
                elapsed,
                '' if not reasons else ': ' + ', '.join(reasons),
            )
        )

    payload = {
        'schema_version': 1,
        'job': 'correctness',
        'mode': args.mode,
        'server_environment_expected': expected_environment(args.mode),
        'endpoint': url,
        'captured_at_unix': time.time(),
        'regime': {
            'model': args.model,
            'context': {
                'kind': 'medium_shared_prefix',
                'target_approx_tokens': CORRECTNESS_CONTEXT_TOKENS,
            },
            'B': CORRECTNESS_BRANCHES,
            'seed_base': args.seed,
            'temperature': 0.0,
        },
        'tasks': tasks,
        'capture_verdict': 'PASS' if all_passed else 'FAIL',
    }
    write_json(args.out, payload)
    print('OVERALL CAPTURE %s -> %s' % (payload['capture_verdict'], args.out))
    return EXIT_PASS if all_passed else EXIT_FAIL


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def task_map(payload):
    tasks = payload.get('tasks') if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        raise ValueError('missing tasks list')
    output = {}
    for task in tasks:
        task_id = task.get('task_id') if isinstance(task, dict) else None
        if not isinstance(task_id, str) or task_id in output:
            raise ValueError('invalid or duplicate task_id')
        output[task_id] = task
    return output


def compare_files(on_path, off_path):
    try:
        on_payload = load_json(on_path)
        off_payload = load_json(off_path)
        if on_payload.get('job') != 'correctness' or off_payload.get('job') != 'correctness':
            raise ValueError('both inputs must be correctness captures')
        on_tasks = task_map(on_payload)
        off_tasks = task_map(off_payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print('COMPARE ERROR: %s' % error)
        return EXIT_USAGE

    task_ids = sorted(set(on_tasks) | set(off_tasks))
    overall = True
    for task_id in task_ids:
        left = on_tasks.get(task_id)
        right = off_tasks.get(task_id)
        reasons = []
        if left is None or right is None:
            reasons.append('task_missing')
        else:
            if left.get('request_body') != right.get('request_body'):
                reasons.append('request_body_differs')
            left_text = left.get('winner_text')
            right_text = right.get('winner_text')
            if not isinstance(left_text, str) or not isinstance(right_text, str):
                reasons.append('winner_text_missing')
            elif left_text.encode('utf-8') != right_text.encode('utf-8'):
                reasons.append('winner_text_differs')
            left_ids = left.get('branch_token_ids')
            right_ids = right.get('branch_token_ids')
            expected_b = left.get('regime', {}).get('B')
            if not isinstance(expected_b, int) or not branch_ids_complete(left_ids, expected_b):
                reasons.append('on_branch_token_ids_missing_or_incomplete')
            if not isinstance(expected_b, int) or not branch_ids_complete(right_ids, expected_b):
                reasons.append('off_branch_token_ids_missing_or_incomplete')
            if left_ids is not None and right_ids is not None and left_ids != right_ids:
                reasons.append('per_branch_token_ids_differ')
        passed = not reasons
        overall = overall and passed
        print(
            '%s %s%s'
            % ('PASS' if passed else 'FAIL', task_id,
               '' if passed else ': ' + ', '.join(reasons))
        )

    verdict = 'PASS' if overall and task_ids else 'FAIL'
    print('OVERALL %s (%s vs %s)' % (verdict, on_path, off_path))
    return EXIT_PASS if verdict == 'PASS' else EXIT_FAIL


def initial_log_offset(path):
    if not path:
        return None
    try:
        with open(path, 'rb') as handle:
            handle.seek(0, 2)
            return handle.tell()
    except OSError:
        return 0


def read_log_growth(path, offset):
    if not path:
        return '', offset, None
    try:
        with open(path, 'rb') as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if offset is None or offset > size:
                offset = 0
            handle.seek(offset)
            data = handle.read()
            new_offset = handle.tell()
        return data.decode('utf-8', errors='replace'), new_offset, None
    except OSError as error:
        return '', offset, str(error)


def parse_gen_throughputs(text):
    marker = 'gen throughput (token/s):'
    values = []
    for line in text.splitlines():
        position = line.find(marker)
        if position < 0:
            continue
        tail = line[position + len(marker):].strip()
        token = tail.split(',', 1)[0].strip().split(' ', 1)[0]
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def aggregate_branch_tokens(response, expected_branches):
    tree = extract_tree(response)
    mapping = tree.get('tokens_spent_per_branch') if tree else None
    if not isinstance(mapping, dict) or not mapping:
        return None
    if tree.get('branch_count') != expected_branches:
        return None
    normalized = {normalize_branch_id(key): value for key, value in mapping.items()}
    expected = {str(index) for index in range(expected_branches)}
    if set(normalized) != expected:
        return None
    total = 0
    for value in normalized.values():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        total += value
    return total


def run_measurement(args):
    url = endpoint_url(args.base_url)
    rows = []
    all_requests_succeeded = True
    log_offset = initial_log_offset(args.server_log)

    for item in measurement_requests(args):
        branches = item['B']
        group_started = time.perf_counter()
        repeats = []
        aggregate_tokens = 0
        aggregate_known = True
        log_start = log_offset
        for repeat_index, body in enumerate(item['bodies']):
            response, status, elapsed, error = post_json(
                url, body, args.api_key, args.timeout
            )
            branch_tokens = aggregate_branch_tokens(response, branches)
            if branch_tokens is None:
                aggregate_known = False
            else:
                aggregate_tokens += branch_tokens
            if error is not None:
                all_requests_succeeded = False
            repeats.append({
                'repeat': repeat_index,
                'seed': body['seed'],
                'request_body': body,
                'http_status': status,
                'wall_seconds': elapsed,
                'aggregate_branch_tokens': branch_tokens,
                'usage': extract_usage(response),
                'tree': extract_tree(response),
                'error': error,
            })
        group_wall = time.perf_counter() - group_started

        if args.server_log and args.log_settle_seconds:
            time.sleep(args.log_settle_seconds)
        log_text, log_offset, log_error = read_log_growth(args.server_log, log_start)
        log_values = parse_gen_throughputs(log_text)
        server_tps = log_values[-1] if log_values else None
        wall_tps = (
            aggregate_tokens / group_wall
            if aggregate_known and group_wall > 0
            else None
        )
        if wall_tps is not None:
            reported_tps = wall_tps
            throughput_source = 'wall_from_tokens_spent_per_branch'
        elif server_tps is not None:
            reported_tps = server_tps
            throughput_source = 'server_log_last_appended_sample'
        else:
            reported_tps = None
            throughput_source = None
            all_requests_succeeded = False

        row = {
            'regime': {
                'model': args.model,
                'context': {
                    'kind': 'long_shared_prefix',
                    'target_approx_tokens': args.context_tokens,
                    'actual_prompt_tokens': (
                        repeats[0]['usage'].get('prompt_tokens')
                        if repeats and isinstance(repeats[0].get('usage'), dict)
                        else None
                    ),
                },
                'B': branches,
                'seed_base': args.seed,
                'temperature': 0.0,
                'repeats': args.repeats,
            },
            'wall_seconds': group_wall,
            'aggregate_branch_tokens': aggregate_tokens if aggregate_known else None,
            'wall_aggregate_tokens_per_second': wall_tps,
            'server_gen_tokens_per_second': server_tps,
            'aggregate_tokens_per_second': reported_tps,
            'throughput_source': throughput_source,
            'server_log_error': log_error,
            'repeats': repeats,
        }
        rows.append(row)
        print(
            '[%s] B=%d wall=%.3fs aggregate_tok/s=%s source=%s'
            % (
                args.mode,
                branches,
                group_wall,
                'null' if reported_tps is None else '%.3f' % reported_tps,
                throughput_source or 'null',
            )
        )

    by_b = {row['regime']['B']: row for row in rows}
    tps_8 = by_b[8]['aggregate_tokens_per_second']
    tps_16 = by_b[16]['aggregate_tokens_per_second']
    source_8 = by_b[8]['throughput_source']
    source_16 = by_b[16]['throughput_source']
    ratio = (
        tps_16 / tps_8
        if (
            tps_8 not in (None, 0)
            and tps_16 is not None
            and source_8 is not None
            and source_8 == source_16
        )
        else None
    )
    payload = {
        'schema_version': 1,
        'job': 'effect',
        'mode': args.mode,
        'server_environment_expected': expected_environment(args.mode),
        'endpoint': url,
        'captured_at_unix': time.time(),
        'regime': {
            'model': args.model,
            'context': {
                'kind': 'long_shared_prefix',
                'target_approx_tokens': args.context_tokens,
            },
            'B': list(MEASURE_BRANCHES),
            'seed_base': args.seed,
            'temperature': 0.0,
            'repeats': args.repeats,
        },
        'results': rows,
        'B16_over_B8_aggregate_tokens_per_second': ratio,
        'B16_over_B8_throughput_source': source_8 if ratio is not None else None,
        'measurement_status': (
            'PASS' if all_requests_succeeded and ratio is not None else 'FAIL'
        ),
    }
    write_json(args.out, payload)
    print(
        'B=16/B=8 aggregate tok/s ratio=%s'
        % ('null' if ratio is None else '%.6f' % ratio)
    )
    print('MEASUREMENT %s -> %s' % (payload['measurement_status'], args.out))
    return EXIT_PASS if payload['measurement_status'] == 'PASS' else EXIT_FAIL


def main():
    args = parse_args()
    if args.compare:
        return compare_files(args.compare[0], args.compare[1])
    if args.dry_run:
        return print_dry_run(args)
    if args.measure:
        return run_measurement(args)
    return run_correctness(args)


if __name__ == '__main__':
    raise SystemExit(main())
