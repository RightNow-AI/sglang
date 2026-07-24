#!/usr/bin/env python3
'''Build a deterministic GPQA Diamond multiple-choice JSONL task set.'''

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ROWS_API = 'https://datasets-server.huggingface.co/rows'
DATASET = 'Idavidrein/gpqa'
CONFIG = 'gpqa_diamond'
SPLIT = 'train'
EXPECTED_ROWS = 198
USER_AGENT = 'sglang-gpqa-task-builder/1.0'
LETTERS = 'ABCD'


class DownloadError(RuntimeError):
    pass


def load_hf_token():
    '''Return an available Hugging Face token without printing it.'''
    for name in ('HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        token = os.environ.get(name, '').strip()
        if token:
            return token

    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        candidates = (os.path.join(hf_home, 'token'),)
    else:
        candidates = (
            os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'token'),
            os.path.join(os.path.expanduser('~'), '.huggingface', 'token'),
        )
    for path in candidates:
        try:
            with open(path, encoding='utf-8') as handle:
                token = handle.read().strip()
        except OSError:
            continue
        if token:
            return token
    return None


def request_json(url, timeout, token):
    headers = {'User-Agent': USER_AGENT}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode('utf-8', 'replace').strip()
            if exc.code in (401, 403):
                raise DownloadError(
                    'GPQA access was denied by Hugging Face. Accept the dataset '
                    'access terms and provide an authorized HF_TOKEN'
                ) from exc
            transient = exc.code == 429 or 500 <= exc.code <= 504
            if transient and attempt < 4:
                retry_after = exc.headers.get('Retry-After')
                try:
                    delay = float(retry_after) if retry_after else 2 ** attempt
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(min(max(delay, 0), 30))
                continue
            raise DownloadError(
                'HTTP {} for {}{}'.format(
                    exc.code, url, ': ' + detail if detail else ''
                )
            ) from exc
        except urllib.error.URLError as exc:
            raise DownloadError(
                'network error for {}: {}'.format(url, exc.reason)
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DownloadError('invalid JSON from {}: {}'.format(url, exc)) from exc


def fetch_page(offset, length, timeout, token):
    query = urllib.parse.urlencode(
        {
            'dataset': DATASET,
            'config': CONFIG,
            'split': SPLIT,
            'offset': offset,
            'length': length,
        }
    )
    payload = request_json(ROWS_API + '?' + query, timeout, token)
    rows = payload.get('rows')
    total = payload.get('num_rows_total')
    if not isinstance(rows, list):
        raise DownloadError('dataset-server response has no rows list')
    if total is not None and (
        not isinstance(total, int) or isinstance(total, bool) or total < 0
    ):
        raise DownloadError('dataset-server returned invalid num_rows_total')
    return payload


def download_rows(page_size=100, timeout=60):
    '''Download all rows and return them sorted by dataset-server row_idx.'''
    token = load_hf_token()
    payload = fetch_page(0, page_size, timeout, token)
    total = payload.get('num_rows_total')
    if total is not None and total != EXPECTED_ROWS:
        raise DownloadError(
            '{} now reports {} rows; expected the locked {}-item set'.format(
                CONFIG, total, EXPECTED_ROWS
            )
        )

    entries = []
    seen = set()
    offset = 0
    while True:
        page = payload['rows']
        for position, entry in enumerate(page):
            if not isinstance(entry, dict) or not isinstance(entry.get('row'), dict):
                raise DownloadError('dataset-server returned a malformed row')
            row_idx = entry.get('row_idx', offset + position)
            if (
                not isinstance(row_idx, int)
                or isinstance(row_idx, bool)
                or row_idx < 0
                or row_idx in seen
            ):
                raise DownloadError('invalid or duplicate row_idx {!r}'.format(row_idx))
            seen.add(row_idx)
            entries.append((row_idx, entry['row']))

        offset += len(page)
        if total is not None and offset >= total:
            break
        if not page:
            raise DownloadError('pagination ended early at offset {}'.format(offset))
        if total is None and len(page) < page_size:
            break
        payload = fetch_page(offset, page_size, timeout, token)
        next_total = payload.get('num_rows_total')
        if total is None:
            total = next_total
            if total is not None and total != EXPECTED_ROWS:
                raise DownloadError(
                    '{} now reports {} rows; expected {}'.format(
                        CONFIG, total, EXPECTED_ROWS
                    )
                )
        elif next_total is not None and next_total != total:
            raise DownloadError('num_rows_total changed during pagination')

    if len(entries) != EXPECTED_ROWS:
        raise DownloadError(
            'downloaded {} rows; expected {}'.format(len(entries), EXPECTED_ROWS)
        )
    entries.sort(key=lambda item: item[0])
    return entries


def normalized_field(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def find_field(row, aliases, required=True):
    for alias in aliases:
        if alias in row:
            return alias
    fields = {
        normalized_field(key): key for key in row if isinstance(key, str)
    }
    for alias in aliases:
        actual = fields.get(normalized_field(alias))
        if actual is not None:
            return actual
    if required:
        raise ValueError(
            'none of {} exist in fields {}'.format(aliases, sorted(row))
        )
    return None


def discover_fields(first_row):
    return {
        'question': find_field(first_row, ('Question', 'question', 'prompt')),
        'correct': find_field(
            first_row,
            ('Correct Answer', 'correct_answer', 'correct answer', 'correct'),
        ),
        'incorrect_1': find_field(
            first_row,
            ('Incorrect Answer 1', 'incorrect_answer_1', 'incorrect 1', 'distractor1'),
        ),
        'incorrect_2': find_field(
            first_row,
            ('Incorrect Answer 2', 'incorrect_answer_2', 'incorrect 2', 'distractor2'),
        ),
        'incorrect_3': find_field(
            first_row,
            ('Incorrect Answer 3', 'incorrect_answer_3', 'incorrect 3', 'distractor3'),
        ),
        'domain': find_field(
            first_row,
            ('Subdomain', 'subdomain', 'Domain', 'domain', 'subject', 'category'),
            required=False,
        ),
    }


def nonempty_string(value, field, row_idx):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            'row {} field {!r} is not a non-empty string'.format(row_idx, field)
        )
    return value.strip()


def shuffled_options(question, correct, incorrect):
    options = [(correct, True)] + [(value, False) for value in incorrect]
    if len(set(value for value, _ in options)) != len(options):
        raise ValueError('question has duplicate answer options')
    seed = int.from_bytes(
        hashlib.sha256(question.encode('utf-8')).digest(), 'big'
    )
    random.Random(seed).shuffle(options)
    return options


def format_prompt(question, options):
    lines = [question, '']
    lines.extend(
        '{}. {}'.format(letter, option)
        for letter, (option, _) in zip(LETTERS, options)
    )
    lines.extend(('', 'Answer with the letter A, B, C, or D.'))
    return '\n'.join(lines)


def build_records(entries):
    if not entries:
        raise ValueError('GPQA Diamond returned no rows')
    fields = discover_fields(entries[0][1])
    records = []
    for number, (row_idx, row) in enumerate(entries, 1):
        question = nonempty_string(row.get(fields['question']), fields['question'], row_idx)
        correct = nonempty_string(row.get(fields['correct']), fields['correct'], row_idx)
        incorrect = [
            nonempty_string(row.get(fields[name]), fields[name], row_idx)
            for name in ('incorrect_1', 'incorrect_2', 'incorrect_3')
        ]
        options = shuffled_options(question, correct, incorrect)
        gold = next(
            letter
            for letter, (_, is_correct) in zip(LETTERS, options)
            if is_correct
        )
        meta = {'set': CONFIG}
        if fields['domain'] is not None:
            domain = row.get(fields['domain'])
            if domain is not None and str(domain).strip():
                meta['domain'] = str(domain).strip()
        records.append(
            {
                'id': 'gpqa-{:04d}'.format(number),
                'prompt': format_prompt(question, options),
                'gold': gold,
                'meta': meta,
            }
        )
    return records, fields


def write_jsonl(path, records):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8', newline='\n') as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')))
            output.write('\n')
    os.replace(temporary, path)


def format_fields(fields):
    return ', '.join(
        '{}={}'.format(logical, actual)
        for logical, actual in fields.items()
        if actual is not None
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'gpqa_diamond.jsonl',
        ),
    )
    parser.add_argument('--page-size', type=int, default=100)
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args(argv)
    if not 1 <= args.page_size <= 100:
        parser.error('--page-size must be between 1 and 100')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        entries = download_rows(args.page_size, args.timeout)
        records, fields = build_records(entries)
        write_jsonl(args.output, records)
    except (DownloadError, OSError, ValueError) as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        return 1

    print('source: dataset={} config={} split={}'.format(DATASET, CONFIG, SPLIT))
    print('fields: {}'.format(format_fields(fields)))
    print('{}: {} rows'.format(os.path.abspath(args.output), len(records)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
