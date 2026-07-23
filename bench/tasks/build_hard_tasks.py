#!/usr/bin/env python3
'''Build deterministic MATH-500 hard and AIME JSONL benchmark inputs.'''

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ROWS_API = 'https://datasets-server.huggingface.co/rows'
HUB_API = 'https://huggingface.co/api/datasets/'
USER_AGENT = 'sglang-hard-task-builder/1.0'

MATH_SPEC = {
    'label': 'MATH-500',
    'dataset': 'HuggingFaceH4/MATH-500',
    'fallback': (),
    'config': 'default',
    'split': 'test',
}
AIME_SPEC = {
    'label': 'AIME 1983-2024',
    'dataset': 'qq8933/AIME_1983_2024',
    # The requested Hub repository currently redirects to this repository.
    'fallback': ('di-zhang-fdu/AIME_1983_2024',),
    'config': 'default',
    'split': 'train',
}


class DownloadError(RuntimeError):
    pass


def request_json(url, timeout):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode('utf-8', 'replace').strip()
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


def hub_dataset_id(dataset, timeout):
    '''Resolve a renamed Hub repository using urllib redirect handling.'''
    url = HUB_API + urllib.parse.quote(dataset, safe='/')
    payload = request_json(url, timeout)
    current = payload.get('id')
    return current if isinstance(current, str) and current else dataset


def fetch_page(dataset, spec, offset, length, timeout):
    query = urllib.parse.urlencode(
        {
            'dataset': dataset,
            'config': spec['config'],
            'split': spec['split'],
            'offset': offset,
            'length': length,
        }
    )
    payload = request_json(ROWS_API + '?' + query, timeout)
    rows = payload.get('rows')
    total = payload.get('num_rows_total')
    if not isinstance(rows, list):
        raise DownloadError('dataset-server response has no rows list')
    if total is not None and (not isinstance(total, int) or total < 0):
        raise DownloadError('dataset-server returned invalid num_rows_total')
    return payload


def dataset_candidates(spec, timeout):
    candidates = []
    try:
        candidates.append(hub_dataset_id(spec['dataset'], timeout))
    except DownloadError:
        # The rows API can still work if Hub metadata is temporarily unavailable.
        pass
    candidates.append(spec['dataset'])
    candidates.extend(spec['fallback'])
    return list(dict.fromkeys(candidates))


def download_rows(spec, page_size=100, timeout=60):
    '''Return the active dataset id and rows sorted by original row_idx.'''
    first = None
    source = None
    errors = []
    for candidate in dataset_candidates(spec, timeout):
        try:
            first = fetch_page(candidate, spec, 0, page_size, timeout)
            source = candidate
            break
        except DownloadError as exc:
            errors.append('{}: {}'.format(candidate, exc))
    if first is None:
        raise DownloadError(
            'could not open {}: {}'.format(spec['label'], ' | '.join(errors))
        )

    entries = []
    seen = set()
    total = first.get('num_rows_total')
    offset = 0
    payload = first
    while True:
        page = payload['rows']
        for position, entry in enumerate(page):
            if not isinstance(entry, dict) or not isinstance(entry.get('row'), dict):
                raise DownloadError('dataset-server returned a malformed row')
            row_idx = entry.get('row_idx', offset + position)
            if not isinstance(row_idx, int) or row_idx < 0 or row_idx in seen:
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
        payload = fetch_page(source, spec, offset, page_size, timeout)
        next_total = payload.get('num_rows_total')
        if total is None:
            total = next_total
        elif next_total is not None and next_total != total:
            raise DownloadError('num_rows_total changed during pagination')

    if total is not None and len(entries) != total:
        raise DownloadError(
            'downloaded {} rows but server reported {}'.format(len(entries), total)
        )
    entries.sort(key=lambda item: item[0])
    return source, entries


def normalized_field(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def find_field(row, aliases, required=True):
    for alias in aliases:
        if alias in row:
            return alias
    fields = {normalized_field(key): key for key in row if isinstance(key, str)}
    for alias in aliases:
        if normalized_field(alias) in fields:
            return fields[normalized_field(alias)]
    if required:
        raise ValueError(
            'none of {} exist in fields {}'.format(aliases, sorted(row))
        )
    return None


def nonempty_string(value, field):
    if value is None or not str(value).strip():
        raise ValueError('field {!r} is empty'.format(field))
    return str(value).strip()


def parse_level(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(r'\b([1-5])\b', str(value))
    if match is None:
        raise ValueError('could not parse level from {!r}'.format(value))
    return int(match.group(1))


def parse_year(row, year_field, id_field):
    for field in (year_field, id_field):
        if field is None:
            continue
        value = row.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        match = re.search(r'(?:19|20)\d{2}', str(value))
        if match:
            return int(match.group(0))
    raise ValueError('could not determine AIME year')


def build_math_hard(entries):
    if not entries:
        raise ValueError('MATH-500 returned no rows')
    first = entries[0][1]
    fields = {
        'prompt': find_field(first, ('problem', 'question')),
        'gold': find_field(first, ('answer', 'gold')),
        'level': find_field(first, ('level', 'difficulty')),
        'subject': find_field(first, ('subject', 'type', 'category')),
    }
    selected = []
    for row_idx, row in entries:
        level = parse_level(row.get(fields['level']))
        if level in (4, 5):
            selected.append(
                (
                    row_idx,
                    nonempty_string(row.get(fields['prompt']), fields['prompt']),
                    nonempty_string(row.get(fields['gold']), fields['gold']),
                    level,
                    nonempty_string(row.get(fields['subject']), fields['subject']),
                )
            )
    records = []
    for number, (_, prompt, gold, level, subject) in enumerate(selected, 1):
        records.append(
            {
                'id': 'math_hard-{:04d}'.format(number),
                'prompt': prompt,
                'gold': gold,
                'meta': {
                    'set': 'math_hard',
                    'level': level,
                    'subject': subject,
                },
            }
        )
    return records, fields


def normalize_aime_gold(value, row_idx):
    gold = nonempty_string(value, 'Answer')
    if re.fullmatch(r'\d{1,3}', gold):
        return gold
    alternatives = re.fullmatch(
        r'(\d{1,3})\s+or\s+(\d{1,3})\s+\(both were accepted\)',
        gold,
        flags=re.IGNORECASE,
    )
    if alternatives is not None:
        # The single-gold format cannot encode alternatives, so use the first
        # accepted answer exactly as the source lists it.
        return alternatives.group(1)
    raise ValueError('AIME row {} has invalid answer {!r}'.format(row_idx, gold))


def build_aime(entries):
    if not entries:
        raise ValueError('AIME returned no rows')
    first = entries[0][1]
    fields = {
        'prompt': find_field(first, ('Question', 'problem', 'question')),
        'gold': find_field(first, ('Answer', 'answer', 'gold')),
        'year': find_field(first, ('Year', 'year'), required=False),
        'source_id': find_field(
            first, ('ID', 'id', 'problem_id'), required=False
        ),
    }
    prepared = []
    for row_idx, row in entries:
        prompt = nonempty_string(row.get(fields['prompt']), fields['prompt'])
        gold = normalize_aime_gold(row.get(fields['gold']), row_idx)
        if not 0 <= int(gold) <= 999:
            raise ValueError('AIME row {} answer is out of range'.format(row_idx))
        year = parse_year(row, fields['year'], fields['source_id'])
        prepared.append((row_idx, prompt, gold, year))
    records = []
    for number, (_, prompt, gold, year) in enumerate(prepared, 1):
        records.append(
            {
                'id': 'aime-{:04d}'.format(number),
                'prompt': prompt,
                'gold': gold,
                'meta': {'set': 'aime', 'year': year},
            }
        )
    return records, fields


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
        '--output-dir',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
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
        math_source, math_rows = download_rows(MATH_SPEC, args.page_size, args.timeout)
        aime_source, aime_rows = download_rows(AIME_SPEC, args.page_size, args.timeout)
        math_records, math_fields = build_math_hard(math_rows)
        aime_records, aime_fields = build_aime(aime_rows)
        math_path = os.path.join(args.output_dir, 'math_hard.jsonl')
        aime_path = os.path.join(args.output_dir, 'aime.jsonl')
        write_jsonl(math_path, math_records)
        write_jsonl(aime_path, aime_records)
    except (DownloadError, OSError, ValueError) as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        return 1

    print('{} source: {}'.format(MATH_SPEC['label'], math_source))
    print('{} fields: {}'.format(MATH_SPEC['label'], format_fields(math_fields)))
    print('{}: {} rows'.format(os.path.abspath(math_path), len(math_records)))
    print('{} source: {}'.format(AIME_SPEC['label'], aime_source))
    print('{} fields: {}'.format(AIME_SPEC['label'], format_fields(aime_fields)))
    print('{}: {} rows'.format(os.path.abspath(aime_path), len(aime_records)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
