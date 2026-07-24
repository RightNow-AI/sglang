#!/usr/bin/env python3
'''Create or verify a content-addressed manifest for task JSONL files.'''

import argparse
import hashlib
import json
import os
import re
import sys


MANIFEST_VERSION = 1
SHA256_PATTERN = re.compile(r'[0-9a-f]{64}\Z')


class ManifestError(RuntimeError):
    pass


def inspect_jsonl(path):
    digest = hashlib.sha256()
    ids = []
    seen_ids = set()
    try:
        with open(path, 'rb') as source:
            for line_number, raw_line in enumerate(source, 1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise ManifestError(
                        '{}:{} is a blank JSONL line'.format(path, line_number)
                    )
                try:
                    record = json.loads(raw_line.decode('utf-8'))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ManifestError(
                        '{}:{} is invalid UTF-8 JSON: {}'.format(
                            path, line_number, exc
                        )
                    ) from exc
                if not isinstance(record, dict):
                    raise ManifestError(
                        '{}:{} is not a JSON object'.format(path, line_number)
                    )
                record_id = record.get('id')
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ManifestError(
                        '{}:{} has no non-empty string id'.format(path, line_number)
                    )
                if record_id in seen_ids:
                    raise ManifestError(
                        '{}:{} repeats id {!r}'.format(path, line_number, record_id)
                    )
                seen_ids.add(record_id)
                ids.append(record_id)
    except OSError as exc:
        raise ManifestError('cannot read {}: {}'.format(path, exc)) from exc
    return {
        'sha256': digest.hexdigest(),
        'line_count': len(ids),
        'ids': ids,
    }


def portable_relative_path(path, manifest_dir):
    absolute = os.path.abspath(path)
    try:
        stored = os.path.relpath(absolute, manifest_dir)
    except ValueError:
        stored = absolute
    return stored.replace(os.sep, '/')


def build_manifest(paths, output_path):
    manifest_dir = os.path.dirname(os.path.abspath(output_path))
    records = []
    seen_paths = set()
    for path in paths:
        absolute = os.path.abspath(path)
        canonical = os.path.normcase(absolute)
        if canonical in seen_paths:
            raise ManifestError('duplicate input path: {}'.format(path))
        seen_paths.add(canonical)
        details = inspect_jsonl(absolute)
        records.append(
            {
                'path': portable_relative_path(absolute, manifest_dir),
                'sha256': details['sha256'],
                'line_count': details['line_count'],
                'ids': details['ids'],
            }
        )
    records.sort(key=lambda record: record['path'])
    return {'manifest_version': MANIFEST_VERSION, 'files': records}


def write_manifest(path, manifest):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + '.tmp'
    try:
        with open(temporary, 'w', encoding='utf-8', newline='\n') as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write('\n')
        os.replace(temporary, path)
    except OSError as exc:
        raise ManifestError('cannot write {}: {}'.format(path, exc)) from exc


def load_manifest(path):
    try:
        with open(path, encoding='utf-8') as source:
            manifest = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError('cannot load {}: {}'.format(path, exc)) from exc
    if not isinstance(manifest, dict):
        raise ManifestError('manifest root must be a JSON object')
    if manifest.get('manifest_version') != MANIFEST_VERSION:
        raise ManifestError(
            'unsupported manifest_version {!r}'.format(
                manifest.get('manifest_version')
            )
        )
    files = manifest.get('files')
    if not isinstance(files, list):
        raise ManifestError('manifest files must be a list')
    return files


def validate_entry(entry, index):
    if not isinstance(entry, dict):
        raise ManifestError('files[{}] must be an object'.format(index))
    path = entry.get('path')
    sha256 = entry.get('sha256')
    line_count = entry.get('line_count')
    ids = entry.get('ids')
    if not isinstance(path, str) or not path:
        raise ManifestError('files[{}].path must be a string'.format(index))
    if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
        raise ManifestError('files[{}].sha256 is invalid'.format(index))
    if (
        not isinstance(line_count, int)
        or isinstance(line_count, bool)
        or line_count < 0
    ):
        raise ManifestError('files[{}].line_count is invalid'.format(index))
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise ManifestError('files[{}].ids must be a string list'.format(index))
    if len(ids) != line_count:
        raise ManifestError(
            'files[{}] ids length does not match line_count'.format(index)
        )
    return path, {'sha256': sha256, 'line_count': line_count, 'ids': ids}


def first_id_difference(expected, actual):
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return 'index {} expected {!r}, got {!r}'.format(index, left, right)
    if len(expected) != len(actual):
        return 'expected {} ids, got {}'.format(len(expected), len(actual))
    return 'unknown difference'


def verify_manifest(path):
    entries = load_manifest(path)
    manifest_dir = os.path.dirname(os.path.abspath(path))
    drifted = 0
    seen_paths = set()
    for index, entry in enumerate(entries):
        stored_path, expected = validate_entry(entry, index)
        native_path = stored_path.replace('/', os.sep)
        if not os.path.isabs(native_path):
            native_path = os.path.join(manifest_dir, native_path)
        canonical = os.path.normcase(os.path.abspath(native_path))
        if canonical in seen_paths:
            raise ManifestError('manifest repeats path {!r}'.format(stored_path))
        seen_paths.add(canonical)
        try:
            actual = inspect_jsonl(native_path)
        except ManifestError as exc:
            print('DRIFT {}: {}'.format(stored_path, exc))
            drifted += 1
            continue

        differences = []
        if actual['sha256'] != expected['sha256']:
            differences.append(
                'sha256 expected {}, got {}'.format(
                    expected['sha256'], actual['sha256']
                )
            )
        if actual['line_count'] != expected['line_count']:
            differences.append(
                'line_count expected {}, got {}'.format(
                    expected['line_count'], actual['line_count']
                )
            )
        if actual['ids'] != expected['ids']:
            differences.append(
                'ids {}'.format(
                    first_id_difference(expected['ids'], actual['ids'])
                )
            )
        if differences:
            print('DRIFT {}: {}'.format(stored_path, '; '.join(differences)))
            drifted += 1
        else:
            print('OK {}'.format(stored_path))

    if drifted:
        print('FAIL: {} of {} files drifted'.format(drifted, len(entries)))
        return 1
    print('PASS: {} files verified'.format(len(entries)))
    return 0


def parse_args(argv=None):
    default_output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data', 'task_manifest.json'
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jsonl', nargs='*', help='task JSONL files to lock')
    parser.add_argument('-o', '--output', default=default_output)
    parser.add_argument(
        '--verify', metavar='MANIFEST', help='verify files recorded by a manifest'
    )
    args = parser.parse_args(argv)
    if args.verify and args.jsonl:
        parser.error('JSONL paths cannot be used with --verify')
    if not args.verify and not args.jsonl:
        parser.error('provide at least one task JSONL or use --verify')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.verify:
            return verify_manifest(args.verify)
        manifest = build_manifest(args.jsonl, args.output)
        write_manifest(args.output, manifest)
    except ManifestError as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        return 1
    print(
        '{}: {} files locked'.format(
            os.path.abspath(args.output), len(manifest['files'])
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
