#!/usr/bin/env python3
"""Measure generated-token economics for AutoTree pruning versus sequential best-of-n."""

import argparse
import collections
import concurrent.futures
import json
import os
import queue
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

ANSWER_SUFFIX = (
    "\nSolve step by step, then give the final numeric answer on the last line as: "
    "Answer: <number>"
)
NUMBER_RE = re.compile(
    r"[-+]?\s*\$?\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)\s*%?"
)


def normalize_number(value):
    """Return a canonical decimal string, or None when value is not numeric."""
    if value is None:
        return None
    cleaned = re.sub(r"\s+", "", str(value))
    cleaned = cleaned.replace("$", "").replace(",", "").replace("%", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned):
        return None
    sign = ""
    if cleaned[0] in "+-":
        sign = cleaned[0]
        cleaned = cleaned[1:]
    if "." in cleaned:
        integer, fraction = cleaned.split(".", 1)
    else:
        integer, fraction = cleaned, ""
    integer = integer.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    normalized = integer if not fraction else integer + "." + fraction
    if sign == "-" and normalized != "0":
        normalized = "-" + normalized
    return normalized


def extract_answer(text):
    """Apply the shared deterministic answer extraction rule."""
    if not isinstance(text, str):
        return None
    marker = "Answer:"
    marker_at = text.rfind(marker)
    if marker_at >= 0:
        match = NUMBER_RE.search(text, marker_at + len(marker))
        return normalize_number(match.group(0)) if match else None
    matches = list(NUMBER_RE.finditer(text))
    if not matches:
        return None
    return normalize_number(matches[-1].group(0))


def majority_vote(answers):
    """Choose the most frequent answer, breaking ties by first occurrence."""
    if not answers:
        return None
    counts = collections.Counter(answers)
    winning_count = max(counts.values())
    for answer in answers:
        if counts[answer] == winning_count:
            return answer
    return None


def parse_seeds(raw):
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--seeds must be a comma-separated list of integers")
    try:
        seeds = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must not contain duplicates")
    return seeds


def load_items(path, offset, limit):
    items = []
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
                raise ValueError("{} line {} is not a JSON object".format(path, line_number))
            for field in ("id", "prompt", "gold"):
                if not isinstance(row.get(field), str):
                    raise ValueError(
                        "{} line {} field {!r} must be a string".format(
                            path, line_number, field
                        )
                    )
            if row["id"] in seen_ids:
                raise ValueError("duplicate item id {!r} in {}".format(row["id"], path))
            normalized_gold = normalize_number(row["gold"])
            if normalized_gold is None:
                raise ValueError(
                    "{} line {} has non-numeric gold {!r}".format(
                        path, line_number, row["gold"]
                    )
                )
            seen_ids.add(row["id"])
            items.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "gold": row["gold"],
                    "normalized_gold": normalized_gold,
                    "item_index": len(items),
                }
            )
    stop = None if limit is None else offset + limit
    selected = items[offset:stop]
    if not selected:
        raise ValueError("--offset/--limit selected no data items")
    return selected


def tree_body(args, item, base_seed):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": item["prompt"] + ANSWER_SUFFIX}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": base_seed + item["item_index"],
        "tree": {
            "policy": "beam",
            "branches": args.branches,
            "budget_tokens": args.branches * args.max_tokens,
        },
    }


def bon_body(args, item, base_seed, sample_index):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": item["prompt"] + ANSWER_SUFFIX}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": base_seed + item["item_index"] * 1000 + sample_index,
    }


def compact_error(value, limit=500):
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def response_detail(payload, raw_text):
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            if key in payload:
                value = payload[key]
                if isinstance(value, dict):
                    value = value.get("message", value)
                return compact_error(value)
    if raw_text:
        return compact_error(raw_text)
    return "no response body"


def decode_json_body(raw):
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return None, text, "empty_response"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, text, "invalid_json_response: {}".format(compact_error(exc))
    if not isinstance(payload, dict):
        return None, text, "json_response_not_object"
    return payload, text, None


def post_json(url, body, timeout):
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        payload, _text, decode_error = decode_json_body(raw)
        return payload, decode_error
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        payload, text, decode_error = decode_json_body(raw)
        detail = response_detail(payload, text)
        error = "http_{}: {}".format(exc.code, detail)
        if decode_error and not raw:
            error = "{} ({})".format(error, decode_error)
        return payload, error
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        label = "timeout" if "timed out" in str(reason).lower() else "url_error"
        return None, "{}: {}".format(label, compact_error(reason))
    except TimeoutError as exc:
        return None, "timeout: {}".format(compact_error(exc))
    except Exception as exc:
        label = "timeout" if "timeout" in type(exc).__name__.lower() else "request_error"
        return None, "{}: {}: {}".format(
            label, type(exc).__name__, compact_error(exc)
        )


def nonnegative_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def completion_content(payload):
    if not isinstance(payload, dict):
        return None, "missing_response_object"
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, "missing_choices_message_content"
    if not isinstance(content, str):
        return None, "non_string_choices_message_content"
    return content, None


def tree_token_report(payload):
    if not isinstance(payload, dict):
        return 0, False
    tree = payload.get("tree")
    if not isinstance(tree, dict):
        return 0, False
    spent = tree.get("tokens_spent_per_branch")
    if not isinstance(spent, dict):
        return 0, False
    total = 0
    valid = True
    for value in spent.values():
        count = nonnegative_int(value)
        if count is None:
            valid = False
        else:
            total += count
    return total, valid


def bon_token_report(payload):
    if not isinstance(payload, dict):
        return 0, False
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, False
    count = nonnegative_int(usage.get("completion_tokens"))
    return (count, True) if count is not None else (0, False)


def tree_metadata(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), dict):
        return None, None
    tree = payload["tree"]
    pruned_count = nonnegative_int(tree.get("pruned_count"))
    winner_branch_id = tree.get("winner_branch_id")
    if winner_branch_id is not None:
        winner_branch_id = str(winner_branch_id)
    return pruned_count, winner_branch_id


def add_error(errors, error):
    if error and error not in errors:
        errors.append(error)


def run_tree_item(args, item, base_seed):
    started = time.perf_counter()
    url = args.base_url.rstrip("/") + "/v1/tree/completions"
    payload, request_error = post_json(
        url, tree_body(args, item, base_seed), args.timeout
    )
    generated_tokens, token_report_valid = tree_token_report(payload)
    content, content_error = completion_content(payload)
    pruned_count, winner_branch_id = tree_metadata(payload)
    errors = []
    add_error(errors, request_error)
    if request_error is None:
        add_error(errors, content_error)
        if generated_tokens == 0:
            add_error(errors, "zero_tokens")
        elif not token_report_valid:
            add_error(errors, "invalid_token_report")
    extracted = extract_answer(content)
    return {
        "bon_parallel": args.bon_parallel,
        "id": item["id"],
        "mode": "tree",
        "seed": base_seed,
        "item_index": item["item_index"],
        "correct": not errors and extracted == item["normalized_gold"],
        "extracted": extracted,
        "gold": item["gold"],
        "generated_tokens": generated_tokens,
        "wall_s": round(time.perf_counter() - started, 6),
        "pruned_count": pruned_count,
        "winner_branch_id": winner_branch_id,
        "error": "; ".join(errors) if errors else None,
    }


def run_bon_item(args, item, base_seed):
    started = time.perf_counter()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    answers = []
    generated_tokens = 0
    errors = []
    sample_results = None
    if args.bon_parallel:

        def request_sample(sample_index):
            return post_json(
                url, bon_body(args, item, base_seed, sample_index), args.timeout
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.branches
        ) as executor:
            sample_results = list(
                executor.map(request_sample, range(args.branches))
            )
    for sample_index in range(args.branches):
        if sample_results is None:
            payload, request_error = post_json(
                url, bon_body(args, item, base_seed, sample_index), args.timeout
            )
        else:
            payload, request_error = sample_results[sample_index]
        sample_tokens, token_report_valid = bon_token_report(payload)
        generated_tokens += sample_tokens
        content, content_error = completion_content(payload)
        answers.append(extract_answer(content))
        if request_error is not None:
            add_error(errors, "sample_{}:{}".format(sample_index, request_error))
        else:
            if content_error:
                add_error(
                    errors, "sample_{}:{}".format(sample_index, content_error)
                )
            if sample_tokens == 0:
                add_error(errors, "zero_tokens")
            elif not token_report_valid:
                add_error(
                    errors,
                    "sample_{}:invalid_token_report".format(sample_index),
                )
    extracted = majority_vote(answers)
    return {
        "bon_parallel": args.bon_parallel,
        "id": item["id"],
        "mode": "bon",
        "seed": base_seed,
        "item_index": item["item_index"],
        "correct": not errors and extracted == item["normalized_gold"],
        "extracted": extracted,
        "gold": item["gold"],
        "generated_tokens": generated_tokens,
        "wall_s": round(time.perf_counter() - started, 6),
        "error": "; ".join(errors) if errors else None,
    }


def error_record(mode, item, base_seed, bon_parallel, exc):
    record = {
        "bon_parallel": bon_parallel,
        "id": item["id"],
        "mode": mode,
        "seed": base_seed,
        "item_index": item["item_index"],
        "correct": False,
        "extracted": None,
        "gold": item["gold"],
        "generated_tokens": 0,
        "wall_s": 0.0,
        "error": "internal_error:{}:{}".format(
            type(exc).__name__, compact_error(exc)
        ),
    }
    if mode == "tree":
        record["pruned_count"] = None
        record["winner_branch_id"] = None
    return record


def record_key(record):
    try:
        return str(record["mode"]), int(record["seed"]), str(record["id"])
    except (KeyError, TypeError, ValueError):
        return None


def valid_resume_record(record):
    required = (
        "id",
        "mode",
        "seed",
        "correct",
        "extracted",
        "gold",
        "generated_tokens",
        "wall_s",
        "error",
    )
    if any(field not in record for field in required):
        return False
    key = record_key(record)
    if key is None or key[0] not in ("tree", "bon"):
        return False
    if not isinstance(record.get("id"), str):
        return False
    if isinstance(record.get("seed"), bool) or not isinstance(record.get("seed"), int):
        return False
    if not isinstance(record.get("correct"), bool):
        return False
    if record.get("extracted") is not None and not isinstance(
        record.get("extracted"), str
    ):
        return False
    if not isinstance(record.get("gold"), str):
        return False
    if nonnegative_int(record.get("generated_tokens")) is None:
        return False
    wall_s = record.get("wall_s")
    if isinstance(wall_s, bool) or not isinstance(wall_s, (int, float)):
        return False
    if wall_s != wall_s or wall_s < 0 or wall_s >= float("inf"):
        return False
    if record.get("error") is not None and not isinstance(record.get("error"), str):
        return False
    if key[0] == "tree":
        if "pruned_count" not in record or "winner_branch_id" not in record:
            return False
        pruned = record.get("pruned_count")
        if pruned is not None and nonnegative_int(pruned) is None:
            return False
        winner = record.get("winner_branch_id")
        if winner is not None and not isinstance(winner, str):
            return False
    return True


def load_existing_records(path):
    records = {}
    invalid_lines = 0
    duplicate_lines = 0
    if not os.path.exists(path):
        return records, invalid_lines, duplicate_lines
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(record, dict):
                invalid_lines += 1
                continue
            key = record_key(record)
            if key is None or not valid_resume_record(record):
                invalid_lines += 1
                continue
            if key in records:
                duplicate_lines += 1
                continue
            records[key] = record
    return records, invalid_lines, duplicate_lines


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_append_boundary(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        last_byte = handle.read(1)
    if last_byte != b"\n":
        with open(path, "ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def append_record(handle, record):
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def numeric_or_zero(value):
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize(mode, seed, records):
    items = len(records)
    correct_count = sum(record.get("correct") is True for record in records)
    total_generated_tokens = sum(
        nonnegative_int(record.get("generated_tokens")) or 0 for record in records
    )
    wall_times = [numeric_or_zero(record.get("wall_s")) for record in records]
    summary = {
        "mode": mode,
        "seed": seed,
        "items": items,
        "correct_count": correct_count,
        "accuracy": correct_count / items if items else 0.0,
        "total_generated_tokens": total_generated_tokens,
        "mean_tokens_per_item": total_generated_tokens / items if items else 0.0,
        "tokens_per_correct": total_generated_tokens / max(correct_count, 1),
        "mean_wall_s": statistics.mean(wall_times) if wall_times else 0.0,
        "error_count": sum(bool(record.get("error")) for record in records),
    }
    if mode == "tree":
        summary["total_pruned"] = sum(
            nonnegative_int(record.get("pruned_count")) or 0 for record in records
        )
    return summary


def print_summary_table(summaries):
    print(
        "MODE  SEED  ITEMS  CORRECT  ACCURACY  TOTAL_TOKENS  TOKENS/ITEM  "
        "TOKENS/CORRECT  MEAN_WALL_S  ERRORS  PRUNED"
    )
    for row in summaries:
        pruned = row.get("total_pruned")
        print(
            "{mode:<5} {seed:>5} {items:>6} {correct:>8} {accuracy:>9.2%} "
            "{tokens:>13} {per_item:>12.2f} {per_correct:>15.2f} "
            "{wall:>12.3f} {errors:>7} {pruned:>7}".format(
                mode=row["mode"],
                seed=row["seed"],
                items=row["items"],
                correct=row["correct_count"],
                accuracy=row["accuracy"],
                tokens=row["total_generated_tokens"],
                per_item=row["mean_tokens_per_item"],
                per_correct=row["tokens_per_correct"],
                wall=row["mean_wall_s"],
                errors=row["error_count"],
                pruned="-" if pruned is None else pruned,
            )
        )


def write_json_atomic(path, value):
    ensure_parent(path)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_benchmark(args, items, seeds):
    existing, invalid_lines, duplicate_lines = load_existing_records(args.out_jsonl)
    if invalid_lines:
        print(
            "warning: ignored {} invalid or truncated line(s) in {}".format(
                invalid_lines, args.out_jsonl
            )
        )
    if duplicate_lines:
        print(
            "warning: ignored {} duplicate mode+seed+id line(s) in {}".format(
                duplicate_lines, args.out_jsonl
            )
        )
    requested_keys = {
        (args.mode, seed, item["id"]) for seed in seeds for item in items
    }
    records_by_key = {
        key: record for key, record in existing.items() if key in requested_keys
    }
    tasks = [
        (item, seed)
        for seed in seeds
        for item in items
        if (args.mode, seed, item["id"]) not in existing
    ]
    print(
        "resume: {} existing, {} pending for mode {}".format(
            len(records_by_key), len(tasks), args.mode
        )
    )

    if tasks:
        ensure_parent(args.out_jsonl)
        ensure_append_boundary(args.out_jsonl)
        task_queue = queue.Queue()
        write_lock = threading.Lock()
        fatal_errors = []
        for task in tasks:
            task_queue.put(task)
        worker_count = min(args.concurrency, len(tasks))
        for _unused in range(worker_count):
            task_queue.put(None)

        with open(args.out_jsonl, "a", encoding="utf-8", newline="\n") as output:

            def worker():
                while True:
                    task = task_queue.get()
                    try:
                        if task is None:
                            return
                        item, seed = task
                        try:
                            if args.mode == "tree":
                                record = run_tree_item(args, item, seed)
                            else:
                                record = run_bon_item(args, item, seed)
                        except Exception as exc:
                            record = error_record(
                                args.mode, item, seed, args.bon_parallel, exc
                            )
                        key = (args.mode, seed, item["id"])
                        with write_lock:
                            if key in existing or key in records_by_key:
                                continue
                            try:
                                append_record(output, record)
                            except Exception as exc:
                                fatal_errors.append(
                                    "{}:{}:{}".format(
                                        type(exc).__name__, item["id"], compact_error(exc)
                                    )
                                )
                                continue
                            records_by_key[key] = record
                    finally:
                        task_queue.task_done()

            threads = [threading.Thread(target=worker) for _ in range(worker_count)]
            for thread in threads:
                thread.start()
            task_queue.join()
            for thread in threads:
                thread.join()

        if fatal_errors:
            raise RuntimeError(
                "failed to append {} result(s): {}".format(
                    len(fatal_errors), "; ".join(fatal_errors[:3])
                )
            )

    missing = requested_keys.difference(records_by_key)
    if missing:
        raise RuntimeError(
            "missing {} requested result record(s) after sweep".format(len(missing))
        )
    summaries = []
    for seed in seeds:
        seed_records = [
            records_by_key[(args.mode, seed, item["id"])] for item in items
        ]
        summaries.append(summarize(args.mode, seed, seed_records))
    output = {
        "config": {
            "mode": args.mode,
            "model": args.model,
            "base_url": args.base_url,
            "data": os.path.abspath(args.data),
            "offset": args.offset,
            "limit": args.limit,
            "seeds": seeds,
            "branches": args.branches,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout": args.timeout,
            "concurrency": args.concurrency,
            "bon_parallel": args.bon_parallel,
            "answer_suffix": ANSWER_SUFFIX,
            "out_jsonl": os.path.abspath(args.out_jsonl),
        },
        "summaries": summaries,
    }
    write_json_atomic(args.out, output)
    print_summary_table(summaries)
    print("summary_json: {}".format(os.path.abspath(args.out)))
    print("items_jsonl: {}".format(os.path.abspath(args.out_jsonl)))


def summary_rows(document):
    if isinstance(document, dict) and isinstance(document.get("summaries"), list):
        return [row for row in document["summaries"] if isinstance(row, dict)]
    if isinstance(document, dict) and "mode" in document and "items" in document:
        return [document]
    raise ValueError("summary JSON must contain a 'summaries' list")


def aggregate_summaries(rows, mode):
    matching = [row for row in rows if row.get("mode") == mode]
    if not matching:
        raise ValueError("summary JSON has no {!r} rows".format(mode))
    items = sum(nonnegative_int(row.get("items")) or 0 for row in matching)
    correct = 0
    for row in matching:
        count = nonnegative_int(row.get("correct_count"))
        if count is None:
            accuracy = numeric_or_zero(row.get("accuracy"))
            count = round(accuracy * (nonnegative_int(row.get("items")) or 0))
        correct += count
    tokens = sum(
        nonnegative_int(row.get("total_generated_tokens")) or 0 for row in matching
    )
    return {
        "items": items,
        "correct_count": correct,
        "accuracy": correct / items if items else 0.0,
        "total_generated_tokens": tokens,
        "tokens_per_correct": tokens / max(correct, 1),
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ratio(numerator, denominator):
    if denominator <= 0:
        return None
    return numerator / denominator


def format_ratio(value):
    return "n/a" if value is None else "{:.6f}".format(value)


def compare_config_mismatches(tree_document, bon_document):
    tree_config = tree_document.get("config") if isinstance(tree_document, dict) else None
    bon_config = bon_document.get("config") if isinstance(bon_document, dict) else None
    if not isinstance(tree_config, dict) or not isinstance(bon_config, dict):
        return []
    keys = (
        "model",
        "base_url",
        "data",
        "offset",
        "limit",
        "seeds",
        "branches",
        "max_tokens",
        "temperature",
        "timeout",
    )
    return [key for key in keys if tree_config.get(key) != bon_config.get(key)]


def compare_summaries(first_path, second_path):
    first_document = load_json(first_path)
    second_document = load_json(second_path)
    first_rows = summary_rows(first_document)
    second_rows = summary_rows(second_document)
    first_modes = {row.get("mode") for row in first_rows}
    second_modes = {row.get("mode") for row in second_rows}
    if "tree" in first_modes and "bon" in second_modes:
        tree_rows, bon_rows = first_rows, second_rows
        tree_document, bon_document = first_document, second_document
    elif "bon" in first_modes and "tree" in second_modes:
        tree_rows, bon_rows = second_rows, first_rows
        tree_document, bon_document = second_document, first_document
    else:
        raise ValueError(
            "--compare needs one tree summary JSON and one bon summary JSON"
        )

    mismatches = compare_config_mismatches(tree_document, bon_document)
    if mismatches:
        raise ValueError(
            "tree and bon configs differ for: {}".format(", ".join(mismatches))
        )
    tree = aggregate_summaries(tree_rows, "tree")
    bon = aggregate_summaries(bon_rows, "bon")
    if tree["items"] != bon["items"]:
        raise ValueError(
            "tree and bon item counts differ: {} versus {}".format(
                tree["items"], bon["items"]
            )
        )
    bon_config = bon_document.get("config")
    if isinstance(bon_config, dict) and bon_config.get("bon_parallel") is True:
        print("NOTE: bon wall times are parallel.")
    accuracy_delta_points = (tree["accuracy"] - bon["accuracy"]) * 100.0
    token_ratio = ratio(
        bon["total_generated_tokens"], tree["total_generated_tokens"]
    )
    tokens_per_correct_ratio = ratio(
        bon["tokens_per_correct"], tree["tokens_per_correct"]
    )
    if token_ratio is not None and token_ratio >= 1.5 and accuracy_delta_points > -1.0:
        verdict = "pruning wins"
    elif token_ratio is not None and 0.9 <= token_ratio < 1.5:
        verdict = "marginal"
    else:
        verdict = "no win"

    print(
        "tree_accuracy: {:.6%} ({}/{})".format(
            tree["accuracy"], tree["correct_count"], tree["items"]
        )
    )
    print(
        "bon_accuracy: {:.6%} ({}/{})".format(
            bon["accuracy"], bon["correct_count"], bon["items"]
        )
    )
    print(
        "accuracy_delta_tree_minus_bon: {:+.6f} pt".format(
            accuracy_delta_points
        )
    )
    print("tree_total_generated_tokens: {}".format(tree["total_generated_tokens"]))
    print("bon_total_generated_tokens: {}".format(bon["total_generated_tokens"]))
    print("token_ratio_bon_over_tree: {}".format(format_ratio(token_ratio)))
    print("tree_tokens_per_correct: {:.6f}".format(tree["tokens_per_correct"]))
    print("bon_tokens_per_correct: {:.6f}".format(bon["tokens_per_correct"]))
    print(
        "tokens_per_correct_ratio_bon_over_tree: {}".format(
            format_ratio(tokens_per_correct_ratio)
        )
    )
    print(
        "verdict: {} (token ratio {}, accuracy delta {:+.6f} pt)".format(
            verdict, format_ratio(token_ratio), accuracy_delta_points
        )
    )


def print_dry_run(args, item, seed):
    base = args.base_url.rstrip("/")
    document = {
        "bon_parallel": args.bon_parallel,
        "item_id": item["id"],
        "item_index": item["item_index"],
        "tree": {
            "url": base + "/v1/tree/completions",
            "body": tree_body(args, item, seed),
        },
        "bon": [
            {
                "sample_index": sample_index,
                "url": base + "/v1/chat/completions",
                "body": bon_body(args, item, seed, sample_index),
            }
            for sample_index in range(args.branches)
        ],
    }
    print(json.dumps(document, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure generated-token economics for AutoTree pruning versus "
            "sequential best-of-n."
        )
    )
    parser.add_argument("--data", help="GSM8K-style JSONL input")
    parser.add_argument("--mode", choices=("tree", "bon"))
    parser.add_argument("--model", help="served model name")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--bon-parallel",
        action="store_true",
        help="issue best-of-n sample requests concurrently",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-jsonl", help="incremental per-item JSONL output")
    parser.add_argument("--out", help="final summary JSON output")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("TREE_JSON", "BON_JSON"),
        help="compare completed tree and bon summary JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print tree and bon request bodies for the first selected item",
    )
    return parser


def validate_measurement_args(parser, args):
    if not args.data:
        parser.error("--data is required unless --compare is used")
    if not args.model:
        parser.error("--model is required unless --compare is used")
    if args.branches <= 0:
        parser.error("--branches must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.offset < 0:
        parser.error("--offset must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if (args.mode == "tree" or args.dry_run) and args.temperature <= 0:
        parser.error("--temperature must be greater than 0 for tree mode")
    if not args.dry_run:
        if not args.mode:
            parser.error("--mode is required unless --compare or --dry-run is used")
        if not args.out_jsonl:
            parser.error("--out-jsonl is required for a benchmark run")
        if not args.out:
            parser.error("--out is required for a benchmark run")
        if os.path.normcase(os.path.abspath(args.out_jsonl)) == os.path.normcase(
            os.path.abspath(args.out)
        ):
            parser.error("--out-jsonl and --out must be different paths")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.compare:
        try:
            compare_summaries(args.compare[0], args.compare[1])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0

    validate_measurement_args(parser, args)
    try:
        seeds = parse_seeds(args.seeds)
        items = load_items(args.data, args.offset, args.limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print_dry_run(args, items[0], seeds[0])
        return 0
    try:
        run_benchmark(args, items, seeds)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
