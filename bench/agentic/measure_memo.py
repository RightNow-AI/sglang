#!/usr/bin/env python3
"""Measure exact prompt memoization on a replayable multi-call trace."""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


TRACE_FORMAT = "agentic-repetition-trace-v1"
BENCHMARK_FORMAT = "agentic-repetition-measurement-v1"
STUB_REQUEST_WALL_S = 0.010
STUB_LOOKUP_WALL_S = 0.0002
FINAL_MARKERS = ("FINAL:", "Final answer:", "Answer:", "####")
NUMBER_RE = re.compile(
    r"[-+]?\s*\$?\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
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


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sampling_params(args, seed):
    params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": seed,
        "endpoint_mode": args.mode,
    }
    if args.mode == "tree":
        params["branches"] = args.branches
        params["tree_policy"] = "beam"
        params["tree_budget_tokens"] = args.branches * args.max_tokens
    return params


def memo_key(prompt, model, params):
    canonical = canonical_json(
        {"model": model, "prompt": prompt, "sampling_params": params}
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical


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


def trace_repetition(calls):
    seen = set()
    repeats = 0
    for call in calls:
        prompt = call["prompt"]
        if prompt in seen:
            repeats += 1
        seen.add(prompt)
    return repeats, repeats / len(calls) if calls else 0.0


def load_trace(path):
    records = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON in {} at line {}: {}".format(path, line_number, exc)
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    "{} line {} is not a JSON object".format(path, line_number)
                )
            records.append((line_number, value))
    if not records:
        raise ValueError("trace is empty: {}".format(path))
    header_line, header = records[0]
    if header.get("type") != "agentic_trace_header":
        raise ValueError("trace line {} is not an agentic trace header".format(header_line))
    if header.get("format") != TRACE_FORMAT:
        raise ValueError(
            "unsupported trace format {!r}".format(header.get("format"))
        )

    calls = []
    seen_ids = set()
    for line_number, call in records[1:]:
        for field in ("call_id", "task_id", "step", "prompt", "gold"):
            if not isinstance(call.get(field), str) or not call[field]:
                raise ValueError(
                    "{} line {} field {!r} must be a nonempty string".format(
                        path, line_number, field
                    )
                )
        if call["call_id"] in seen_ids:
            raise ValueError("duplicate call_id {!r}".format(call["call_id"]))
        seen_ids.add(call["call_id"])
        calls.append(call)
    if not calls:
        raise ValueError("trace contains no calls")

    repeat_calls, realized_rate = trace_repetition(calls)
    if header.get("total_calls") != len(calls):
        raise ValueError(
            "trace header total_calls {} does not match {} call records".format(
                header.get("total_calls"), len(calls)
            )
        )
    if header.get("repeat_calls") != repeat_calls:
        raise ValueError(
            "trace header repeat_calls {} does not match measured {}".format(
                header.get("repeat_calls"), repeat_calls
            )
        )
    header_rate = header.get("realized_repeat_rate")
    if not isinstance(header_rate, (int, float)) or not math.isclose(
        float(header_rate), realized_rate, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "trace header realized_repeat_rate {!r} does not match measured {:.12f}".format(
                header_rate, realized_rate
            )
        )
    return header, calls


def request_body(args, prompt, seed):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": seed,
    }
    if args.mode == "tree":
        body["tree"] = {
            "policy": "beam",
            "branches": args.branches,
            "budget_tokens": args.branches * args.max_tokens,
        }
    return body


def endpoint_url(args):
    suffix = "/v1/chat/completions" if args.mode == "chat" else "/v1/tree/completions"
    return args.base_url.rstrip("/") + suffix


def decode_json_body(raw):
    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, "invalid_json_response: {}".format(exc)
    if not isinstance(payload, dict):
        return None, "json_response_not_object"
    return payload, None


def post_json(url, body, timeout):
    encoded = canonical_json(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        return decode_json_body(raw)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = ""
        return None, "http_{}: {}".format(exc.code, detail)
    except urllib.error.URLError as exc:
        return None, "url_error: {}".format(getattr(exc, "reason", exc))
    except TimeoutError as exc:
        return None, "timeout: {}".format(exc)
    except Exception as exc:
        return None, "request_error: {}: {}".format(type(exc).__name__, exc)


def branch_value_text(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    for key in ("text", "content", "continuation", "output_text", "answer"):
        text = value.get(key)
        if isinstance(text, str) and text:
            return text
    message = value.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return None


def completion_content(payload):
    if not isinstance(payload, dict):
        return None, "missing_response_object"
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = None
    if isinstance(content, str) and content:
        return content, None

    tree = payload.get("tree")
    if isinstance(tree, dict):
        raw_answers = tree.get("branch_answers")
        if isinstance(raw_answers, dict):
            answers = [branch_value_text(value) for value in raw_answers.values()]
            answers = [answer for answer in answers if answer]
            if answers:
                counts = Counter(answers)
                best_count = max(counts.values())
                for answer in answers:
                    if counts[answer] == best_count:
                        return answer, None
    return None, "missing_choices_message_content"


def nonnegative_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def generated_tokens(payload, mode):
    if not isinstance(payload, dict):
        return 0, False, "missing"
    if mode == "tree":
        tree = payload.get("tree")
        if isinstance(tree, dict):
            spent = tree.get("tokens_spent_per_branch")
            if isinstance(spent, dict) and spent:
                values = [nonnegative_int(value) for value in spent.values()]
                if all(value is not None for value in values):
                    return sum(values), True, "tree.tokens_spent_per_branch"
    usage = payload.get("usage")
    if isinstance(usage, dict):
        count = nonnegative_int(usage.get("completion_tokens"))
        if count is not None:
            return count, True, "usage.completion_tokens"
    return 0, False, "missing"


def boxed_value(text):
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    position = start + len(marker)
    depth = 1
    output = []
    while position < len(text):
        character = text[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return "".join(output).strip() or None
        output.append(character)
        position += 1
    return None


def extract_answer(text):
    if not isinstance(text, str) or not text.strip():
        return None
    best_position = -1
    best_marker = None
    for marker in FINAL_MARKERS:
        position = text.rfind(marker)
        if position > best_position:
            best_position = position
            best_marker = marker
    if best_marker is not None:
        tail = text[best_position + len(best_marker) :].strip()
        first_line = tail.splitlines()[0].strip() if tail else ""
        if first_line:
            return first_line
    boxed = boxed_value(text)
    if boxed:
        return boxed
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    matches = list(NUMBER_RE.finditer(text))
    return matches[-1].group(0).strip() if matches else None


def unwrap_boxed(text):
    stripped = text.strip()
    if stripped.startswith("\\boxed{") and stripped.endswith("}"):
        inner = boxed_value(stripped)
        if inner is not None:
            return inner
    return stripped


def numeric_fraction(text):
    cleaned = text.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.replace(" ", "")
    fraction_match = re.fullmatch(r"\\(?:d?frac)\{([-+]?\d+)\}\{(\d+)\}", cleaned)
    if fraction_match:
        denominator = int(fraction_match.group(2))
        if denominator:
            return Fraction(int(fraction_match.group(1)), denominator)
        return None
    if re.fullmatch(r"[-+]?\d+/\d+", cleaned):
        numerator, denominator = cleaned.split("/", 1)
        if int(denominator):
            return Fraction(int(numerator), int(denominator))
        return None
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned):
        try:
            return Fraction(cleaned)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def normalize_symbolic(text):
    value = unwrap_boxed(text)
    value = value.strip().strip("$.")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "")
    value = value.replace("−", "-")
    value = re.sub(r"\\text\{\s*degrees?\s*\}", "deg", value, flags=re.I)
    value = re.sub(r"\^?\\circ", "deg", value)
    value = value.replace("°", "deg")
    value = re.sub(r"\bdegrees?\b", "deg", value, flags=re.I)
    value = re.sub(r"\s+", "", value)
    return value.lower()


def is_correct(answer, gold):
    if not isinstance(answer, str):
        return False
    answer_fraction = numeric_fraction(unwrap_boxed(answer))
    gold_fraction = numeric_fraction(unwrap_boxed(gold))
    if answer_fraction is not None and gold_fraction is not None:
        return answer_fraction == gold_fraction
    return normalize_symbolic(answer) == normalize_symbolic(gold)


def stub_payload(args, call):
    per_branch_tokens = 20
    content = "STUB PLUMBING RESPONSE\nFINAL: {}".format(call["gold"])
    if args.mode == "chat":
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"completion_tokens": per_branch_tokens},
        }
    spent = {str(index): per_branch_tokens for index in range(args.branches)}
    return {
        "choices": [{"message": {"content": content}}],
        "tree": {
            "tokens_spent_per_branch": spent,
            "branch_answers": {
                str(index): content for index in range(args.branches)
            },
        },
        "usage": {"completion_tokens": per_branch_tokens * args.branches},
    }


def perform_request(args, call, seed):
    body = request_body(args, call["prompt"], seed)
    if args.stub:
        return stub_payload(args, call), None, STUB_REQUEST_WALL_S, body
    started = time.perf_counter()
    payload, error = post_json(endpoint_url(args), body, args.timeout)
    return payload, error, time.perf_counter() - started, body


def make_record(args, arm, call, call_index, seed, memo):
    params = sampling_params(args, seed)
    lookup_started = time.perf_counter()
    key, _canonical = memo_key(call["prompt"], args.model, params)
    lookup_wall = (
        STUB_LOOKUP_WALL_S if args.stub else time.perf_counter() - lookup_started
    )

    if arm == "memo" and key in memo:
        cached = memo[key]
        return {
            "benchmark": BENCHMARK_FORMAT,
            "config_hash": args.config_hash,
            "arm": arm,
            "seed": seed,
            "call_index": call_index,
            "call_id": call["call_id"],
            "task_id": call["task_id"],
            "step": call["step"],
            "memo_key": key,
            "served_from_memo": True,
            "cache_stored": False,
            "generated_tokens": 0,
            "tokens_saved": cached["generated_tokens"],
            "token_report_valid": True,
            "token_report_source": "memo_hit",
            "wall_s": lookup_wall,
            "answer": cached["answer"],
            "correct": cached["correct"],
            "error": None,
        }

    payload, request_error, request_wall, _body = perform_request(args, call, seed)
    content, content_error = completion_content(payload)
    answer = extract_answer(content)
    tokens, token_valid, token_source = generated_tokens(payload, args.mode)
    error_parts = [part for part in (request_error, content_error) if part]
    error = "; ".join(error_parts) if error_parts else None
    correct = is_correct(answer, call["gold"])
    cache_stored = arm == "memo" and request_error is None
    record = {
        "benchmark": BENCHMARK_FORMAT,
        "config_hash": args.config_hash,
        "arm": arm,
        "seed": seed,
        "call_index": call_index,
        "call_id": call["call_id"],
        "task_id": call["task_id"],
        "step": call["step"],
        "memo_key": key,
        "served_from_memo": False,
        "cache_stored": cache_stored,
        "generated_tokens": tokens,
        "tokens_saved": 0,
        "token_report_valid": token_valid,
        "token_report_source": token_source,
        "wall_s": request_wall + (lookup_wall if arm == "memo" else 0.0),
        "answer": answer,
        "correct": correct,
        "error": error,
    }
    if cache_stored:
        memo[key] = {
            "answer": answer,
            "correct": correct,
            "generated_tokens": tokens,
        }
    return record


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_append_boundary(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")


def load_existing(path, config_hash):
    existing = {}
    if not path or not os.path.exists(path):
        return existing
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON in {} at line {}: {}".format(path, line_number, exc)
                ) from exc
            if record.get("config_hash") != config_hash:
                raise ValueError(
                    "{} contains a different run configuration at line {}".format(
                        path, line_number
                    )
                )
            key = (record.get("arm"), record.get("seed"), record.get("call_id"))
            if key in existing:
                raise ValueError("duplicate resumable record key {!r}".format(key))
            existing[key] = record
    return existing


def restore_memo_record(record, memo):
    key = record["memo_key"]
    if record.get("served_from_memo"):
        if key not in memo:
            raise ValueError(
                "cannot resume memo hit {} before its cached miss".format(
                    record.get("call_id")
                )
            )
        return
    if record.get("cache_stored"):
        memo[key] = {
            "answer": record.get("answer"),
            "correct": bool(record.get("correct")),
            "generated_tokens": int(record.get("generated_tokens") or 0),
        }


def run_arm(args, arm, calls, seeds, existing, output_handle):
    records = []
    pending = 0
    for seed in seeds:
        for call in calls:
            if (arm, seed, call["call_id"]) not in existing:
                pending += 1
    print("resume {}: {} existing, {} pending".format(arm, len(calls) * len(seeds) - pending, pending))

    for seed in seeds:
        memo = {}
        for call_index, call in enumerate(calls):
            key = (arm, seed, call["call_id"])
            if key in existing:
                record = existing[key]
                expected_key, _canonical = memo_key(
                    call["prompt"], args.model, sampling_params(args, seed)
                )
                if record.get("memo_key") != expected_key:
                    raise ValueError(
                        "resumed memo key mismatch for {}".format(call["call_id"])
                    )
                if arm == "memo":
                    restore_memo_record(record, memo)
            else:
                record = make_record(args, arm, call, call_index, seed, memo)
                output_handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                output_handle.flush()
            records.append(record)
    return records


def summarize_arm(arm, records):
    total = len(records)
    hits = sum(bool(record.get("served_from_memo")) for record in records)
    correct = sum(bool(record.get("correct")) for record in records)
    requested = [record for record in records if not record.get("served_from_memo")]
    token_report_complete = all(
        bool(record.get("token_report_valid")) for record in requested
    )
    total_tokens = sum(int(record.get("generated_tokens") or 0) for record in records)
    return {
        "arm": arm,
        "total_calls": total,
        "real_requests": len(requested),
        "memo_hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "total_generated_tokens": total_tokens,
        "total_tokens_saved": sum(
            int(record.get("tokens_saved") or 0) for record in records
        ),
        "total_wall_s": sum(float(record.get("wall_s") or 0.0) for record in records),
        "correct_calls": correct,
        "accuracy": correct / total if total else 0.0,
        "cost_units": float(total_tokens),
        "cost_units_definition": "one unit per server-reported generated token",
        "token_report_complete": token_report_complete,
        "errors": sum(bool(record.get("error")) for record in records),
    }


def ratio(numerator, denominator):
    if denominator <= 0:
        return None
    return numerator / denominator


def compare_arms(nomemo, memo):
    if nomemo["total_calls"] != memo["total_calls"]:
        raise ValueError(
            "arm total_calls differ: {} versus {}".format(
                nomemo["total_calls"], memo["total_calls"]
            )
        )
    hit_rate = memo["hit_rate"]
    theoretical = 1.0 / (1.0 - hit_rate) if hit_rate < 1.0 else None
    token_multiplier = None
    if nomemo["token_report_complete"] and memo["token_report_complete"]:
        token_multiplier = ratio(
            nomemo["total_generated_tokens"], memo["total_generated_tokens"]
        )
    return {
        "hit_rate": hit_rate,
        "theoretical_multiplier": theoretical,
        "token_multiplier": token_multiplier,
        "wall_multiplier": ratio(nomemo["total_wall_s"], memo["total_wall_s"]),
        "cost_multiplier": (
            ratio(nomemo["cost_units"], memo["cost_units"])
            if nomemo["token_report_complete"] and memo["token_report_complete"]
            else None
        ),
        "accuracy_delta_points": (memo["accuracy"] - nomemo["accuracy"]) * 100.0,
    }


def format_multiplier(value):
    return "unavailable" if value is None else "{:.3f}x".format(value)


def print_arm(summary):
    print("ARM {}".format(summary["arm"]))
    print("  total calls: {}".format(summary["total_calls"]))
    print("  real requests: {}".format(summary["real_requests"]))
    print("  memo hits: {}".format(summary["memo_hits"]))
    print("  HIT RATE: {:.2%}".format(summary["hit_rate"]))
    print("  total generated tokens: {}".format(summary["total_generated_tokens"]))
    print("  total tokens saved: {}".format(summary["total_tokens_saved"]))
    print("  total wall: {:.6f}s".format(summary["total_wall_s"]))
    print("  accuracy: {:.2%}".format(summary["accuracy"]))
    print("  cost units: {:.3f}".format(summary["cost_units"]))
    print("  errors: {}".format(summary["errors"]))
    if not summary["token_report_complete"]:
        print("  WARNING: generated token reporting is incomplete")


def print_comparison(comparison):
    hit_rate = comparison["hit_rate"]
    print("COMPARISON")
    print(
        "  theoretical 1/(1-h) at hit rate {:.2%}: {}".format(
            hit_rate, format_multiplier(comparison["theoretical_multiplier"])
        )
    )
    print(
        "  token multiplier nomemo/memo at hit rate {:.2%}: {}".format(
            hit_rate, format_multiplier(comparison["token_multiplier"])
        )
    )
    print(
        "  wall multiplier nomemo/memo at hit rate {:.2%}: {}".format(
            hit_rate, format_multiplier(comparison["wall_multiplier"])
        )
    )
    print(
        "  cost multiplier nomemo/memo at hit rate {:.2%}: {}".format(
            hit_rate, format_multiplier(comparison["cost_multiplier"])
        )
    )
    print(
        "  accuracy delta memo minus nomemo: {:+.3f} percentage points".format(
            comparison["accuracy_delta_points"]
        )
    )
    if comparison["accuracy_delta_points"] < -1.0:
        print("  " + "!" * 72)
        print(
            "  WARNING: MEMO ACCURACY IS MORE THAN 1 PERCENTAGE POINT BELOW NOMEMO."
        )
        print("  WARNING: THE MULTIPLIER IS NOT FREE.")
        print("  " + "!" * 72)
    print(
        "  Gap note: 1/(1-h) assumes equal-cost calls and free lookups. Every miss "
        "pays full request cost, lookup and harness time remain, and generated "
        "lengths can differ. Wall and cost gains should therefore trail the ideal; "
        "the raw token ratio can move either way when call lengths vary."
    )


def print_honesty(trace_header):
    print(
        "REALIZED TRACE REPETITION RATE: {:.2%} ({}/{} calls)".format(
            trace_header["realized_repeat_rate"],
            trace_header["repeat_calls"],
            trace_header["total_calls"],
        )
    )
    print(
        "HONESTY: the multiplier is a function of workload repetition, NOT of the engine."
    )
    print(
        "HONESTY: on novel non-repeating traffic the hit rate is approximately zero "
        "and the multiplier is 1x."
    )


def write_json_atomic(path, value):
    ensure_parent(path)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_config(args, trace_path, trace_sha, trace_header, seeds):
    return {
        "trace_path": str(Path(trace_path).resolve()),
        "trace_sha256": trace_sha,
        "trace_format": trace_header["format"],
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "mode": args.mode,
        "branches": args.branches if args.mode == "tree" else None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seeds": seeds,
        "stub": args.stub,
    }


def dry_run(args, header, calls, seeds):
    params = sampling_params(args, seeds[0])
    key, canonical = memo_key(calls[0]["prompt"], args.model, params)
    document = {
        "plumbing_test_only": bool(args.stub),
        "trace": {
            "total_calls": header["total_calls"],
            "repeat_calls": header["repeat_calls"],
            "realized_repeat_rate": header["realized_repeat_rate"],
        },
        "first_call": {
            "call_id": calls[0]["call_id"],
            "url": "STUB_LOCAL_NO_NETWORK" if args.stub else endpoint_url(args),
            "body": request_body(args, calls[0]["prompt"], seeds[0]),
            "memo_key": key,
            "memo_key_canonical_json": canonical,
        },
    }
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    print_honesty(header)


def extract_named_arm(document, name, path):
    arms = document.get("arms")
    if isinstance(arms, dict) and isinstance(arms.get(name), dict):
        return arms[name]
    raise ValueError("{} does not contain arm {!r}".format(path, name))


def compare_files(nomemo_path, memo_path):
    with open(nomemo_path, "r", encoding="utf-8-sig") as handle:
        nomemo_document = json.load(handle)
    with open(memo_path, "r", encoding="utf-8-sig") as handle:
        memo_document = json.load(handle)
    for document, path in (
        (nomemo_document, nomemo_path),
        (memo_document, memo_path),
    ):
        if document.get("benchmark") != BENCHMARK_FORMAT:
            raise ValueError("unsupported benchmark document: {}".format(path))
    if nomemo_document.get("config") != memo_document.get("config"):
        raise ValueError("nomemo and memo measurement configs differ")
    if nomemo_document.get("trace") != memo_document.get("trace"):
        raise ValueError("nomemo and memo trace metadata differ")
    nomemo = extract_named_arm(nomemo_document, "nomemo", nomemo_path)
    memo = extract_named_arm(memo_document, "memo", memo_path)
    comparison = compare_arms(nomemo, memo)
    if nomemo_document.get("plumbing_test_only") or memo_document.get(
        "plumbing_test_only"
    ):
        print(
            "PLUMBING TEST ONLY: one or both inputs used fabricated --stub responses."
        )
        print("DO NOT REPORT THESE RESULTS AS MODEL OR ENGINE PERFORMANCE.")
    print_honesty(nomemo_document["trace"])
    print_arm(nomemo)
    print_arm(memo)
    print_comparison(comparison)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Replay an agentic trace with and without an exact-match memo."
    )
    parser.add_argument("--trace")
    parser.add_argument("--model")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--mode", choices=("chat", "tree"), default="chat")
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out-jsonl")
    parser.add_argument("--out")
    parser.add_argument(
        "--arm", choices=("nomemo", "memo", "both"), default="both"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("NOMEMO_JSON", "MEMO_JSON")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="fabricate deterministic local responses for plumbing tests only",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.compare:
        try:
            compare_files(*args.compare)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0

    if not args.trace or not args.model:
        parser.error("--trace and --model are required unless --compare is used")
    if args.branches <= 0 or args.branches > 64:
        parser.error("--branches must be between 1 and 64")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not math.isfinite(args.temperature) or not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be finite and between 0.0 and 2.0")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        seeds = parse_seeds(args.seeds)
        header, calls = load_trace(args.trace)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        dry_run(args, header, calls, seeds)
        return 0
    if not args.out_jsonl or not args.out:
        parser.error("--out-jsonl and --out are required for a replay")
    if os.path.abspath(args.out_jsonl) == os.path.abspath(args.out):
        parser.error("--out-jsonl and --out must be different paths")

    trace_sha = sha256_file(args.trace)
    config = run_config(args, args.trace, trace_sha, header, seeds)
    args.config_hash = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    try:
        existing = load_existing(args.out_jsonl, args.config_hash)
        ensure_parent(args.out_jsonl)
        ensure_append_boundary(args.out_jsonl)
        selected_arms = ("nomemo", "memo") if args.arm == "both" else (args.arm,)
        summaries = {}
        with open(args.out_jsonl, "a", encoding="utf-8", newline="\n") as output:
            for arm in selected_arms:
                records = run_arm(args, arm, calls, seeds, existing, output)
                summaries[arm] = summarize_arm(arm, records)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    trace_metadata = {
        "path": str(Path(args.trace).resolve()),
        "sha256": trace_sha,
        "format": header["format"],
        "mode": header.get("mode"),
        "total_calls": header["total_calls"],
        "repeat_calls": header["repeat_calls"],
        "realized_repeat_rate": header["realized_repeat_rate"],
    }
    comparison = None
    if "nomemo" in summaries and "memo" in summaries:
        comparison = compare_arms(summaries["nomemo"], summaries["memo"])
    document = {
        "benchmark": BENCHMARK_FORMAT,
        "plumbing_test_only": bool(args.stub),
        "reportable_result": not args.stub,
        "trace": trace_metadata,
        "config": config,
        "arms": summaries,
        "comparison": comparison,
        "honesty": {
            "workload_property": (
                "the multiplier is a function of workload repetition, not of the engine"
            ),
            "novel_traffic": (
                "on novel non-repeating traffic hit rate is approximately zero and "
                "the multiplier is 1x"
            ),
            "accuracy_warning_threshold_points": -1.0,
        },
    }
    try:
        write_json_atomic(args.out, document)
    except OSError as exc:
        parser.error(str(exc))

    if args.stub:
        print("PLUMBING TEST ONLY: --stub fabricated deterministic local responses.")
        print("DO NOT REPORT THESE RESULTS AS MODEL OR ENGINE PERFORMANCE.")
    print_honesty(trace_metadata)
    for arm in ("nomemo", "memo"):
        if arm in summaries:
            print_arm(summaries[arm])
    if comparison is not None:
        print_comparison(comparison)
    print("records: {}".format(Path(args.out_jsonl).resolve()))
    print("summary: {}".format(Path(args.out).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
