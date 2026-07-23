#!/usr/bin/env python3
"""Measure small-tree to large-model cascade economics."""

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
MODES = ("cascade", "large_bo8", "large_greedy", "cascade_score")


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
                raise ValueError(
                    "{} line {} is not a JSON object".format(path, line_number)
                )
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


def prompt_messages(item):
    return [{"role": "user", "content": item["prompt"] + ANSWER_SUFFIX}]


def item_seed(base_seed, item):
    return base_seed + item["item_index"]


def sample_seed(base_seed, item, sample_index):
    return base_seed + item["item_index"] * 1000 + sample_index


def small_tree_body(args, item, base_seed):
    return {
        "model": args.small_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": item_seed(base_seed, item),
        "tree": {
            "policy": "beam",
            "branches": args.branches,
            "budget_tokens": args.branches * args.max_tokens,
        },
    }


def large_sample_body(args, item, base_seed, sample_index):
    return {
        "model": args.large_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": sample_seed(base_seed, item, sample_index),
    }


def large_greedy_body(args, item, base_seed):
    return {
        "model": args.large_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": item_seed(base_seed, item),
    }


def large_score_body(item, candidate):
    return {
        "text": item["prompt"] + ANSWER_SUFFIX + "\nAnswer: " + candidate,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
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
        label = (
            "timeout" if "timeout" in type(exc).__name__.lower() else "request_error"
        )
        return None, "{}: {}: {}".format(label, type(exc).__name__, compact_error(exc))


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


def nonnegative_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number < 0 or number >= float("inf"):
        return None
    return number


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


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


def chat_token_report(payload):
    if not isinstance(payload, dict):
        return 0, False
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, False
    count = nonnegative_int(usage.get("completion_tokens"))
    return (count, True) if count is not None else (0, False)


def branch_answer_report(payload):
    if not isinstance(payload, dict):
        return {}, None, 0, 0, False
    tree = payload.get("tree")
    if not isinstance(tree, dict) or "branch_answers" not in tree:
        return {}, None, 0, 0, False
    raw_answers = tree.get("branch_answers")
    if not isinstance(raw_answers, dict):
        return {}, None, 0, 0, False
    normalized = {}
    voters = []
    valid = True
    for branch_id, value in raw_answers.items():
        key = str(branch_id)
        if value is None:
            normalized[key] = None
            continue
        if not isinstance(value, str):
            normalized[key] = None
            valid = False
            continue
        answer = normalize_number(value)
        normalized[key] = answer
        if answer is None:
            valid = False
        else:
            voters.append(answer)
    leader = majority_vote(voters)
    leader_count = voters.count(leader) if leader is not None else 0
    return normalized, leader, leader_count, len(voters), valid


def tree_winner_branch_id(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), dict):
        return None
    value = payload["tree"].get("winner_branch_id")
    return None if value is None else str(value)


def ranked_branch_candidates(payload, branch_answers):
    counts = collections.Counter(
        answer for answer in branch_answers.values() if answer is not None
    )
    best_scores = {}
    tree = payload.get("tree") if isinstance(payload, dict) else None
    final_scores = tree.get("final_scores") if isinstance(tree, dict) else None
    if isinstance(final_scores, dict):
        for branch_id, answer in branch_answers.items():
            if answer is None:
                continue
            score = finite_number(final_scores.get(branch_id))
            if score is None:
                continue
            if answer not in best_scores or score > best_scores[answer]:
                best_scores[answer] = score
    ranked = sorted(
        counts,
        key=lambda answer: (
            -counts[answer],
            0 if answer in best_scores else 1,
            -best_scores.get(answer, 0.0),
            answer,
        ),
    )
    return ranked[:2]


def estimated_candidate_token_count(candidate):
    return max(1, (len(candidate) + 3) // 4)


def run_native_score_request(url, body, candidate, timeout):
    payload, request_error = post_json(url, body, timeout)
    meta_info = payload.get("meta_info") if isinstance(payload, dict) else None
    raw_logprobs = (
        meta_info.get("input_token_logprobs")
        if isinstance(meta_info, dict)
        else None
    )
    raw_logprob_length = len(raw_logprobs) if isinstance(raw_logprobs, list) else 0
    prompt_tokens = (
        nonnegative_int(meta_info.get("prompt_tokens"))
        if isinstance(meta_info, dict)
        else None
    )
    prompt_token_source = "meta_info.prompt_tokens"
    if prompt_tokens is None and isinstance(raw_logprobs, list):
        prompt_tokens = raw_logprob_length
        prompt_token_source = "input_token_logprobs_length"
    if prompt_tokens is None:
        prompt_tokens = 0
        prompt_token_source = "unavailable"

    errors = []
    add_error(errors, request_error)
    if request_error is None and not isinstance(meta_info, dict):
        add_error(errors, "missing_meta_info")
    if not isinstance(raw_logprobs, list) or not raw_logprobs:
        add_error(errors, "missing_input_token_logprobs")

    estimated_tokens = estimated_candidate_token_count(candidate)
    values = []
    if isinstance(raw_logprobs, list) and raw_logprobs:
        if len(raw_logprobs) < estimated_tokens:
            add_error(errors, "candidate_span_exceeds_logprob_array")
        else:
            for entry in raw_logprobs[-estimated_tokens:]:
                if not isinstance(entry, (list, tuple)) or not entry:
                    add_error(errors, "malformed_input_token_logprob")
                    break
                value = finite_number(entry[0])
                if value is None:
                    add_error(errors, "non_numeric_input_token_logprob")
                    break
                values.append(value)
    score = sum(values) / len(values) if not errors and values else None
    return {
        "score": score,
        "prompt_tokens": prompt_tokens,
        "prompt_token_source": prompt_token_source,
        "raw_logprob_length": raw_logprob_length,
        "estimated_candidate_tokens": estimated_tokens,
        "errors": errors,
    }


def add_error(errors, error):
    if error and error not in errors:
        errors.append(error)


def request_errors(request_error, content_error, tokens, token_report_valid):
    errors = []
    add_error(errors, request_error)
    if request_error is None:
        add_error(errors, content_error)
        if tokens == 0:
            add_error(errors, "zero_tokens")
        elif not token_report_valid:
            add_error(errors, "invalid_token_report")
    return errors


def run_chat_request(url, body, timeout):
    payload, request_error = post_json(url, body, timeout)
    tokens, token_report_valid = chat_token_report(payload)
    content, content_error = completion_content(payload)
    return {
        "answer": extract_answer(content),
        "tokens": tokens,
        "errors": request_errors(
            request_error, content_error, tokens, token_report_valid
        ),
    }


def prefixed_errors(prefix, errors):
    return ["{}:{}".format(prefix, error) for error in errors]


def item_cost(args, small_tokens, large_tokens):
    return small_tokens * args.small_cost + large_tokens * args.large_cost


def common_record(
    args,
    mode,
    item,
    base_seed,
    started,
    answer,
    small_tokens,
    large_tokens,
    errors,
):
    return {
        "id": item["id"],
        "mode": mode,
        "seed": base_seed,
        "item_index": item["item_index"],
        "correct": not errors and answer == item["normalized_gold"],
        "extracted": answer,
        "gold": item["gold"],
        "small_tokens": small_tokens,
        "large_tokens": large_tokens,
        "total_cost_units": item_cost(args, small_tokens, large_tokens),
        "wall_s": round(time.perf_counter() - started, 6),
        "error": "; ".join(errors) if errors else None,
    }


def run_cascade_item(args, item, base_seed):
    started = time.perf_counter()
    small_url = args.small_url.rstrip("/") + "/v1/tree/completions"
    payload, request_error = post_json(
        small_url, small_tree_body(args, item, base_seed), args.timeout
    )
    small_tokens, token_report_valid = tree_token_report(payload)
    content, content_error = completion_content(payload)
    winner_answer = extract_answer(content)
    (
        branch_answers,
        branch_leader,
        leader_count,
        voter_count,
        branch_report_valid,
    ) = branch_answer_report(payload)
    errors = prefixed_errors(
        "small",
        request_errors(request_error, content_error, small_tokens, token_report_valid),
    )
    if request_error is None and not branch_report_valid:
        add_error(errors, "small:invalid_branch_answers")

    small_response_valid = not errors
    confident = (
        small_response_valid
        and leader_count >= args.agree_threshold
        and voter_count >= 4
    )
    escalated = not confident
    large_tokens = 0
    answer = winner_answer
    if escalated:
        large_url = args.large_url.rstrip("/") + "/v1/chat/completions"
        large = run_chat_request(
            large_url, large_greedy_body(args, item, base_seed), args.timeout
        )
        large_tokens = large["tokens"]
        answer = large["answer"]
        for error in prefixed_errors("large", large["errors"]):
            add_error(errors, error)

    record = common_record(
        args,
        "cascade",
        item,
        base_seed,
        started,
        answer,
        small_tokens,
        large_tokens,
        errors,
    )
    record.update(
        {
            "escalated": escalated,
            "small_winner_answer": winner_answer,
            "branch_answers": branch_answers,
            "branch_answer_leader": branch_leader,
            "leader_count": leader_count,
            "voter_count": voter_count,
            "leader_matches_winner": (
                branch_leader is not None and branch_leader == winner_answer
            ),
            "winner_branch_id": tree_winner_branch_id(payload),
        }
    )
    return record


def run_cascade_score_item(args, item, base_seed):
    started = time.perf_counter()
    small_url = args.small_url.rstrip("/") + "/v1/tree/completions"
    payload, request_error = post_json(
        small_url, small_tree_body(args, item, base_seed), args.timeout
    )
    small_tokens, token_report_valid = tree_token_report(payload)
    content, content_error = completion_content(payload)
    winner_answer = extract_answer(content)
    (
        branch_answers,
        branch_leader,
        leader_count,
        voter_count,
        branch_report_valid,
    ) = branch_answer_report(payload)
    errors = prefixed_errors(
        "small",
        request_errors(request_error, content_error, small_tokens, token_report_valid),
    )
    if request_error is None and not branch_report_valid:
        add_error(errors, "small:invalid_branch_answers")

    small_response_valid = not errors
    confident = (
        small_response_valid
        and leader_count >= args.agree_threshold
        and voter_count >= 4
    )
    escalated = not confident
    candidates = ranked_branch_candidates(payload, branch_answers)
    scored_candidates = []
    span_scores = {}
    raw_logprob_lengths = {}
    estimated_candidate_tokens = {}
    scoring_prompt_tokens = {}
    scoring_prompt_token_sources = {}
    scoring_errors = []
    large_prefill_tokens = 0
    large_tokens = 0
    answer = winner_answer
    chosen_by = "vote"

    if escalated:
        if len(candidates) == 2:
            score_url = args.large_url.rstrip("/") + "/generate"
            for candidate in candidates:
                score_result = run_native_score_request(
                    score_url,
                    large_score_body(item, candidate),
                    candidate,
                    args.timeout,
                )
                scored_candidates.append(candidate)
                span_scores[candidate] = score_result["score"]
                raw_logprob_lengths[candidate] = score_result["raw_logprob_length"]
                estimated_candidate_tokens[candidate] = score_result[
                    "estimated_candidate_tokens"
                ]
                scoring_prompt_tokens[candidate] = score_result["prompt_tokens"]
                scoring_prompt_token_sources[candidate] = score_result[
                    "prompt_token_source"
                ]
                large_prefill_tokens += score_result["prompt_tokens"]
                for error in score_result["errors"]:
                    add_error(
                        scoring_errors,
                        "candidate_{}:{}".format(candidate, error),
                    )
        else:
            add_error(
                scoring_errors,
                "selection:need_two_distinct_candidates_got_{}".format(
                    len(candidates)
                ),
            )

        scores_available = len(candidates) == 2 and all(
            span_scores.get(candidate) is not None for candidate in candidates
        )
        if scores_available:
            answer = max(candidates, key=lambda candidate: span_scores[candidate])
            chosen_by = "score"
        else:
            large_url = args.large_url.rstrip("/") + "/v1/chat/completions"
            large = run_chat_request(
                large_url, large_greedy_body(args, item, base_seed), args.timeout
            )
            large_tokens = large["tokens"]
            answer = large["answer"]
            chosen_by = "fallback"
            for error in prefixed_errors("large", large["errors"]):
                add_error(errors, error)

    record = common_record(
        args,
        "cascade_score",
        item,
        base_seed,
        started,
        answer,
        small_tokens,
        large_tokens,
        errors,
    )
    record["large_prefill_tokens"] = large_prefill_tokens
    record["total_cost_units"] += large_prefill_tokens * args.large_prefill_cost
    record.update(
        {
            "escalated": escalated,
            "small_winner_answer": winner_answer,
            "branch_answers": branch_answers,
            "branch_answer_leader": branch_leader,
            "leader_count": leader_count,
            "voter_count": voter_count,
            "leader_matches_winner": (
                branch_leader is not None and branch_leader == winner_answer
            ),
            "winner_branch_id": tree_winner_branch_id(payload),
            "scored_candidates": scored_candidates,
            "span_scores": span_scores,
            "raw_logprob_lengths": raw_logprob_lengths,
            "estimated_candidate_tokens": estimated_candidate_tokens,
            "scoring_prompt_tokens": scoring_prompt_tokens,
            "scoring_prompt_token_sources": scoring_prompt_token_sources,
            "scoring_errors": scoring_errors,
            "chosen_by": chosen_by,
        }
    )
    return record


def run_large_bo8_item(args, item, base_seed):
    started = time.perf_counter()
    url = args.large_url.rstrip("/") + "/v1/chat/completions"

    def run_sample(sample_index):
        return run_chat_request(
            url,
            large_sample_body(args, item, base_seed, sample_index),
            args.timeout,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.branches) as executor:
        futures = [executor.submit(run_sample, index) for index in range(args.branches)]
        samples = [future.result() for future in futures]

    answers = []
    large_tokens = 0
    errors = []
    for sample_index, sample in enumerate(samples):
        answers.append(sample["answer"])
        large_tokens += sample["tokens"]
        for error in prefixed_errors(
            "sample_{}".format(sample_index), sample["errors"]
        ):
            add_error(errors, error)
    answer = majority_vote(answers)
    record = common_record(
        args,
        "large_bo8",
        item,
        base_seed,
        started,
        answer,
        0,
        large_tokens,
        errors,
    )
    record["sample_answers"] = answers
    return record


def run_large_greedy_item(args, item, base_seed):
    started = time.perf_counter()
    url = args.large_url.rstrip("/") + "/v1/chat/completions"
    sample = run_chat_request(
        url, large_greedy_body(args, item, base_seed), args.timeout
    )
    errors = prefixed_errors("large", sample["errors"])
    return common_record(
        args,
        "large_greedy",
        item,
        base_seed,
        started,
        sample["answer"],
        0,
        sample["tokens"],
        errors,
    )


def run_mode_item(args, item, base_seed):
    if args.mode == "cascade":
        return run_cascade_item(args, item, base_seed)
    if args.mode == "cascade_score":
        return run_cascade_score_item(args, item, base_seed)
    if args.mode == "large_bo8":
        return run_large_bo8_item(args, item, base_seed)
    return run_large_greedy_item(args, item, base_seed)


def error_record(args, item, base_seed, exc):
    errors = ["internal_error:{}:{}".format(type(exc).__name__, compact_error(exc))]
    record = common_record(
        args,
        args.mode,
        item,
        base_seed,
        time.perf_counter(),
        None,
        0,
        0,
        errors,
    )
    record["wall_s"] = 0.0
    if args.mode == "cascade":
        record.update(
            {
                "escalated": None,
                "small_winner_answer": None,
                "branch_answers": {},
                "branch_answer_leader": None,
                "leader_count": 0,
                "voter_count": 0,
                "leader_matches_winner": False,
                "winner_branch_id": None,
            }
        )
    elif args.mode == "cascade_score":
        record.update(
            {
                "large_prefill_tokens": 0,
                "escalated": None,
                "small_winner_answer": None,
                "branch_answers": {},
                "branch_answer_leader": None,
                "leader_count": 0,
                "voter_count": 0,
                "leader_matches_winner": False,
                "winner_branch_id": None,
                "scored_candidates": [],
                "span_scores": {},
                "raw_logprob_lengths": {},
                "estimated_candidate_tokens": {},
                "scoring_prompt_tokens": {},
                "scoring_prompt_token_sources": {},
                "scoring_errors": ["internal_error:scoring_not_completed"],
                "chosen_by": "fallback",
            }
        )
    elif args.mode == "large_bo8":
        record["sample_answers"] = []
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
        "small_tokens",
        "large_tokens",
        "total_cost_units",
        "wall_s",
        "error",
    )
    if any(field not in record for field in required):
        return False
    key = record_key(record)
    if key is None or key[0] not in MODES:
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
    if nonnegative_int(record.get("small_tokens")) is None:
        return False
    if nonnegative_int(record.get("large_tokens")) is None:
        return False
    if nonnegative_number(record.get("total_cost_units")) is None:
        return False
    if nonnegative_number(record.get("wall_s")) is None:
        return False
    if record.get("error") is not None and not isinstance(record.get("error"), str):
        return False
    if key[0] in ("cascade", "cascade_score"):
        cascade_fields = (
            "escalated",
            "small_winner_answer",
            "branch_answers",
            "branch_answer_leader",
            "leader_count",
            "voter_count",
            "leader_matches_winner",
            "winner_branch_id",
        )
        if any(field not in record for field in cascade_fields):
            return False
        if record.get("escalated") is not None and not isinstance(
            record.get("escalated"), bool
        ):
            return False
        for field in (
            "small_winner_answer",
            "branch_answer_leader",
            "winner_branch_id",
        ):
            if record.get(field) is not None and not isinstance(record.get(field), str):
                return False
        if not isinstance(record.get("branch_answers"), dict):
            return False
        if nonnegative_int(record.get("leader_count")) is None:
            return False
        if nonnegative_int(record.get("voter_count")) is None:
            return False
        if not isinstance(record.get("leader_matches_winner"), bool):
            return False
    if key[0] == "cascade_score":
        cascade_score_fields = (
            "large_prefill_tokens",
            "scored_candidates",
            "span_scores",
            "raw_logprob_lengths",
            "estimated_candidate_tokens",
            "scoring_prompt_tokens",
            "scoring_prompt_token_sources",
            "scoring_errors",
            "chosen_by",
        )
        if any(field not in record for field in cascade_score_fields):
            return False
        if (
            nonnegative_int(record.get("large_prefill_tokens")) is None
            or not isinstance(record.get("scored_candidates"), list)
            or not isinstance(record.get("scoring_errors"), list)
            or record.get("chosen_by") not in ("vote", "score", "fallback")
        ):
            return False
        for field in (
            "span_scores",
            "raw_logprob_lengths",
            "estimated_candidate_tokens",
            "scoring_prompt_tokens",
            "scoring_prompt_token_sources",
        ):
            if not isinstance(record.get(field), dict):
                return False
    if key[0] == "large_bo8" and not isinstance(record.get("sample_answers"), list):
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


def summarize(args, seed, records):
    items = len(records)
    correct_count = sum(record.get("correct") is True for record in records)
    total_small_tokens = sum(
        nonnegative_int(record.get("small_tokens")) or 0 for record in records
    )
    total_large_tokens = sum(
        nonnegative_int(record.get("large_tokens")) or 0 for record in records
    )
    total_large_prefill_tokens = sum(
        nonnegative_int(record.get("large_prefill_tokens")) or 0
        for record in records
    )
    if args.mode == "cascade_score":
        total_cost_units = item_cost(args, total_small_tokens, total_large_tokens)
        total_cost_units += total_large_prefill_tokens * args.large_prefill_cost
    else:
        total_cost_units = item_cost(args, total_small_tokens, total_large_tokens)
    wall_times = [nonnegative_number(record.get("wall_s")) or 0.0 for record in records]
    summary = {
        "mode": args.mode,
        "seed": seed,
        "items": items,
        "correct_count": correct_count,
        "accuracy": correct_count / items if items else 0.0,
        "total_small_tokens": total_small_tokens,
        "total_large_tokens": total_large_tokens,
        "total_cost_units": total_cost_units,
        "cost_per_correct": total_cost_units / max(correct_count, 1),
        "mean_wall_s": statistics.mean(wall_times) if wall_times else 0.0,
        "error_count": sum(bool(record.get("error")) for record in records),
    }
    if args.mode == "cascade_score":
        summary["total_large_prefill_tokens"] = total_large_prefill_tokens
    if args.mode in ("cascade", "cascade_score"):
        observed = [
            record["escalated"]
            for record in records
            if isinstance(record.get("escalated"), bool)
        ]
        escalated_count = sum(value is True for value in observed)
        summary.update(
            {
                "escalated_count": escalated_count,
                "escalation_observed_items": len(observed),
                "escalation_rate": (
                    escalated_count / len(observed) if observed else 0.0
                ),
            }
        )
    return summary


def print_summary_table(summaries):
    print(
        "MODE          SEED  ITEMS  CORRECT  ACCURACY  ESC_RATE  SMALL_TOKENS  "
        "LARGE_TOKENS  COST_UNITS  COST/CORRECT  MEAN_WALL_S  ERRORS"
    )
    for row in summaries:
        escalation_rate = row.get("escalation_rate")
        print(
            "{mode:<13} {seed:>5} {items:>6} {correct:>8} {accuracy:>9.2%} "
            "{escalation:>9} {small:>13} {large:>13} {cost:>11.3f} "
            "{per_correct:>13.3f} {wall:>12.3f} {errors:>7}".format(
                mode=row["mode"],
                seed=row["seed"],
                items=row["items"],
                correct=row["correct_count"],
                accuracy=row["accuracy"],
                escalation=(
                    "-" if escalation_rate is None else "{:.2%}".format(escalation_rate)
                ),
                small=row["total_small_tokens"],
                large=row["total_large_tokens"],
                cost=row["total_cost_units"],
                per_correct=row["cost_per_correct"],
                wall=row["mean_wall_s"],
                errors=row["error_count"],
            )
        )
        if row["mode"] == "cascade_score":
            print(
                "cascade_score_seed_{}_large_prefill_tokens: {}".format(
                    row["seed"], row["total_large_prefill_tokens"]
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
    requested_keys = {(args.mode, seed, item["id"]) for seed in seeds for item in items}
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
                            record = run_mode_item(args, item, seed)
                        except Exception as exc:
                            record = error_record(args, item, seed, exc)
                        key = (args.mode, seed, item["id"])
                        with write_lock:
                            if key in existing or key in records_by_key:
                                continue
                            try:
                                append_record(output, record)
                            except Exception as exc:
                                fatal_errors.append(
                                    "{}:{}:{}".format(
                                        type(exc).__name__,
                                        item["id"],
                                        compact_error(exc),
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
        seed_records = [records_by_key[(args.mode, seed, item["id"])] for item in items]
        summaries.append(summarize(args, seed, seed_records))
    output = {
        "config": {
            "mode": args.mode,
            "small_model": args.small_model,
            "large_model": args.large_model,
            "small_url": args.small_url,
            "large_url": args.large_url,
            "data": os.path.abspath(args.data),
            "offset": args.offset,
            "limit": args.limit,
            "seeds": seeds,
            "branches": args.branches,
            "agree_threshold": args.agree_threshold,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "greedy_temperature": 0.0,
            "small_cost": args.small_cost,
            "large_cost": args.large_cost,
            "timeout": args.timeout,
            "concurrency": args.concurrency,
            "answer_suffix": ANSWER_SUFFIX,
            "out_jsonl": os.path.abspath(args.out_jsonl),
        },
        "summaries": summaries,
    }
    if args.mode == "cascade_score":
        output["config"]["large_prefill_cost"] = args.large_prefill_cost
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{} does not contain a JSON object".format(path))
    return value


def rows_for_mode(document, expected_mode, path):
    rows = summary_rows(document)
    matching = [row for row in rows if row.get("mode") == expected_mode]
    if not matching:
        raise ValueError("{} has no {!r} summary rows".format(path, expected_mode))
    return matching


def aggregate_summaries(rows, mode):
    items = sum(nonnegative_int(row.get("items")) or 0 for row in rows)
    correct_count = sum(nonnegative_int(row.get("correct_count")) or 0 for row in rows)
    total_small_tokens = sum(
        nonnegative_int(row.get("total_small_tokens")) or 0 for row in rows
    )
    total_large_tokens = sum(
        nonnegative_int(row.get("total_large_tokens")) or 0 for row in rows
    )
    total_cost_units = sum(
        nonnegative_number(row.get("total_cost_units")) or 0.0 for row in rows
    )
    aggregate = {
        "items": items,
        "correct_count": correct_count,
        "accuracy": correct_count / items if items else 0.0,
        "total_small_tokens": total_small_tokens,
        "total_large_tokens": total_large_tokens,
        "total_cost_units": total_cost_units,
        "cost_per_correct": total_cost_units / max(correct_count, 1),
    }
    if mode == "cascade_score":
        aggregate["total_large_prefill_tokens"] = sum(
            nonnegative_int(row.get("total_large_prefill_tokens")) or 0
            for row in rows
        )
    if mode in ("cascade", "cascade_score"):
        escalated_count = sum(
            nonnegative_int(row.get("escalated_count")) or 0 for row in rows
        )
        observed_items = sum(
            nonnegative_int(row.get("escalation_observed_items")) or 0 for row in rows
        )
        aggregate.update(
            {
                "escalated_count": escalated_count,
                "escalation_observed_items": observed_items,
                "escalation_rate": (
                    escalated_count / observed_items if observed_items else 0.0
                ),
            }
        )
    return aggregate


def compare_config_mismatches(documents):
    configs = []
    for document in documents:
        config = document.get("config")
        if not isinstance(config, dict):
            return []
        configs.append(config)
    keys = (
        "large_model",
        "large_url",
        "data",
        "offset",
        "limit",
        "seeds",
        "branches",
        "max_tokens",
        "temperature",
        "small_cost",
        "large_cost",
        "timeout",
        "concurrency",
        "answer_suffix",
    )
    first = configs[0]
    return [
        key
        for key in keys
        if any(config.get(key) != first.get(key) for config in configs[1:])
    ]


def ratio(numerator, denominator):
    if denominator <= 0:
        return None
    return numerator / denominator


def format_ratio(value):
    return "n/a" if value is None else "{:.6f}".format(value)


def print_comparison(mode, baseline, tree_summary, tree_mode="cascade"):
    accuracy_delta_points = (
        tree_summary["accuracy"] - baseline["accuracy"]
    ) * 100.0
    cost_ratio = ratio(
        baseline["cost_per_correct"], tree_summary["cost_per_correct"]
    )
    print(
        "{}_accuracy: {:.6%} ({}/{})".format(
            mode, baseline["accuracy"], baseline["correct_count"], baseline["items"]
        )
    )
    print(
        "accuracy_delta_{}_minus_{}: {:+.6f} pt".format(
            tree_mode, mode, accuracy_delta_points
        )
    )
    print("{}_total_small_tokens: {}".format(mode, baseline["total_small_tokens"]))
    print("{}_total_large_tokens: {}".format(mode, baseline["total_large_tokens"]))
    print("{}_total_cost_units: {:.6f}".format(mode, baseline["total_cost_units"]))
    print("{}_cost_per_correct: {:.6f}".format(mode, baseline["cost_per_correct"]))
    print(
        "cost_per_correct_ratio_{}_over_{}: {}".format(
            mode, tree_mode, format_ratio(cost_ratio)
        )
    )
    if cost_ratio is None:
        conclusion = "cost-per-correct ratio is unavailable"
    elif cost_ratio > 1.0:
        conclusion = "{} is cheaper per correct in this run".format(tree_mode)
    elif cost_ratio < 1.0:
        conclusion = "{} is not cheaper per correct in this run".format(tree_mode)
    else:
        conclusion = "cost per correct is equal in this run"
    print(
        "verdict_{}: {}; baseline/{} ratio {}, {} cost/correct {:.6f}, "
        "baseline cost/correct {:.6f}, {} accuracy delta {:+.6f} pt, {} "
        "escalation rate {:.6%}".format(
            mode,
            conclusion,
            tree_mode,
            format_ratio(cost_ratio),
            tree_mode,
            tree_summary["cost_per_correct"],
            baseline["cost_per_correct"],
            tree_mode,
            accuracy_delta_points,
            tree_mode,
            tree_summary["escalation_rate"],
        )
    )


def compare_summaries(paths):
    documents = [load_json(path) for path in paths]
    first_modes = {
        row.get("mode")
        for row in summary_rows(documents[0])
        if row.get("mode") is not None
    }
    if len(first_modes) != 1:
        raise ValueError("first summary JSON must contain exactly one mode")
    tree_mode = next(iter(first_modes))
    if tree_mode not in ("cascade", "cascade_score"):
        raise ValueError("first summary mode must be 'cascade' or 'cascade_score'")
    expected_modes = [tree_mode]
    for document in documents[1:]:
        baseline_modes = {
            row.get("mode")
            for row in summary_rows(document)
            if row.get("mode") is not None
        }
        if len(baseline_modes) != 1:
            raise ValueError("baseline summary JSON must contain exactly one mode")
        baseline_mode = next(iter(baseline_modes))
        if baseline_mode not in ("large_bo8", "large_greedy"):
            raise ValueError(
                "baseline summary mode must be 'large_bo8' or 'large_greedy'"
            )
        if baseline_mode in expected_modes:
            raise ValueError("duplicate baseline mode {!r}".format(baseline_mode))
        expected_modes.append(baseline_mode)
    mismatches = compare_config_mismatches(documents)
    if mismatches:
        raise ValueError("summary configs differ for: {}".format(", ".join(mismatches)))
    aggregates = {}
    for path, document, mode in zip(paths, documents, expected_modes):
        aggregates[mode] = aggregate_summaries(
            rows_for_mode(document, mode, path), mode
        )
    tree_summary = aggregates[tree_mode]
    for mode in expected_modes[1:]:
        if aggregates[mode]["items"] != tree_summary["items"]:
            raise ValueError(
                "{} and {} item counts differ: {} versus {}".format(
                    tree_mode,
                    mode,
                    tree_summary["items"],
                    aggregates[mode]["items"],
                )
            )

    print(
        "{}_accuracy: {:.6%} ({}/{})".format(
            tree_mode,
            tree_summary["accuracy"],
            tree_summary["correct_count"],
            tree_summary["items"],
        )
    )
    print(
        "{}_total_small_tokens: {}".format(
            tree_mode, tree_summary["total_small_tokens"]
        )
    )
    print(
        "{}_total_large_tokens: {}".format(
            tree_mode, tree_summary["total_large_tokens"]
        )
    )
    if tree_mode == "cascade_score":
        print(
            "cascade_score_total_large_prefill_tokens: {}".format(
                tree_summary["total_large_prefill_tokens"]
            )
        )
    print(
        "{}_total_cost_units: {:.6f}".format(
            tree_mode, tree_summary["total_cost_units"]
        )
    )
    print(
        "{}_cost_per_correct: {:.6f}".format(
            tree_mode, tree_summary["cost_per_correct"]
        )
    )
    print(
        "{}_escalation_rate: {:.6%}".format(
            tree_mode, tree_summary["escalation_rate"]
        )
    )
    for mode in expected_modes[1:]:
        print_comparison(mode, aggregates[mode], tree_summary, tree_mode)


def print_dry_run(args, item, seed):
    small_base = args.small_url.rstrip("/")
    large_base = args.large_url.rstrip("/")
    document = {
        "item_id": item["id"],
        "item_index": item["item_index"],
        "seed": seed,
        "cascade": {
            "small_tree": {
                "url": small_base + "/v1/tree/completions",
                "body": small_tree_body(args, item, seed),
            },
            "large_fallback_if_needed": {
                "url": large_base + "/v1/chat/completions",
                "body": large_greedy_body(args, item, seed),
            },
        },
        "cascade_score": {
            "small_tree": {
                "url": small_base + "/v1/tree/completions",
                "body": small_tree_body(args, item, seed),
            },
            "large_scoring_if_needed": [
                {
                    "candidate": candidate,
                    "url": large_base + "/generate",
                    "body": large_score_body(item, candidate),
                }
                for candidate in ("<candidate_1>", "<candidate_2>")
            ],
            "large_fallback_if_scoring_fails": {
                "url": large_base + "/v1/chat/completions",
                "body": large_greedy_body(args, item, seed),
            },
        },
        "large_bo8": [
            {
                "sample_index": sample_index,
                "url": large_base + "/v1/chat/completions",
                "body": large_sample_body(args, item, seed, sample_index),
            }
            for sample_index in range(args.branches)
        ],
        "large_greedy": {
            "url": large_base + "/v1/chat/completions",
            "body": large_greedy_body(args, item, seed),
        },
    }
    print(json.dumps(document, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure a small reasoning-tree to large-model confidence cascade "
            "against large best-of-n and greedy baselines."
        )
    )
    parser.add_argument("--data", help="GSM8K-style JSONL input")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--small-model", help="served small model name")
    parser.add_argument("--large-model", help="served large model name")
    parser.add_argument("--small-url", default="http://127.0.0.1:30000")
    parser.add_argument("--large-url", default="http://127.0.0.1:30001")
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--agree-threshold", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--small-cost", type=float, default=1.0)
    parser.add_argument("--large-cost", type=float, default=10.0)
    parser.add_argument(
        "--large-prefill-cost",
        type=float,
        default=2.0,
        help="large-model prefill cost units per token for cascade_score",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-jsonl", help="incremental per-item JSONL output")
    parser.add_argument("--out", help="final summary JSON output")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="SUMMARY_JSON",
        help=(
            "compare cascade or cascade_score JSON with one or both of "
            "large_bo8 and large_greedy JSON"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print all four modes' request bodies for the first selected item",
    )
    return parser


def validate_measurement_args(parser, args):
    if not args.data:
        parser.error("--data is required unless --compare is used")
    if (args.mode in ("cascade", "cascade_score") or args.dry_run) and not args.small_model:
        parser.error(
            "--small-model is required for cascade, cascade_score, and --dry-run"
        )
    if not args.large_model:
        parser.error("--large-model is required unless --compare is used")
    if args.branches <= 0 or args.branches > 64:
        parser.error("--branches must be between 1 and 64")
    if args.agree_threshold <= 0:
        parser.error("--agree-threshold must be positive")
    if args.max_tokens <= 0 or args.max_tokens > 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    if args.temperature <= 0 or args.temperature > 2.0:
        parser.error("--temperature must be greater than 0 and at most 2.0")
    if args.small_cost < 0:
        parser.error("--small-cost must be nonnegative")
    if args.large_cost < 0:
        parser.error("--large-cost must be nonnegative")
    if args.large_prefill_cost < 0:
        parser.error("--large-prefill-cost must be nonnegative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.offset < 0:
        parser.error("--offset must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
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
        if len(args.compare) not in (2, 3):
            parser.error("--compare requires two or three summary JSON files")
        try:
            compare_summaries(args.compare)
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
