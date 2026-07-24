#!/usr/bin/env python3
"""Measure small-tree to large-model cascade economics."""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
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
ANSWER_SUFFIX_MATH = (
    "\nSolve step by step, then give the final answer on the last line as: "
    "Answer: <answer>"
)
# Answer handling mode, set once by main() before any item work. "numeric"
# is the exact historical behavior; "math" routes extraction, equivalence,
# and vote keying through bench/tasks/math_equiv.py so LaTeX answers
# (\frac{3}{2}, 90^\circ) are judged by mathematical equivalence.
ANSWER_MODE = "numeric"
_MATH_EQUIV = None


def set_answer_mode(mode):
    global ANSWER_MODE, _MATH_EQUIV
    ANSWER_MODE = mode
    if mode == "math" and _MATH_EQUIV is None:
        import importlib.util

        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "tasks",
            "math_equiv.py",
        )
        spec = importlib.util.spec_from_file_location("math_equiv", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MATH_EQUIV = module


def answer_suffix():
    return ANSWER_SUFFIX_MATH if ANSWER_MODE == "math" else ANSWER_SUFFIX


def answers_match(candidate, reference):
    if candidate is None or reference is None:
        return False
    if ANSWER_MODE == "math":
        try:
            return bool(_MATH_EQUIV.is_equiv(candidate, reference))
        except Exception:
            return False
    return candidate == reference


def vote_key(answer):
    """Group key for majority voting: math mode groups by normalized
    equivalence class so \\frac{1}{2} and 0.5 vote together."""
    if answer is None:
        return None
    if ANSWER_MODE == "math":
        try:
            return _MATH_EQUIV.normalize_answer(answer) or answer
        except Exception:
            return answer
    return answer


NUMBER_RE = re.compile(
    r"[-+]?\s*\$?\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)\s*%?"
)
MODES = (
    "cascade",
    "large_bo8",
    "large_greedy",
    "cascade_score",
    "large_tree",
    "adaptive_k",
    "spec_tree",
)
DEFAULT_SPEC_TREE_SMALL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GATE_FEATURE_NAMES = (
    "leader_count",
    "voter_count",
    "leader_share",
    "n_distinct_answers",
    "second_count",
    "margin",
    "null_fraction",
    "leader_matches_winner",
    "small_tokens_per_branch",
)
GATE_MODEL_VERSION = 1


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
    if ANSWER_MODE == "math":
        return _MATH_EQUIV.extract_final_answer(text)
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
    """Choose the most frequent answer, breaking ties by first occurrence.
    Math mode groups by equivalence class but returns the original string of
    the first voter in the winning class."""
    if not answers:
        return None
    keyed = [(vote_key(answer), answer) for answer in answers]
    counts = collections.Counter(key for key, _ in keyed)
    winning_count = max(counts.values())
    for key, answer in keyed:
        if counts[key] == winning_count:
            return answer
    return None


def plurality_report(answers):
    """Return plurality answer and top two counts with first-seen tie breaks."""
    keyed = [(vote_key(answer), answer) for answer in answers if answer is not None]
    if not keyed:
        return None, 0, 0
    counts = collections.Counter(key for key, _answer in keyed)
    ranked_counts = sorted(counts.values(), reverse=True)
    leader_count = ranked_counts[0]
    runner_up_count = ranked_counts[1] if len(ranked_counts) > 1 else 0
    for key, answer in keyed:
        if counts[key] == leader_count:
            return answer, leader_count, runner_up_count
    return None, 0, 0


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


def parse_k_ladder(raw):
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--k-ladder must be a comma-separated list of integers")
    try:
        ladder = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--k-ladder must be a comma-separated list of integers") from exc
    if any(k <= 0 or k > 64 for k in ladder):
        raise ValueError("--k-ladder values must be between 1 and 64")
    return ladder


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
            if ANSWER_MODE == "math":
                normalized_gold = row["gold"].strip()
                if not normalized_gold:
                    raise ValueError(
                        "{} line {} has empty gold".format(path, line_number)
                    )
            else:
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
    return [{"role": "user", "content": item["prompt"] + answer_suffix()}]


def item_seed(base_seed, item):
    return base_seed + item["item_index"]


def sample_seed(base_seed, item, sample_index):
    return base_seed + item["item_index"] * 1000 + sample_index


def small_tree_body(args, item, base_seed):
    tree = {
        "policy": "beam",
        "branches": args.branches,
        "budget_tokens": args.branches * args.max_tokens,
    }
    if args.fork_at_entropy is not None:
        # Entropy-triggered adaptive forking: the parent decodes alone until
        # its windowed uncertainty proxy crosses this threshold (nats), then
        # all branches fork at that point (engine feature fork_at_entropy).
        tree["fork_at_entropy"] = args.fork_at_entropy
    return {
        "model": args.small_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": item_seed(base_seed, item),
        "tree": tree,
    }


def large_sample_body(args, item, base_seed, sample_index):
    return {
        "model": args.large_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": sample_seed(base_seed, item, sample_index),
    }


def large_tree_body(args, item, base_seed):
    """Tree request pointed at the LARGE model: fork/prune/vote/majority-lock
    as a cheaper best-of-k on one capable model. No small model, no escalation.
    Its own vote answer is the response; its decode tokens are charged at the
    large-decode rate so cost-per-correct is directly comparable to large_bo8."""
    tree = {
        "policy": "beam",
        "branches": args.branches,
        "budget_tokens": args.branches * args.max_tokens,
    }
    if args.fork_at_entropy is not None:
        tree["fork_at_entropy"] = args.fork_at_entropy
    return {
        "model": args.large_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": item_seed(base_seed, item),
        "tree": tree,
    }


def adaptive_k_tree_body(args, item, base_seed, rung_index, branches):
    tree = {
        "policy": "beam",
        "branches": branches,
        "budget_tokens": branches * args.max_tokens,
    }
    if args.fork_at_entropy is not None:
        tree["fork_at_entropy"] = args.fork_at_entropy
    return {
        "model": args.large_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": item_seed(base_seed, item) + rung_index,
        "tree": tree,
    }


def large_greedy_body(args, item, base_seed):
    return {
        "model": args.large_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": item_seed(base_seed, item),
    }


def small_sample_body(args, item, base_seed, sample_index):
    return {
        "model": args.small_model,
        "messages": prompt_messages(item),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": sample_seed(base_seed, item, sample_index),
    }


def spec_tree_verify_prompt(item):
    return item["prompt"] + answer_suffix() + "\n"


def spec_tree_verify_body(args, item, continuations):
    return {
        "model": args.large_model,
        "prompt": spec_tree_verify_prompt(item),
        "continuations": continuations,
    }


def spec_tree_tokenize_body(args, item):
    return {
        "model": args.large_model,
        "prompt": spec_tree_verify_prompt(item),
        "add_special_tokens": True,
    }


def spec_tree_native_verify_body(item, continuation, prompt_tokens):
    return {
        "text": spec_tree_verify_prompt(item) + continuation,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": prompt_tokens,
    }


def large_score_body(item, candidate):
    return {
        "text": item["prompt"] + answer_suffix() + "\nAnswer: " + candidate,
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


def gate_nonnegative_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def require_gate_features(record, path, line_number):
    """Mirror bench/valuehead/train_gate.py require_cascade_features exactly."""
    where = "{} line {}".format(path, line_number)
    if record.get("mode") != "cascade":
        raise ValueError("{} field 'mode' must be 'cascade'".format(where))
    branch_answers = record.get("branch_answers")
    if not isinstance(branch_answers, dict):
        raise ValueError("{} field 'branch_answers' must be an object".format(where))
    branches = len(branch_answers)
    if branches <= 0:
        raise ValueError("{} has no branches in 'branch_answers'".format(where))

    votes = []
    for branch_id, answer in branch_answers.items():
        if answer is None:
            continue
        normalized = normalize_number(answer)
        if normalized is None:
            raise ValueError(
                "{} branch {!r} has a non-numeric answer".format(where, branch_id)
            )
        votes.append(normalized)
    voter_count = gate_nonnegative_int(record.get("voter_count"))
    if voter_count is None or voter_count != len(votes):
        raise ValueError(
            "{} voter_count must match numeric branch answers".format(where)
        )
    counts = collections.Counter(votes)
    ranked_counts = sorted(counts.values(), reverse=True)
    computed_leader_count = ranked_counts[0] if ranked_counts else 0
    leader_count = gate_nonnegative_int(record.get("leader_count"))
    if leader_count is None or leader_count != computed_leader_count:
        raise ValueError("{} leader_count must match branch answers".format(where))
    leader_matches_winner = record.get("leader_matches_winner")
    if not isinstance(leader_matches_winner, bool):
        raise ValueError("{} leader_matches_winner must be bool".format(where))
    small_tokens = nonnegative_number(record.get("small_tokens"))
    if small_tokens is None:
        raise ValueError("{} small_tokens must be nonnegative".format(where))

    second_count = ranked_counts[1] if len(ranked_counts) > 1 else 0
    voter_denominator = max(voter_count, 1)
    return [
        float(leader_count),
        float(voter_count),
        leader_count / voter_denominator,
        float(len(counts)),
        float(second_count),
        (leader_count - second_count) / voter_denominator,
        (branches - voter_count) / branches,
        1.0 if leader_matches_winner else 0.0,
        (small_tokens / branches) / 512.0,
    ]


def gate_sigmoid(value):
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def load_gate_model(path):
    with open(path, "r", encoding="utf-8") as handle:
        model = json.load(handle)
    if not isinstance(model, dict):
        raise ValueError("{} does not contain a JSON object".format(path))
    if (
        model.get("model_type") != "logistic_regression"
        or model.get("version") != GATE_MODEL_VERSION
        or model.get("feature_names") != list(GATE_FEATURE_NAMES)
    ):
        raise ValueError("{} is not a supported gate model".format(path))
    standardization = model.get("standardization")
    weights_by_name = model.get("weights")
    if not isinstance(standardization, dict) or not isinstance(weights_by_name, dict):
        raise ValueError("{} is missing standardization or weights".format(path))
    try:
        means = [float(standardization["means"][name]) for name in GATE_FEATURE_NAMES]
        stds = [float(standardization["stds"][name]) for name in GATE_FEATURE_NAMES]
        weights = [float(weights_by_name[name]) for name in GATE_FEATURE_NAMES]
        intercept = float(model["intercept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("{} has invalid model parameters".format(path)) from exc
    values = means + stds + weights + [intercept]
    if not all(math.isfinite(value) for value in values) or any(
        value < 0.0 for value in stds
    ):
        raise ValueError("{} has non-finite parameters or negative stds".format(path))
    return {
        "intercept": intercept,
        "means": means,
        "stds": stds,
        "weights": weights,
    }


def gate_model_config(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {"basename": os.path.basename(path), "sha256": digest.hexdigest()}


def gate_probability(features, model):
    standardized = [
        (value - model["means"][index])
        / (model["stds"][index] if model["stds"][index] > 0.0 else 1.0)
        for index, value in enumerate(features)
    ]
    logit = model["intercept"] + sum(
        weight * value for weight, value in zip(model["weights"], standardized)
    )
    return gate_sigmoid(logit)


def confidence_decision(
    args,
    small_response_valid,
    branch_answers,
    leader_count,
    voter_count,
    leader_matches_winner,
    small_tokens,
):
    if not args.gate_model:
        return (
            small_response_valid
            and leader_count >= args.agree_threshold
            and voter_count >= 4,
            None,
            False,
        )
    if not isinstance(branch_answers, dict) or not branch_answers:
        return False, None, True
    feature_record = {
        "mode": "cascade",
        "branch_answers": branch_answers,
        "leader_count": leader_count,
        "voter_count": voter_count,
        "leader_matches_winner": leader_matches_winner,
        "small_tokens": small_tokens,
    }
    try:
        features = require_gate_features(feature_record, "<runtime>", 1)
    except (TypeError, ValueError):
        return False, None, False
    probability = gate_probability(features, args.gate_model_data)
    return (
        small_response_valid and probability >= args.gate_threshold,
        probability,
        False,
    )


def add_gate_fields(record, args, probability):
    if args.gate_model:
        record.update(
            {
                "gate_p": probability,
                "gate_threshold": args.gate_threshold,
                "gated": True,
            }
        )


def format_selftest_float(value):
    return "{:.12f}".format(value)


def run_gate_selftest(model_path, items_path):
    model = load_gate_model(model_path)
    record_index = 0
    with open(items_path, "r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON in {} line {}: {}".format(
                        items_path, line_number, exc
                    )
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    "{} line {} is not a JSON object".format(items_path, line_number)
                )
            record_index += 1
            output = {
                "feature_names": list(GATE_FEATURE_NAMES),
                "id": record.get("id"),
                "index": record_index,
                "line": line_number,
            }
            if record.get("error") is not None:
                output.update(
                    {"error": "input_error", "features": None, "gate_p": None}
                )
            else:
                try:
                    features = require_gate_features(record, items_path, line_number)
                    probability = gate_probability(features, model)
                    output.update(
                        {
                            "error": None,
                            "features": {
                                name: format_selftest_float(features[index])
                                for index, name in enumerate(GATE_FEATURE_NAMES)
                            },
                            "gate_p": format_selftest_float(probability),
                        }
                    )
                except (TypeError, ValueError) as exc:
                    output.update({"error": str(exc), "features": None, "gate_p": None})
            print(
                json.dumps(
                    output,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            if record_index == 5:
                break
    if record_index == 0:
        raise ValueError("{} contains no JSON records".format(items_path))


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


def branch_id_sort_key(branch_id):
    text = str(branch_id)
    try:
        return 0, int(text), text
    except ValueError:
        return 1, text, text


def branch_text_value(value):
    if isinstance(value, str):
        return value if value else None
    if not isinstance(value, dict):
        return None
    for key in ("text", "content", "continuation", "output_text"):
        text = value.get(key)
        if isinstance(text, str) and text:
            return text
    message = value.get("message")
    if isinstance(message, dict):
        text = message.get("content")
        if isinstance(text, str) and text:
            return text
    return None


def branch_text_report(payload):
    if not isinstance(payload, dict):
        return {}
    tree = payload.get("tree")
    containers = [tree, payload] if isinstance(tree, dict) else [payload]
    for container in containers:
        for key in ("branch_texts", "branch_continuations", "branch_outputs"):
            raw = container.get(key)
            texts = {}
            if isinstance(raw, dict):
                for branch_id, value in raw.items():
                    text = branch_text_value(value)
                    if text is not None:
                        texts[str(branch_id)] = text
            elif isinstance(raw, list):
                for branch_id, value in enumerate(raw):
                    text = branch_text_value(value)
                    if text is not None:
                        texts[str(branch_id)] = text
            if texts:
                return texts
    if isinstance(tree, dict) and isinstance(tree.get("branches"), dict):
        texts = {}
        for branch_id, value in tree["branches"].items():
            text = branch_text_value(value)
            if text is not None:
                texts[str(branch_id)] = text
        if texts:
            return texts
    return {}


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
            # Non-string, non-null values are protocol corruption.
            normalized[key] = None
            valid = False
            continue
        if ANSWER_MODE == "math":
            answer = value.strip() or None
        else:
            # Numeric mode: the server may legitimately send non-numeric
            # string votes (word answers). They abstain rather than
            # invalidating the whole report.
            answer = normalize_number(value)
        normalized[key] = answer
        if answer is not None:
            voters.append(answer)
    leader = majority_vote(voters)
    if leader is None:
        leader_count = 0
    else:
        leader_key = vote_key(leader)
        leader_count = sum(1 for v in voters if vote_key(v) == leader_key)
    return normalized, leader, leader_count, len(voters), valid


def tree_winner_branch_id(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), dict):
        return None
    value = payload["tree"].get("winner_branch_id")
    return None if value is None else str(value)


def tree_pruned_count(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("tree"), dict):
        return None
    return nonnegative_int(payload["tree"].get("pruned_count"))


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


def first_mapping_value(mapping, names):
    if not isinstance(mapping, dict):
        return None
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def batched_verify_report(payload, expected_count):
    errors = []
    scores = []
    token_counts = []
    rows = first_mapping_value(payload, ("scores", "results", "verifications"))
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                score = finite_number(
                    first_mapping_value(
                        row, ("mean_logprob", "target_mean_logprob", "score")
                    )
                )
                count = nonnegative_int(
                    first_mapping_value(row, ("n_tokens", "token_count", "tokens"))
                )
            else:
                score = finite_number(row)
                count = None
            scores.append(score)
            token_counts.append(count)
    else:
        raw_scores = first_mapping_value(
            payload, ("mean_logprobs", "target_scores")
        )
        if isinstance(raw_scores, list):
            scores = [finite_number(value) for value in raw_scores]
            token_counts = [None] * len(scores)

    raw_counts = first_mapping_value(payload, ("n_tokens", "token_counts"))
    total_tokens = None
    if isinstance(raw_counts, list):
        parsed_counts = [nonnegative_int(value) for value in raw_counts]
        if len(parsed_counts) == len(token_counts):
            token_counts = [
                parsed if existing is None else existing
                for existing, parsed in zip(token_counts, parsed_counts)
            ]
        else:
            add_error(errors, "n_tokens_length_mismatch")
    else:
        total_tokens = nonnegative_int(raw_counts)
    if total_tokens is None:
        total_tokens = nonnegative_int(
            first_mapping_value(payload, ("total_n_tokens", "total_tokens"))
        )
    if total_tokens is None and token_counts and all(
        count is not None for count in token_counts
    ):
        total_tokens = sum(token_counts)

    if len(scores) != expected_count:
        add_error(
            errors,
            "score_count_mismatch_expected_{}_got_{}".format(
                expected_count, len(scores)
            ),
        )
    if any(score is None for score in scores):
        add_error(errors, "invalid_mean_logprob")
    if total_tokens is None:
        add_error(errors, "missing_n_tokens")
        total_tokens = 0
    elif expected_count and total_tokens == 0:
        add_error(errors, "zero_n_tokens")

    scores = (scores + [None] * expected_count)[:expected_count]
    token_counts = (token_counts + [None] * expected_count)[:expected_count]
    return {
        "scores": scores,
        "token_counts": token_counts,
        "total_tokens": total_tokens,
        "errors": errors,
    }


def run_tokenize_count_request(url, body, timeout):
    payload, request_error = post_json(url, body, timeout)
    errors = []
    add_error(errors, request_error)
    count = (
        nonnegative_int(payload.get("count")) if isinstance(payload, dict) else None
    )
    if request_error is None and count is None:
        add_error(errors, "missing_tokenize_count")
    return count, errors


def input_logprob_value(entry):
    if isinstance(entry, (list, tuple)) and entry:
        return finite_number(entry[0])
    if isinstance(entry, dict):
        return finite_number(first_mapping_value(entry, ("logprob", "value")))
    return finite_number(entry)


def run_native_verify_request(url, body, timeout):
    payload, request_error = post_json(url, body, timeout)
    meta_info = payload.get("meta_info") if isinstance(payload, dict) else None
    raw_logprobs = (
        meta_info.get("input_token_logprobs")
        if isinstance(meta_info, dict)
        else None
    )
    n_tokens = (
        nonnegative_int(meta_info.get("prompt_tokens"))
        if isinstance(meta_info, dict)
        else None
    )
    errors = []
    add_error(errors, request_error)
    if request_error is None and not isinstance(meta_info, dict):
        add_error(errors, "missing_meta_info")
    if n_tokens is None:
        add_error(errors, "missing_prompt_tokens")
        n_tokens = 0
    elif n_tokens == 0:
        add_error(errors, "zero_prompt_tokens")
    values = []
    if not isinstance(raw_logprobs, list) or not raw_logprobs:
        add_error(errors, "missing_input_token_logprobs")
    else:
        for entry in raw_logprobs:
            value = input_logprob_value(entry)
            if value is None:
                add_error(errors, "non_numeric_input_token_logprob")
                break
            values.append(value)
    return {
        "score": sum(values) / len(values) if not errors and values else None,
        "n_tokens": n_tokens,
        "raw_logprob_length": len(raw_logprobs) if isinstance(raw_logprobs, list) else 0,
        "errors": errors,
    }


def request_may_have_spent_tokens(request_error):
    if not request_error:
        return False
    return not request_error.startswith(("http_400:", "http_404:", "http_422:"))


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


def run_chat_text_request(url, body, timeout):
    payload, request_error = post_json(url, body, timeout)
    tokens, token_report_valid = chat_token_report(payload)
    content, content_error = completion_content(payload)
    return {
        "content": content,
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
        "correct": not errors and answers_match(answer, item["normalized_gold"]),
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

    leader_matches_winner = branch_leader is not None and answers_match(branch_leader, winner_answer)
    small_response_valid = not errors
    confident, gate_p, no_votes = confidence_decision(
        args,
        small_response_valid,
        branch_answers,
        leader_count,
        voter_count,
        leader_matches_winner,
        small_tokens,
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
            "leader_matches_winner": leader_matches_winner,
            "winner_branch_id": tree_winner_branch_id(payload),
        }
    )
    add_gate_fields(record, args, gate_p)
    if args.gate_model and no_votes:
        record["chosen_by"] = "no_votes"
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

    leader_matches_winner = branch_leader is not None and answers_match(branch_leader, winner_answer)
    small_response_valid = not errors
    confident, gate_p, no_votes = confidence_decision(
        args,
        small_response_valid,
        branch_answers,
        leader_count,
        voter_count,
        leader_matches_winner,
        small_tokens,
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
                "selection:need_two_distinct_candidates_got_{}".format(len(candidates)),
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
        if no_votes:
            chosen_by = "no_votes"

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
            "leader_matches_winner": leader_matches_winner,
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
    add_gate_fields(record, args, gate_p)
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


def run_large_tree_item(args, item, base_seed):
    started = time.perf_counter()
    url = args.large_url.rstrip("/") + "/v1/tree/completions"
    payload, request_error = post_json(
        url, large_tree_body(args, item, base_seed), args.timeout
    )
    large_tokens, token_report_valid = tree_token_report(payload)
    content, content_error = completion_content(payload)
    answer = extract_answer(content)
    (
        branch_answers,
        branch_leader,
        leader_count,
        voter_count,
        branch_report_valid,
    ) = branch_answer_report(payload)
    errors = prefixed_errors(
        "large_tree",
        request_errors(request_error, content_error, large_tokens, token_report_valid),
    )
    if request_error is None and not branch_report_valid:
        add_error(errors, "large_tree:invalid_branch_answers")
    # All generated tokens are large-model decode tokens: charge via large_tokens
    # so cost-per-correct is directly comparable to large_bo8.
    record = common_record(
        args, "large_tree", item, base_seed, started, answer, 0, large_tokens, errors
    )
    record.update(
        {
            "escalated": None,
            "branch_answers": branch_answers,
            "branch_answer_leader": branch_leader,
            "leader_count": leader_count,
            "voter_count": voter_count,
            "leader_matches_winner": (
                branch_leader is not None and answers_match(branch_leader, answer)
            ),
            "winner_branch_id": tree_winner_branch_id(payload),
            "pruned_count": tree_pruned_count(payload),
        }
    )
    return record


def run_adaptive_k_item(args, item, base_seed):
    started = time.perf_counter()
    url = args.large_url.rstrip("/") + "/v1/tree/completions"
    rungs_used = []
    answers_by_rung = []
    accumulated_answers = []
    large_tokens = 0
    errors = []
    answer = None
    leader_count = 0
    runner_up_count = 0
    stopped_early = False

    for rung_index, branches in enumerate(args.k_ladder_values):
        payload, request_error = post_json(
            url,
            adaptive_k_tree_body(args, item, base_seed, rung_index, branches),
            args.timeout,
        )
        rung_tokens, token_report_valid = tree_token_report(payload)
        large_tokens += rung_tokens
        _content, content_error = completion_content(payload)
        branch_answers, _branch_leader, _leader_count, _voter_count, valid = (
            branch_answer_report(payload)
        )
        prefix = "adaptive_k_rung_{}".format(branches)
        for error in prefixed_errors(
            prefix,
            request_errors(
                request_error, content_error, rung_tokens, token_report_valid
            ),
        ):
            add_error(errors, error)
        if request_error is None and not valid:
            add_error(errors, prefix + ":invalid_branch_answers")

        rung_answers = [
            branch_answers[branch_id]
            for branch_id in sorted(branch_answers, key=branch_id_sort_key)
        ]
        rungs_used.append(branches)
        answers_by_rung.append(rung_answers)
        accumulated_answers.extend(
            rung_answer for rung_answer in rung_answers if rung_answer is not None
        )
        answer, leader_count, runner_up_count = plurality_report(accumulated_answers)
        agreed = (
            leader_count - runner_up_count >= args.agree_margin
            and leader_count >= args.min_leader
        )
        if agreed:
            stopped_early = rung_index < len(args.k_ladder_values) - 1
            break

    if answer is None:
        add_error(errors, "adaptive_k:no_final_answer")
    record = common_record(
        args,
        "adaptive_k",
        item,
        base_seed,
        started,
        answer,
        0,
        large_tokens,
        errors,
    )
    record.update(
        {
            "rungs_used": rungs_used,
            "total_branches_sampled": sum(rungs_used),
            "answers_by_rung": answers_by_rung,
            "leader_count": leader_count,
            "runner_up_count": runner_up_count,
            "stopped_early": stopped_early,
        }
    )
    return record


def run_spec_tree_draft(args, item, base_seed):
    tree_url = args.small_url.rstrip("/") + "/v1/tree/completions"
    payload, request_error = post_json(
        tree_url, small_tree_body(args, item, base_seed), args.timeout
    )
    tree_tokens, tree_token_report_valid = tree_token_report(payload)
    branch_texts = branch_text_report(payload)
    branch_answers = branch_answer_report(payload)[0]
    ordered_branch_ids = sorted(branch_texts, key=branch_id_sort_key)
    draft_errors = []
    fatal_errors = []
    add_error(draft_errors, request_error)
    if request_error is None:
        if tree_tokens == 0:
            add_error(draft_errors, "tree:zero_tokens")
            add_error(fatal_errors, "draft:tree_cost_unknown_zero_tokens")
        elif not tree_token_report_valid:
            add_error(draft_errors, "tree:invalid_token_report")
            add_error(fatal_errors, "draft:tree_cost_unknown_invalid_report")
    elif request_may_have_spent_tokens(request_error):
        add_error(fatal_errors, "draft:tree_cost_unknown_after_request_error")

    if len(ordered_branch_ids) >= args.branches:
        draft_source = "tree"
        branch_ids = ordered_branch_ids[: args.branches]
        continuations = [branch_texts[branch_id] for branch_id in branch_ids]
        draft_answers = [
            extract_answer(text) or branch_answers.get(branch_id)
            for branch_id, text in zip(branch_ids, continuations)
        ]
        sample_tokens = 0
    else:
        draft_source = "samples"
        branch_ids = [str(index) for index in range(args.branches)]
        add_error(
            draft_errors,
            "tree:branch_texts_unavailable_expected_{}_got_{}".format(
                args.branches, len(ordered_branch_ids)
            ),
        )
        sample_url = args.small_url.rstrip("/") + "/v1/chat/completions"

        def run_sample(sample_index):
            return run_chat_text_request(
                sample_url,
                small_sample_body(args, item, base_seed, sample_index),
                args.timeout,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.branches
        ) as executor:
            futures = [
                executor.submit(run_sample, index) for index in range(args.branches)
            ]
            samples = [future.result() for future in futures]

        continuations = []
        draft_answers = []
        sample_tokens = 0
        for sample_index, sample in enumerate(samples):
            continuations.append(sample["content"])
            draft_answers.append(sample["answer"])
            sample_tokens += sample["tokens"]
            for error in prefixed_errors(
                "sample_{}".format(sample_index), sample["errors"]
            ):
                add_error(draft_errors, error)
                add_error(fatal_errors, "draft:" + error)

    return {
        "continuations": continuations,
        "draft_answers": draft_answers,
        "draft_source": draft_source,
        "branch_ids": branch_ids,
        "tree_tokens": tree_tokens,
        "sample_tokens": sample_tokens,
        "draft_tokens": tree_tokens + sample_tokens,
        "draft_errors": draft_errors,
        "fatal_errors": fatal_errors,
    }


def run_spec_tree_verify(args, item, continuations):
    target_scores = [None] * len(continuations)
    target_score_tokens = [None] * len(continuations)
    verify_errors = []
    fatal_errors = []
    valid = [
        (index, continuation)
        for index, continuation in enumerate(continuations)
        if isinstance(continuation, str) and continuation
    ]
    verify_path = "batched"
    verify_prefill_tokens = 0
    if not valid:
        add_error(verify_errors, "selection:no_draft_continuations")
        add_error(fatal_errors, "verify:no_draft_continuations")
        return {
            "target_scores": target_scores,
            "target_score_tokens": target_score_tokens,
            "verify_prefill_tokens": verify_prefill_tokens,
            "verify_path": verify_path,
            "verify_errors": verify_errors,
            "fatal_errors": fatal_errors,
        }

    batched_url = args.large_url.rstrip("/") + "/v1/tree/verify"
    batched_body = spec_tree_verify_body(
        args, item, [continuation for _index, continuation in valid]
    )
    payload, request_error = post_json(batched_url, batched_body, args.timeout)
    if request_error and request_error.startswith("http_404:"):
        add_error(verify_errors, "batched:" + request_error)
        verify_path = "fallback"
        tokenize_url = args.large_url.rstrip("/") + "/v1/tokenize"
        prompt_tokens, tokenize_errors = run_tokenize_count_request(
            tokenize_url, spec_tree_tokenize_body(args, item), args.timeout
        )
        for error in prefixed_errors("tokenize", tokenize_errors):
            add_error(verify_errors, error)
            add_error(fatal_errors, "verify:" + error)
        if prompt_tokens is not None and not tokenize_errors:
            native_url = args.large_url.rstrip("/") + "/generate"

            def run_continuation(continuation):
                return run_native_verify_request(
                    native_url,
                    spec_tree_native_verify_body(
                        item, continuation, prompt_tokens
                    ),
                    args.timeout,
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(valid)
            ) as executor:
                futures = [
                    executor.submit(run_continuation, continuation)
                    for _index, continuation in valid
                ]
                results = [future.result() for future in futures]
            for (draft_index, _continuation), result in zip(valid, results):
                target_scores[draft_index] = result["score"]
                target_score_tokens[draft_index] = result["n_tokens"]
                verify_prefill_tokens += result["n_tokens"]
                for error in prefixed_errors(
                    "continuation_{}".format(draft_index), result["errors"]
                ):
                    add_error(verify_errors, error)
                    add_error(fatal_errors, "verify:" + error)
    else:
        add_error(verify_errors, request_error)
        if request_error is not None:
            add_error(fatal_errors, "verify:batched:" + request_error)
        else:
            report = batched_verify_report(payload, len(valid))
            verify_prefill_tokens = report["total_tokens"]
            for (draft_index, _continuation), score, count in zip(
                valid, report["scores"], report["token_counts"]
            ):
                target_scores[draft_index] = score
                target_score_tokens[draft_index] = count
            for error in report["errors"]:
                add_error(verify_errors, "batched:" + error)
                add_error(fatal_errors, "verify:batched:" + error)

    return {
        "target_scores": target_scores,
        "target_score_tokens": target_score_tokens,
        "verify_prefill_tokens": verify_prefill_tokens,
        "verify_path": verify_path,
        "verify_errors": verify_errors,
        "fatal_errors": fatal_errors,
    }


def run_spec_tree_item(args, item, base_seed):
    started = time.perf_counter()
    draft = run_spec_tree_draft(args, item, base_seed)
    verify = run_spec_tree_verify(args, item, draft["continuations"])
    errors = list(draft["fatal_errors"])
    for error in verify["fatal_errors"]:
        add_error(errors, error)

    extractable = [
        index
        for index, answer in enumerate(draft["draft_answers"])
        if answer is not None
    ]
    ranked = sorted(
        (
            index
            for index in extractable
            if verify["target_scores"][index] is not None
        ),
        key=lambda index: (-verify["target_scores"][index], index),
    )
    chosen_index = ranked[0] if ranked else None
    repair_reason = None
    if not extractable:
        repair_reason = "no_extractable_draft"
    elif draft["fatal_errors"]:
        repair_reason = "draft_error"
    elif verify["fatal_errors"]:
        repair_reason = "verify_error"
    elif chosen_index is None:
        repair_reason = "no_scored_extractable_draft"
    elif len(ranked) >= 2:
        score_gap = (
            verify["target_scores"][ranked[0]]
            - verify["target_scores"][ranked[1]]
        )
        if score_gap <= args.repair_margin:
            repair_reason = "top_two_within_margin"

    repaired = repair_reason is not None
    repair_tokens = 0
    repair_errors = []
    if repaired:
        repair_url = args.large_url.rstrip("/") + "/v1/chat/completions"
        repair = run_chat_request(
            repair_url, large_greedy_body(args, item, base_seed), args.timeout
        )
        repair_tokens = repair["tokens"]
        answer = repair["answer"]
        for error in prefixed_errors("repair", repair["errors"]):
            add_error(repair_errors, error)
            add_error(errors, error)
    else:
        answer = draft["draft_answers"][chosen_index]
    if answer is None:
        add_error(errors, "selection:no_final_answer")

    record = common_record(
        args,
        "spec_tree",
        item,
        base_seed,
        started,
        answer,
        draft["draft_tokens"],
        repair_tokens,
        errors,
    )
    record["large_prefill_tokens"] = verify["verify_prefill_tokens"]
    record["total_cost_units"] += (
        verify["verify_prefill_tokens"] * args.large_prefill_cost
    )
    record.update(
        {
            "draft_tokens": draft["draft_tokens"],
            "verify_prefill_tokens": verify["verify_prefill_tokens"],
            "repair_tokens": repair_tokens,
            "tree_draft_tokens": draft["tree_tokens"],
            "sample_draft_tokens": draft["sample_tokens"],
            "draft_answers": draft["draft_answers"],
            "target_scores": verify["target_scores"],
            "target_score_tokens": verify["target_score_tokens"],
            "chosen_index": chosen_index,
            "repaired": repaired,
            "repair_reason": repair_reason,
            "verify_path": verify["verify_path"],
            "draft_source": draft["draft_source"],
            "draft_branch_ids": draft["branch_ids"],
            "draft_errors": draft["draft_errors"],
            "verify_errors": verify["verify_errors"],
            "repair_errors": repair_errors,
        }
    )
    return record


def run_mode_item(args, item, base_seed):
    if args.mode == "cascade":
        return run_cascade_item(args, item, base_seed)
    if args.mode == "cascade_score":
        return run_cascade_score_item(args, item, base_seed)
    if args.mode == "large_bo8":
        return run_large_bo8_item(args, item, base_seed)
    if args.mode == "large_tree":
        return run_large_tree_item(args, item, base_seed)
    if args.mode == "adaptive_k":
        return run_adaptive_k_item(args, item, base_seed)
    if args.mode == "spec_tree":
        return run_spec_tree_item(args, item, base_seed)
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
        add_gate_fields(record, args, None)
        if args.gate_model:
            record["chosen_by"] = "no_votes"
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
                "chosen_by": "no_votes" if args.gate_model else "fallback",
            }
        )
        add_gate_fields(record, args, None)
    elif args.mode == "spec_tree":
        record.update(
            {
                "large_prefill_tokens": 0,
                "draft_tokens": 0,
                "verify_prefill_tokens": 0,
                "repair_tokens": 0,
                "tree_draft_tokens": 0,
                "sample_draft_tokens": 0,
                "draft_answers": [],
                "target_scores": [],
                "target_score_tokens": [],
                "chosen_index": None,
                "repaired": False,
                "repair_reason": None,
                "verify_path": "batched",
                "draft_source": "tree",
                "draft_branch_ids": [],
                "draft_errors": ["internal_error:draft_not_completed"],
                "verify_errors": ["internal_error:verify_not_completed"],
                "repair_errors": [],
            }
        )
    elif args.mode == "adaptive_k":
        record.update(
            {
                "rungs_used": [],
                "total_branches_sampled": 0,
                "answers_by_rung": [],
                "leader_count": 0,
                "runner_up_count": 0,
                "stopped_early": False,
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


def dedupe_records(records):
    unique = []
    seen = set()
    duplicate_count = 0
    for record in records:
        key = record_key(record)
        if key is not None and key in seen:
            duplicate_count += 1
            continue
        if key is not None:
            seen.add(key)
        unique.append(record)
    return unique, duplicate_count


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
            or record.get("chosen_by") not in ("vote", "score", "fallback", "no_votes")
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
    if key[0] == "spec_tree":
        spec_tree_fields = (
            "large_prefill_tokens",
            "draft_tokens",
            "verify_prefill_tokens",
            "repair_tokens",
            "tree_draft_tokens",
            "sample_draft_tokens",
            "draft_answers",
            "target_scores",
            "target_score_tokens",
            "chosen_index",
            "repaired",
            "repair_reason",
            "verify_path",
            "draft_source",
            "draft_branch_ids",
            "draft_errors",
            "verify_errors",
            "repair_errors",
        )
        if any(field not in record for field in spec_tree_fields):
            return False
        for field in (
            "large_prefill_tokens",
            "draft_tokens",
            "verify_prefill_tokens",
            "repair_tokens",
            "tree_draft_tokens",
            "sample_draft_tokens",
        ):
            if nonnegative_int(record.get(field)) is None:
                return False
        if (
            record.get("small_tokens") != record.get("draft_tokens")
            or record.get("large_tokens") != record.get("repair_tokens")
            or record.get("large_prefill_tokens")
            != record.get("verify_prefill_tokens")
            or record.get("draft_tokens")
            != record.get("tree_draft_tokens")
            + record.get("sample_draft_tokens")
        ):
            return False
        draft_answers = record.get("draft_answers")
        target_scores = record.get("target_scores")
        target_score_tokens = record.get("target_score_tokens")
        if (
            not isinstance(draft_answers, list)
            or not isinstance(target_scores, list)
            or not isinstance(target_score_tokens, list)
            or len(draft_answers) != len(target_scores)
            or len(draft_answers) != len(target_score_tokens)
        ):
            return False
        if any(
            answer is not None and not isinstance(answer, str)
            for answer in draft_answers
        ):
            return False
        if any(
            score is not None and finite_number(score) is None
            for score in target_scores
        ):
            return False
        if any(
            count is not None and nonnegative_int(count) is None
            for count in target_score_tokens
        ):
            return False
        chosen_index = record.get("chosen_index")
        if chosen_index is not None and (
            nonnegative_int(chosen_index) is None
            or chosen_index >= len(draft_answers)
        ):
            return False
        if (
            not isinstance(record.get("repaired"), bool)
            or record.get("verify_path") not in ("batched", "fallback")
            or record.get("draft_source") not in ("tree", "samples")
        ):
            return False
        if record.get("repair_reason") is not None and not isinstance(
            record.get("repair_reason"), str
        ):
            return False
        for field in (
            "draft_branch_ids",
            "draft_errors",
            "verify_errors",
            "repair_errors",
        ):
            if not isinstance(record.get(field), list):
                return False
    if key[0] == "adaptive_k":
        adaptive_fields = (
            "rungs_used",
            "total_branches_sampled",
            "answers_by_rung",
            "leader_count",
            "runner_up_count",
            "stopped_early",
        )
        if any(field not in record for field in adaptive_fields):
            return False
        rungs_used = record.get("rungs_used")
        answers_by_rung = record.get("answers_by_rung")
        if (
            not isinstance(rungs_used, list)
            or any(nonnegative_int(rung) is None or rung == 0 for rung in rungs_used)
            or nonnegative_int(record.get("total_branches_sampled")) is None
            or record.get("total_branches_sampled") != sum(rungs_used)
            or not isinstance(answers_by_rung, list)
            or len(answers_by_rung) != len(rungs_used)
            or nonnegative_int(record.get("leader_count")) is None
            or nonnegative_int(record.get("runner_up_count")) is None
            or not isinstance(record.get("stopped_early"), bool)
        ):
            return False
        for answers in answers_by_rung:
            if not isinstance(answers, list) or any(
                answer is not None and not isinstance(answer, str)
                for answer in answers
            ):
                return False
    gate_fields = ("gate_p", "gate_threshold", "gated")
    if any(field in record for field in gate_fields):
        if any(field not in record for field in gate_fields):
            return False
        gate_p = record.get("gate_p")
        gate_threshold = finite_number(record.get("gate_threshold"))
        if (
            record.get("gated") is not True
            or (gate_p is not None and finite_number(gate_p) is None)
            or gate_threshold is None
            or gate_threshold < 0.0
            or gate_threshold > 1.0
        ):
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


def records_cost_units(args, records):
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
    total_cost_units = item_cost(args, total_small_tokens, total_large_tokens)
    if args.mode in ("cascade_score", "spec_tree"):
        total_cost_units += total_large_prefill_tokens * args.large_prefill_cost
    return (
        total_cost_units,
        total_small_tokens,
        total_large_tokens,
        total_large_prefill_tokens,
    )


def seed_result_hash(records):
    tuples = [
        (str(record.get("id")), record.get("extracted"), record.get("correct"))
        for record in sorted(records, key=lambda record: str(record.get("id")))
    ]
    encoded = json.dumps(
        tuples, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize(args, seed, records):
    records, duplicate_count = dedupe_records(records)
    if duplicate_count:
        print(
            "WARNING: MEASUREMENT INTEGRITY: dropped {} duplicate record(s) by "
            "(mode, seed, id) during final summarization for mode={} seed={}".format(
                duplicate_count, args.mode, seed
            )
        )
    items = len(records)
    correct_count = sum(record.get("correct") is True for record in records)
    error_count = sum(bool(record.get("error")) for record in records)
    error_n = sum(record.get("error") is not None for record in records)
    non_error_records = [
        record for record in records if record.get("error") is None
    ]
    non_error_correct_count = sum(
        record.get("correct") is True for record in non_error_records
    )
    (
        total_cost_units,
        total_small_tokens,
        total_large_tokens,
        total_large_prefill_tokens,
    ) = records_cost_units(args, records)
    non_error_cost_units = records_cost_units(args, non_error_records)[0]
    wall_times = [nonnegative_number(record.get("wall_s")) or 0.0 for record in records]
    wall_s_total = sum(wall_times)
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
        "wall_s_total": wall_s_total,
        "wall_per_correct": wall_s_total / max(correct_count, 1),
        "error_count": error_count,
        "error_n": error_n,
        "error_rate": error_n / items if items else 0.0,
        "accuracy_excluding_errors": (
            non_error_correct_count / len(non_error_records)
            if non_error_records
            else 0.0
        ),
        "cost_per_correct_excluding_errors": (
            non_error_cost_units / max(non_error_correct_count, 1)
        ),
        "_seed_result_hash": seed_result_hash(records),
    }
    if args.mode == "cascade_score":
        summary["total_large_prefill_tokens"] = total_large_prefill_tokens
    elif args.mode == "spec_tree":
        summary["total_large_prefill_tokens"] = total_large_prefill_tokens
        observed = [
            record["repaired"]
            for record in records
            if isinstance(record.get("repaired"), bool)
        ]
        repaired_count = sum(value is True for value in observed)
        summary.update(
            {
                "repaired_count": repaired_count,
                "repair_observed_items": len(observed),
                "repair_rate": repaired_count / len(observed) if observed else 0.0,
            }
        )
    elif args.mode == "adaptive_k":
        observed = [
            record["stopped_early"]
            for record in records
            if isinstance(record.get("stopped_early"), bool)
        ]
        stopped_early_count = sum(value is True for value in observed)
        summary.update(
            {
                "stopped_early_count": stopped_early_count,
                "stopped_early_observed_items": len(observed),
                "stopped_early_rate": (
                    stopped_early_count / len(observed) if observed else 0.0
                ),
                "total_branches_sampled": sum(
                    nonnegative_int(record.get("total_branches_sampled")) or 0
                    for record in records
                ),
            }
        )
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
        if row["mode"] == "spec_tree":
            escalation_rate = row.get("repair_rate")
        elif row["mode"] == "adaptive_k":
            escalation_rate = row.get("stopped_early_rate")
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
        print(
            "integrity_seed_{seed}: error_n={error_n} error_rate={error_rate:.6%} "
            "accuracy_excluding_errors={accuracy_excluding_errors:.6%} "
            "cost_per_correct_excluding_errors="
            "{cost_per_correct_excluding_errors:.6f} wall_s_total={wall_s_total:.6f} "
            "wall_per_correct={wall_per_correct:.6f}".format(**row)
        )
        if row["mode"] == "cascade_score":
            print(
                "cascade_score_seed_{}_large_prefill_tokens: {}".format(
                    row["seed"], row["total_large_prefill_tokens"]
                )
            )
        elif row["mode"] == "spec_tree":
            print(
                "spec_tree_seed_{}_large_prefill_tokens: {}".format(
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
    if args.fresh:
        ensure_parent(args.out_jsonl)
        with open(args.out_jsonl, "w", encoding="utf-8", newline="\n") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        print("fresh: truncated out-jsonl before benchmark: {}".format(args.out_jsonl))
    existing, invalid_lines, duplicate_lines = load_existing_records(args.out_jsonl)
    if invalid_lines:
        print(
            "warning: ignored {} invalid or truncated line(s) in {}".format(
                invalid_lines, args.out_jsonl
            )
        )
    if duplicate_lines:
        print(
            "WARNING: MEASUREMENT INTEGRITY: dropped {} duplicate record(s) by "
            "(mode, seed, id) while loading {}".format(
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
    seed_hashes = [summary.pop("_seed_result_hash") for summary in summaries]
    seeds_nominal = len(seed_hashes)
    seeds_effective = len(set(seed_hashes))
    for summary in summaries:
        summary["seeds_nominal"] = seeds_nominal
        summary["seeds_effective"] = seeds_effective
    print(
        "seed_evidence: mode={} seeds_nominal={} seeds_effective={}".format(
            args.mode, seeds_nominal, seeds_effective
        )
    )
    if seeds_effective < seeds_nominal:
        print(
            "FLAG: EFFECTIVE SEED COUNT: mode={} seeds_nominal={} "
            "seeds_effective={}; nominal seeds are not independent evidence".format(
                args.mode, seeds_nominal, seeds_effective
            )
        )
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
            "answer_mode": args.answer_mode,
            "fork_at_entropy": args.fork_at_entropy,
            "answer_suffix": answer_suffix(),
            "out_jsonl": os.path.abspath(args.out_jsonl),
            "fresh": args.fresh,
        },
        "summaries": summaries,
    }
    if args.mode == "cascade_score":
        output["config"]["large_prefill_cost"] = args.large_prefill_cost
    elif args.mode == "spec_tree":
        output["config"].update(
            {
                "large_prefill_cost": args.large_prefill_cost,
                "repair_margin": args.repair_margin,
                "small_model_defaulted": args.small_model_defaulted,
            }
        )
    elif args.mode == "adaptive_k":
        output["config"].update(
            {
                "k_ladder": args.k_ladder_values,
                "agree_margin": args.agree_margin,
                "min_leader": args.min_leader,
            }
        )
    if args.gate_model:
        output["config"].update(
            {
                "gate_model": args.gate_model_config,
                "gate_threshold": args.gate_threshold,
            }
        )
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
    error_n = sum(
        nonnegative_int(row.get("error_n"))
        if nonnegative_int(row.get("error_n")) is not None
        else nonnegative_int(row.get("error_count")) or 0
        for row in rows
    )
    total_small_tokens = sum(
        nonnegative_int(row.get("total_small_tokens")) or 0 for row in rows
    )
    total_large_tokens = sum(
        nonnegative_int(row.get("total_large_tokens")) or 0 for row in rows
    )
    total_cost_units = sum(
        nonnegative_number(row.get("total_cost_units")) or 0.0 for row in rows
    )
    cost_per_correct_excluding_values = [
        nonnegative_number(row.get("cost_per_correct_excluding_errors"))
        for row in rows
    ]
    if all(value is not None for value in cost_per_correct_excluding_values):
        cost_units_excluding_errors = sum(
            value * max(nonnegative_int(row.get("correct_count")) or 0, 1)
            for row, value in zip(rows, cost_per_correct_excluding_values)
        )
    else:
        cost_units_excluding_errors = None
    wall_s_total = sum(
        (
            nonnegative_number(row.get("wall_s_total"))
            if nonnegative_number(row.get("wall_s_total")) is not None
            else (nonnegative_number(row.get("mean_wall_s")) or 0.0)
            * (nonnegative_int(row.get("items")) or 0)
        )
        for row in rows
    )
    seeds_nominal = max(
        [nonnegative_int(row.get("seeds_nominal")) or 0 for row in rows]
        + [len(rows)]
    )
    effective_seed_values = [
        nonnegative_int(row.get("seeds_effective"))
        for row in rows
        if nonnegative_int(row.get("seeds_effective")) is not None
    ]
    seeds_effective = (
        max(effective_seed_values) if effective_seed_values else seeds_nominal
    )
    non_error_items = max(items - error_n, 0)
    aggregate = {
        "items": items,
        "correct_count": correct_count,
        "accuracy": correct_count / items if items else 0.0,
        "total_small_tokens": total_small_tokens,
        "total_large_tokens": total_large_tokens,
        "total_cost_units": total_cost_units,
        "cost_per_correct": total_cost_units / max(correct_count, 1),
        "error_n": error_n,
        "error_rate": error_n / items if items else 0.0,
        "accuracy_excluding_errors": (
            correct_count / non_error_items if non_error_items else 0.0
        ),
        "cost_per_correct_excluding_errors": (
            cost_units_excluding_errors / max(correct_count, 1)
            if cost_units_excluding_errors is not None
            else None
        ),
        "wall_s_total": wall_s_total,
        "wall_per_correct": wall_s_total / max(correct_count, 1),
        "seeds_nominal": seeds_nominal,
        "seeds_effective": seeds_effective,
    }
    if mode == "cascade_score":
        aggregate["total_large_prefill_tokens"] = sum(
            nonnegative_int(row.get("total_large_prefill_tokens")) or 0
            for row in rows
        )
    elif mode == "spec_tree":
        aggregate["total_large_prefill_tokens"] = sum(
            nonnegative_int(row.get("total_large_prefill_tokens")) or 0
            for row in rows
        )
        repaired_count = sum(
            nonnegative_int(row.get("repaired_count")) or 0 for row in rows
        )
        observed_items = sum(
            nonnegative_int(row.get("repair_observed_items")) or 0 for row in rows
        )
        aggregate.update(
            {
                "repaired_count": repaired_count,
                "repair_observed_items": observed_items,
                "repair_rate": (
                    repaired_count / observed_items if observed_items else 0.0
                ),
            }
        )
    elif mode == "adaptive_k":
        stopped_early_count = sum(
            nonnegative_int(row.get("stopped_early_count")) or 0 for row in rows
        )
        observed_items = sum(
            nonnegative_int(row.get("stopped_early_observed_items")) or 0
            for row in rows
        )
        aggregate.update(
            {
                "stopped_early_count": stopped_early_count,
                "stopped_early_observed_items": observed_items,
                "stopped_early_rate": (
                    stopped_early_count / observed_items if observed_items else 0.0
                ),
                "total_branches_sampled": sum(
                    nonnegative_int(row.get("total_branches_sampled")) or 0
                    for row in rows
                ),
            }
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
        "answer_mode",
        "fork_at_entropy",
        "answer_suffix",
    )
    first = configs[0]
    mismatches = [
        key
        for key in keys
        if any(config.get(key) != first.get(key) for config in configs[1:])
    ]
    tree_configs = []
    for document, config in zip(documents, configs):
        modes = {
            row.get("mode")
            for row in summary_rows(document)
            if row.get("mode")
            in ("cascade", "cascade_score", "spec_tree", "adaptive_k")
        }
        if modes:
            tree_configs.append(config)
    if len(tree_configs) > 1:
        first_tree = tree_configs[0]
        for key in ("gate_model", "gate_threshold"):
            if any(
                config.get(key) != first_tree.get(key) for config in tree_configs[1:]
            ):
                mismatches.append(key)
    return mismatches


def ratio(numerator, denominator):
    if denominator <= 0:
        return None
    return numerator / denominator


def format_ratio(value):
    return "n/a" if value is None else "{:.6f}".format(value)


def ratio_winner(value, baseline_mode, tree_mode):
    if value is None or value == 1.0:
        return None
    return tree_mode if value > 1.0 else baseline_mode


def print_integrity_metrics(mode, summary):
    print("{}_error_n: {}".format(mode, summary["error_n"]))
    print("{}_error_rate: {:.6%}".format(mode, summary["error_rate"]))
    print(
        "{}_accuracy_excluding_errors: {:.6%}".format(
            mode, summary["accuracy_excluding_errors"]
        )
    )
    print(
        "{}_cost_per_correct_excluding_errors: {}".format(
            mode,
            (
                "n/a"
                if summary["cost_per_correct_excluding_errors"] is None
                else "{:.6f}".format(
                    summary["cost_per_correct_excluding_errors"]
                )
            ),
        )
    )
    print("{}_wall_s_total: {:.6f}".format(mode, summary["wall_s_total"]))
    print("{}_wall_per_correct: {:.6f}".format(mode, summary["wall_per_correct"]))
    print(
        "{}_seed_evidence: seeds_nominal={} seeds_effective={}".format(
            mode, summary["seeds_nominal"], summary["seeds_effective"]
        )
    )
    if summary["seeds_effective"] < summary["seeds_nominal"]:
        print(
            "FLAG: EFFECTIVE SEED COUNT: mode={} seeds_nominal={} "
            "seeds_effective={}; nominal seeds are not independent evidence".format(
                mode, summary["seeds_nominal"], summary["seeds_effective"]
            )
        )


def print_comparison(mode, baseline, tree_summary, tree_mode="cascade"):
    accuracy_delta_points = (
        tree_summary["accuracy"] - baseline["accuracy"]
    ) * 100.0
    accuracy_excluding_errors_delta_points = (
        tree_summary["accuracy_excluding_errors"]
        - baseline["accuracy_excluding_errors"]
    ) * 100.0
    cost_ratio = ratio(
        baseline["cost_per_correct"], tree_summary["cost_per_correct"]
    )
    wall_ratio = ratio(
        baseline["wall_per_correct"], tree_summary["wall_per_correct"]
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
    print_integrity_metrics(mode, baseline)
    print(
        "accuracy_excluding_errors_delta_{}_minus_{}: {:+.6f} pt".format(
            tree_mode, mode, accuracy_excluding_errors_delta_points
        )
    )
    error_rate_delta_points = abs(
        tree_summary["error_rate"] - baseline["error_rate"]
    ) * 100.0
    if error_rate_delta_points > 5.0:
        print(
            "FLAG: ERROR RATE IMBALANCE: {} error_rate={:.6%} and {} "
            "error_rate={:.6%} differ by {:.6f} pt; comparison is not apples "
            "to apples".format(
                tree_mode,
                tree_summary["error_rate"],
                mode,
                baseline["error_rate"],
                error_rate_delta_points,
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
    print(
        "wall_per_correct_ratio_{}_over_{}: {}".format(
            mode, tree_mode, format_ratio(wall_ratio)
        )
    )
    cost_winner = ratio_winner(cost_ratio, mode, tree_mode)
    wall_winner = ratio_winner(wall_ratio, mode, tree_mode)
    if (
        cost_winner is not None
        and wall_winner is not None
        and cost_winner != wall_winner
    ):
        print(
            "FLAG: COST/WALL REVERSAL: cost_per_correct favors {} but "
            "wall_per_correct favors {} in {} versus {}".format(
                cost_winner, wall_winner, tree_mode, mode
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
    if tree_mode == "spec_tree":
        print(
            "verdict_{}: {}; baseline/{} ratio {}, {} cost/correct {:.6f}, "
            "baseline cost/correct {:.6f}, {} accuracy delta {:+.6f} pt, {} "
            "repair rate {:.6%}".format(
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
                tree_summary["repair_rate"],
            )
        )
    elif tree_mode == "adaptive_k":
        print(
            "verdict_{}: {}; baseline/{} ratio {}, {} cost/correct {:.6f}, "
            "baseline cost/correct {:.6f}, {} accuracy delta {:+.6f} pt, {} "
            "early-stop rate {:.6%}".format(
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
                tree_summary["stopped_early_rate"],
            )
        )
    else:
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
    if tree_mode not in ("cascade", "cascade_score", "spec_tree", "adaptive_k"):
        raise ValueError(
            "first summary mode must be 'cascade', 'cascade_score', 'spec_tree', "
            "or 'adaptive_k'"
        )
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
    print_integrity_metrics(tree_mode, tree_summary)
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
    elif tree_mode == "spec_tree":
        print(
            "spec_tree_total_large_prefill_tokens: {}".format(
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
    if tree_mode == "spec_tree":
        print(
            "spec_tree_repair_rate: {:.6%}".format(tree_summary["repair_rate"])
        )
    elif tree_mode == "adaptive_k":
        print(
            "adaptive_k_stopped_early_rate: {:.6%}".format(
                tree_summary["stopped_early_rate"]
            )
        )
        print(
            "adaptive_k_total_branches_sampled: {}".format(
                tree_summary["total_branches_sampled"]
            )
        )
    else:
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
    adaptive_rungs = [
        {
            "rung": rung_index + 1,
            "k": branches,
            "url": large_base + "/v1/tree/completions",
            "body": adaptive_k_tree_body(
                args, item, seed, rung_index, branches
            ),
        }
        for rung_index, branches in enumerate(args.k_ladder_values[:2])
    ]
    if args.mode == "adaptive_k":
        print(
            json.dumps(
                {
                    "item_id": item["id"],
                    "item_index": item["item_index"],
                    "seed": seed,
                    "adaptive_k": adaptive_rungs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    draft_placeholders = [
        "<draft_{}>".format(index) for index in range(args.branches)
    ]
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
        "large_tree": {
            "url": large_base + "/v1/tree/completions",
            "body": large_tree_body(args, item, seed),
        },
        "adaptive_k": adaptive_rungs,
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
        "spec_tree": {
            "draft_tree": {
                "url": small_base + "/v1/tree/completions",
                "body": small_tree_body(args, item, seed),
            },
            "draft_samples_if_branch_texts_unavailable": [
                {
                    "sample_index": sample_index,
                    "url": small_base + "/v1/chat/completions",
                    "body": small_sample_body(args, item, seed, sample_index),
                }
                for sample_index in range(args.branches)
            ],
            "verify_batched": {
                "url": large_base + "/v1/tree/verify",
                "body": spec_tree_verify_body(args, item, draft_placeholders),
            },
            "verify_fallback_tokenize": {
                "url": large_base + "/v1/tokenize",
                "body": spec_tree_tokenize_body(args, item),
            },
            "verify_fallback_generate": [
                {
                    "draft_index": draft_index,
                    "url": large_base + "/generate",
                    "body": spec_tree_native_verify_body(
                        item, continuation, "<tokenized_prompt_count>"
                    ),
                }
                for draft_index, continuation in enumerate(draft_placeholders)
            ],
            "repair_if_needed": {
                "url": large_base + "/v1/chat/completions",
                "body": large_greedy_body(args, item, seed),
            },
        },
    }
    if args.gate_model:
        document["gate"] = {
            "model": args.gate_model_config,
            "threshold": args.gate_threshold,
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
    parser.add_argument(
        "--small-model",
        help=(
            "served small model name; spec_tree defaults to "
            + DEFAULT_SPEC_TREE_SMALL_MODEL
        ),
    )
    parser.add_argument("--large-model", help="served large model name")
    parser.add_argument("--small-url", default="http://127.0.0.1:30000")
    parser.add_argument("--large-url", default="http://127.0.0.1:30001")
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument(
        "--k-ladder",
        default="2,4,8,16",
        help="adaptive_k branch counts issued sequentially",
    )
    parser.add_argument(
        "--agree-margin",
        type=int,
        default=2,
        help="adaptive_k minimum leader minus runner-up vote count",
    )
    parser.add_argument(
        "--min-leader",
        type=int,
        default=2,
        help="adaptive_k minimum plurality leader vote count",
    )
    parser.add_argument("--agree-threshold", type=int, default=6)
    parser.add_argument(
        "--answer-mode",
        choices=("numeric", "math"),
        default="numeric",
        help="math routes extraction/equivalence through bench/tasks/math_equiv.py",
    )
    parser.add_argument(
        "--fork-at-entropy",
        type=float,
        help="entropy-fork threshold (nats) passed to the small tree server",
    )
    parser.add_argument("--gate-model", help="learned escalation gate model JSON")
    parser.add_argument("--gate-threshold", type=float, default=0.75)
    parser.add_argument(
        "--gate-selftest",
        nargs=2,
        metavar=("MODEL_JSON", "ITEMS_JSONL"),
        help="print harness gate features and probabilities for the first 5 records",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--small-cost", type=float, default=1.0)
    parser.add_argument("--large-cost", type=float, default=10.0)
    parser.add_argument(
        "--large-prefill-cost",
        type=float,
        default=2.0,
        help="large-model prefill cost units per token for target verification",
    )
    parser.add_argument(
        "--repair-margin",
        type=float,
        default=0.05,
        help="spec_tree top-two target-score margin in nats that triggers repair",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-jsonl", help="incremental per-item JSONL output")
    parser.add_argument("--out", help="final summary JSON output")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="truncate --out-jsonl before starting a clean benchmark run",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="SUMMARY_JSON",
        help=(
            "compare cascade, cascade_score, spec_tree, or adaptive_k JSON with "
            "one or both of large_bo8 and large_greedy JSON"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every mode's request bodies for the first selected item",
    )
    return parser


def validate_measurement_args(parser, args):
    if not args.data:
        parser.error("--data is required unless --compare is used")
    if (
        args.mode in ("cascade", "cascade_score")
        or (args.dry_run and args.mode != "adaptive_k")
    ) and not args.small_model:
        parser.error(
            "--small-model is required for cascade, cascade_score, and --dry-run"
        )
    if not args.large_model:
        parser.error("--large-model is required unless --compare is used")
    if args.branches <= 0 or args.branches > 64:
        parser.error("--branches must be between 1 and 64")
    try:
        args.k_ladder_values = parse_k_ladder(args.k_ladder)
    except ValueError as exc:
        parser.error(str(exc))
    if args.agree_margin < 0:
        parser.error("--agree-margin must be nonnegative")
    if args.min_leader <= 0:
        parser.error("--min-leader must be positive")
    if args.agree_threshold <= 0:
        parser.error("--agree-threshold must be positive")
    if not math.isfinite(args.gate_threshold) or not 0.0 <= args.gate_threshold <= 1.0:
        parser.error("--gate-threshold must be finite and between 0 and 1")
    if (
        args.gate_model
        and not args.dry_run
        and args.mode not in ("cascade", "cascade_score")
    ):
        parser.error("--gate-model requires mode cascade or cascade_score")
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
    if not math.isfinite(args.repair_margin) or args.repair_margin < 0:
        parser.error("--repair-margin must be finite and nonnegative")
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
        if args.fresh and os.path.normcase(
            os.path.abspath(args.out_jsonl)
        ) == os.path.normcase(os.path.abspath(args.data)):
            parser.error("--fresh --out-jsonl must differ from --data")
        if (
            args.fresh
            and args.gate_model
            and os.path.normcase(os.path.abspath(args.out_jsonl))
            == os.path.normcase(os.path.abspath(args.gate_model))
        ):
            parser.error("--fresh --out-jsonl must differ from --gate-model")


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.small_model_defaulted = False
    if args.fresh and (args.gate_selftest or args.compare or args.dry_run):
        parser.error("--fresh requires a benchmark run")
    if args.mode == "spec_tree" and not args.small_model:
        args.small_model = DEFAULT_SPEC_TREE_SMALL_MODEL
        args.small_model_defaulted = True
    if args.gate_selftest:
        if args.compare:
            parser.error("--gate-selftest cannot be combined with --compare")
        try:
            run_gate_selftest(*args.gate_selftest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
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
        set_answer_mode(args.answer_mode)
    except (OSError, AttributeError, FileNotFoundError) as exc:
        parser.error(
            "--answer-mode math requires bench/tasks/math_equiv.py: {}".format(exc)
        )
    args.gate_model_data = None
    args.gate_model_config = None
    if args.gate_model:
        try:
            args.gate_model_data = load_gate_model(args.gate_model)
            args.gate_model_config = gate_model_config(args.gate_model)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
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
