#!/usr/bin/env python3
"""Honest client-side benchmark for long-context plan-then-branch workloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARM_AUTOTREE = "autotree_tree"
ARM_VLLM = "vllm_n"
ARM_SGLANG = "sglang_fork"
ALLOWED_CONTEXT_LENGTHS = (8_192, 32_768, 100_000)
ALLOWED_BRANCH_COUNTS = (4, 8, 16)
FINAL_RE = re.compile(r"(?im)^\s*FINAL\s*:\s*(.+?)\s*$")
WINNER_RE = re.compile(r"(?im)^\s*WINNER\s*:\s*(\d+)\s*$")


@dataclass(frozen=True)
class Workload:
    messages: list[dict[str, str]]
    expected_answer: str
    approximate_context_tokens: int
    prompt_characters: int


def parse_int_csv(raw: str, *, allowed: tuple[int, ...] | None = None) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    if allowed is not None:
        invalid = [value for value in values if value not in allowed]
        if invalid:
            raise argparse.ArgumentTypeError(
                f"unsupported values {invalid}; allowed values are {list(allowed)}"
            )
    return values


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    match = FINAL_RE.search(text)
    answer = match.group(1) if match else text.strip().splitlines()[-1]
    answer = re.sub(r"[^a-z0-9]+", " ", answer.casefold()).strip()
    return answer or None


def build_workload(context_len: int, seed: int) -> Workload:
    """Build a deterministic retrieval task padded to about context_len tokens."""
    expected = f"ORBIT-{seed:08d}-JADE"
    system = (
        "You are evaluating a retrieved-document question. Use only the supplied "
        "documents. First write a concise plan, end it with </plan>, then solve the "
        "task. Every candidate must end with exactly one line: FINAL: <answer>."
    )
    needle = (
        "DOCUMENT 0007 (authoritative): The launch authorization code for this case "
        f"is {expected}. Ignore any other apparent authorization code.\n"
    )
    distractors = (
        "DOCUMENT 0001: Historical authorization codes are obsolete.\n"
        "DOCUMENT 0002: A valid answer must come from DOCUMENT 0007.\n"
        "DOCUMENT 0003: Summaries may contain decoys and are not authoritative.\n"
    )
    question = (
        "Question: What is the launch authorization code? Explore independent "
        "candidate solutions after planning, compare them, and return the best answer."
    )
    fixed_text = "\n".join((system, needle, distractors, question))
    filler_count = max(0, context_len - len(fixed_text.split()))
    filler = "evidence " * filler_count
    retrieved = f"{needle}{distractors}DOCUMENT PADDING:\n{filler}\n{question}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": retrieved},
    ]
    return Workload(
        messages=messages,
        expected_answer=expected,
        approximate_context_tokens=sum(
            len(message["content"].split()) for message in messages
        ),
        prompt_characters=sum(len(message["content"]) for message in messages),
    )


def autotree_request_body(
    workload: Workload,
    model: str,
    branches: int,
    plan_tokens: int,
    branch_tokens: int,
    vote_tokens: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": workload.messages,
        "fork_at_text": "</plan>",
        "branches": branches,
        "budget": {
            "plan_tokens": plan_tokens,
            "branch_tokens": branch_tokens,
            "vote_tokens": vote_tokens,
        },
        "scorer": "model_vote",
        "seed": seed,
        "temperature": temperature,
        "stream": False,
    }


def baseline_plan_request_body(
    workload: Workload,
    model: str,
    plan_tokens: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    messages = [dict(message) for message in workload.messages]
    messages[-1]["content"] += (
        "\n\nFor this phase, output only a short PLAN and terminate it with </plan>. "
        "Do not answer the question yet."
    )
    return {
        "model": model,
        "messages": messages,
        "n": 1,
        "max_tokens": plan_tokens,
        "seed": seed,
        "temperature": temperature,
        "stream": False,
    }


def ensure_plan_delimiter(plan: str) -> str:
    return plan if "</plan>" in plan else f"{plan.rstrip()}\n</plan>"


def baseline_branch_request_body(
    workload: Workload,
    model: str,
    plan: str,
    branches: int,
    branch_tokens: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    messages = [dict(message) for message in workload.messages]
    messages.extend(
        [
            {"role": "assistant", "content": ensure_plan_delimiter(plan)},
            {
                "role": "user",
                "content": (
                    "Generate one independent candidate solution following the shared "
                    "plan. End with exactly one line: FINAL: <answer>."
                ),
            },
        ]
    )
    return {
        "model": model,
        "messages": messages,
        "n": branches,
        "max_tokens": branch_tokens,
        "seed": seed,
        "temperature": temperature,
        "stream": False,
    }


def baseline_vote_request_body(
    model: str,
    question: str,
    candidates: list[str],
    vote_tokens: int,
    seed: int,
) -> dict[str, Any]:
    rendered = "\n\n".join(
        f"CANDIDATE {index}:\n{candidate}" for index, candidate in enumerate(candidates)
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Select the candidate best supported by the retrieved evidence. "
                    "Reply with exactly two lines: WINNER: <zero-based id> and "
                    "FINAL: <answer>."
                ),
            },
            {"role": "user", "content": f"{question}\n\n{rendered}"},
        ],
        "n": 1,
        "max_tokens": vote_tokens,
        "seed": seed,
        "temperature": 0,
        "stream": False,
    }


def response_contents(response: dict[str, Any]) -> list[str]:
    contents: list[str] = []
    for choice in response.get("choices", []):
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            contents.append(content)
    return contents


def response_completion_tokens(response: dict[str, Any]) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens")
    return value if isinstance(value, int) and value >= 0 else None


def sum_nullable(values: Iterable[int | None]) -> int | None:
    materialized = list(values)
    if not materialized or any(value is None for value in materialized):
        return None
    return sum(value for value in materialized if value is not None)


def post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def compute_trial_metrics(
    elapsed_seconds: float,
    completion_tokens: int | None,
    gpu_cost_per_hour: float | None,
    *,
    completed: bool = True,
) -> dict[str, float | int | None]:
    if not completed:
        return {
            "e2e_wall_seconds": None,
            "total_completion_tokens": None,
            "throughput_tokens_per_second": None,
            "gpu_hours_per_1k_trees": None,
            "gpu_cost_per_1k_trees_usd": None,
        }
    gpu_hours = elapsed_seconds * 1_000 / 3_600
    throughput = (
        completion_tokens / elapsed_seconds
        if completion_tokens is not None and elapsed_seconds > 0
        else None
    )
    return {
        "e2e_wall_seconds": elapsed_seconds,
        "total_completion_tokens": completion_tokens,
        "throughput_tokens_per_second": throughput,
        "gpu_hours_per_1k_trees": gpu_hours,
        "gpu_cost_per_1k_trees_usd": (
            gpu_hours * gpu_cost_per_hour if gpu_cost_per_hour is not None else None
        ),
    }


def choose_autotree_content(response: dict[str, Any]) -> tuple[str | None, int | None]:
    contents = response_contents(response)
    tree = response.get("tree") if isinstance(response.get("tree"), dict) else {}
    winner = tree.get("winner_branch_id")
    if not isinstance(winner, int):
        winner = None
    if winner is not None and 0 <= winner < len(contents):
        return contents[winner], winner
    return (contents[0] if contents else None), winner


def choose_voted_content(
    vote_text: str | None, candidates: list[str]
) -> tuple[str | None, int | None]:
    if vote_text:
        match = WINNER_RE.search(vote_text)
        winner = int(match.group(1)) if match else None
        if FINAL_RE.search(vote_text):
            return vote_text, winner
        if winner is not None and 0 <= winner < len(candidates):
            return candidates[winner], winner
    return (candidates[0] if candidates else None), None


def run_autotree_trial(
    base_url: str,
    body: dict[str, Any],
    workload: Workload,
    timeout: float,
    api_key: str | None,
    gpu_cost_per_hour: float | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = post_json(
        f"{normalize_base_url(base_url)}/v1/tree/completions",
        body,
        timeout=timeout,
        api_key=api_key,
    )
    elapsed = time.perf_counter() - started
    selected, winner = choose_autotree_content(response)
    tree = response.get("tree") if isinstance(response.get("tree"), dict) else {}
    reported_count = tree.get("branch_count")
    expected_count = body["branches"]
    if reported_count is not None and reported_count != expected_count:
        raise RuntimeError(
            f"tree branch_count={reported_count}, expected {expected_count}"
        )
    metrics = compute_trial_metrics(
        elapsed, response_completion_tokens(response), gpu_cost_per_hour
    )
    return {
        **metrics,
        "request_count": 1,
        "selected_answer": normalize_answer(selected),
        "expected_answer": normalize_answer(workload.expected_answer),
        "exact_match": (
            normalize_answer(selected) == normalize_answer(workload.expected_answer)
            if selected is not None
            else None
        ),
        "winner_branch_id": winner,
        "tree": {
            key: tree.get(key)
            for key in (
                "branch_count",
                "tokens_spent_per_branch",
                "winner_branch_id",
                "scorer",
                "pruned_count",
            )
        },
    }


def run_baseline_trial(
    base_url: str,
    plan_body: dict[str, Any],
    workload: Workload,
    branches: int,
    branch_tokens: int,
    vote_tokens: int,
    seed: int,
    temperature: float,
    timeout: float,
    api_key: str | None,
    gpu_cost_per_hour: float | None,
) -> dict[str, Any]:
    url = f"{normalize_base_url(base_url)}/v1/chat/completions"
    started = time.perf_counter()
    plan_response = post_json(url, plan_body, timeout=timeout, api_key=api_key)
    plans = response_contents(plan_response)
    if not plans:
        raise RuntimeError("plan response had no choices[].message.content")
    branch_body = baseline_branch_request_body(
        workload,
        plan_body["model"],
        plans[0],
        branches,
        branch_tokens,
        seed,
        temperature,
    )
    branch_response = post_json(url, branch_body, timeout=timeout, api_key=api_key)
    candidates = response_contents(branch_response)
    if len(candidates) != branches:
        raise RuntimeError(
            f"branch response returned {len(candidates)} choices, expected {branches}"
        )
    vote_body = baseline_vote_request_body(
        plan_body["model"],
        "What is the launch authorization code?",
        candidates,
        vote_tokens,
        seed,
    )
    vote_response = post_json(url, vote_body, timeout=timeout, api_key=api_key)
    vote_contents = response_contents(vote_response)
    selected, winner = choose_voted_content(
        vote_contents[0] if vote_contents else None, candidates
    )
    elapsed = time.perf_counter() - started
    completion_tokens = sum_nullable(
        response_completion_tokens(response)
        for response in (plan_response, branch_response, vote_response)
    )
    metrics = compute_trial_metrics(elapsed, completion_tokens, gpu_cost_per_hour)
    return {
        **metrics,
        "request_count": 3,
        "selected_answer": normalize_answer(selected),
        "expected_answer": normalize_answer(workload.expected_answer),
        "exact_match": (
            normalize_answer(selected) == normalize_answer(workload.expected_answer)
            if selected is not None
            else None
        ),
        "winner_branch_id": winner,
        "tree": None,
    }


def numeric_stats(
    values: Iterable[float | int | None],
) -> dict[str, float | int | None]:
    present = [
        value
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not present:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(present),
        "mean": statistics.fmean(present),
        "median": statistics.median(present),
    }


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "e2e_wall_seconds",
        "total_completion_tokens",
        "throughput_tokens_per_second",
        "gpu_hours_per_1k_trees",
        "gpu_cost_per_1k_trees_usd",
    )
    successful = [trial for trial in trials if trial.get("status") == "ok"]
    result = {
        metric: numeric_stats(trial.get(metric) for trial in successful)
        for metric in metrics
    }
    accuracies = [
        1.0 if trial["exact_match"] else 0.0
        for trial in successful
        if trial.get("exact_match") is not None
    ]
    result["accuracy_rate"] = statistics.fmean(accuracies) if accuracies else None
    result["successful_trials"] = len(successful)
    result["failed_trials"] = len(trials) - len(successful)
    return result


def evaluate_publish_guards(
    *,
    speedup_x: float | None,
    baseline_prefix_caching: dict[str, str],
    cold_count: int,
    warm_count: int,
    accuracy_delta_pp: float | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(code: str, triggered: bool, message: str, observed: Any) -> None:
        checks.append(
            {
                "code": code,
                "triggered": triggered,
                "message": message,
                "observed": observed,
            }
        )

    add(
        "SPEEDUP_UNAVAILABLE",
        speedup_x is None,
        "Speedup is missing; refuse to publish.",
        speedup_x,
    )
    add(
        "SPEEDUP_LT_2X",
        speedup_x is not None and speedup_x < 2.0,
        "Speedup is below 2x: not dominant, do not headline.",
        speedup_x,
    )
    caching_off = sorted(
        arm for arm, state in baseline_prefix_caching.items() if state != "on"
    )
    add(
        "BASELINE_PREFIX_CACHING_NOT_ON",
        bool(caching_off),
        "A baseline did not have prefix caching confirmed on; refuse to publish.",
        caching_off,
    )
    add(
        "COLD_AND_WARM_NOT_BOTH_REPORTED",
        cold_count < 1 or warm_count < 1,
        "Cold and warm results must both be reported; refuse to publish.",
        {"cold": cold_count, "warm": warm_count},
    )
    add(
        "ACCURACY_DELTA_UNAVAILABLE",
        accuracy_delta_pp is None,
        "Accuracy delta is missing; refuse to publish.",
        accuracy_delta_pp,
    )
    add(
        "ACCURACY_DELTA_GT_0_5PP",
        accuracy_delta_pp is not None and abs(accuracy_delta_pp) > 0.5,
        "Absolute accuracy delta exceeds 0.5 percentage points; refuse to publish.",
        accuracy_delta_pp,
    )
    triggered = [check for check in checks if check["triggered"]]
    return {
        "publishable": not triggered,
        "refuse_to_publish": bool(triggered),
        "checks": checks,
        "triggered_codes": [check["code"] for check in triggered],
    }


def add_answer_agreement(trials: list[dict[str, Any]]) -> None:
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for trial in trials:
        key = (
            trial["context_len"],
            trial["branches"],
            trial["seed"],
            trial["rep"],
        )
        groups[key].append(trial)
    for grouped in groups.values():
        answers = {
            trial["arm"]: trial.get("selected_answer") for trial in grouped
        }
        for trial in grouped:
            own = trial.get("selected_answer")
            trial["answer_agreement"] = {
                arm: (
                    own == answer
                    if own is not None and answer is not None
                    else None
                )
                for arm, answer in answers.items()
                if arm != trial["arm"]
            }


def pairwise_agreement(
    trials: list[dict[str, Any]], left_arm: str, right_arm: str
) -> float | None:
    grouped: dict[
        tuple[int, int, int, int], dict[str, str | None]
    ] = defaultdict(dict)
    for trial in trials:
        key = (
            trial["context_len"],
            trial["branches"],
            trial["seed"],
            trial["rep"],
        )
        grouped[key][trial["arm"]] = trial.get("selected_answer")
    values = []
    for answers in grouped.values():
        left = answers.get(left_arm)
        right = answers.get(right_arm)
        if left is not None and right is not None:
            values.append(1.0 if left == right else 0.0)
    return statistics.fmean(values) if values else None


def build_summary(
    trials: list[dict[str, Any]], baseline_prefix_caching: dict[str, str]
) -> dict[str, Any]:
    groups = []
    all_publishable = True
    for context_len, branches in sorted(
        {
            (trial["context_len"], trial["branches"])
            for trial in trials
        }
    ):
        scoped = [
            trial
            for trial in trials
            if trial["context_len"] == context_len
            and trial["branches"] == branches
        ]
        arms: dict[str, Any] = {}
        for arm in sorted({trial["arm"] for trial in scoped}):
            arm_trials = [trial for trial in scoped if trial["arm"] == arm]
            cold = [trial for trial in arm_trials if trial["rep"] == 0]
            warm = [trial for trial in arm_trials if trial["rep"] > 0]
            arms[arm] = {
                "cold": summarize_trials(cold),
                "warm": summarize_trials(warm),
                "all": summarize_trials(arm_trials),
            }

        auto = arms.get(ARM_AUTOTREE, {}).get("warm", {})
        vllm = arms.get(ARM_VLLM, {}).get("warm", {})
        auto_wall = auto.get("e2e_wall_seconds", {}).get("median")
        vllm_wall = vllm.get("e2e_wall_seconds", {}).get("median")
        speedup = (
            vllm_wall / auto_wall
            if auto_wall is not None
            and vllm_wall is not None
            and auto_wall > 0
            else None
        )
        auto_accuracy = auto.get("accuracy_rate")
        vllm_accuracy = vllm.get("accuracy_rate")
        accuracy_delta_pp = (
            (auto_accuracy - vllm_accuracy) * 100
            if auto_accuracy is not None and vllm_accuracy is not None
            else None
        )
        warm_scoped = [trial for trial in scoped if trial["rep"] > 0]
        comparison = {
            "autotree_tree_vs_vllm_n": {
                "speedup_x": speedup,
                "latency_reduction_percent": (
                    (1 - auto_wall / vllm_wall) * 100
                    if auto_wall is not None
                    and vllm_wall is not None
                    and vllm_wall > 0
                    else None
                ),
                "accuracy_delta_pp": accuracy_delta_pp,
                "answer_agreement_rate": pairwise_agreement(
                    warm_scoped, ARM_AUTOTREE, ARM_VLLM
                ),
            }
        }
        latency_by_arm = {
            arm: data["warm"]["e2e_wall_seconds"]["median"]
            for arm, data in arms.items()
            if data["warm"]["e2e_wall_seconds"]["median"] is not None
        }
        winner = None
        if latency_by_arm:
            winner_arm = min(latency_by_arm, key=latency_by_arm.get)
            ordered = sorted(
                latency_by_arm.items(), key=lambda item: item[1]
            )
            margin = (
                ordered[1][1] / ordered[0][1]
                if len(ordered) > 1
                else None
            )
            winner = {
                "metric": "warm_median_e2e_wall_seconds",
                "arm": winner_arm,
                "margin_x_vs_runner_up": margin,
            }
        included_baselines = {
            arm: state
            for arm, state in baseline_prefix_caching.items()
            if arm in arms
        }
        guards = evaluate_publish_guards(
            speedup_x=speedup,
            baseline_prefix_caching=included_baselines,
            cold_count=min(
                (
                    data["cold"]["successful_trials"]
                    for data in arms.values()
                ),
                default=0,
            ),
            warm_count=min(
                (
                    data["warm"]["successful_trials"]
                    for data in arms.values()
                ),
                default=0,
            ),
            accuracy_delta_pp=accuracy_delta_pp,
        )
        all_publishable = all_publishable and guards["publishable"]
        groups.append(
            {
                "context_len": context_len,
                "branches": branches,
                "arms": arms,
                "comparisons": comparison,
                "winner": winner,
                "publish_guards": guards,
            }
        )
    return {
        "groups": groups,
        "publishable": all_publishable and bool(groups),
        "refuse_to_publish": not (all_publishable and bool(groups)),
    }


def failed_trial_metrics() -> dict[str, Any]:
    return {
        **compute_trial_metrics(0, None, None, completed=False),
        "request_count": None,
        "selected_answer": None,
        "expected_answer": None,
        "exact_match": None,
        "winner_branch_id": None,
        "tree": None,
    }


def dry_run_requests(args: argparse.Namespace) -> int:
    context_len = args.context_lens[0]
    branches = args.branches[0]
    seed = args.seeds[0]
    workload = build_workload(context_len, seed)
    mock_plan = (
        "PLAN: locate DOCUMENT 0007 and verify the authoritative code.\n"
        "</plan>"
    )
    mock_candidates = [
        (
            f"Candidate {index} checked DOCUMENT 0007.\n"
            f"FINAL: {workload.expected_answer}"
        )
        for index in range(branches)
    ]
    requests = [
        (
            ARM_AUTOTREE,
            "POST",
            f"{normalize_base_url(args.autotree_base_url)}"
            "/v1/tree/completions",
            autotree_request_body(
                workload,
                args.model,
                branches,
                args.plan_tokens,
                args.branch_tokens,
                args.vote_tokens,
                seed,
                args.temperature,
            ),
        )
    ]
    for arm, base_url in (
        (ARM_VLLM, args.vllm_base_url),
        (ARM_SGLANG, args.sglang_base_url),
    ):
        url = f"{normalize_base_url(base_url)}/v1/chat/completions"
        requests.extend(
            [
                (
                    f"{arm}.phase_1_plan",
                    "POST",
                    url,
                    baseline_plan_request_body(
                        workload,
                        args.model,
                        args.plan_tokens,
                        seed,
                        args.temperature,
                    ),
                ),
                (
                    f"{arm}.phase_2_branches",
                    "POST",
                    url,
                    baseline_branch_request_body(
                        workload,
                        args.model,
                        mock_plan,
                        branches,
                        args.branch_tokens,
                        seed,
                        args.temperature,
                    ),
                ),
                (
                    f"{arm}.phase_3_vote",
                    "POST",
                    url,
                    baseline_vote_request_body(
                        args.model,
                        "What is the launch authorization code?",
                        mock_candidates,
                        args.vote_tokens,
                        seed,
                    ),
                ),
            ]
        )
    print(
        json.dumps(
            {
                "dry_run": True,
                "network_requests_sent": 0,
                "note": (
                    "Dynamic plan/candidate outputs are replaced with "
                    "deterministic mock outputs so every downstream request "
                    "body is exact and complete."
                ),
                "approximate_context_tokens": (
                    workload.approximate_context_tokens
                ),
                "requests": [
                    {
                        "arm_phase": arm,
                        "method": method,
                        "url": url,
                        "body": body,
                    }
                    for arm, method, url, body in requests
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def markdown_table(results: dict[str, Any]) -> str:
    lines = [
        (
            "| Context | B | Arm | Warm median E2E (s) | Warm tok/s | "
            "Accuracy | vs vLLM | Publish? |"
        ),
        "|---:|---:|---|---:|---:|---:|---:|:---:|",
    ]

    def fmt(value: float | None, digits: int = 3) -> str:
        return "null" if value is None else f"{value:.{digits}f}"

    for group in results.get("summary", {}).get("groups", []):
        comparison = group["comparisons"]["autotree_tree_vs_vllm_n"]
        speedup = comparison["speedup_x"]
        publish = (
            "yes" if group["publish_guards"]["publishable"] else "NO"
        )
        for arm, data in group["arms"].items():
            warm = data["warm"]
            wall = warm["e2e_wall_seconds"]["median"]
            throughput = warm["throughput_tokens_per_second"]["median"]
            accuracy = warm["accuracy_rate"]
            versus = speedup if arm == ARM_AUTOTREE else None
            lines.append(
                f"| {group['context_len']} | {group['branches']} | {arm} | "
                f"{fmt(wall)} | {fmt(throughput, 1)} | "
                f"{fmt(accuracy, 4)} | {fmt(versus, 2)} | {publish} |"
            )
    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    endpoints = {
        ARM_AUTOTREE: args.autotree_base_url,
        ARM_VLLM: args.vllm_base_url,
    }
    if args.include_sglang:
        endpoints[ARM_SGLANG] = args.sglang_base_url
    trials: list[dict[str, Any]] = []
    trial_id = 0
    for context_len in args.context_lens:
        for branches in args.branches:
            for seed in args.seeds:
                workload = build_workload(context_len, seed)
                for arm, base_url in endpoints.items():
                    for rep in range(args.reps):
                        trial_id += 1
                        common = {
                            "trial_id": trial_id,
                            "arm": arm,
                            "context_len": context_len,
                            "approximate_context_tokens": (
                                workload.approximate_context_tokens
                            ),
                            "prompt_characters": workload.prompt_characters,
                            "branches": branches,
                            "seed": seed,
                            "rep": rep,
                            "temperature_state": (
                                "cold" if rep == 0 else "warm"
                            ),
                            "included_in_primary_summary": rep > 0,
                            "prefix_caching": (
                                "n/a"
                                if arm == ARM_AUTOTREE
                                else (
                                    args.vllm_prefix_caching
                                    if arm == ARM_VLLM
                                    else args.sglang_prefix_caching
                                )
                            ),
                        }
                        try:
                            if arm == ARM_AUTOTREE:
                                outcome = run_autotree_trial(
                                    base_url,
                                    autotree_request_body(
                                        workload,
                                        args.model,
                                        branches,
                                        args.plan_tokens,
                                        args.branch_tokens,
                                        args.vote_tokens,
                                        seed,
                                        args.temperature,
                                    ),
                                    workload,
                                    args.timeout,
                                    api_key,
                                    args.gpu_cost_per_hour,
                                )
                            else:
                                outcome = run_baseline_trial(
                                    base_url,
                                    baseline_plan_request_body(
                                        workload,
                                        args.model,
                                        args.plan_tokens,
                                        seed,
                                        args.temperature,
                                    ),
                                    workload,
                                    branches,
                                    args.branch_tokens,
                                    args.vote_tokens,
                                    seed,
                                    args.temperature,
                                    args.timeout,
                                    api_key,
                                    args.gpu_cost_per_hour,
                                )
                            trials.append(
                                {
                                    **common,
                                    "status": "ok",
                                    "error": None,
                                    **outcome,
                                }
                            )
                        except Exception as exc:
                            trials.append(
                                {
                                    **common,
                                    "status": "error",
                                    "error": (
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                    **failed_trial_metrics(),
                                }
                            )
    add_answer_agreement(trials)
    caching = {ARM_VLLM: args.vllm_prefix_caching}
    if args.include_sglang:
        caching[ARM_SGLANG] = args.sglang_prefix_caching
    return {
        "meta": {
            "schema_version": 1,
            "benchmark": "autotree_plan_then_branch",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "config": {
                "context_lens": args.context_lens,
                "branches": args.branches,
                "plan_tokens": args.plan_tokens,
                "branch_tokens": args.branch_tokens,
                "vote_tokens": args.vote_tokens,
                "reps": args.reps,
                "seeds": args.seeds,
                "temperature": args.temperature,
                "gpu_cost_per_hour_usd": args.gpu_cost_per_hour,
            },
            "endpoints": endpoints,
            "baseline_prefix_caching": caching,
            "methodology": {
                "warm_definition": (
                    "rep > 0; rep 0 is retained as cold and excluded "
                    "from primary comparisons"
                ),
                "vllm_best_case": (
                    "two phase: n=1 plan, then plan appended with n=B; "
                    "prefix caching declared warm"
                ),
                "token_padding": (
                    "approximate whitespace tokens using repeated "
                    "common-word filler; no tokenizer-exact claim"
                ),
                "gpu_accounting": (
                    "one occupied GPU per endpoint; GPU-hours = "
                    "wall-hours, with optional hourly cost"
                ),
                "missing_values": "null, never zero-filled",
            },
        },
        "trials": trials,
        "summary": build_summary(trials, caching),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autotree-base-url", default="http://127.0.0.1:30000"
    )
    parser.add_argument(
        "--vllm-base-url", default="http://127.0.0.1:8000"
    )
    parser.add_argument(
        "--sglang-base-url", default="http://127.0.0.1:30001"
    )
    parser.add_argument("--include-sglang", action="store_true")
    parser.add_argument("--model", default="model")
    parser.add_argument(
        "--context-lens",
        type=lambda raw: parse_int_csv(
            raw, allowed=ALLOWED_CONTEXT_LENGTHS
        ),
        default=[8_192],
        metavar="8192,32768,100000",
    )
    parser.add_argument(
        "--branches",
        type=lambda raw: parse_int_csv(
            raw, allowed=ALLOWED_BRANCH_COUNTS
        ),
        default=[4],
        metavar="4,8,16",
    )
    parser.add_argument(
        "--seeds",
        type=lambda raw: parse_int_csv(raw),
        default=[1],
        metavar="1,2,3",
    )
    parser.add_argument("--plan-tokens", type=int, default=128)
    parser.add_argument("--branch-tokens", type=int, default=256)
    parser.add_argument("--vote-tokens", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gpu-cost-per-hour", type=float)
    parser.add_argument(
        "--vllm-prefix-caching",
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument(
        "--sglang-prefix-caching",
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument("--timeout", type=float, default=3_600)
    parser.add_argument("--api-key", help="defaults to OPENAI_API_KEY")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--table-from", type=Path)
    return parser


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    for name in (
        "plan_tokens",
        "branch_tokens",
        "vote_tokens",
        "reps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.gpu_cost_per_hour is not None
        and args.gpu_cost_per_hour < 0
    ):
        parser.error("--gpu-cost-per-hour must be non-negative")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.table_from:
        with args.table_from.open(encoding="utf-8") as handle:
            print(markdown_table(json.load(handle)))
        return 0
    if args.dry_run:
        return dry_run_requests(args)
    results = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(markdown_table(results))
    print(f"\nWrote {args.output}")
    return (
        0
        if all(trial["status"] == "ok" for trial in results["trials"])
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())
