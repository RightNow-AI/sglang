"""OpenAI-compatible request builders and arm execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .answers import extract_answer, majority_vote
from .bench_models import HarnessConfig


@dataclass(frozen=True)
class ArmOutcome:
    answer: str | None
    completion_tokens: int | None


def endpoint_path(arm: str) -> str:
    return "/v1/tree/completions" if arm == "tree" else "/v1/chat/completions"


def endpoint_url(base_url: str, arm: str) -> str:
    parsed = urlsplit(base_url)
    base_path = parsed.path.rstrip("/")
    suffix = "/tree/completions" if arm == "tree" else "/chat/completions"
    path = f"{base_path}{suffix}" if base_path.endswith("/v1") else f"{base_path}/v1{suffix}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def build_request_body(
    config: HarnessConfig,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "seed": seed,
    }
    if config.arm == "best_of_n":
        body["n"] = config.n
    elif config.arm == "tree":
        body["tree"] = {
            "policy": config.policy,
            "branches": config.branches,
            "budget_tokens": config.budget_tokens,
        }
    return body


def _completion_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens")
    return value if type(value) is int and value >= 0 else None


def _tree_tokens(payload: dict[str, Any]) -> int | None:
    tree = payload.get("tree")
    if isinstance(tree, dict):
        per_branch = tree.get("tokens_spent_per_branch")
        values: list[Any] | None = None
        if isinstance(per_branch, dict) and per_branch:
            values = list(per_branch.values())
        elif isinstance(per_branch, list) and per_branch:
            values = per_branch
        if values is not None and all(type(value) is int and value >= 0 for value in values):
            return sum(values)
    return _completion_tokens(payload)


def _choice_texts(payload: dict[str, Any]) -> list[str]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise ValueError("endpoint response is missing choices")
    texts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("endpoint response contains a non-object choice")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("endpoint response choice is missing text content")
        texts.append(content)
    return texts


def execute_arm(
    client: httpx.Client,
    config: HarnessConfig,
    prompt: str,
    seed: int,
) -> ArmOutcome:
    response = client.post(
        endpoint_url(str(config.base_url), config.arm),
        json=build_request_body(config, prompt, seed),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("endpoint response must be a JSON object")
    texts = _choice_texts(payload)
    if config.arm == "best_of_n":
        if len(texts) != config.n:
            raise ValueError(f"best_of_n expected {config.n} choices, received {len(texts)}")
        answer = majority_vote(texts)
    else:
        if not texts:
            raise ValueError("endpoint returned no choices")
        answer = extract_answer(texts[0])
    tokens = _tree_tokens(payload) if config.arm == "tree" else _completion_tokens(payload)
    return ArmOutcome(answer=answer, completion_tokens=tokens)
