"""Batched teacher-forced verification for AutoTree continuations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from sglang.srt.managers.io_struct import GenerateReqInput


class TreeVerifyRequest(BaseModel):
    """Request body for ``POST /v1/tree/verify``."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    prompt: str
    continuations: list[str]


class BranchScore(TypedDict):
    mean_logprob: float | None
    sum_logprob: float | None
    n_scored_tokens: int
    error: str | None


@dataclass(frozen=True)
class _PreparedBranch:
    original_index: int
    continuation_ids: list[int]
    response_offset: int


def _error_score(error: str) -> BranchScore:
    return {
        "mean_logprob": None,
        "sum_logprob": None,
        "n_scored_tokens": 0,
        "error": error,
    }


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) for token_id in token_ids
    ):
        raise ValueError("tokenizer.encode did not return a list of token ids")
    return token_ids


def _continuation_logprob_span(prompt_token_count: int) -> tuple[int, int]:
    """Return the request start and response offset for continuation logprobs.

    SGLang aligns input logprobs to tokens and returns ``None`` for the first
    token at ``logprob_start_len``. Starting at the prompt's final token and
    skipping that boundary entry keeps every continuation token scoreable.
    """
    if prompt_token_count < 1:
        raise ValueError("the tokenized prompt must contain at least one token")
    return prompt_token_count - 1, 1


def _build_batched_request(
    tokenizer_manager: Any,
    prompt_text: str,
    continuations: Sequence[Any],
) -> tuple[GenerateReqInput | None, list[_PreparedBranch], list[BranchScore]]:
    """Build one GenerateReqInput containing every valid continuation."""
    scores = [_error_score("verification did not complete") for _ in continuations]
    tokenizer = getattr(tokenizer_manager, "tokenizer", None)
    if tokenizer is None:
        error = "tokenizer_manager has no initialized tokenizer"
        return None, [], [_error_score(error) for _ in continuations]

    if not isinstance(prompt_text, str):
        error = "prompt must be a string"
        return None, [], [_error_score(error) for _ in continuations]

    try:
        prompt_ids = _encode(tokenizer, prompt_text, add_special_tokens=True)
        logprob_start_len, response_offset = _continuation_logprob_span(len(prompt_ids))
    except Exception as exc:
        error = f"failed to tokenize prompt: {exc}"
        return None, [], [_error_score(error) for _ in continuations]

    input_ids: list[list[int]] = []
    prepared: list[_PreparedBranch] = []
    for index, continuation in enumerate(continuations):
        if not isinstance(continuation, str):
            scores[index] = _error_score("continuation must be a string")
            continue
        if not continuation:
            scores[index] = _error_score("continuation must not be empty")
            continue

        try:
            continuation_ids = _encode(
                tokenizer, continuation, add_special_tokens=False
            )
        except Exception as exc:
            scores[index] = _error_score(f"failed to tokenize continuation: {exc}")
            continue

        if not continuation_ids:
            scores[index] = _error_score("continuation tokenization produced no tokens")
            continue

        input_ids.append(prompt_ids + continuation_ids)
        prepared.append(
            _PreparedBranch(
                original_index=index,
                continuation_ids=continuation_ids,
                response_offset=response_offset,
            )
        )

    if not prepared:
        return None, [], scores

    request = GenerateReqInput(
        input_ids=input_ids,
        sampling_params=[{"max_new_tokens": 0} for _ in prepared],
        return_logprob=[True for _ in prepared],
        logprob_start_len=[logprob_start_len for _ in prepared],
        stream=False,
    )
    return request, prepared, scores


def _score_result(result: Any, branch: _PreparedBranch) -> BranchScore:
    try:
        input_logprobs = result["meta_info"]["input_token_logprobs"]
        continuation_logprobs = input_logprobs[branch.response_offset :]
        expected_tokens = branch.continuation_ids
        if len(continuation_logprobs) != len(expected_tokens):
            raise ValueError(
                "expected "
                f"{len(expected_tokens)} continuation logprobs, got "
                f"{len(continuation_logprobs)}"
            )

        values: list[float] = []
        for expected_token, entry in zip(expected_tokens, continuation_logprobs):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError("malformed input_token_logprobs entry")
            logprob, token_id = entry[0], entry[1]
            if token_id != expected_token:
                raise ValueError(
                    f"continuation token mismatch: expected {expected_token}, got {token_id}"
                )
            if not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
                raise ValueError("continuation logprob is missing or non-finite")
            values.append(float(logprob))

        sum_logprob = sum(values)
        return {
            "mean_logprob": sum_logprob / len(values),
            "sum_logprob": sum_logprob,
            "n_scored_tokens": len(values),
            "error": None,
        }
    except Exception as exc:
        return _error_score(f"failed to score continuation: {exc}")


async def verify_branches(
    tokenizer_manager: Any,
    prompt_text: str,
    continuations: Sequence[Any],
    *,
    request: Any = None,
) -> list[BranchScore]:
    """Score continuations in one batched, prefill-only generate request.

    Invalid branches are reported independently. A batch-level failure is
    copied to every otherwise valid branch so callers always receive one result
    per input continuation.
    """
    try:
        continuations = list(continuations)
    except Exception as exc:
        return [_error_score(f"continuations must be a sequence: {exc}")]

    try:
        batch_request, prepared, scores = _build_batched_request(
            tokenizer_manager, prompt_text, continuations
        )
    except Exception as exc:
        return [
            _error_score(f"failed to build verification request: {exc}")
            for _ in continuations
        ]

    if batch_request is None:
        return scores

    try:
        results = await tokenizer_manager.generate_request(
            batch_request, request
        ).__anext__()
    except Exception as exc:
        error = f"batched verification failed: {exc}"
        for branch in prepared:
            scores[branch.original_index] = _error_score(error)
        return scores

    if isinstance(results, dict) and len(prepared) == 1:
        results = [results]
    if not isinstance(results, list):
        error = "batched verification returned a non-list response"
        for branch in prepared:
            scores[branch.original_index] = _error_score(error)
        return scores

    for result_index, branch in enumerate(prepared):
        if result_index >= len(results):
            scores[branch.original_index] = _error_score(
                "batched verification returned no result for this branch"
            )
            continue
        scores[branch.original_index] = _score_result(results[result_index], branch)

    return scores
