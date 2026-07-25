"""Small SSE decoder for httpx streaming responses."""

from __future__ import annotations

from collections.abc import Iterator
import json
from typing import Any

import httpx

from .errors import SSEParseError


def iter_sse_data(response: httpx.Response) -> Iterator[str]:
    """Yield complete SSE ``data`` payloads.

    httpx buffers arbitrary byte chunks into complete lines, so JSON split
    across network chunks is handled without assuming transport boundaries.
    Comment keep-alives and non-data fields are ignored. ``[DONE]`` terminates
    the stream and is not yielded.
    """

    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines.clear()
                if payload == "[DONE]":
                    return
                yield payload
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data" or not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        data_lines.append(value)
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            yield payload


def decode_sse_json(payload: str) -> dict[str, Any]:
    """Decode one SSE data payload and require a JSON object."""

    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SSEParseError("invalid_sse_json", str(exc)) from exc
    if not isinstance(value, dict):
        raise SSEParseError("invalid_sse_payload", "expected a JSON object")
    return value
