"""Synchronous typed client for AutoTree's OpenAI-compatible HTTP API."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import json
from typing import Any, Literal, overload

import httpx
from pydantic import ValidationError

from .errors import SSEParseError, TreeHTTPError, TreeNotSupportedError
from .models import (
    ChatCompletionResponse,
    TreeCompletion,
    TreeCompletionResponse,
    TreeEvent,
    TreeParameters,
    TreePolicy,
    parse_tree_event,
)
from .sse import decode_sse_json, iter_sse_data
from .trace import TraceAssembler


class TreeClient:
    """Typed synchronous HTTP client with no implicit POST retries."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float | httpx.Timeout = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TreeClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @overload
    def completions(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        tree: TreeParameters | Mapping[str, Any] | None = None,
        stream: Literal[False] = False,
        **parameters: Any,
    ) -> ChatCompletionResponse: ...

    @overload
    def completions(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        tree: TreeParameters | Mapping[str, Any] | None = None,
        stream: Literal[True],
        **parameters: Any,
    ) -> Iterator[dict[str, Any]]: ...

    def completions(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str | None = None,
        tree: TreeParameters | Mapping[str, Any] | None = None,
        stream: bool = False,
        **parameters: Any,
    ) -> ChatCompletionResponse | Iterator[dict[str, Any]]:
        """Call ``/v1/chat/completions`` with an optional ``tree`` extra."""

        body = self._body(messages, model, stream, tree, parameters)
        if stream:
            return self._stream_json("/v1/chat/completions", body)
        response = self._client.post(self._url("/v1/chat/completions"), json=body)
        self._raise_for_status(response)
        return self._parse_response(response, ChatCompletionResponse)

    def tree_completions(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str,
        branches: int = 4,
        budget_tokens: int,
        policy: TreePolicy = "beam",
        scorer: str | None = None,
        **sampling: Any,
    ) -> TreeCompletion:
        """Run first-class tree completion without hand-built ``extra_body``."""

        tree = TreeParameters(
            policy=policy,
            branches=branches,
            budget_tokens=budget_tokens,
            scorer=scorer,
        )
        body = self._body(messages, model, False, tree, sampling)
        response = self._client.post(self._url("/v1/tree/completions"), json=body)
        self._raise_for_status(response, tree_endpoint=True)
        wire_response = self._parse_response(response, TreeCompletionResponse)
        try:
            return TreeCompletion.from_response(wire_response)
        except ValidationError as exc:
            raise SSEParseError("invalid_response", str(exc)) from exc

    def stream_tree_completions(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tree: TreeParameters | Mapping[str, Any],
        model: str | None = None,
        **parameters: Any,
    ) -> Iterator[TreeEvent]:
        """Yield typed tree events and validate the complete consumed stream."""

        body = self._body(messages, model, True, tree, parameters)
        assembler = TraceAssembler()
        with self._client.stream(
            "POST", self._url("/v1/tree/completions"), json=body
        ) as response:
            self._raise_for_status(response, tree_endpoint=True)
            for payload in iter_sse_data(response):
                event = parse_tree_event(decode_sse_json(payload))
                assembler.add(event)
                yield event
        assembler.finish()

    def _stream_json(
        self, path: str, body: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        with self._client.stream("POST", self._url(path), json=body) as response:
            self._raise_for_status(response)
            for payload in iter_sse_data(response):
                yield decode_sse_json(payload)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _body(
        messages: Sequence[Mapping[str, Any]],
        model: str | None,
        stream: bool,
        tree: TreeParameters | Mapping[str, Any] | None,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": list(messages), **parameters}
        body["stream"] = stream
        if model is not None:
            body["model"] = model
        if tree is not None:
            body["tree"] = (
                tree.model_dump(exclude_none=False)
                if isinstance(tree, TreeParameters)
                else dict(tree)
            )
        return body

    @staticmethod
    def _raise_for_status(
        response: httpx.Response, *, tree_endpoint: bool = False
    ) -> None:
        if response.is_success:
            return
        if tree_endpoint and response.status_code == 404:
            raise TreeNotSupportedError()
        try:
            body = response.json()
            detail = body.get("error", body) if isinstance(body, dict) else body
            rendered = detail if isinstance(detail, str) else json.dumps(detail)
        except (ValueError, UnicodeDecodeError):
            rendered = response.text
        raise TreeHTTPError(response.status_code, rendered)

    @staticmethod
    def _parse_response(response: httpx.Response, model: type[Any]) -> Any:
        try:
            payload = response.json()
        except ValueError as exc:
            raise SSEParseError("invalid_json_response", str(exc)) from exc
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise SSEParseError("invalid_response", str(exc)) from exc
