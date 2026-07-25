"""Test-only ASGI implementation of the documented AutoTree wire contract."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx


class ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks


class ASGISyncTransport(httpx.BaseTransport):
    """Run an ASGI app in-process while preserving response body chunks."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request_body = request.read()
        status_code = 500
        response_headers: list[tuple[bytes, bytes]] = []
        response_chunks: list[bytes] = []
        request_sent = False

        async def receive() -> dict[str, Any]:
            nonlocal request_sent
            if request_sent:
                return {"type": "http.disconnect"}
            request_sent = True
            return {"type": "http.request", "body": request_body, "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            nonlocal status_code, response_headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                response_chunks.append(message.get("body", b""))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path,
            "query_string": request.url.query,
            "headers": request.headers.raw,
            "client": ("test", 123),
            "server": (request.url.host, request.url.port or 80),
        }
        asyncio.run(self.app(scope, receive, send))
        return httpx.Response(
            status_code,
            headers=response_headers,
            stream=ChunkStream(response_chunks),
            request=request,
        )


class MockAutoTreeASGI:
    """Small protocol double kept under tests and excluded from the wheel."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, scope, receive, send) -> None:
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        payload = json.loads(body or b"{}")
        self.requests.append({"path": scope["path"], "body": payload})

        if payload.get("scenario") == "http_error":
            await self._json(send, 503, {"error": {"message": "capacity unavailable"}})
            return
        if (
            payload.get("scenario") == "tree_not_supported"
            and scope["path"] == "/v1/tree/completions"
        ):
            await self._json(send, 404, {"error": "not found"})
            return
        if scope["path"] == "/v1/chat/completions":
            if payload.get("stream"):
                await self._chat_stream(payload, send)
            else:
                await self._chat(payload, send)
            return
        if scope["path"] == "/v1/tree/completions":
            if payload.get("stream"):
                await self._tree_stream(payload, send)
            else:
                await self._tree_response(payload, send)
            return
        await self._json(send, 404, {"error": "not found"})

    async def _chat(self, payload, send) -> None:
        tree = self._summary(payload, {"root": 2}, {"root": 0.9})
        await self._json(
            send,
            200,
            {
                "id": "chat-1",
                "object": "chat.completion",
                "model": payload.get("model", "server-default"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "tree": tree,
            },
        )

    async def _tree_response(self, payload, send) -> None:
        summary = self._summary(
            payload,
            {"root": 2, "alternate": 1},
            {"root": 0.9, "alternate": 0.25},
        )
        if payload.get("scenario") == "null_kv_reuse":
            summary["kv_reuse_ratio"] = None
        await self._json(
            send,
            200,
            {
                "id": "tree-1",
                "object": "chat.completion",
                "model": payload.get("model", "server-default"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                "tree": summary,
            },
        )

    async def _chat_stream(self, payload, send) -> None:
        chunks = [
            {
                "id": "chat-stream-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            },
            {
                "id": "chat-stream-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "ok"}}],
                "tree": self._summary(payload, {"root": 1}, {"root": 0.9}),
            },
        ]
        wire = b": keep-alive\r\n\r\n" + b"".join(
            f"data: {json.dumps(chunk)}\r\n\r\n".encode() for chunk in chunks
        ) + b"data: [DONE]\r\n\r\n"
        await self._sse(send, wire)

    async def _tree_stream(self, payload, send) -> None:
        scenario = payload.get("scenario", "valid")
        if scenario == "invalid_json":
            await self._sse(send, b'data: {"type":\r\n\r\ndata: [DONE]\r\n\r\n')
            return
        if scenario == "invalid_event":
            await self._sse(
                send,
                b'data: {"type":"token","branch_id":"root",'
                b'"token_index":0,"token":"x"}\r\n\r\ndata: [DONE]\r\n\r\n',
            )
            return
        if scenario == "capacity_error":
            event = {
                "type": "error",
                "error": {
                    "message": "Tree-KV capacity is exhausted.",
                    "type": "rate_limit_error",
                    "param": "kv_pages",
                    "code": "kv_capacity_exhausted",
                },
                "retry_after_seconds": 2,
            }
            wire = (
                f"data: {json.dumps(event)}\r\n\r\ndata: [DONE]\r\n\r\n".encode()
            )
            await self._sse(send, wire)
            return
        events = [
            {"type": "branch_started", "branch_id": "root", "parent_id": None},
            {"type": "token", "branch_id": "root", "token_index": 0, "token": "an", "token_id": 101, "logprob": -0.1},
            {"type": "branch_started", "branch_id": "alt", "parent_id": "root"},
            {"type": "token", "branch_id": "alt", "token_index": 0, "token": "no", "token_id": 201, "logprob": -1.0},
            {"type": "token", "branch_id": "root", "token_index": 1, "token": "swer", "token_id": 102, "logprob": -0.2},
            {"type": "branch_pruned", "branch_id": "alt", "reason": "low_score"},
        ]
        if scenario == "unknown_branch":
            events = [
                {"type": "token", "branch_id": "ghost", "token_index": 0, "token": "x", "logprob": -1.0}
            ]
        if scenario != "missing_done":
            completion_tokens = 4 if scenario == "usage_mismatch" else 3
            events.append(
                {
                    "type": "done",
                    "branch_id": "root",
                    "text": "answer",
                    "finish_reason": "length",
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": completion_tokens,
                        "total_tokens": completion_tokens + 2,
                    },
                    "counters": {
                        "logical_tokens": 5,
                        "physical_tokens": 5,
                        "useful_tokens": 2,
                        "elapsed_seconds": 0.01,
                        "ttft_seconds": 0.001,
                    },
                    "tree": self._summary(
                        payload,
                        {"root": 2, "alt": 1},
                        {"root": 0.9, "alt": 0.1},
                        pruned_count=1,
                    ),
                }
            )
        frames = [f"data: {json.dumps(event)}\r\n\r\n".encode() for event in events]
        wire = b": keep-alive\r\n\r\n" + b"".join(frames) + b"data: [DONE]\r\n\r\n"
        # Split inside JSON tokens and CRLF boundaries to exercise incremental parsing.
        cuts = [1, 7, 19, 41, 88, 133, 211]
        chunks: list[bytes] = []
        start = 0
        for cut in cuts:
            chunks.append(wire[start:cut])
            start = cut
        chunks.append(wire[start:])
        await self._sse(send, b"".join(chunks), chunks=chunks)

    @staticmethod
    def _summary(payload, token_counts, scores, pruned_count=0):
        final_scores = (
            list(scores.values())
            if payload.get("scenario") == "positional_scores"
            else scores
        )
        return {
            "policy": payload.get("tree", {}).get("policy", "beam"),
            "branch_count": len(token_counts),
            "pruned_count": pruned_count,
            "merged_count": 0,
            "winner_branch_id": max(scores, key=scores.get),
            "tokens_spent_per_branch": token_counts,
            "final_scores": final_scores,
            "scorer": payload.get("tree", {}).get("scorer"),
            "kv_reuse_ratio": 1.0,
        }

    @staticmethod
    async def _json(send, status, payload) -> None:
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    @staticmethod
    async def _sse(send, wire, *, chunks=None) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for chunk in chunks or [wire]:
            await send(
                {"type": "http.response.body", "body": chunk, "more_body": True}
            )
        await send({"type": "http.response.body", "body": b"", "more_body": False})


def make_http_client(app: MockAutoTreeASGI) -> httpx.Client:
    return httpx.Client(transport=ASGISyncTransport(app))
