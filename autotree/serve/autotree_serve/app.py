"""FastAPI application factory and OpenAI-compatible wire adapters."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from .engine import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    DeterministicEngine,
    EngineEvent,
    EngineProtocol,
    GenerationDone,
    GenerationRequest,
    KVCapacityExceededError,
    Message,
    TokenGenerated,
    TreeExecution,
)
from .enterprise import EnterpriseConfig, EnterpriseMiddleware
from .metrics import ServeMetrics
from .runner import DEFAULT_MAX_CONCURRENT_REQUESTS, EngineRunner
from .schema import ChatCompletionRequest, TreeCompletionRequest


_UNSUPPORTED_SEMANTIC_FIELDS = frozenset(
    {
        "audio",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "modalities",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "tool_choice",
        "tools",
        "top_logprobs",
        "web_search_options",
    }
)

_PLAYGROUND_PATH = Path(__file__).with_name("playground") / "index.html"
_PLAYGROUND_CSP = (
    "default-src 'self'; style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline'; connect-src 'self'; "
    "img-src 'self' data:; base-uri 'none'; form-action 'self'"
)
_CAPACITY_RETRY_AFTER_SECONDS = 1


class EngineContractError(RuntimeError):
    """Raised when an engine violates the public event/accounting contract."""


class EventAccumulator:
    def __init__(self) -> None:
        self.started: set[str] = set()
        self.terminal: set[str] = set()
        self.parents: dict[str, str | None] = {}
        self.tokens: dict[str, list[str]] = {}
        self.next_token_index: dict[str, int] = {}
        self.token_count = 0
        self.pruned_count = 0
        self.merged_count = 0
        self.finished = False

    def accept(self, event: EngineEvent) -> None:
        if self.finished:
            raise EngineContractError("engine emitted an event after done")
        if isinstance(event, BranchStarted):
            if event.branch_id in self.started:
                raise EngineContractError(f"branch {event.branch_id} started more than once")
            if event.parent_id is not None and event.parent_id not in self.started:
                raise EngineContractError(
                    f"parent_id {event.parent_id!r} for branch {event.branch_id} "
                    "does not reference an existing branch"
                )
            self.started.add(event.branch_id)
            self.parents[event.branch_id] = event.parent_id
            self.tokens[event.branch_id] = []
            self.next_token_index[event.branch_id] = 0
            return
        if isinstance(event, TokenGenerated):
            if event.branch_id not in self.started:
                raise EngineContractError(f"token emitted for unknown branch {event.branch_id}")
            if event.branch_id in self.terminal:
                raise EngineContractError(f"token emitted after branch {event.branch_id} terminated")
            expected_index = self.next_token_index[event.branch_id]
            if event.token_index != expected_index:
                raise EngineContractError(
                    f"branch {event.branch_id} token_index must be {expected_index}, "
                    f"got {event.token_index}"
                )
            if not math.isfinite(event.logprob):
                raise EngineContractError(
                    f"branch {event.branch_id} emitted a non-finite token logprob"
                )
            self.tokens[event.branch_id].append(event.token)
            self.next_token_index[event.branch_id] += 1
            self.token_count += 1
            return
        if isinstance(event, BranchMerged):
            self._validate_merge_target(event)
            self._terminate(event.branch_id)
            self.merged_count += 1
            return
        if isinstance(event, BranchPruned):
            if not event.reason:
                raise EngineContractError(
                    f"branch {event.branch_id} emitted an empty prune reason"
                )
            self._terminate(event.branch_id)
            self.pruned_count += 1
            return
        if isinstance(event, GenerationDone):
            self._terminate(event.branch_id)
            if self.started != self.terminal:
                missing = sorted(self.started - self.terminal)
                raise EngineContractError(f"branches without terminal events: {missing}")
            if event.usage.completion_tokens != self.token_count:
                raise EngineContractError(
                    "completion token usage does not match emitted token events: "
                    f"usage={event.usage.completion_tokens}, events={self.token_count}"
                )
            self._validate_done_counters(event)
            if event.tree_summary is not None:
                expected_kv_reuse_ratio = (
                    event.counters.logical_tokens / event.counters.physical_tokens
                    if event.counters.physical_tokens > 0
                    else math.inf
                )
                if (
                    not math.isfinite(event.tree_summary.kv_reuse_ratio)
                    or event.tree_summary.kv_reuse_ratio < 1
                    or not math.isclose(
                        event.tree_summary.kv_reuse_ratio,
                        expected_kv_reuse_ratio,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    raise EngineContractError(
                        "tree summary kv_reuse_ratio must equal "
                        "counters.logical_tokens / counters.physical_tokens and be >= 1"
                    )
                if event.tree_summary.branch_count != len(self.started):
                    raise EngineContractError(
                        "tree summary branch_count does not match branch_started events: "
                        f"summary={event.tree_summary.branch_count}, events={len(self.started)}"
                    )
                if event.tree_summary.pruned_count != self.pruned_count:
                    raise EngineContractError(
                        "tree summary pruned_count does not match branch_pruned events: "
                        f"summary={event.tree_summary.pruned_count}, "
                        f"events={self.pruned_count}"
                    )
                if event.tree_summary.merged_count != self.merged_count:
                    raise EngineContractError(
                        "tree summary merged_count does not match branch_merged events: "
                        f"summary={event.tree_summary.merged_count}, "
                        f"events={self.merged_count}"
                    )
                emitted_per_branch = {
                    branch_id: len(self.tokens[branch_id])
                    for branch_id in sorted(self.started)
                }
                if event.tree_summary.tokens_spent_per_branch != emitted_per_branch:
                    raise EngineContractError(
                        "tree summary tokens_spent_per_branch does not match emitted "
                        f"token events: summary={event.tree_summary.tokens_spent_per_branch}, "
                        f"events={emitted_per_branch}"
                    )
                score_branches = set(event.tree_summary.final_scores)
                if score_branches != self.started:
                    raise EngineContractError(
                        "tree summary final_scores must be a branch_id-keyed mapping for "
                        f"every branch: scores={sorted(score_branches)}, "
                        f"events={sorted(self.started)}"
                    )
                if any(
                    not math.isfinite(score)
                    for score in event.tree_summary.final_scores.values()
                ):
                    raise EngineContractError(
                        "tree summary final_scores must contain only finite values"
                    )
                if event.tree_summary.winner_branch_id != event.branch_id:
                    raise EngineContractError(
                        "tree summary winner_branch_id does not match done branch_id"
                    )
            emitted_text = "".join(
                token
                for branch_id in self._branch_path(event.branch_id)
                for token in self.tokens[branch_id]
            )
            if emitted_text != event.text:
                raise EngineContractError("winning text does not match winner token events")
            self.finished = True

    @staticmethod
    def _validate_done_counters(event: GenerationDone) -> None:
        if event.usage.prompt_tokens < 0:
            raise EngineContractError("usage.prompt_tokens must be non-negative")
        counters = event.counters
        if counters.logical_tokens < 0:
            raise EngineContractError("counters.logical_tokens must be non-negative")
        if counters.physical_tokens < 0:
            raise EngineContractError("counters.physical_tokens must be non-negative")
        if counters.useful_tokens < 0:
            raise EngineContractError("counters.useful_tokens must be non-negative")
        if not math.isfinite(counters.elapsed_seconds) or counters.elapsed_seconds <= 0:
            raise EngineContractError("counters.elapsed_seconds must be finite and positive")
        if not math.isfinite(counters.ttft_seconds) or counters.ttft_seconds < 0:
            raise EngineContractError("counters.ttft_seconds must be finite and non-negative")

    def _branch_path(self, branch_id: str) -> tuple[str, ...]:
        reversed_path: list[str] = []
        current: str | None = branch_id
        while current is not None:
            reversed_path.append(current)
            current = self.parents[current]
        return tuple(reversed(reversed_path))

    def _validate_merge_target(self, event: BranchMerged) -> None:
        if event.into_branch_id == event.branch_id:
            raise EngineContractError(
                f"into_branch_id {event.into_branch_id!r} cannot be the merged branch itself"
            )
        if event.into_branch_id not in self.started:
            raise EngineContractError(
                f"into_branch_id {event.into_branch_id!r} does not reference an existing branch"
            )
        if event.into_branch_id in self.terminal:
            raise EngineContractError(
                f"into_branch_id {event.into_branch_id!r} references a terminated branch"
            )

    def _terminate(self, branch_id: str) -> None:
        if branch_id not in self.started:
            raise EngineContractError(f"terminal event for unknown branch {branch_id}")
        if branch_id in self.terminal:
            raise EngineContractError(f"branch {branch_id} terminated more than once")
        self.terminal.add(branch_id)


def create_app(
    engine: EngineProtocol | None = None,
    *,
    model_id: str = "autotree-deterministic",
    registry: CollectorRegistry | None = None,
    enterprise_config: EnterpriseConfig | None = None,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
) -> FastAPI:
    selected_engine = engine or DeterministicEngine(model_id=model_id)
    metrics = ServeMetrics(registry)
    engine_runner = EngineRunner(
        selected_engine,
        max_concurrent_requests=max_concurrent_requests,
        metrics=metrics,
    )
    enterprise = enterprise_config or EnterpriseConfig.from_env()
    started_at = time.monotonic()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await engine_runner.shutdown()

    app = FastAPI(title="autotree-serve", version="0.1.0", lifespan=lifespan)
    app.state.engine = selected_engine
    app.state.engine_runner = engine_runner
    app.state.metrics = metrics
    app.state.enterprise_config = enterprise
    app.add_middleware(EnterpriseMiddleware, config=enterprise, metrics=metrics)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = first.get("loc", ())
        param = ".".join(str(part) for part in location if part != "body") or None
        return _openai_error(
            status_code=422,
            message=first.get("msg", "Request validation failed"),
            param=param,
            code="validation_error",
        )

    @app.exception_handler(EngineContractError)
    async def engine_contract_error_handler(
        _request: Request,
        exc: EngineContractError,
    ) -> JSONResponse:
        return _openai_error(
            status_code=500,
            message=f"Engine event contract violation: {exc}",
            error_type="server_error",
            code="engine_contract_error",
        )

    @app.exception_handler(KVCapacityExceededError)
    async def kv_capacity_error_handler(
        _request: Request,
        exc: KVCapacityExceededError,
    ) -> JSONResponse:
        metrics.capacity_rejections_total.inc()
        return _openai_error(
            status_code=429,
            message=str(exc),
            param="kv_pages",
            error_type="rate_limit_error",
            code="kv_capacity_exhausted",
            headers={"Retry-After": str(_CAPACITY_RETRY_AFTER_SECONDS)},
        )

    @app.middleware("http")
    async def count_requests(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", None) or "unmatched"
        metrics.requests_total.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        return response

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        metadata = selected_engine.model_metadata
        return {
            "object": "list",
            "data": [
                {
                    "id": metadata.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "autotree",
                    "metadata": {
                        "engine": metadata.engine,
                        "description": metadata.description,
                        "real_model_weights": metadata.real_model_weights,
                        "tree_policies": list(metadata.tree_policies),
                    },
                }
            ],
        }

    @app.get("/health")
    async def health() -> dict[str, object]:
        metadata = engine_runner.model_metadata
        return {
            "engine_kind": metadata.engine,
            "model_id": metadata.id,
            "uptime_seconds": max(0.0, time.monotonic() - started_at),
            "ready": engine_runner.ready,
        }

    @app.get("/playground", response_class=HTMLResponse)
    @app.get("/playground/", response_class=HTMLResponse)
    async def playground() -> HTMLResponse:
        return HTMLResponse(
            _PLAYGROUND_PATH.read_text(encoding="utf-8"),
            headers={"Content-Security-Policy": _PLAYGROUND_CSP},
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        http_request: Request,
        body: ChatCompletionRequest,
    ) -> Response:
        feature_error = _validate_supported_features(body)
        if feature_error:
            return feature_error
        model_error = _validate_model(body.model, selected_engine)
        if model_error:
            return model_error
        request = _to_engine_request(body)
        if body.stream:
            return StreamingResponse(
                _chat_stream(
                    engine_runner,
                    request,
                    metrics,
                    request_scope=http_request.scope,
                    include_usage=(
                        body.stream_options.include_usage
                        if body.stream_options is not None
                        else False
                    ),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        _events, done = await _collect_events(
            engine_runner,
            request,
            metrics,
            request_scope=http_request.scope,
        )
        return JSONResponse(_completion_response(done, body.model))

    @app.post("/v1/tree/completions")
    async def tree_completions(
        http_request: Request,
        body: TreeCompletionRequest,
    ) -> Response:
        feature_error = _validate_supported_features(body)
        if feature_error:
            return feature_error
        model_error = _validate_model(body.model, selected_engine)
        if model_error:
            return model_error
        request = _to_engine_request(body)
        if body.stream:
            return StreamingResponse(
                _tree_stream(
                    engine_runner,
                    request,
                    metrics,
                    request_scope=http_request.scope,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        _events, done = await _collect_events(
            engine_runner,
            request,
            metrics,
            request_scope=http_request.scope,
        )
        return JSONResponse(_completion_response(done, body.model))

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


def _validate_model(model: str, engine: EngineProtocol) -> JSONResponse | None:
    if model == engine.model_metadata.id:
        return None
    return _openai_error(
        status_code=404,
        message=f"Model '{model}' is not served by this process.",
        param="model",
        code="model_not_found",
    )


def _validate_supported_features(body: ChatCompletionRequest) -> JSONResponse | None:
    if body.n != 1:
        return _openai_error(
            status_code=400,
            message="AutoTree currently supports exactly one completion; 'n' must be 1.",
            param="n",
            code="unsupported_feature",
        )
    extras = body.model_extra or {}
    unsupported = next(
        (name for name in extras if name in _UNSUPPORTED_SEMANTIC_FIELDS),
        None,
    )
    if unsupported is None:
        return None
    return _openai_error(
        status_code=400,
        message=f"Unsupported chat completion feature: '{unsupported}'.",
        param=unsupported,
        code="unsupported_feature",
    )


def _to_engine_request(body: ChatCompletionRequest) -> GenerationRequest:
    tree = None
    if body.tree is not None:
        tree = TreeExecution(
            policy=body.tree.policy,
            branches=body.tree.branches,
            budget_tokens=body.tree.budget_tokens,
            scorer=body.tree.scorer,
            consensus_interval=body.tree.consensus_interval,
            consensus_warmup=body.tree.consensus_warmup,
            min_survivors=body.tree.min_survivors,
        )
    return GenerationRequest(
        model=body.model,
        messages=tuple(Message(role=item.role, content=item.content) for item in body.messages),
        max_tokens=body.resolved_max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stop=(body.stop,) if isinstance(body.stop, str) else tuple(body.stop or ()),
        seed=body.seed,
        user=body.user,
        tree=tree,
    )


async def _collect_events(
    engine: EngineProtocol,
    request: GenerationRequest,
    metrics: ServeMetrics,
    *,
    request_scope: dict[str, Any] | None = None,
) -> tuple[list[EngineEvent], GenerationDone]:
    accumulator = EventAccumulator()
    events: list[EngineEvent] = []
    done: GenerationDone | None = None
    async for event in engine.generate(request):
        accumulator.accept(event)
        metrics.observe_event(event)
        if isinstance(event, TokenGenerated) and request_scope is not None:
            request_scope["autotree.audit_tokens"] = accumulator.token_count
        events.append(event)
        if isinstance(event, GenerationDone):
            done = event
    if done is None:
        raise EngineContractError("engine stream ended without a done event")
    return events, done


def _completion_response(
    done: GenerationDone,
    model: str,
) -> dict[str, object]:
    response: dict[str, object] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": done.text},
                "logprobs": None,
                "finish_reason": done.finish_reason,
            }
        ],
        "usage": done.usage.to_dict(),
    }
    if done.tree_summary is not None:
        response["tree"] = done.tree_summary.to_dict()
    return response


async def _chat_stream(
    engine: EngineProtocol,
    request: GenerationRequest,
    metrics: ServeMetrics,
    *,
    include_usage: bool,
    request_scope: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    yield _sse_data(
        _chat_chunk(
            stream_id,
            created,
            request.model,
            choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        )
    )

    if request.tree is not None:
        accumulator = EventAccumulator()
        done = None
        try:
            async for event in engine.generate(request):
                accumulator.accept(event)
                metrics.observe_event(event)
                if isinstance(event, TokenGenerated) and request_scope is not None:
                    request_scope["autotree.audit_tokens"] = accumulator.token_count
                if not isinstance(event, GenerationDone):
                    yield _sse_data(
                        _chat_chunk(
                            stream_id,
                            created,
                            request.model,
                            choices=[],
                            tree_event=_event_payload(event),
                        )
                    )
                    continue
                done = event
        except KVCapacityExceededError as error:
            metrics.capacity_rejections_total.inc()
            yield _sse_data(
                _chat_capacity_error_chunk(stream_id, created, request.model, error)
            )
            yield "data: [DONE]\n\n"
            return
        if done is None:
            raise EngineContractError("engine stream ended without a done event")
        if done.text:
            yield _sse_data(
                _chat_chunk(
                    stream_id,
                    created,
                    request.model,
                    choices=[{"index": 0, "delta": {"content": done.text}, "finish_reason": None}],
                )
            )
        yield _sse_data(
            _chat_chunk(
                stream_id,
                created,
                request.model,
                choices=[{"index": 0, "delta": {}, "finish_reason": done.finish_reason}],
                tree=done.tree_summary.to_dict() if done.tree_summary else None,
                tree_event=_event_payload(done),
            )
        )
    else:
        accumulator = EventAccumulator()
        done = None
        try:
            async for event in engine.generate(request):
                accumulator.accept(event)
                metrics.observe_event(event)
                if isinstance(event, TokenGenerated) and request_scope is not None:
                    request_scope["autotree.audit_tokens"] = accumulator.token_count
                if isinstance(event, TokenGenerated):
                    yield _sse_data(
                        _chat_chunk(
                            stream_id,
                            created,
                            request.model,
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {"content": event.token},
                                    "finish_reason": None,
                                }
                            ],
                        )
                    )
                elif isinstance(event, GenerationDone):
                    done = event
                    yield _sse_data(
                        _chat_chunk(
                            stream_id,
                            created,
                            request.model,
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": event.finish_reason,
                                }
                            ],
                        )
                    )
        except KVCapacityExceededError as error:
            metrics.capacity_rejections_total.inc()
            yield _sse_data(
                _chat_capacity_error_chunk(stream_id, created, request.model, error)
            )
            yield "data: [DONE]\n\n"
            return
        if done is None:
            raise EngineContractError("engine stream ended without a done event")

    if include_usage:
        yield _sse_data(
            _chat_chunk(
                stream_id,
                created,
                request.model,
                choices=[],
                usage=done.usage.to_dict(),
            )
        )
    yield "data: [DONE]\n\n"


async def _tree_stream(
    engine: EngineProtocol,
    request: GenerationRequest,
    metrics: ServeMetrics,
    *,
    request_scope: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    accumulator = EventAccumulator()
    saw_done = False
    try:
        async for event in engine.generate(request):
            accumulator.accept(event)
            metrics.observe_event(event)
            if isinstance(event, TokenGenerated) and request_scope is not None:
                request_scope["autotree.audit_tokens"] = accumulator.token_count
            saw_done = saw_done or isinstance(event, GenerationDone)
            payload = _event_payload(event)
            yield (
                f"event: {event.type}\n"
                f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            )
    except KVCapacityExceededError as error:
        metrics.capacity_rejections_total.inc()
        payload = _capacity_error_event(error)
        yield (
            "event: error\n"
            f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
        )
        yield "data: [DONE]\n\n"
        return
    if not saw_done:
        raise EngineContractError("engine stream ended without a done event")
    yield "data: [DONE]\n\n"


def _chat_chunk(
    stream_id: str,
    created: int,
    model: str,
    *,
    choices: list[dict[str, object]],
    usage: dict[str, int] | None = None,
    tree: dict[str, object] | None = None,
    tree_event: dict[str, object] | None = None,
) -> dict[str, object]:
    chunk: dict[str, object] = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    if tree is not None:
        chunk["tree"] = tree
    if tree_event is not None:
        chunk["tree_event"] = tree_event
    return chunk


def _event_payload(event: EngineEvent) -> dict[str, object]:
    if isinstance(event, GenerationDone):
        payload: dict[str, object] = {
            "type": event.type,
            "branch_id": event.branch_id,
            "text": event.text,
            "finish_reason": event.finish_reason,
            "usage": event.usage.to_dict(),
            "counters": asdict(event.counters),
        }
        if event.tree_summary is not None:
            payload["tree"] = event.tree_summary.to_dict()
        return payload
    return {"type": event.type, **asdict(event)}


def _sse_data(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _capacity_error_event(error: KVCapacityExceededError) -> dict[str, object]:
    return {
        "type": "error",
        "error": _capacity_error_details(error),
        "retry_after_seconds": _CAPACITY_RETRY_AFTER_SECONDS,
    }


def _chat_capacity_error_chunk(
    stream_id: str,
    created: int,
    model: str,
    error: KVCapacityExceededError,
) -> dict[str, object]:
    return _chat_chunk(
        stream_id,
        created,
        model,
        choices=[
            {
                "index": 0,
                "delta": {
                    "error": _capacity_error_details(error),
                    "retry_after_seconds": _CAPACITY_RETRY_AFTER_SECONDS,
                },
                "finish_reason": "length",
            }
        ],
    )


def _capacity_error_details(error: KVCapacityExceededError) -> dict[str, object]:
    return {
        "message": str(error),
        "type": "rate_limit_error",
        "param": "kv_pages",
        "code": "kv_capacity_exhausted",
    }


def _openai_error(
    *,
    status_code: int,
    message: str,
    param: str | None = None,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )
