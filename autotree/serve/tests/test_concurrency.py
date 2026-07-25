from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import asdict

import httpx
from autotree_serve import create_app
from autotree_serve.engine import (
    BranchStarted,
    EngineCounters,
    EngineUsage,
    GenerationDone,
    GenerationRequest,
    Message,
    ModelMetadata,
    TokenGenerated,
)
from autotree_serve.runner import EngineRunner
from prometheus_client.parser import text_string_to_metric_families


MODEL_ID = "concurrent-test"


class RecordingTreeKVEngine:
    def __init__(
        self,
        *,
        steps: int = 4,
        step_delay: float = 0.02,
        gate: threading.Event | None = None,
    ) -> None:
        self.steps = steps
        self.step_delay = step_delay
        self.gate = gate
        self._metadata = ModelMetadata(
            id=MODEL_ID,
            engine="treekv",
            description="Controllable blocking engine for concurrency tests.",
            real_model_weights=True,
            tree_policies=("beam",),
        )
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.trace: list[tuple[str, str, float]] = []

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    async def generate(self, request: GenerationRequest):
        label = request.messages[-1].content
        branch_id = "branch-0"
        tokens: list[str] = []
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.trace.append((label, "start", time.perf_counter()))

        try:
            yield BranchStarted(branch_id=branch_id, parent_id=None)
            if self.gate is not None and not self.gate.wait(timeout=5):
                raise TimeoutError("test generation gate was not released")

            for index in range(self.steps):
                time.sleep(self.step_delay)  # noqa: ASYNC251
                token = f"{label}:{index}|"
                tokens.append(token)
                with self._lock:
                    self.trace.append((label, f"step-{index}", time.perf_counter()))
                yield TokenGenerated(
                    branch_id=branch_id,
                    token=token,
                    token_index=index,
                    logprob=-0.5,
                    token_id=index,
                )

            with self._lock:
                self.trace.append((label, "done", time.perf_counter()))
            elapsed = max(self.steps * self.step_delay, 1e-6)
            yield GenerationDone(
                branch_id=branch_id,
                text="".join(tokens),
                finish_reason="length",
                usage=EngineUsage(prompt_tokens=1, completion_tokens=self.steps),
                counters=EngineCounters(
                    logical_tokens=1 + self.steps,
                    physical_tokens=1 + self.steps,
                    useful_tokens=self.steps,
                    elapsed_seconds=elapsed,
                    ttft_seconds=max(self.step_delay, 0.0),
                ),
                tree_summary=None,
            )
        finally:
            with self._lock:
                self._in_flight -= 1


def request_for(label: str) -> GenerationRequest:
    return GenerationRequest(
        model=MODEL_ID,
        messages=(Message(role="user", content=label),),
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        stop=(),
        seed=7,
        user=None,
        tree=None,
    )


async def collect(runner: EngineRunner, label: str):
    return [event async for event in runner.generate(request_for(label))]


async def post_completion(client: httpx.AsyncClient, label: str) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": label}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
    )


def completion_text(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]["content"]


def metric_values(scrape: str) -> dict[str, float]:
    return {
        sample.name: sample.value
        for family in text_string_to_metric_families(scrape)
        for sample in family.samples
        if not sample.labels
    }


async def test_two_server_requests_interleave_progress() -> None:
    engine = RecordingTreeKVEngine(steps=5, step_delay=0.03)
    app = create_app(engine=engine)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            post_completion(client, "alpha"),
            post_completion(client, "beta"),
        )

    assert all(response.status_code == 200 for response in responses)
    starts = {label: at for label, kind, at in engine.trace if kind == "start"}
    dones = {label: at for label, kind, at in engine.trace if kind == "done"}
    assert engine.max_in_flight >= 2
    assert max(starts.values()) < min(dones.values())
    for label in starts:
        other = "beta" if label == "alpha" else "alpha"
        assert any(
            trace_label == other and kind.startswith("step-") and at < dones[label]
            for trace_label, kind, at in engine.trace
        )


async def test_four_requests_raise_aggregate_throughput_materially() -> None:
    async def measure(request_count: int) -> float:
        engine = RecordingTreeKVEngine(steps=5, step_delay=0.03)
        app = create_app(engine=engine)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            started_at = time.perf_counter()
            responses = await asyncio.gather(
                *(
                    post_completion(client, f"load-{index}")
                    for index in range(request_count)
                )
            )
            elapsed = time.perf_counter() - started_at
        assert all(response.status_code == 200 for response in responses)
        return request_count * engine.steps / elapsed

    single_throughput = await measure(1)
    four_request_throughput = await measure(4)

    assert four_request_throughput >= single_throughput * 2.5


async def test_concurrent_outputs_remain_isolated_across_repeated_races() -> None:
    engine = RecordingTreeKVEngine(steps=3, step_delay=0.002)
    app = create_app(engine=engine)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for iteration in range(8):
            labels = [f"request-{iteration}-{index}" for index in range(8)]
            responses = await asyncio.gather(
                *(post_completion(client, label) for label in labels)
            )
            actual = [completion_text(response) for response in responses]
            expected = [
                "".join(f"{label}:{step}|" for step in range(engine.steps))
                for label in labels
            ]
            assert actual == expected

    assert engine.max_in_flight >= 2


async def test_single_request_event_bytes_are_unchanged_by_runner_concurrency() -> None:
    baseline_engine = RecordingTreeKVEngine(steps=3, step_delay=0.0)
    baseline = [
        event async for event in baseline_engine.generate(request_for("stable"))
    ]

    runner = EngineRunner(
        RecordingTreeKVEngine(steps=3, step_delay=0.0),
        max_concurrent_requests=4,
    )
    actual = await collect(runner, "stable")
    await runner.shutdown()

    def encoded(events: list[object]) -> bytes:
        return json.dumps(
            [asdict(event) for event in events],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    assert encoded(actual) == encoded(baseline)


async def test_concurrency_cap_queues_excess_and_exports_metrics() -> None:
    release = threading.Event()
    engine = RecordingTreeKVEngine(steps=1, step_delay=0.0, gate=release)
    app = create_app(engine=engine, max_concurrent_requests=2)
    transport = httpx.ASGITransport(app=app)
    tasks: list[asyncio.Task[httpx.Response]] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        try:
            tasks = [
                asyncio.create_task(post_completion(client, f"capped-{index}"))
                for index in range(3)
            ]
            deadline = time.perf_counter() + 2
            values: dict[str, float] = {}
            while time.perf_counter() < deadline:
                scrape = await client.get("/metrics")
                values = metric_values(scrape.text)
                if (
                    engine.in_flight == 2
                    and values.get("in_flight_requests") == 2
                    and values.get("queued_requests") == 1
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError(
                    f"cap metrics did not settle: engine={engine.in_flight}, metrics={values}"
                )

            assert engine.max_in_flight == 2
            assert values["concurrency_rejections_total"] == 0
        finally:
            release.set()

        responses = await asyncio.gather(*tasks)
        assert all(response.status_code == 200 for response in responses)
        final_values = metric_values((await client.get("/metrics")).text)

    assert engine.max_in_flight == 2
    assert final_values["in_flight_requests"] == 0
    assert final_values["queued_requests"] == 0
