"""Responsive execution adapter for engines with blocking generation steps."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .engine import EngineEvent, EngineProtocol, GenerationRequest, ModelMetadata

if TYPE_CHECKING:
    from .metrics import ServeMetrics


_WORKER_DONE = object()
DEFAULT_MAX_CONCURRENT_REQUESTS = 8


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    error: BaseException


class EngineRunner:
    """Bound generation concurrency and keep blocking engine work off-loop."""

    def __init__(
        self,
        engine: EngineProtocol,
        *,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
        metrics: ServeMetrics | None = None,
    ) -> None:
        if (
            isinstance(max_concurrent_requests, bool)
            or not isinstance(max_concurrent_requests, int)
            or max_concurrent_requests <= 0
        ):
            raise ValueError("max_concurrent_requests must be a positive integer")
        self._engine = engine
        self._max_concurrent_requests = max_concurrent_requests
        self._generation_slots = asyncio.Semaphore(max_concurrent_requests)
        self._metrics = metrics
        self._accepting = True
        self._active_generations = 0
        self._in_flight_generations = 0
        self._queued_generations = 0
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._engine.model_metadata

    @property
    def ready(self) -> bool:
        return self._accepting

    @property
    def max_concurrent_requests(self) -> int:
        return self._max_concurrent_requests

    @property
    def in_flight_requests(self) -> int:
        return self._in_flight_generations

    @property
    def queued_requests(self) -> int:
        return self._queued_generations

    async def generate(self, request: GenerationRequest):
        if not self._accepting:
            if self._metrics is not None:
                self._metrics.concurrency_rejections_total.inc()
            raise RuntimeError("engine runner is shutting down")
        self._active_generations += 1
        self._drained.clear()
        queued = self._in_flight_generations >= self._max_concurrent_requests
        if queued:
            self._queued_generations += 1
            self._update_concurrency_metrics()
        acquired = False
        try:
            try:
                await self._generation_slots.acquire()
                acquired = True
            finally:
                if queued:
                    self._queued_generations -= 1
                    self._update_concurrency_metrics()

            self._in_flight_generations += 1
            self._update_concurrency_metrics()
            try:
                if self.model_metadata.engine != "treekv":
                    async for event in self._engine.generate(request):
                        yield event
                    return

                async for event in self._generate_in_worker(request):
                    yield event
            finally:
                self._in_flight_generations -= 1
                self._update_concurrency_metrics()
                if acquired:
                    self._generation_slots.release()
                    acquired = False
        finally:
            if acquired:
                self._generation_slots.release()
            self._active_generations -= 1
            if self._active_generations == 0:
                self._drained.set()

    async def shutdown(self) -> None:
        """Stop admission and wait for every admitted generation to finish."""
        self._accepting = False
        await self._drained.wait()

    def _update_concurrency_metrics(self) -> None:
        if self._metrics is None:
            return
        self._metrics.in_flight_requests.set(self._in_flight_generations)
        self._metrics.queued_requests.set(self._queued_generations)

    async def _generate_in_worker(self, request: GenerationRequest):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[EngineEvent | _WorkerFailure | object] = asyncio.Queue()

        def publish(item: EngineEvent | _WorkerFailure | object) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                # The interpreter/event loop is already being force-closed.
                pass

        async def consume() -> None:
            async for event in self._engine.generate(request):
                publish(event)

        def worker() -> None:
            try:
                asyncio.run(consume())
            except BaseException as error:
                publish(_WorkerFailure(error))
            finally:
                publish(_WORKER_DONE)

        thread = threading.Thread(
            target=worker,
            name=f"autotree-{self.model_metadata.engine}-generation",
            daemon=False,
        )
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is _WORKER_DONE:
                    break
                if isinstance(item, _WorkerFailure):
                    raise item.error
                yield item
        finally:
            if thread.is_alive():
                await asyncio.shield(asyncio.to_thread(thread.join))


__all__ = ["DEFAULT_MAX_CONCURRENT_REQUESTS", "EngineRunner"]
