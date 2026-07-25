from __future__ import annotations

import asyncio

import pytest

from autotree_serve.engine import (
    DeterministicEngine,
    GenerationDone,
    GenerationRequest,
    Message,
)
from autotree_serve.runner import EngineRunner


class GatedEngine:
    def __init__(self) -> None:
        self._delegate = DeterministicEngine()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def model_metadata(self):
        return self._delegate.model_metadata

    async def generate(self, request):
        self.started.set()
        await self.release.wait()
        async for event in self._delegate.generate(request):
            yield event


def request() -> GenerationRequest:
    return GenerationRequest(
        model="deterministic-demo",
        messages=(Message(role="user", content="finish before shutdown"),),
        max_tokens=1,
        temperature=1.0,
        top_p=1.0,
        stop=(),
        seed=1,
        user=None,
        tree=None,
    )


async def collect(runner: EngineRunner):
    return [event async for event in runner.generate(request())]


async def test_shutdown_drains_admitted_stream_and_rejects_new_work():
    engine = GatedEngine()
    runner = EngineRunner(engine)
    generation = asyncio.create_task(collect(runner))
    await engine.started.wait()

    shutdown = asyncio.create_task(runner.shutdown())
    await asyncio.sleep(0)

    assert runner.ready is False
    assert shutdown.done() is False
    with pytest.raises(RuntimeError, match="shutting down"):
        await anext(runner.generate(request()))

    engine.release.set()
    events = await generation
    await shutdown

    assert isinstance(events[-1], GenerationDone)
