from __future__ import annotations

import asyncio
import time

import httpx

from autotree_serve import create_app
from autotree_serve.engine import DeterministicEngine, ModelMetadata


class SlowTreeKVEngine:
    def __init__(self) -> None:
        self._delegate = DeterministicEngine("slow-treekv")
        self._metadata = ModelMetadata(
            id="slow-treekv",
            engine="treekv",
            description="Test engine with blocking generation work.",
            real_model_weights=True,
            tree_policies=("beam", "best_first", "mcts"),
        )

    @property
    def model_metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(self, request):
        time.sleep(0.35)
        async for event in self._delegate.generate(request):
            yield event


async def test_models_stay_responsive_during_slow_treekv_generation():
    app = create_app(engine=SlowTreeKVEngine())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        started_at = time.perf_counter()
        generation = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "slow-treekv",
                    "messages": [{"role": "user", "content": "block once"}],
                    "max_tokens": 1,
                },
            )
        )
        await asyncio.sleep(0.02)

        models = await client.get("/v1/models")
        responsive_elapsed = time.perf_counter() - started_at
        completion = await generation

    assert models.status_code == 200
    assert responsive_elapsed < 0.2
    assert completion.status_code == 200
