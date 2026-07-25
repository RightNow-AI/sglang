from __future__ import annotations

import asyncio
import time

import httpx

from autotree_serve import create_app

from test_responsiveness import SlowTreeKVEngine


async def test_health_is_cheap_and_reports_engine_identity():
    app = create_app(engine=SlowTreeKVEngine())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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

        started_at = time.perf_counter()
        health = await client.get("/health")
        responsive_elapsed = time.perf_counter() - started_at
        completion = await generation

    assert health.status_code == 200
    assert health.json()["engine_kind"] == "treekv"
    assert health.json()["model_id"] == "slow-treekv"
    assert health.json()["ready"] is True
    assert health.json()["uptime_seconds"] >= 0
    assert responsive_elapsed < 0.2
    assert completion.status_code == 200
