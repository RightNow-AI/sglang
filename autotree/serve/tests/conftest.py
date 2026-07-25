from __future__ import annotations

import httpx
import pytest
from openai import AsyncOpenAI

from autotree_serve import create_app


MODEL_ID = "deterministic-demo"


@pytest.fixture
def app():
    return create_app(model_id=MODEL_ID)


@pytest.fixture
async def http_client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def openai_client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as raw_client:
        yield AsyncOpenAI(
            api_key="test-key",
            base_url="http://test/v1",
            http_client=raw_client,
        )
