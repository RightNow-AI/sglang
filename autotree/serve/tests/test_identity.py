from __future__ import annotations

import httpx

from autotree_serve import create_app


async def test_deterministic_engine_uses_demo_identity_even_when_given_real_model_name():
    app = create_app(model_id="gpt2")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        models = await client.get("/v1/models")
        completion = await client.post(
            "/v1/chat/completions",
            json={
                "model": "deterministic-demo",
                "messages": [{"role": "user", "content": "identify yourself"}],
                "max_tokens": 1,
            },
        )

    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "deterministic-demo"
    assert completion.status_code == 200
    assert completion.json()["model"] == "deterministic-demo"
    assert app.state.engine.model_metadata.real_model_weights is False
