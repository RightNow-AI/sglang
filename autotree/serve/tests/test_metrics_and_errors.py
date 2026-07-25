from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

from conftest import MODEL_ID


async def test_metrics_scrape_parses_and_moves_after_generation(http_client):
    before = await http_client.get("/metrics")
    assert before.status_code == 200
    list(text_string_to_metric_families(before.text))

    completion = await http_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "measure this tree"}],
            "max_tokens": 4,
            "tree": {
                "policy": "beam",
                "branches": 3,
                "budget_tokens": 9,
                "scorer": None,
            },
        },
    )
    assert completion.status_code == 200

    after = await http_client.get("/metrics")
    families = {family.name: family for family in text_string_to_metric_families(after.text)}
    assert families["kv_reuse_ratio"].samples[0].value > 1
    assert 0 < families["useful_token_ratio"].samples[0].value <= 1
    assert families["active_branches"].samples[0].value == 0
    assert families["tokens_per_second"].samples[0].value > 0
    assert families["ttft_seconds"].samples
    request_samples = families["requests"].samples
    assert any(
        sample.name == "requests_total"
        and sample.labels == {"endpoint": "/v1/chat/completions", "status": "200"}
        and sample.value >= 1
        for sample in request_samples
    )


async def test_unmatched_paths_share_one_metrics_label(http_client):
    first = await http_client.get("/random-probe-one")
    second = await http_client.get("/random-probe-two")

    assert first.status_code == 404
    assert second.status_code == 404

    scrape = await http_client.get("/metrics")
    families = {family.name: family for family in text_string_to_metric_families(scrape.text)}
    unmatched_labels = {
        sample.labels["endpoint"]
        for sample in families["requests"].samples
        if sample.name == "requests_total" and sample.labels["status"] == "404"
    }
    assert unmatched_labels == {"unmatched"}


async def test_bad_tree_parameters_return_openai_error_body(http_client):
    response = await http_client.post(
        "/v1/tree/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "bad policy"}],
            "tree": {
                "policy": "random",
                "branches": 0,
                "budget_tokens": -1,
                "unexpected": True,
            },
        },
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "validation_error"
    assert error["param"].startswith("tree.")


async def test_malformed_body_returns_openai_error_body(http_client):
    response = await http_client.post(
        "/v1/chat/completions",
        content="{not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_unknown_model_fails_without_fallback(http_client):
    response = await http_client.post(
        "/v1/chat/completions",
        json={
            "model": "not-served",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_n_greater_than_one_fails_honestly(http_client):
    response = await http_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "do not ignore n"}],
            "n": 2,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_feature"
    assert error["param"] == "n"


async def test_unimplemented_semantic_field_names_feature_in_400(http_client):
    response = await http_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "call a tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_feature"
    assert error["param"] == "tools"
    assert "tools" in error["message"]
