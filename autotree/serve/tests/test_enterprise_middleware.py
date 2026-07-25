from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import jwt

from autotree_serve import create_app
from autotree_serve.enterprise import EnterpriseConfig, OIDCConfig, TokenQuota
from conftest import MODEL_ID


def _completion_payload(*, budget_tokens: int = 4) -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "enterprise request"}],
        "max_tokens": 4,
        "tree": {
            "policy": "beam",
            "branches": 2,
            "budget_tokens": budget_tokens,
        },
    }


async def _client(config: EnterpriseConfig):
    app = create_app(model_id=MODEL_ID, enterprise_config=config)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


async def test_auth_is_off_by_default(http_client):
    response = await http_client.get("/v1/models")
    assert response.status_code == 200


async def test_api_key_auth_rejects_missing_and_accepts_configured_key():
    config = EnterpriseConfig(api_keys={"secret-a": "tenant-a"})
    async with await _client(config) as client:
        missing = await client.get("/v1/models")
        invalid = await client.get("/v1/models", headers={"X-API-Key": "wrong"})
        accepted = await client.get(
            "/v1/models",
            headers={"X-API-Key": "secret-a"},
        )

    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "invalid_authentication"
    assert invalid.status_code == 401
    assert accepted.status_code == 200


async def test_api_keys_load_from_file_and_environment_overrides(tmp_path):
    key_file = tmp_path / "keys.json"
    key_file.write_text(json.dumps({"file-key": "tenant-file"}), encoding="utf-8")
    config = EnterpriseConfig.from_env(
        {
            "AUTOTREE_API_KEYS_FILE": str(key_file),
            "AUTOTREE_API_KEYS": json.dumps({"env-key": "tenant-env"}),
        }
    )

    assert config.api_keys == {
        "file-key": "tenant-file",
        "env-key": "tenant-env",
    }


async def test_per_tenant_generated_token_quota_returns_typed_429():
    config = EnterpriseConfig(
        api_keys={"secret-a": "tenant-a"},
        quotas={"tenant-a": 4},
        quota_window_seconds=60,
    )
    headers = {"X-API-Key": "secret-a"}
    async with await _client(config) as client:
        accepted = await client.post(
            "/v1/tree/completions",
            headers=headers,
            json=_completion_payload(budget_tokens=4),
        )
        rejected = await client.post(
            "/v1/tree/completions",
            headers=headers,
            json=_completion_payload(budget_tokens=1),
        )

    assert accepted.status_code == 200
    assert accepted.json()["usage"]["completion_tokens"] == 4
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"]
    error = rejected.json()["error"]
    assert error == {
        "message": (
            "Tenant generated-token quota exceeded; 0 of 4 tokens remain in the "
            "current window."
        ),
        "type": "rate_limit_error",
        "param": "max_tokens",
        "code": "tenant_token_quota_exceeded",
    }


def test_quota_settlement_cannot_refund_a_new_window(monkeypatch):
    clock = iter((0.0, 61.0))
    monkeypatch.setattr("autotree_serve.enterprise.time.monotonic", lambda: next(clock))
    quota = TokenQuota({"tenant-a": 10}, window_seconds=60)
    old_reservation = quota.reserve("tenant-a", 10)
    new_reservation = quota.reserve("tenant-a", 7)

    assert old_reservation is not None
    assert new_reservation is not None
    old_reservation.settle(1)
    assert quota.states["tenant-a"].used == 7


async def test_audit_log_records_actual_generated_tokens(tmp_path):
    audit_log = tmp_path / "audit" / "events.jsonl"
    config = EnterpriseConfig(
        api_keys={"secret-a": "tenant-a"},
        audit_log=audit_log,
    )
    async with await _client(config) as client:
        response = await client.post(
            "/v1/tree/completions",
            headers={"X-API-Key": "secret-a"},
            json=_completion_payload(budget_tokens=3),
        )

    assert response.status_code == 200
    records = [json.loads(line) for line in audit_log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["tenant"] == "tenant-a"
    assert records[0]["route"] == "/v1/tree/completions"
    assert records[0]["model"] == MODEL_ID
    assert records[0]["tokens"] == response.json()["usage"]["completion_tokens"]
    assert records[0]["status"] == 200
    assert datetime.fromisoformat(records[0]["timestamp"].replace("Z", "+00:00"))


async def test_streaming_audit_log_records_generated_tokens(tmp_path):
    audit_log = tmp_path / "stream-audit.jsonl"
    config = EnterpriseConfig(audit_log=audit_log)
    payload = _completion_payload(budget_tokens=5)
    payload["stream"] = True
    async with await _client(config) as client:
        response = await client.post("/v1/tree/completions", json=payload)

    record = json.loads(audit_log.read_text())
    token_events = response.text.count("event: token\n")
    assert response.status_code == 200
    assert record["tokens"] == token_events == 5


async def test_oidc_verifies_signature_issuer_and_audience(monkeypatch):
    now = datetime.now(UTC)
    claims = {
        "sub": "tenant-oidc",
        "iss": "https://issuer.example",
        "aud": "autotree",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    signing_secret = "oidc-test-secret-with-at-least-32-bytes"
    token = jwt.encode(claims, signing_secret, algorithm="HS256")
    config = EnterpriseConfig(
        oidc=OIDCConfig(
            jwks_url="https://issuer.example/.well-known/jwks.json",
            issuer="https://issuer.example",
            audience="autotree",
            algorithms=("HS256",),
        )
    )
    monkeypatch.setattr(
        "jwt.PyJWKClient.get_signing_key_from_jwt",
        lambda _self, _token: SimpleNamespace(key=signing_secret),
    )
    async with await _client(config) as client:
        accepted = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {token}"},
        )

        wrong_audience = jwt.encode(
            {**claims, "aud": "other-service"},
            signing_secret,
            algorithm="HS256",
        )
        rejected = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {wrong_audience}"},
        )

    assert accepted.status_code == 200
    assert rejected.status_code == 401
    assert rejected.json()["error"]["code"] == "invalid_authentication"
