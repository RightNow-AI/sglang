"""Opt-in enterprise authentication, quota, and audit middleware."""

from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import jwt
from jwt import PyJWKClient


_PROTECTED_PREFIX = "/v1/"
_COMPLETION_ROUTES = frozenset({"/v1/chat/completions", "/v1/tree/completions"})


@dataclass(frozen=True)
class OIDCConfig:
    jwks_url: str
    issuer: str
    audience: str
    algorithms: tuple[str, ...] = ("RS256",)


@dataclass(frozen=True)
class EnterpriseConfig:
    api_keys: Mapping[str, str] = field(default_factory=dict)
    oidc: OIDCConfig | None = None
    quotas: Mapping[str, int] = field(default_factory=dict)
    quota_window_seconds: int = 60
    audit_log: Path | None = None

    def __post_init__(self) -> None:
        if self.quota_window_seconds <= 0:
            raise ValueError("quota_window_seconds must be positive")
        if any(not key or not tenant for key, tenant in self.api_keys.items()):
            raise ValueError("API keys and tenant names must be non-empty")
        if any(not tenant or limit <= 0 for tenant, limit in self.quotas.items()):
            raise ValueError(
                "quota tenants must be non-empty and limits must be positive"
            )
        if self.quotas and not self.auth_enabled:
            raise ValueError("per-tenant quotas require API-key or OIDC authentication")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys) or self.oidc is not None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> EnterpriseConfig:
        source = os.environ if env is None else env
        api_keys = _load_string_mapping(
            source.get("AUTOTREE_API_KEYS"),
            source.get("AUTOTREE_API_KEYS_FILE"),
            "API keys",
        )
        quotas = _load_int_mapping(
            source.get("AUTOTREE_TENANT_QUOTAS"),
            source.get("AUTOTREE_TENANT_QUOTAS_FILE"),
            "tenant quotas",
        )
        oidc_values = {
            "jwks_url": source.get("AUTOTREE_OIDC_JWKS_URL"),
            "issuer": source.get("AUTOTREE_OIDC_ISSUER"),
            "audience": source.get("AUTOTREE_OIDC_AUDIENCE"),
        }
        configured_oidc_values = [value for value in oidc_values.values() if value]
        if configured_oidc_values and len(configured_oidc_values) != len(oidc_values):
            raise ValueError(
                "OIDC requires AUTOTREE_OIDC_JWKS_URL, AUTOTREE_OIDC_ISSUER, "
                "and AUTOTREE_OIDC_AUDIENCE"
            )
        oidc = None
        if configured_oidc_values:
            algorithms = tuple(
                item.strip()
                for item in source.get("AUTOTREE_OIDC_ALGORITHMS", "RS256").split(",")
                if item.strip()
            )
            if not algorithms:
                raise ValueError("AUTOTREE_OIDC_ALGORITHMS must not be empty")
            oidc = OIDCConfig(algorithms=algorithms, **oidc_values)  # type: ignore[arg-type]

        audit_value = source.get("AUTOTREE_AUDIT_LOG")
        return cls(
            api_keys=api_keys,
            oidc=oidc,
            quotas=quotas,
            quota_window_seconds=int(source.get("AUTOTREE_QUOTA_WINDOW_SECONDS", "60")),
            audit_log=Path(audit_value) if audit_value else None,
        )


class AuthenticationError(Exception):
    pass


class OIDCVerifier:
    def __init__(self, config: OIDCConfig) -> None:
        self.config = config
        self.client = PyJWKClient(config.jwks_url)

    async def verify(self, token: str) -> str:
        claims = await asyncio.to_thread(self._decode_token, token)
        for claim_name in ("tenant", "tenant_id", "tid", "sub"):
            tenant = claims.get(claim_name)
            if isinstance(tenant, str) and tenant:
                return tenant
        raise AuthenticationError(
            "OIDC token does not contain a tenant or subject claim"
        )

    def _decode_token(self, token: str) -> dict[str, Any]:
        signing_key = self.client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=list(self.config.algorithms),
            audience=self.config.audience,
            issuer=self.config.issuer,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )


class Authenticator:
    def __init__(self, config: EnterpriseConfig) -> None:
        self.api_keys = tuple(config.api_keys.items())
        self.oidc = OIDCVerifier(config.oidc) if config.oidc is not None else None

    async def authenticate(self, headers: Mapping[str, str]) -> str:
        api_key = headers.get("x-api-key")
        if api_key:
            for expected, tenant in self.api_keys:
                if hmac.compare_digest(api_key, expected):
                    return tenant
            raise AuthenticationError("Invalid API key")

        authorization = headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token and self.oidc is not None:
            try:
                return await self.oidc.verify(token)
            except (jwt.PyJWTError, OSError, ValueError) as error:
                raise AuthenticationError("Invalid OIDC bearer token") from error

        raise AuthenticationError("Provide X-API-Key or a configured OIDC bearer token")


@dataclass
class _QuotaState:
    window_started: float
    used: int = 0


class QuotaReservation:
    def __init__(
        self,
        quota: TokenQuota,
        tenant: str,
        reserved: int,
        window_started: float,
    ) -> None:
        self.quota = quota
        self.tenant = tenant
        self.reserved = reserved
        self.window_started = window_started

    def settle(self, actual_tokens: int) -> None:
        self.quota.settle(
            self.tenant,
            self.reserved,
            actual_tokens,
            window_started=self.window_started,
        )


class QuotaExceeded(Exception):
    def __init__(self, *, limit: int, remaining: int, retry_after: int) -> None:
        self.limit = limit
        self.remaining = remaining
        self.retry_after = retry_after
        super().__init__("Tenant generated-token quota exceeded")


class TokenQuota:
    """In-memory fixed-window quotas for generated tokens per tenant."""

    def __init__(self, limits: Mapping[str, int], window_seconds: int) -> None:
        self.limits = dict(limits)
        self.window_seconds = window_seconds
        self.states: dict[str, _QuotaState] = {}
        self.lock = threading.Lock()

    def reserve(self, tenant: str, requested_tokens: int) -> QuotaReservation | None:
        limit = self.limits.get(tenant)
        if limit is None or requested_tokens <= 0:
            return None
        now = time.monotonic()
        with self.lock:
            state = self.states.get(tenant)
            if state is None or now - state.window_started >= self.window_seconds:
                state = _QuotaState(window_started=now)
                self.states[tenant] = state
            remaining = max(0, limit - state.used)
            if requested_tokens > remaining:
                retry_after = max(
                    1,
                    math.ceil(self.window_seconds - (now - state.window_started)),
                )
                raise QuotaExceeded(
                    limit=limit,
                    remaining=remaining,
                    retry_after=retry_after,
                )
            state.used += requested_tokens
        return QuotaReservation(self, tenant, requested_tokens, state.window_started)

    def settle(
        self,
        tenant: str,
        reserved: int,
        actual_tokens: int,
        *,
        window_started: float,
    ) -> None:
        limit = self.limits.get(tenant)
        if limit is None:
            return
        with self.lock:
            state = self.states.get(tenant)
            if state is not None and state.window_started == window_started:
                state.used = max(0, state.used - reserved + max(0, actual_tokens))


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(
        self,
        *,
        tenant: str,
        route: str,
        model: str | None,
        tokens: int,
        status: int,
    ) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "tenant": tenant,
            "route": route,
            "model": model,
            "tokens": max(0, tokens),
            "status": status,
        }
        payload = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        with self.lock:
            descriptor = os.open(self.path, flags, 0o600)
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)


class EnterpriseMiddleware:
    def __init__(self, app: Any, *, config: EnterpriseConfig, metrics: Any) -> None:
        self.app = app
        self.config = config
        self.metrics = metrics
        self.authenticator = Authenticator(config) if config.auth_enabled else None
        self.quota = TokenQuota(config.quotas, config.quota_window_seconds)
        self.audit = (
            AuditLogger(config.audit_log) if config.audit_log is not None else None
        )

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(
            _PROTECTED_PREFIX
        ):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        tenant = "anonymous"
        status = 500
        reservation: QuotaReservation | None = None
        body = b""
        payload: dict[str, Any] = {}
        model: str | None = None
        scope["autotree.audit_tokens"] = 0

        body_replayed = False

        async def replay_body() -> dict[str, Any]:
            nonlocal body_replayed
            if not body_replayed:
                body_replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        async def capture_status(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            if self.authenticator is not None:
                try:
                    tenant = await self.authenticator.authenticate(headers)
                except AuthenticationError as error:
                    status = 401
                    await _send_error(
                        send,
                        status=401,
                        message=str(error),
                        error_type="authentication_error",
                        code="invalid_authentication",
                        headers=((b"www-authenticate", b"Bearer"),),
                    )
                    return
            scope["autotree.tenant"] = tenant

            body = await _read_body(receive)
            payload = _json_object(body)
            model_value = payload.get("model")
            model = model_value if isinstance(model_value, str) else None

            requested_tokens = _requested_tokens(scope.get("path", ""), payload)
            try:
                reservation = self.quota.reserve(tenant, requested_tokens)
            except QuotaExceeded as error:
                status = 429
                self.metrics.quota_rejections_total.inc()
                await _send_error(
                    send,
                    status=429,
                    message=(
                        f"Tenant generated-token quota exceeded; {error.remaining} of "
                        f"{error.limit} tokens remain in the current window."
                    ),
                    error_type="rate_limit_error",
                    code="tenant_token_quota_exceeded",
                    param="max_tokens",
                    headers=((b"retry-after", str(error.retry_after).encode("ascii")),),
                )
                return

            await self.app(scope, replay_body, capture_status)
        finally:
            actual_tokens = int(scope.get("autotree.audit_tokens", 0))
            if reservation is not None:
                reservation.settle(actual_tokens)
            if self.audit is not None:
                await asyncio.to_thread(
                    self.audit.write,
                    tenant=tenant,
                    route=scope.get("path", ""),
                    model=model,
                    tokens=actual_tokens,
                    status=status,
                )


async def _read_body(receive: Any) -> bytes:
    chunks: list[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        chunks.append(message.get("body", b""))
        more_body = bool(message.get("more_body", False))
    return b"".join(chunks)


def _json_object(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _requested_tokens(route: str, payload: Mapping[str, Any]) -> int:
    if route not in _COMPLETION_ROUTES:
        return 0
    tree = payload.get("tree")
    if isinstance(tree, dict):
        budget = tree.get("budget_tokens")
        if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0:
            return budget
    for key in ("max_completion_tokens", "max_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 16


async def _send_error(
    send: Any,
    *,
    status: int,
    message: str,
    error_type: str,
    code: str,
    param: str | None = None,
    headers: tuple[tuple[bytes, bytes], ...] = (),
) -> None:
    body = json.dumps(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    response_headers = (
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *headers,
    )
    await send(
        {"type": "http.response.start", "status": status, "headers": response_headers}
    )
    await send({"type": "http.response.body", "body": body})


def _load_string_mapping(
    raw: str | None, filename: str | None, label: str
) -> dict[str, str]:
    value = _load_mapping(raw, filename, label)
    if any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError(f"{label} must be a JSON object of string keys and values")
    return dict(value)


def _load_int_mapping(
    raw: str | None, filename: str | None, label: str
) -> dict[str, int]:
    value = _load_mapping(raw, filename, label)
    if any(
        not isinstance(key, str) or not isinstance(item, int) or isinstance(item, bool)
        for key, item in value.items()
    ):
        raise ValueError(
            f"{label} must be a JSON object of string keys and integer values"
        )
    return dict(value)


def _load_mapping(raw: str | None, filename: str | None, label: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if filename:
        file_value = json.loads(Path(filename).read_text(encoding="utf-8"))
        if not isinstance(file_value, dict):
            raise ValueError(f"{label} file must contain a JSON object")
        merged.update(file_value)
    if raw:
        env_value = json.loads(raw)
        if not isinstance(env_value, dict):
            raise ValueError(f"{label} environment value must be a JSON object")
        merged.update(env_value)
    return merged
