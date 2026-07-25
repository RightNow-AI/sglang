"""Deterministic verifier evaluation for final tree branch selection."""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Mapping, Sequence
from urllib.request import HTTPRedirectHandler, Request, build_opener


CALLBACK_MAX_RESPONSE_BYTES = 64 * 1024
CALLBACK_MAX_INFLIGHT = 8

_CALLBACK_SLOTS = threading.BoundedSemaphore(CALLBACK_MAX_INFLIGHT)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _post_callback(
    url: str,
    candidates: Sequence[tuple[str, str]],
    timeout_s: float,
) -> list[str]:
    """Make one bounded callback request and return approved branch IDs."""
    payload = json.dumps(
        {
            "candidates": [
                {"branch_id": branch_id, "answer": answer}
                for branch_id, answer in candidates
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout_s) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise ValueError(
                    "callback verifier returned an invalid Content-Length"
                ) from error
            if declared_length > CALLBACK_MAX_RESPONSE_BYTES:
                raise ValueError("callback verifier response is too large")
        body = response.read(CALLBACK_MAX_RESPONSE_BYTES + 1)
    if len(body) > CALLBACK_MAX_RESPONSE_BYTES:
        raise ValueError("callback verifier response is too large")

    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("callback verifier response must be an object")
    approved = decoded.get("approved_branch_ids")
    if not isinstance(approved, list) or not all(
        isinstance(branch_id, str) for branch_id in approved
    ):
        raise ValueError(
            "callback verifier response must contain approved_branch_ids strings"
        )
    approved_lookup = set(approved)
    return [
        branch_id for branch_id, _answer in candidates if branch_id in approved_lookup
    ]


def _callback_approved(
    verifier: Mapping[str, Any], candidates: Sequence[tuple[str, str]]
) -> list[str]:
    """Apply a hard total wait and bound abandoned callback workers."""
    if not _CALLBACK_SLOTS.acquire(blocking=False):
        return []

    completed = threading.Event()
    result: dict[str, list[str]] = {"approved": []}

    def worker() -> None:
        try:
            result["approved"] = _post_callback(
                str(verifier["url"]),
                candidates,
                float(verifier["timeout_s"]),
            )
        except Exception:
            result["approved"] = []
        finally:
            _CALLBACK_SLOTS.release()
            completed.set()

    thread = threading.Thread(
        target=worker,
        name="autotree-verifier-callback",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        _CALLBACK_SLOTS.release()
        return []
    if not completed.wait(float(verifier["timeout_s"])):
        return []
    return result["approved"]


def evaluate_verifier(
    verifier: Mapping[str, Any],
    candidates: Sequence[tuple[str, str]],
) -> list[str]:
    """Return approved branch IDs in deterministic branch order."""
    ordered = sorted(candidates, key=lambda candidate: int(candidate[0]))
    verifier_type = verifier.get("type")
    if verifier_type == "regex":
        flags = re.IGNORECASE if verifier.get("flags", "") == "i" else 0
        pattern = re.compile(str(verifier["pattern"]), flags)
        return [
            branch_id
            for branch_id, answer in ordered
            if pattern.search(answer) is not None
        ]
    if verifier_type == "numeric":
        expected = float(verifier["equals"])
        tolerance = float(verifier["tolerance"])
        approved = []
        for branch_id, answer in ordered:
            try:
                value = float(answer)
            except (TypeError, ValueError):
                continue
            if abs(value - expected) <= tolerance:
                approved.append(branch_id)
        return approved
    if verifier_type == "callback":
        return _callback_approved(verifier, ordered)
    raise ValueError(f"unsupported verifier type: {verifier_type!r}")
