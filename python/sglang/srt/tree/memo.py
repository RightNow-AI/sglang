"""Exact-match storage for previously verified reasoning results.

The memo deliberately supports exact matches only. It normalizes a small set
of cosmetic whitespace differences, then hashes the complete prompt or message
payload together with the model, answer-shaping sampling parameters, and a
caller-controlled context version. Semantic, fuzzy, and embedding similarity
are intentionally excluded because they can reuse an answer for a meaningfully
different problem without an auditable correctness boundary.

``verify_always`` callbacks receive a copy of the stored entry and must return
``True`` only after a fresh generation agrees with it. The callback owns both
the fresh generation and the workload-specific agreement judgment.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any


TRUST = "trust"
VERIFY_CHEAP = "verify_cheap"
VERIFY_ALWAYS = "verify_always"

_VERIFICATION_POLICIES = {TRUST, VERIFY_CHEAP, VERIFY_ALWAYS}
_SAMPLING_KEYS = ("temperature", "top_p", "max_tokens")
_ENTRY_FIELDS = (
    "key",
    "answer_text",
    "extracted_answer",
    "model",
    "context_version",
    "created_at_index",
    "hits",
    "verified",
    "n_tokens_saved",
)


def _normalize_text(text: str) -> str:
    """Normalize cosmetic whitespace without changing leading indentation."""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized: list[str] = []
    previous_was_blank = False
    for line in lines:
        line = line.rstrip()
        is_blank = not line
        if is_blank and previous_was_blank:
            continue
        normalized.append(line)
        previous_was_blank = is_blank

    while normalized and not normalized[0]:
        normalized.pop(0)
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized)


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, Mapping):
        return {key: _normalize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError("messages_or_prompt must contain only JSON-compatible values")


def canonical_key(
    messages_or_prompt: Any,
    model: str,
    sampling_relevant_params: Mapping[str, Any] | None,
    context_version: str = "",
) -> str:
    """Return the deterministic SHA-256 key for one exact memo context.

    Only ``temperature``, ``top_p``, and ``max_tokens`` are copied from the
    sampling mapping. ``seed`` is copied only when either
    ``seeded_determinism_requested`` or ``seeded_determinism`` is true. For
    compatibility with the three-argument form, ``context_version`` may also
    be supplied inside the sampling mapping.
    """

    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if sampling_relevant_params is None:
        params: dict[str, Any] = {}
    elif isinstance(sampling_relevant_params, Mapping):
        params = dict(sampling_relevant_params)
    else:
        raise TypeError("sampling_relevant_params must be a mapping or None")

    params_context_version = params.pop("context_version", None)
    if params_context_version is not None:
        if context_version and params_context_version != context_version:
            raise ValueError("context_version was supplied with two values")
        context_version = params_context_version
    if not isinstance(context_version, str):
        raise TypeError("context_version must be a string")

    sampling = {key: params[key] for key in _SAMPLING_KEYS if key in params}
    seeded = bool(
        params.get("seeded_determinism_requested") or params.get("seeded_determinism")
    )
    if seeded:
        if "seed" not in params:
            raise ValueError("seeded determinism requires a seed")
        sampling["seed"] = params["seed"]

    payload = {
        "context_version": context_version,
        "messages_or_prompt": _normalize_payload(messages_or_prompt),
        "model": model,
        "sampling": sampling,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MemoStore:
    """Thread-safe JSONL memo store with per-namespace LRU eviction."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        max_entries: int | None = None,
        namespace: str | None = None,
        verification_policy: str = TRUST,
    ) -> None:
        if max_entries is not None and (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 0
        ):
            raise ValueError("max_entries must be a nonnegative integer or None")
        if namespace is not None and not isinstance(namespace, str):
            raise TypeError("namespace must be a string or None")
        self._validate_policy(verification_policy)

        self.path = os.fspath(path)
        self.max_entries = max_entries
        self.namespace = namespace or ""
        self.verification_policy = verification_policy
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._next_index = 0
        self._hits = 0
        self._misses = 0
        self._tokens_saved_estimate = 0
        self._agreement_checks = 0
        self._agreements = 0
        self._verification_needed = 0

        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._load()
        with self._lock:
            self._enforce_max_entries_locked()

    @staticmethod
    def _validate_policy(policy: str) -> None:
        if policy not in _VERIFICATION_POLICIES:
            choices = ", ".join(sorted(_VERIFICATION_POLICIES))
            raise ValueError(f"verification policy must be one of: {choices}")

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return

        saw_stats = False
        with open(self.path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid memo JSONL at line {line_number}"
                    ) from exc
                if record.get("namespace", "") != self.namespace:
                    continue

                operation = record.get("op", "upsert")
                if operation == "upsert":
                    entry = {field: record[field] for field in _ENTRY_FIELDS}
                    key = entry["key"]
                    self._entries.pop(key, None)
                    self._entries[key] = entry
                    self._next_index = max(
                        self._next_index, int(entry["created_at_index"]) + 1
                    )
                elif operation == "delete":
                    self._entries.pop(record["key"], None)
                elif operation == "clear":
                    self._entries.clear()
                elif operation == "stats":
                    saw_stats = True
                    stats = record["stats"]
                    self._hits = int(stats.get("hits", 0))
                    self._misses = int(stats.get("misses", 0))
                    self._tokens_saved_estimate = int(
                        stats.get("tokens_saved_estimate", 0)
                    )
                    self._agreement_checks = int(stats.get("agreement_checks", 0))
                    self._agreements = int(stats.get("agreements", 0))
                    self._verification_needed = int(stats.get("verification_needed", 0))
                else:
                    raise ValueError(
                        f"unknown memo operation at line {line_number}: {operation}"
                    )

        if not saw_stats:
            self._hits = sum(int(entry["hits"]) for entry in self._entries.values())

    def _append_locked(self, record: Mapping[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            json.dump(
                record,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()

    def _append_entry_locked(self, entry: Mapping[str, Any]) -> None:
        self._append_locked({"op": "upsert", "namespace": self.namespace, **entry})

    def _append_stats_locked(self) -> None:
        self._append_locked(
            {
                "op": "stats",
                "namespace": self.namespace,
                "stats": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "tokens_saved_estimate": self._tokens_saved_estimate,
                    "agreement_checks": self._agreement_checks,
                    "agreements": self._agreements,
                    "verification_needed": self._verification_needed,
                },
            }
        )

    def _enforce_max_entries_locked(self) -> None:
        if self.max_entries is None:
            return
        while len(self._entries) > self.max_entries:
            oldest_key = next(iter(self._entries))
            del self._entries[oldest_key]
            self._append_locked(
                {
                    "op": "delete",
                    "namespace": self.namespace,
                    "key": oldest_key,
                }
            )

    def put(
        self,
        key: str,
        answer_text: str,
        extracted_answer: str | None = None,
        model: str | None = None,
        context_version: str = "",
        verified: bool = True,
        n_tokens_saved: int = 0,
    ) -> dict[str, Any]:
        """Insert or replace an entry and return a copy of the stored value."""

        if not isinstance(key, str):
            raise TypeError("key must be a string")
        if not isinstance(answer_text, str):
            raise TypeError("answer_text must be a string")
        if extracted_answer is not None and not isinstance(extracted_answer, str):
            raise TypeError("extracted_answer must be a string or None")
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        if not isinstance(context_version, str):
            raise TypeError("context_version must be a string")
        if not isinstance(verified, bool):
            raise TypeError("verified must be a boolean")
        if (
            isinstance(n_tokens_saved, bool)
            or not isinstance(n_tokens_saved, int)
            or n_tokens_saved < 0
        ):
            raise ValueError("n_tokens_saved must be a nonnegative integer")

        with self._lock:
            entry = {
                "key": key,
                "answer_text": answer_text,
                "extracted_answer": extracted_answer,
                "model": model,
                "context_version": context_version,
                "created_at_index": self._next_index,
                "hits": 0,
                "verified": verified,
                "n_tokens_saved": n_tokens_saved,
            }
            self._next_index += 1
            self._entries.pop(key, None)
            self._entries[key] = entry
            self._append_entry_locked(entry)
            self._enforce_max_entries_locked()
            return dict(entry)

    def get(
        self,
        key: str,
        model: str,
        context_version: str = "",
        verification_policy: str | None = None,
        verify_callback: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any] | None:
        """Look up an entry under one of the three verification policies.

        ``trust`` returns the memo directly. ``verify_cheap`` also returns it,
        while incrementing ``verification_needed``. ``verify_always`` requires
        a callback. Agreement returns the memo and disagreement returns ``None``.
        """

        policy = verification_policy or self.verification_policy
        self._validate_policy(policy)
        if policy == VERIFY_ALWAYS and verify_callback is None:
            raise ValueError("verify_always requires verify_callback")

        with self._lock:
            entry = self._entries.get(key)
            if (
                entry is None
                or entry["model"] != model
                or entry["context_version"] != context_version
            ):
                self._misses += 1
                self._append_stats_locked()
                return None

            self._entries.pop(key)
            self._entries[key] = entry
            entry["hits"] += 1
            self._hits += 1
            if policy == VERIFY_CHEAP:
                self._verification_needed += 1
            if policy != VERIFY_ALWAYS:
                self._tokens_saved_estimate += entry["n_tokens_saved"]
            self._append_entry_locked(entry)
            self._append_stats_locked()
            result = dict(entry)

        if policy != VERIFY_ALWAYS:
            return result

        agreed = verify_callback(dict(result))
        if type(agreed) is not bool:
            raise TypeError("verify_callback must return a boolean")

        with self._lock:
            self._agreement_checks += 1
            if agreed:
                self._agreements += 1
            current = self._entries.get(key)
            if (
                current is not None
                and current["created_at_index"] == result["created_at_index"]
                and current["model"] == model
                and current["context_version"] == context_version
            ):
                current["verified"] = agreed
                self._append_entry_locked(current)
                result = dict(current)
            else:
                result["verified"] = agreed
            self._append_stats_locked()
        return result if agreed else None

    def invalidate(self, prefix: str) -> int:
        """Invalidate every entry whose SHA key starts with ``prefix``."""

        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        with self._lock:
            keys = [key for key in self._entries if key.startswith(prefix)]
            for key in keys:
                del self._entries[key]
                self._append_locked(
                    {
                        "op": "delete",
                        "namespace": self.namespace,
                        "key": key,
                    }
                )
            return len(keys)

    def clear(self) -> None:
        """Clear this namespace and reset its cumulative metrics."""

        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._tokens_saved_estimate = 0
            self._agreement_checks = 0
            self._agreements = 0
            self._verification_needed = 0
            self._append_locked({"op": "clear", "namespace": self.namespace})
            self._append_stats_locked()

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def hit_rate(self) -> float:
        with self._lock:
            lookups = self._hits + self._misses
            return self._hits / lookups if lookups else 0.0

    @property
    def tokens_saved_estimate(self) -> int:
        with self._lock:
            return self._tokens_saved_estimate

    @property
    def agreement_rate(self) -> float | None:
        with self._lock:
            if not self._agreement_checks:
                return None
            return self._agreements / self._agreement_checks

    def stats(self) -> dict[str, Any]:
        """Return cumulative lookup, savings, and verification statistics."""

        with self._lock:
            lookups = self._hits + self._misses
            agreement_rate = (
                self._agreements / self._agreement_checks
                if self._agreement_checks
                else None
            )
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / lookups if lookups else 0.0,
                "tokens_saved_estimate": self._tokens_saved_estimate,
                "agreement_rate": agreement_rate,
                "agreement_checks": self._agreement_checks,
                "agreements": self._agreements,
                "disagreements": self._agreement_checks - self._agreements,
                "verification_needed": self._verification_needed,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
