"""Low-overhead, process-local profiling for the AutoTree request path."""

from __future__ import annotations

import json
import logging
import os
import threading
from time import perf_counter as _perf_counter
from typing import Any, Dict


ENABLED = os.environ.get("AUTOTREE_PROFILE") == "1"

try:
    _PROFILE_EVERY = max(1, int(os.environ.get("AUTOTREE_PROFILE_EVERY", "200")))
except ValueError:
    _PROFILE_EVERY = 200

_LOCK = threading.Lock()
_COUNTERS: Dict[str, int] = {}
_SPANS: Dict[str, list] = {}
_LOGGER = logging.getLogger(__name__)


class _NoOpSpan:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


_NO_OP_SPAN = _NoOpSpan()


if ENABLED:

    class _Span:
        __slots__ = ("name", "started_at")

        def __init__(self, name: str) -> None:
            self.name = name
            self.started_at = 0.0

        def __enter__(self):
            self.started_at = _perf_counter()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            elapsed = _perf_counter() - self.started_at
            with _LOCK:
                entry = _SPANS.get(self.name)
                if entry is None:
                    _SPANS[self.name] = [1, elapsed]
                else:
                    entry[0] += 1
                    entry[1] += elapsed
            return False

    def span(name: str):
        return _Span(name)

    def incr(name: str, n: int = 1) -> None:
        with _LOCK:
            _COUNTERS[name] = _COUNTERS.get(name, 0) + n

else:

    def span(name: str):
        return _NO_OP_SPAN

    def incr(name: str, n: int = 1) -> None:
        return None


def _dump_locked() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if _COUNTERS:
        result["counters"] = dict(_COUNTERS)
    if _SPANS:
        result["spans"] = {
            name: {"count": values[0], "total_seconds": values[1]}
            for name, values in _SPANS.items()
        }
    return result


def dump() -> Dict[str, Any]:
    if not ENABLED:
        return {}
    with _LOCK:
        return _dump_locked()


def note_decode_step() -> None:
    """Count one decode batch and periodically log the process-local dump."""
    if not ENABLED:
        return
    with _LOCK:
        step = _COUNTERS.get("forward_batch.decode_steps", 0) + 1
        _COUNTERS["forward_batch.decode_steps"] = step
        if step % _PROFILE_EVERY:
            return
        snapshot = _dump_locked()
    _LOGGER.info(
        "[autotree-profile] %s",
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
    )
