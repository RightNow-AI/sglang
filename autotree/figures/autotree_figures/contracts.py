"""Input honesty contracts shared by the figure pipeline."""

from __future__ import annotations

from typing import Any


class ProvenanceError(ValueError):
    """Raised when a figure input lacks usable provenance."""


def require_provenance(
    payload: dict[str, Any], *, location: str = "results"
) -> dict[str, Any]:
    """Return validated top-level or ThoughtBench task-set provenance."""

    candidate = payload.get("provenance")
    if candidate is None:
        task_set = payload.get("task_set")
        if isinstance(task_set, dict):
            candidate = task_set.get("provenance")
    if not isinstance(candidate, dict):
        raise ProvenanceError(f"{location} is missing provenance")
    kind = candidate.get("kind")
    source = candidate.get("source")
    if not isinstance(kind, str) or not kind.strip():
        raise ProvenanceError(f"{location} provenance.kind must be non-empty")
    if not isinstance(source, str) or not source.strip():
        raise ProvenanceError(f"{location} provenance.source must be non-empty")
    return candidate


def is_fixture(provenance: dict[str, Any]) -> bool:
    """Return whether provenance requires fixture-only labeling."""

    return provenance["kind"].strip().lower() == "fixture"
