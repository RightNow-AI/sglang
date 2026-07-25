"""Load and validate figure bundles without depending on ThoughtBench internals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import ProvenanceError, require_provenance
from .models import FigureBundle, ResultReference


class FigureInputError(ValueError):
    """Raised when a figure bundle or referenced result is invalid."""


@dataclass(frozen=True)
class LoadedRun:
    reference: ResultReference
    path: Path
    sha256: str
    provenance: dict[str, Any]
    payload: dict[str, Any]


@dataclass(frozen=True)
class LoadedBundle:
    path: Path
    sha256: str
    spec: FigureBundle
    runs: tuple[LoadedRun, ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FigureInputError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FigureInputError(f"{path} must contain a JSON object")
    return payload


def _validate_thoughtbench_result(payload: dict[str, Any], path: Path) -> None:
    required = {
        "schema_version",
        "engine_config",
        "task_set",
        "per_seed_metrics",
        "aggregate_metrics",
        "samples",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise FigureInputError(f"{path} is missing ThoughtBench fields: {', '.join(missing)}")
    if not str(payload["schema_version"]).startswith("thoughtbench.results."):
        raise FigureInputError(f"{path} is not a ThoughtBench results artifact")
    if not isinstance(payload["per_seed_metrics"], list) or not payload["per_seed_metrics"]:
        raise FigureInputError(f"{path} has no per-seed metrics")
    if not isinstance(payload["aggregate_metrics"], list) or not payload["aggregate_metrics"]:
        raise FigureInputError(f"{path} has no aggregate metrics")


def load_bundle(path: Path) -> LoadedBundle:
    """Load a bundle plus every referenced ThoughtBench results file."""

    resolved = path.resolve()
    payload = _read_object(resolved)
    try:
        top_provenance = require_provenance(payload, location=str(resolved))
        spec = FigureBundle.model_validate(payload)
    except (ProvenanceError, ValidationError) as exc:
        raise FigureInputError(str(exc)) from exc

    runs: list[LoadedRun] = []
    for reference in spec.thoughtbench_results:
        result_path = (resolved.parent / reference.path).resolve()
        result_payload = _read_object(result_path)
        try:
            provenance = require_provenance(result_payload, location=str(result_path))
        except ProvenanceError as exc:
            raise FigureInputError(str(exc)) from exc
        if provenance["kind"].strip().lower() != top_provenance["kind"].strip().lower():
            raise FigureInputError(
                f"{result_path} provenance kind does not match the figure bundle"
            )
        _validate_thoughtbench_result(result_payload, result_path)
        runs.append(
            LoadedRun(
                reference=reference,
                path=result_path,
                sha256=sha256_file(result_path),
                provenance=provenance,
                payload=result_payload,
            )
        )
    return LoadedBundle(
        path=resolved,
        sha256=sha256_file(resolved),
        spec=spec,
        runs=tuple(runs),
    )
