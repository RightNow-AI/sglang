"""Versioned JSON Schema validation for results artifacts."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from .models import RESULTS_SCHEMA_VERSION, ResultsDocument


def results_json_schema() -> dict[str, Any]:
    """Return the v1 downloadable-results schema."""

    schema = ResultsDocument.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"urn:autotree:{RESULTS_SCHEMA_VERSION}"
    return schema


def validate_results_payload(payload: dict[str, Any]) -> None:
    """Validate a serialized payload against the current public schema."""

    schema = results_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
