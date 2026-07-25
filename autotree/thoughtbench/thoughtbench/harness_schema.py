"""JSON Schema helpers for landing-page result artifacts."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from .bench_models import BenchmarkResults


def benchmark_results_schema() -> dict[str, Any]:
    schema = BenchmarkResults.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:autotree:thoughtbench.results.v1"
    return schema


def validate_benchmark_results(payload: dict[str, Any]) -> None:
    schema = benchmark_results_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
