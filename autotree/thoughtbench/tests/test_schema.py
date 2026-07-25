import json

import pytest
from jsonschema import ValidationError

from thoughtbench.models import RESULTS_SCHEMA_VERSION
from thoughtbench.schema import results_json_schema, validate_results_payload


def test_results_schema_is_versioned_and_forbids_unknown_top_level_fields() -> None:
    schema = results_json_schema()

    assert schema["$id"].endswith(RESULTS_SCHEMA_VERSION)
    assert schema["additionalProperties"] is False


def test_schema_rejects_a_payload_without_fixture_honesty_stamp() -> None:
    with pytest.raises(ValidationError):
        validate_results_payload(
            {
                "schema_version": RESULTS_SCHEMA_VERSION,
                "benchmark_claims_allowed": False,
            }
        )


def test_schema_can_be_serialized_for_downstream_consumers() -> None:
    rendered = json.dumps(results_json_schema())

    assert "thoughtbench.results.v2" in rendered
    assert "artifact_notice" in rendered


def test_real_provenance_unlocks_claims_and_notice_must_match() -> None:
    from pydantic import ValidationError as PydanticValidationError

    from thoughtbench.models import (
        FIXTURE_NOTICE,
        REAL_NOTICE,
        FixtureProvenance,
        RealProvenance,
        TaskSetStamp,
    )

    real_stamp = TaskSetStamp(
        name="aime-real",
        sha256="0" * 64,
        task_count=60,
        provenance=RealProvenance(
            source="MAA AIME 2024+2025 via public HF datasets",
            license="MAA competition problems, publicly distributed",
        ),
    )
    assert real_stamp.provenance.kind == "real"
    assert real_stamp.provenance.notice == REAL_NOTICE

    fixture_stamp = TaskSetStamp(
        name="fixtures",
        sha256="0" * 64,
        task_count=1,
        provenance=FixtureProvenance(
            source="bundled synthetic fixtures", license="repository"
        ),
    )
    assert fixture_stamp.provenance.notice == FIXTURE_NOTICE

    with pytest.raises(PydanticValidationError):
        RealProvenance(
            source="x", license="y", notice=FIXTURE_NOTICE  # wrong notice for kind
        )
