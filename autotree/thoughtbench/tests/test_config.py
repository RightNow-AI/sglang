from pathlib import Path

import pytest
from pydantic import ValidationError

from thoughtbench.models import FIXTURE_NOTICE, RunConfig


def _payload(tmp_path: Path) -> dict:
    return {
        "model": "model",
        "base_url": "http://127.0.0.1:8000",
        "mode": "sequential",
        "task_set": {
            "name": "fixtures",
            "path": str(tmp_path / "tasks.jsonl"),
            "provenance": {
                "kind": "fixture",
                "source": "synthetic",
                "license": "repository",
                "notice": FIXTURE_NOTICE,
            },
        },
        "output_path": str(tmp_path / "results.json"),
        "budgets": [{"name": "tiny", "max_tokens": 4}],
        "k_samples": 4,
        "seeds": [1, 2, 3],
    }


def test_config_enforces_three_unique_protocol_seeds(tmp_path) -> None:
    payload = _payload(tmp_path)
    payload["seeds"] = [1, 1, 2]

    with pytest.raises(ValidationError, match="exactly three unique seeds"):
        RunConfig.model_validate(payload)


def test_tree_mode_requires_explicit_tree_budget(tmp_path) -> None:
    payload = _payload(tmp_path)
    payload["mode"] = "tree"
    payload["tree"] = {"policy": "beam", "branches": 2}

    with pytest.raises(ValidationError, match="tree_budget_tokens"):
        RunConfig.model_validate(payload)


def test_metric_ks_only_include_fully_sampled_protocol_points(tmp_path) -> None:
    config = RunConfig.model_validate(_payload(tmp_path))

    assert config.metric_ks == (1, 4)
