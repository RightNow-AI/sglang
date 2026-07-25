from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autotree_figures.pipeline import generate_all


FIGURES_ROOT = Path(__file__).parents[1]
BUNDLE_PATH = FIGURES_ROOT / "fixtures" / "all-figures.fixture.json"
EXPECTED_IDS = {
    "accuracy_vs_tokens",
    "cost_accuracy_pareto",
    "kv_reuse_heatmap",
    "tree_topology",
    "branching_factor_ablation",
    "rollout_throughput",
    "thinking_effort_sweep",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    output = tmp_path_factory.mktemp("figures")
    _, manifest = generate_all(BUNDLE_PATH, output)
    return output, manifest


@pytest.mark.parametrize("figure_id", sorted(EXPECTED_IDS))
def test_each_renderer_writes_nontrivial_watermarked_pdf_and_svg(
    generated: tuple[Path, dict], figure_id: str
) -> None:
    output, manifest = generated
    record = next(item for item in manifest["figures"] if item["id"] == figure_id)
    files = {item["format"]: output / item["path"] for item in record["files"]}

    assert set(files) == {"pdf", "svg"}
    assert files["pdf"].stat().st_size > 4_000
    assert files["svg"].stat().st_size > 4_000
    assert "FIXTURE DATA" in files["svg"].read_text(encoding="utf-8")


def test_manifest_records_inputs_sources_and_output_hashes(
    generated: tuple[Path, dict]
) -> None:
    output, manifest = generated
    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert on_disk == manifest
    assert manifest["input"]["sha256"] == _sha256(BUNDLE_PATH)
    assert {item["id"] for item in manifest["figures"]} == EXPECTED_IDS
    assert len(manifest["sources"]) == 22
    for source in manifest["sources"]:
        assert source["sha256"] == _sha256(BUNDLE_PATH.parent / source["path"])
    for figure in manifest["figures"]:
        assert figure["provenance"]["kind"] == "fixture"
        for artifact in figure["files"]:
            path = output / artifact["path"]
            assert artifact["sha256"] == _sha256(path)
            assert artifact["bytes"] == path.stat().st_size
            assert artifact["dpi"] == 300


def test_generation_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _, first_manifest = generate_all(BUNDLE_PATH, first)
    _, second_manifest = generate_all(BUNDLE_PATH, second)

    first_hashes = {
        item["path"]: item["sha256"]
        for figure in first_manifest["figures"]
        for item in figure["files"]
    }
    second_hashes = {
        item["path"]: item["sha256"]
        for figure in second_manifest["figures"]
        for item in figure["files"]
    }
    assert first_hashes == second_hashes
