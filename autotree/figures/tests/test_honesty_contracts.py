from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from autotree_figures.contracts import ProvenanceError, require_provenance
from autotree_figures.theme import save_figure


def test_results_without_provenance_are_refused() -> None:
    with pytest.raises(ProvenanceError, match="provenance"):
        require_provenance({"schema_version": "autotree.figures.v1"})


def test_fixture_figure_contains_visible_watermark(tmp_path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    _, svg_path = save_figure(
        figure,
        tmp_path / "watermarked",
        provenance={"kind": "fixture", "source": "contract test"},
    )
    plt.close(figure)

    assert "FIXTURE DATA" in svg_path.read_text(encoding="utf-8")
