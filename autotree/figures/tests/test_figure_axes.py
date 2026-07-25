from __future__ import annotations

from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure
import pytest

from autotree_figures.loaders import LoadedBundle, load_bundle
from autotree_figures import renderers


FIGURES_ROOT = Path(__file__).parents[1]
BUNDLE_PATH = FIGURES_ROOT / "fixtures" / "all-figures.fixture.json"


@pytest.fixture(scope="module")
def bundle() -> LoadedBundle:
    return load_bundle(BUNDLE_PATH)


def _capture(
    monkeypatch: pytest.MonkeyPatch,
    bundle: LoadedBundle,
    renderer: Callable,
    tmp_path: Path,
) -> Figure:
    captured: list[Figure] = []

    def fake_save(figure: Figure, stem: Path, *, provenance: dict) -> tuple[Path, Path]:
        assert provenance["kind"] == "fixture"
        captured.append(figure)
        return stem.with_suffix(".pdf"), stem.with_suffix(".svg")

    monkeypatch.setattr(renderers, "save_figure", fake_save)
    renderer(bundle, tmp_path / renderer.__name__)
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize(
    ("renderer", "title", "xlabel", "ylabel", "axis_count"),
    [
        (renderers.render_scaling, "Accuracy vs. token budget", "Mean tokens across tasks", "Accuracy@k", 4),
        (renderers.render_pareto, "Cost-accuracy Pareto frontier", "Cost per correct answer (USD)", "Accuracy@k", 4),
        (renderers.render_branching, "Branching-factor ablation", "Branching factor k", "Accuracy@k", 1),
        (renderers.render_throughput, "Rollout throughput", "Execution mode", "Rollouts per hour per GPU", 1),
        (renderers.render_thinking_effort, "Thinking-effort sweep", "thinking_effort", "Accuracy@k", 1),
    ],
)
def test_line_and_bar_figures_have_publication_axes(
    monkeypatch: pytest.MonkeyPatch,
    bundle: LoadedBundle,
    tmp_path: Path,
    renderer: Callable,
    title: str,
    xlabel: str,
    ylabel: str,
    axis_count: int,
) -> None:
    figure = _capture(monkeypatch, bundle, renderer, tmp_path)
    data_axes = figure.axes[:axis_count]
    assert len(data_axes) == axis_count
    assert all(axis.get_xlabel() == xlabel for axis in data_axes)
    assert all(axis.get_ylabel() == ylabel for axis in data_axes)
    visible_title = figure._suptitle.get_text() if figure._suptitle else data_axes[0].get_title()
    assert title in visible_title


def test_kv_heatmap_uses_multiplier_definition(
    monkeypatch: pytest.MonkeyPatch, bundle: LoadedBundle, tmp_path: Path
) -> None:
    figure = _capture(monkeypatch, bundle, renderers.render_kv_reuse, tmp_path)
    axis, colorbar_axis = figure.axes
    assert axis.get_xlabel() == "Branching factor"
    assert axis.get_ylabel() == "Tree depth"
    assert axis.get_title() == "KV reuse by tree shape"
    assert colorbar_axis.get_ylabel() == "KV reuse ratio (logical / physical tokens)"
    assert all(float(text.get_text().removesuffix("×")) >= 1 for text in axis.texts)


def test_tree_topology_encodes_value_and_pruning(
    monkeypatch: pytest.MonkeyPatch, bundle: LoadedBundle, tmp_path: Path
) -> None:
    figure = _capture(monkeypatch, bundle, renderers.render_topology, tmp_path)
    axis, colorbar_axis = figure.axes
    assert "renderer contract fixture" in axis.get_title()
    assert colorbar_axis.get_ylabel() == "Value estimate"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["Pruned"]


def test_blueprint_series_and_categories_are_present(
    monkeypatch: pytest.MonkeyPatch, bundle: LoadedBundle, tmp_path: Path
) -> None:
    scaling = _capture(monkeypatch, bundle, renderers.render_scaling, tmp_path)
    assert {text.get_text() for text in scaling.legends[0].get_texts()} == {
        "Sequential",
        "AutoTree",
    }

    pareto = _capture(monkeypatch, bundle, renderers.render_pareto, tmp_path)
    assert {text.get_text() for text in pareto.legends[0].get_texts()} == {
        "vLLM",
        "SGLang",
        "AutoTree",
    }

    throughput = _capture(monkeypatch, bundle, renderers.render_throughput, tmp_path)
    assert [tick.get_text() for tick in throughput.axes[0].get_xticklabels()] == [
        "Sequential",
        "vLLM",
        "SGLang",
        "AutoTree",
    ]
