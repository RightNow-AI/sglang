"""Shared publication styling and deterministic figure writers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.figure import Figure

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


def apply_publication_theme() -> None:
    """Install the shared colorblind-safe publication style."""

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "autotree-figures-v1",
        }
    )


def save_figure(
    figure: Figure,
    output_stem: Path,
    *,
    provenance: dict[str, Any],
) -> tuple[Path, Path]:
    """Write a figure as 300 dpi PDF and SVG."""

    from .contracts import is_fixture

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    if is_fixture(provenance):
        figure.text(
            0.5,
            0.5,
            "FIXTURE DATA",
            ha="center",
            va="center",
            rotation=32,
            fontsize=42,
            weight="bold",
            color="#B2182B",
            alpha=0.17,
            zorder=1000,
        )
    fixed_time = datetime(1970, 1, 1, tzinfo=UTC)
    common = {"dpi": 300, "bbox_inches": "tight"}
    figure.savefig(
        pdf_path,
        metadata={
            "Creator": "AutoTree figure pipeline",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
        **common,
    )
    figure.savefig(
        svg_path,
        metadata={"Creator": "AutoTree figure pipeline", "Date": "1970-01-01"},
        **common,
    )
    return pdf_path, svg_path
