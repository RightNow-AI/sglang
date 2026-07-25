"""One-command deterministic orchestration and output manifest generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .loaders import LoadedBundle, load_bundle, sha256_file
from .renderers import RENDERERS, RenderedFigure
from .theme import apply_publication_theme


def _file_record(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "format": path.suffix.lstrip("."),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dpi": 300,
    }


def _manifest(bundle: LoadedBundle, output_dir: Path, rendered: list[RenderedFigure]) -> dict[str, Any]:
    return {
        "schema_version": "autotree.figures.manifest.v1",
        "generator": "autotree-figures 0.1.0",
        "input": {"path": bundle.path.as_posix(), "sha256": bundle.sha256},
        "provenance": bundle.spec.provenance.model_dump(exclude_none=True),
        "sources": [
            {
                "path": run.reference.path,
                "sha256": run.sha256,
                "provenance": run.provenance,
                "panels": run.reference.panels,
            }
            for run in bundle.runs
        ],
        "figures": [
            {
                "id": item.figure_id,
                "title": item.title,
                "sources": list(item.sources),
                "provenance": item.provenance,
                "files": [_file_record(path, output_dir) for path in item.files],
            }
            for item in rendered
        ],
    }


def generate_all(results_path: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Generate all seven figures and a deterministic provenance manifest."""

    bundle = load_bundle(results_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_theme()
    rendered: list[RenderedFigure] = []
    for filename, _title, renderer in RENDERERS:
        rendered.append(renderer(bundle, output_dir / filename))
    manifest = _manifest(bundle, output_dir, rendered)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path, manifest
