"""Command-line entry point for the Phase-4 figure pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _bootstrap_project() -> int | None:
    try:
        import jsonschema  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pydantic  # noqa: F401
    except ModuleNotFoundError:
        if os.environ.get("AUTOTREE_FIGURES_BOOTSTRAPPED") == "1":
            raise
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("matplotlib is unavailable and uv was not found")
        script = Path(__file__).resolve()
        env = os.environ.copy()
        env["AUTOTREE_FIGURES_BOOTSTRAPPED"] = "1"
        command = [uv, "run", "--project", str(script.parent), "python", str(script), *sys.argv[1:]]
        return subprocess.run(command, env=env, check=False).returncode
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate all AutoTree paper figures")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures" / "all-figures.fixture.json",
        help="figure bundle JSON (default: bundled publication fixture)",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    from autotree_figures.pipeline import generate_all

    args = _parser().parse_args(argv)
    manifest_path, manifest = generate_all(args.results, args.out)
    print(f"produced {len(manifest['figures'])} figures (PDF + SVG, 300 dpi):")
    for figure in manifest["figures"]:
        for artifact in figure["files"]:
            print(f"  {artifact['path']}")
    print(f"  {manifest_path.name}")
    return 0


if __name__ == "__main__":
    bootstrap_code = _bootstrap_project()
    raise SystemExit(bootstrap_code if bootstrap_code is not None else main())
