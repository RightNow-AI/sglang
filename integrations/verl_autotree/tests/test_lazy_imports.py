from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_package_imports_without_verl_or_sglang():
    package_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_root)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; import verl_autotree, verl_autotree.adapters; "
            "assert 'verl' not in sys.modules; assert 'sglang' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
