"""AutoTree rollout adapters for verl.

The adapter classes are resolved lazily so importing this package does not
require verl, Ray, SGLang, PyTorch, or CUDA.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["AutoTreeServerAdapter", "AutoTreeHttpServer", "AutoTreeReplica"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module("verl_autotree.adapters"), name)
