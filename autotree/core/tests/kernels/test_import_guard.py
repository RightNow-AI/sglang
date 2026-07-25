"""Optional-Triton import behavior."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from inspect import signature

import pytest
import torch


def test_triton_module_imports_cleanly() -> None:
    module = import_module("autotree_core.kernels.triton_kernel")

    assert module.TRITON_AVAILABLE is (find_spec("triton") is not None)


@pytest.mark.skipif(find_spec("triton") is not None, reason="Triton is installed")
def test_unavailable_triton_kernel_raises_a_clear_error() -> None:
    module = import_module("autotree_core.kernels.triton_kernel")
    q = torch.zeros((1, 1, 4))
    cache = torch.zeros((1, 4, 1, 4))

    with pytest.raises(RuntimeError, match="Triton is not available"):
        module.triton_tree_attention_decode(
            q,
            cache,
            cache,
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        )


def test_public_kernel_package_imports_cleanly() -> None:
    package = import_module("autotree_core.kernels")

    assert callable(package.reference_tree_attention_decode)
    assert callable(package.tree_attention_decode)


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("autotree_core.kernels.dispatch", "tree_attention_decode"),
        ("autotree_core.kernels.triton_kernel", "triton_tree_attention_decode"),
    ],
)
def test_public_decode_functions_keep_the_normative_signature(
    module_name: str, function_name: str
) -> None:
    function = getattr(import_module(module_name), function_name)
    parameters = signature(function).parameters

    assert list(parameters) == [
        "q",
        "k_cache",
        "v_cache",
        "block_tables",
        "context_lens",
        "scale",
    ]
    assert parameters["scale"].default is None
