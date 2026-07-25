"""Backend routing tests for public tree-attention dispatch."""

from __future__ import annotations

from importlib import import_module
import logging

import pytest
import torch

from .cases import build_random_tree_case


def test_cpu_dispatch_matches_reference_and_logs_backend(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dispatch = import_module("autotree_core.kernels.dispatch")
    case = build_random_tree_case(
        seed=5,
        num_branches=3,
        gqa_ratio=4,
        context_remainder=1,
        dtype=torch.float32,
        page_size=4,
    )

    with caplog.at_level(logging.INFO, logger=dispatch.__name__):
        actual = dispatch.tree_attention_decode(
            case.q,
            case.k_cache,
            case.v_cache,
            case.block_tables,
            case.context_lens,
        )
    expected = dispatch.reference_tree_attention_decode(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )

    torch.testing.assert_close(actual, expected)
    assert "reference" in caplog.text.lower()


class _FakeCudaQuery:
    is_cuda = True
    device = torch.device("cuda")


def test_cuda_dispatch_calls_low_level_triton_directly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dispatch = import_module("autotree_core.kernels.dispatch")
    sentinel = object()
    captured: tuple[object, ...] | None = None

    def fake_triton(*args: object) -> object:
        nonlocal captured
        captured = args
        return sentinel

    monkeypatch.setattr(dispatch, "TRITON_AVAILABLE", True)
    monkeypatch.setattr(dispatch, "triton_tree_attention_decode", fake_triton)
    query = _FakeCudaQuery()
    arguments = (query, object(), object(), object(), object(), 0.25)

    with caplog.at_level(logging.INFO, logger=dispatch.__name__):
        result = dispatch.tree_attention_decode(*arguments)

    assert result is sentinel
    assert captured == arguments
    assert "triton" in caplog.text.lower()


def test_triton_launch_errors_are_not_silently_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = import_module("autotree_core.kernels.dispatch")

    def failing_triton(*_args: object) -> None:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(dispatch, "TRITON_AVAILABLE", True)
    monkeypatch.setattr(dispatch, "triton_tree_attention_decode", failing_triton)

    with pytest.raises(RuntimeError, match="compile failed"):
        dispatch.tree_attention_decode(
            _FakeCudaQuery(), object(), object(), object(), object()
        )


def test_cuda_without_triton_uses_logged_reference_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dispatch = import_module("autotree_core.kernels.dispatch")
    sentinel = object()
    monkeypatch.setattr(dispatch, "TRITON_AVAILABLE", False)
    monkeypatch.setattr(
        dispatch, "reference_tree_attention_decode", lambda *_args: sentinel
    )

    with caplog.at_level(logging.WARNING, logger=dispatch.__name__):
        result = dispatch.tree_attention_decode(
            _FakeCudaQuery(), object(), object(), object(), object()
        )

    assert result is sentinel
    assert "triton" in caplog.text.lower()
    assert "reference" in caplog.text.lower()
