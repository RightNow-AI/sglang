from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest
import torch

from autotree_core.kernels import reference_tree_attention_decode

from .conftest import ModelCase


def test_real_decode_model_attention_matches_reference_tree_kernel(
    model_case: ModelCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = model_case.executor
    execution = executor.prefill(model_case.prompt[:5])
    model_module = import_module(executor.model.__class__.__module__)
    original_attention = getattr(model_module, "eager_attention_forward", None)
    if original_attention is None:
        pytest.fail(
            f"{model_case.model_id} does not expose eager_attention_forward"
        )
    captured: dict[str, Any] = {}

    def capture_attention(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        result = original_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        )
        if not captured:
            captured["query"] = query.detach().clone()
            captured["attention"] = result[0].detach().clone()
            captured["scale"] = kwargs.get("scaling")
        return result

    monkeypatch.setattr(model_module, "eager_attention_forward", capture_attention)
    token_id = int(torch.argmax(execution.next_logits(execution.root_id)).item())
    executor.decode(execution, execution.root_id, token_id)

    query = captured["query"][:, :, -1, :]
    block_tables, context_lens = execution.attention_metadata([execution.root_id])
    scale = captured["scale"]
    kernel_output = reference_tree_attention_decode(
        query,
        execution.pool.k_cache[0],
        execution.pool.v_cache[0],
        block_tables,
        context_lens,
        scale=None if scale is None else float(scale),
    )
    model_attention = captured["attention"][:, -1]

    tolerance = 1e-6 if executor.config.dtype is torch.float32 else 2e-2
    torch.testing.assert_close(
        kernel_output,
        model_attention,
        rtol=tolerance,
        atol=tolerance,
    )
