"""Public backend dispatch for tree-attention decode."""

from __future__ import annotations

import logging

import torch

from .reference import reference_tree_attention_decode
from .triton_kernel import TRITON_AVAILABLE, triton_tree_attention_decode


LOGGER = logging.getLogger(__name__)


def tree_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Use fused Triton on CUDA when available, otherwise log and use PyTorch."""
    if q.is_cuda and TRITON_AVAILABLE:
        LOGGER.info("tree_attention_decode backend=triton device=%s", q.device)
        return triton_tree_attention_decode(
            q, k_cache, v_cache, block_tables, context_lens, scale
        )

    if q.is_cuda:
        LOGGER.warning(
            "Triton is unavailable for CUDA input; "
            "tree_attention_decode backend=reference"
        )
    else:
        LOGGER.info("tree_attention_decode backend=reference device=%s", q.device)
    return reference_tree_attention_decode(
        q, k_cache, v_cache, block_tables, context_lens, scale
    )


__all__ = ["tree_attention_decode"]
