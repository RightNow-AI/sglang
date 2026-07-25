"""Normative pure-PyTorch tree-attention decode implementation."""

from __future__ import annotations

from math import isfinite, sqrt

import torch


def _validate_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None,
) -> tuple[int, int, int, float]:
    if q.ndim != 3:
        raise ValueError("q must have shape [num_branches, num_q_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache and v_cache must have shape "
            "[num_pages, PAGE_SIZE, num_kv_heads, head_dim]"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [num_branches, max_pages]")
    if context_lens.ndim != 1:
        raise ValueError("context_lens must have shape [num_branches]")

    num_branches, num_q_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, cache_head_dim = k_cache.shape
    if block_tables.shape[0] != num_branches or context_lens.shape[0] != num_branches:
        raise ValueError("q, block_tables, and context_lens must agree on num_branches")
    if cache_head_dim != head_dim:
        raise ValueError("q and KV caches must agree on head_dim")
    if num_kv_heads < 1 or num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    if num_pages < 1 or page_size < 1:
        raise ValueError("KV caches must contain at least one non-empty page")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("tree attention supports fp16, bf16, and fp32 tensors")
    if q.dtype != k_cache.dtype or k_cache.dtype != v_cache.dtype:
        raise TypeError("q, k_cache, and v_cache must have the same dtype")
    if q.device != k_cache.device or k_cache.device != v_cache.device:
        raise ValueError("q, k_cache, and v_cache must be on the same device")
    if block_tables.device != q.device or context_lens.device != q.device:
        raise ValueError(
            "block_tables and context_lens must be on the attention device"
        )
    if block_tables.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise TypeError("block_tables and context_lens must use int32")

    effective_scale = 1.0 / sqrt(head_dim) if scale is None else float(scale)
    if not isfinite(effective_scale):
        raise ValueError("scale must be finite")
    return num_kv_heads, page_size, num_pages, effective_scale


def reference_tree_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode one query per branch over only that branch's root-to-node path.

    KV pages are gathered in block-table order. Scores, softmax probabilities,
    and value accumulation are evaluated in fp32, including for fp16/bf16
    inputs. The result is converted back to the query dtype.
    """
    num_kv_heads, page_size, num_pages, effective_scale = _validate_inputs(
        q, k_cache, v_cache, block_tables, context_lens, scale
    )
    num_branches, num_q_heads, _ = q.shape
    queries_per_kv_head = num_q_heads // num_kv_heads
    kv_head_for_query = torch.arange(num_q_heads, device=q.device).div(
        queries_per_kv_head, rounding_mode="floor"
    )
    branch_outputs: list[torch.Tensor] = []

    for branch_idx in range(num_branches):
        context_len = int(context_lens[branch_idx].item())
        if context_len <= 0:
            raise ValueError("every context_len must be positive for decode attention")
        pages_needed = (context_len + page_size - 1) // page_size
        if pages_needed > block_tables.shape[1]:
            raise ValueError(
                "context_len requires more pages than block_tables provides"
            )

        page_ids = block_tables[branch_idx, :pages_needed].to(torch.long)
        if torch.any(page_ids == -1):
            raise ValueError("block_tables contains -1 padding inside a branch context")
        if torch.any(page_ids < 0) or torch.any(page_ids >= num_pages):
            raise ValueError("block_tables contains an out-of-range physical page id")

        branch_k = k_cache.index_select(0, page_ids).reshape(
            pages_needed * page_size, num_kv_heads, q.shape[-1]
        )[:context_len]
        branch_v = v_cache.index_select(0, page_ids).reshape(
            pages_needed * page_size, num_kv_heads, q.shape[-1]
        )[:context_len]
        expanded_k = branch_k[:, kv_head_for_query, :].float()
        expanded_v = branch_v[:, kv_head_for_query, :].float()
        scores = torch.einsum("hd,thd->ht", q[branch_idx].float(), expanded_k)
        probabilities = torch.softmax(scores * effective_scale, dim=-1)
        branch_output = torch.einsum("ht,thd->hd", probabilities, expanded_v)
        branch_outputs.append(branch_output.to(q.dtype))

    if not branch_outputs:
        return torch.empty_like(q)
    return torch.stack(branch_outputs, dim=0)


__all__ = ["reference_tree_attention_decode"]
