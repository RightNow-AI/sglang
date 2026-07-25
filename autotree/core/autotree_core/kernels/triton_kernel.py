"""Fused Triton decode attention over paged branch-local KV paths."""

from __future__ import annotations

from importlib.util import find_spec
from math import isfinite, sqrt

import torch


TRITON_IMPORT_ERROR: BaseException | None = None
if find_spec("triton") is not None:
    try:
        import triton
        import triton.language as tl
    except (ImportError, OSError, ValueError) as error:
        TRITON_AVAILABLE = False
        TRITON_IMPORT_ERROR = error
    else:
        TRITON_AVAILABLE = True
else:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _tree_attention_decode_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        block_tables_ptr,
        context_lens_ptr,
        out_ptr,
        scale,
        stride_q_branch,
        stride_q_head,
        stride_q_dim,
        stride_k_page,
        stride_k_token,
        stride_k_head,
        stride_k_dim,
        stride_v_page,
        stride_v_token,
        stride_v_head,
        stride_v_dim,
        stride_bt_branch,
        stride_bt_page,
        stride_context_branch,
        stride_out_branch,
        stride_out_head,
        stride_out_dim,
        NUM_Q_HEADS: tl.constexpr,
        QUERIES_PER_KV: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_PAGES: tl.constexpr,
        MAX_PAGES: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        branch_idx = tl.program_id(0)
        kv_head_idx = tl.program_id(1)

        offsets_q = tl.arange(0, BLOCK_Q)
        offsets_t = tl.arange(0, BLOCK_T)
        offsets_d = tl.arange(0, BLOCK_D)
        branch_idx_64 = branch_idx.to(tl.int64)
        kv_head_idx_64 = kv_head_idx.to(tl.int64)
        offsets_q_64 = offsets_q.to(tl.int64)
        offsets_t_64 = offsets_t.to(tl.int64)
        offsets_d_64 = offsets_d.to(tl.int64)
        q_head_indices_64 = kv_head_idx_64 * QUERIES_PER_KV + offsets_q_64
        q_mask = (offsets_q < QUERIES_PER_KV) & (q_head_indices_64 < NUM_Q_HEADS)
        dim_mask = offsets_d < HEAD_DIM

        q_offsets = (
            branch_idx_64 * stride_q_branch
            + q_head_indices_64[:, None] * stride_q_head
            + offsets_d_64[None, :] * stride_q_dim
        )
        q = tl.load(
            q_ptr + q_offsets,
            mask=q_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        context_len = tl.load(context_lens_ptr + branch_idx_64 * stride_context_branch)
        metadata_valid = (context_len > 0) & (context_len <= MAX_PAGES * PAGE_SIZE)
        running_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_Q,), tl.float32)
        accumulator = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)
        log2e: tl.constexpr = 1.4426950408889634

        for page_slot in range(MAX_PAGES):
            page_active = page_slot * PAGE_SIZE < context_len
            token_indices_64 = page_slot * PAGE_SIZE + offsets_t_64
            page_id = tl.load(
                block_tables_ptr
                + branch_idx_64 * stride_bt_branch
                + page_slot * stride_bt_page,
                mask=page_active,
                other=0,
            ).to(tl.int64)
            page_id_valid = (page_id >= 0) & (page_id < NUM_PAGES)
            metadata_valid = metadata_valid & ((~page_active) | page_id_valid)
            valid_page_active = page_active & metadata_valid
            safe_page_id = tl.where(page_id_valid, page_id, 0)
            token_mask = (
                valid_page_active
                & (offsets_t < PAGE_SIZE)
                & (token_indices_64 < context_len)
            )

            k_offsets = (
                safe_page_id * stride_k_page
                + offsets_t_64[:, None] * stride_k_token
                + kv_head_idx_64 * stride_k_head
                + offsets_d_64[None, :] * stride_k_dim
            )
            v_offsets = (
                safe_page_id * stride_v_page
                + offsets_t_64[:, None] * stride_v_token
                + kv_head_idx_64 * stride_v_head
                + offsets_d_64[None, :] * stride_v_dim
            )
            load_mask = token_mask[:, None] & dim_mask[None, :]
            k = tl.load(k_ptr + k_offsets, mask=load_mask, other=0.0)
            v = tl.load(v_ptr + v_offsets, mask=load_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k), input_precision="ieee")
            scores = scores * scale * log2e
            score_mask = q_mask[:, None] & token_mask[None, :]
            scores = tl.where(score_mask, scores, -float("inf"))
            page_max = tl.max(scores, axis=1)
            next_max = tl.maximum(running_max, page_max)
            correction = tl.where(
                valid_page_active, tl.math.exp2(running_max - next_max), 1.0
            )
            probabilities = tl.where(
                score_mask,
                tl.math.exp2(scores - next_max[:, None]),
                0.0,
            )

            accumulator = accumulator * correction[:, None]
            accumulator = tl.dot(
                probabilities,
                v.to(tl.float32),
                accumulator,
                input_precision="ieee",
            )
            running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
            running_max = tl.where(valid_page_active, next_max, running_max)

        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        output = accumulator / denominator[:, None]
        output = tl.where(metadata_valid, output, float("nan"))
        out_offsets = (
            branch_idx_64 * stride_out_branch
            + q_head_indices_64[:, None] * stride_out_head
            + offsets_d_64[None, :] * stride_out_dim
        )
        tl.store(
            out_ptr + out_offsets,
            output,
            mask=q_mask[:, None] & dim_mask[None, :],
        )


def _validate_triton_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None,
) -> float:
    if not TRITON_AVAILABLE:
        detail = f" ({TRITON_IMPORT_ERROR})" if TRITON_IMPORT_ERROR is not None else ""
        raise RuntimeError(f"Triton is not available in this environment{detail}")
    if q.ndim != 3:
        raise ValueError("q must have shape [num_branches, num_q_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "k_cache and v_cache must have shape "
            "[num_pages, PAGE_SIZE, num_kv_heads, head_dim]"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    if block_tables.ndim != 2 or context_lens.ndim != 1:
        raise ValueError("block_tables must be rank 2 and context_lens rank 1")

    num_branches, num_q_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, cache_head_dim = k_cache.shape
    if block_tables.shape[0] != num_branches or context_lens.shape[0] != num_branches:
        raise ValueError("q, block_tables, and context_lens must agree on num_branches")
    if cache_head_dim != head_dim:
        raise ValueError("q and KV caches must agree on head_dim")
    if num_kv_heads < 1 or num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    if num_pages < 1 or page_size < 1 or head_dim < 1 or block_tables.shape[1] < 1:
        raise ValueError(
            "num_pages, PAGE_SIZE, head_dim, and max_pages must be positive"
        )
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Triton tree attention supports fp16, bf16, and fp32")
    if q.dtype != k_cache.dtype or k_cache.dtype != v_cache.dtype:
        raise TypeError("q, k_cache, and v_cache must have the same dtype")
    if block_tables.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise TypeError("block_tables and context_lens must use int32")
    if not q.is_cuda:
        raise ValueError("Triton tree attention requires CUDA tensors")
    tensors = (k_cache, v_cache, block_tables, context_lens)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all attention tensors must be on the same CUDA device")

    effective_scale = 1.0 / sqrt(head_dim) if scale is None else float(scale)
    if not isfinite(effective_scale):
        raise ValueError("scale must be finite")
    return effective_scale


def triton_tree_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Launch fused branch-aware decode attention on CUDA via Triton.

    Malformed device-resident context metadata is bounds-masked in the kernel
    and produces NaN for the affected branch instead of an unsafe GPU read.
    """
    effective_scale = _validate_triton_inputs(
        q, k_cache, v_cache, block_tables, context_lens, scale
    )
    num_branches, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    if num_branches == 0:
        return torch.empty_like(q)

    queries_per_kv = num_q_heads // num_kv_heads
    block_q = triton.next_power_of_2(max(16, queries_per_kv))
    block_t = triton.next_power_of_2(max(16, k_cache.shape[1]))
    block_d = triton.next_power_of_2(max(16, head_dim))
    output = torch.empty_like(q)
    grid = (num_branches, num_kv_heads)
    num_warps = 4 if max(block_q, block_d) <= 64 else 8

    _tree_attention_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        output,
        effective_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        context_lens.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_Q_HEADS=num_q_heads,
        QUERIES_PER_KV=queries_per_kv,
        PAGE_SIZE=k_cache.shape[1],
        HEAD_DIM=head_dim,
        NUM_PAGES=k_cache.shape[0],
        MAX_PAGES=block_tables.shape[1],
        BLOCK_Q=block_q,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


__all__ = [
    "TRITON_AVAILABLE",
    "TRITON_IMPORT_ERROR",
    "triton_tree_attention_decode",
]
