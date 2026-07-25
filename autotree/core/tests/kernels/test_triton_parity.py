"""GPU parity tests that invoke the fused Triton kernel directly."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec

import pytest
import torch

from autotree_core.kernels.reference import reference_tree_attention_decode

from .cases import (
    BRANCH_COUNTS,
    CONTEXT_REMAINDERS,
    GQA_RATIOS,
    REFERENCE_DTYPES,
    build_accumulation_stress_case,
    build_random_tree_case,
    matrix_seed,
    rounded_probability_flash_output,
)


GPU_TRITON_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("triton") is None,
    reason="requires CUDA and Triton",
)


def _run_triton(case, scale: float | None = None) -> torch.Tensor:
    kernel = import_module("autotree_core.kernels.triton_kernel")
    return kernel.triton_tree_attention_decode(
        case.q.cuda(),
        case.k_cache.cuda(),
        case.v_cache.cuda(),
        case.block_tables.cuda(),
        case.context_lens.cuda(),
        scale,
    ).cpu()


@GPU_TRITON_REQUIRED
@pytest.mark.parametrize("dtype", REFERENCE_DTYPES, ids=str)
@pytest.mark.parametrize("gqa_ratio", GQA_RATIOS)
@pytest.mark.parametrize("context_remainder", CONTEXT_REMAINDERS)
@pytest.mark.parametrize("num_branches", BRANCH_COUNTS)
def test_triton_matches_reference_on_the_full_oracle_grid(
    dtype: torch.dtype,
    gqa_ratio: int,
    context_remainder: int,
    num_branches: int,
) -> None:
    case = build_random_tree_case(
        seed=matrix_seed(num_branches, gqa_ratio, context_remainder),
        num_branches=num_branches,
        gqa_ratio=gqa_ratio,
        context_remainder=context_remainder,
        dtype=dtype,
    )
    expected = reference_tree_attention_decode(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )

    actual = _run_triton(case)

    tolerance = 1e-3 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=tolerance, atol=tolerance
    )


@GPU_TRITON_REQUIRED
def test_triton_matches_reference_with_custom_scale_and_non_power_of_two_layout() -> (
    None
):
    case = build_random_tree_case(
        seed=91,
        num_branches=5,
        gqa_ratio=4,
        context_remainder=2,
        dtype=torch.float16,
        page_size=3,
        head_dim=10,
    )
    scale = 0.125
    expected = reference_tree_attention_decode(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
        scale,
    )

    actual = _run_triton(case, scale)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@GPU_TRITON_REQUIRED
@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires CUDA bfloat16 support",
)
def test_triton_matches_reference_for_bfloat16() -> None:
    case = build_random_tree_case(
        seed=92,
        num_branches=5,
        gqa_ratio=4,
        context_remainder=15,
        dtype=torch.bfloat16,
    )
    expected = reference_tree_attention_decode(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )

    actual = _run_triton(case)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@GPU_TRITON_REQUIRED
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=("fp16", "bf16"),
)
def test_triton_keeps_probability_and_value_accumulation_in_fp32(
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA bfloat16 support")
    case = build_accumulation_stress_case(dtype)
    expected = reference_tree_attention_decode(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )
    rounded_probability_output = rounded_probability_flash_output(case)

    actual = _run_triton(case)

    actual_error = torch.abs(actual.float() - expected.float())
    rounded_error = torch.abs(rounded_probability_output.float() - expected.float())
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert torch.all(actual_error < rounded_error * 0.5)


@GPU_TRITON_REQUIRED
def test_triton_honors_noncontiguous_context_length_stride() -> None:
    case = build_random_tree_case(
        seed=93,
        num_branches=5,
        gqa_ratio=4,
        context_remainder=1,
        dtype=torch.float16,
        page_size=4,
    )
    expected = reference_tree_attention_decode(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )
    backing = torch.empty(
        case.context_lens.numel() * 2, dtype=torch.int32, device="cuda"
    )
    backing[::2] = case.context_lens.cuda()
    strided_context_lens = backing[::2]
    assert strided_context_lens.stride() == (2,)

    actual = _run_triton(replace(case, context_lens=strided_context_lens))

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@GPU_TRITON_REQUIRED
def test_triton_honors_query_cache_and_block_table_strides() -> None:
    case = build_random_tree_case(
        seed=94,
        num_branches=5,
        gqa_ratio=4,
        context_remainder=3,
        dtype=torch.float16,
        page_size=4,
        head_dim=7,
    )
    expected = reference_tree_attention_decode(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )

    def strided_last_dimension(tensor: torch.Tensor) -> torch.Tensor:
        backing = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] * 2),
            dtype=tensor.dtype,
            device="cuda",
        )
        view = backing[..., ::2]
        view.copy_(tensor.cuda())
        assert not view.is_contiguous()
        return view

    actual = _run_triton(
        replace(
            case,
            q=strided_last_dimension(case.q),
            k_cache=strided_last_dimension(case.k_cache),
            v_cache=strided_last_dimension(case.v_cache),
            block_tables=strided_last_dimension(case.block_tables),
        )
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@GPU_TRITON_REQUIRED
@pytest.mark.parametrize(
    ("block_tables", "context_lens"),
    [
        (torch.tensor([[-1]], dtype=torch.int32), torch.tensor([1], dtype=torch.int32)),
        (torch.tensor([[0]], dtype=torch.int32), torch.tensor([17], dtype=torch.int32)),
        (torch.tensor([[0]], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)),
    ],
    ids=("negative-page", "oversized-context", "empty-context"),
)
def test_triton_signals_invalid_metadata_without_out_of_bounds_access(
    block_tables: torch.Tensor, context_lens: torch.Tensor
) -> None:
    kernel = import_module("autotree_core.kernels.triton_kernel")
    q = torch.zeros((1, 1, 1), dtype=torch.float16, device="cuda")
    cache = torch.zeros((1, 16, 1, 1), dtype=torch.float16, device="cuda")

    actual = kernel.triton_tree_attention_decode(
        q, cache, cache, block_tables.cuda(), context_lens.cuda()
    )

    assert torch.isnan(actual).all()
