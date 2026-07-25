"""Normative correctness tests for pure-PyTorch tree attention."""

from __future__ import annotations

from importlib import import_module
from inspect import signature

import pytest
import torch

from .cases import (
    BRANCH_COUNTS,
    CONTEXT_REMAINDERS,
    GQA_RATIOS,
    REFERENCE_DTYPES,
    build_accumulation_stress_case,
    build_random_tree_case,
    dense_sdpa_oracle,
    matrix_seed,
    rounded_probability_flash_output,
)


def _reference():
    return import_module(
        "autotree_core.kernels.reference"
    ).reference_tree_attention_decode


@pytest.mark.parametrize("dtype", REFERENCE_DTYPES, ids=str)
@pytest.mark.parametrize("gqa_ratio", GQA_RATIOS)
@pytest.mark.parametrize("context_remainder", CONTEXT_REMAINDERS)
@pytest.mark.parametrize("num_branches", BRANCH_COUNTS)
def test_reference_matches_dense_sdpa_for_seeded_random_trees(
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

    actual = _reference()(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )
    expected = dense_sdpa_oracle(case)

    tolerance = 1e-3 if dtype == torch.float32 else 2e-2
    assert actual.shape == case.q.shape
    assert actual.dtype == case.q.dtype
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=tolerance, atol=tolerance
    )


def test_reference_has_the_normative_signature() -> None:
    parameters = signature(_reference()).parameters

    assert list(parameters) == [
        "q",
        "k_cache",
        "v_cache",
        "block_tables",
        "context_lens",
        "scale",
    ]
    assert parameters["scale"].default is None


def test_reference_honors_custom_scale() -> None:
    case = build_random_tree_case(
        seed=42,
        num_branches=5,
        gqa_ratio=4,
        context_remainder=1,
        dtype=torch.float32,
        page_size=4,
    )
    scale = 0.125

    actual = _reference()(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
        scale,
    )

    torch.testing.assert_close(actual, dense_sdpa_oracle(case, scale=scale))


def test_reference_supports_post_dedup_shared_physical_pages() -> None:
    case = build_random_tree_case(
        seed=7,
        num_branches=8,
        gqa_ratio=1,
        context_remainder=1,
        dtype=torch.float32,
        page_size=4,
    )
    first_page_id = int(case.block_tables[0, 0].item())

    assert first_page_id == int(case.block_tables[1, 0].item())
    actual = _reference()(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )

    torch.testing.assert_close(actual, dense_sdpa_oracle(case))


def test_reference_supports_bfloat16_with_fp32_accumulation() -> None:
    case = build_random_tree_case(
        seed=81,
        num_branches=5,
        gqa_ratio=4,
        context_remainder=3,
        dtype=torch.bfloat16,
        page_size=4,
    )

    actual = _reference()(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.float(),
        dense_sdpa_oracle(case).float(),
        rtol=2e-2,
        atol=2e-2,
    )


def test_reference_excludes_sibling_branch_tokens() -> None:
    page_size = 4
    q = torch.zeros((2, 1, 2), dtype=torch.float32)
    k_cache = torch.zeros((3, page_size, 1, 2), dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    v_cache[1, 0] = 1.0
    v_cache[2, 0] = 100.0
    block_tables = torch.tensor([[0, 1, -1], [0, 2, -1]], dtype=torch.int32)
    context_lens = torch.tensor([5, 5], dtype=torch.int32)

    actual = _reference()(q, k_cache, v_cache, block_tables, context_lens)

    torch.testing.assert_close(actual[0], torch.full((1, 2), 0.2))
    torch.testing.assert_close(actual[1], torch.full((1, 2), 20.0))


def test_reference_stops_exactly_at_a_full_page_boundary() -> None:
    page_size = 4
    q = torch.zeros((1, 1, 2), dtype=torch.float32)
    k_cache = torch.zeros((2, page_size, 1, 2), dtype=torch.float32)
    v_cache = torch.full_like(k_cache, 100.0)
    v_cache[0] = 2.0
    block_tables = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    context_lens = torch.tensor([page_size], dtype=torch.int32)

    actual = _reference()(q, k_cache, v_cache, block_tables, context_lens)

    torch.testing.assert_close(actual, torch.full_like(actual, 2.0))


def test_reference_includes_the_last_token_at_a_full_page_boundary() -> None:
    q = torch.zeros((1, 1, 1), dtype=torch.float32)
    k_cache = torch.zeros((1, 4, 1, 1), dtype=torch.float32)
    v_cache = torch.tensor([[[[0.0]], [[0.0]], [[0.0]], [[8.0]]]])

    actual = _reference()(
        q,
        k_cache,
        v_cache,
        torch.tensor([[0, -1]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
    )

    torch.testing.assert_close(actual, torch.full_like(actual, 2.0))


def test_reference_uses_each_branch_context_length() -> None:
    q = torch.zeros((2, 1, 1), dtype=torch.float32)
    k_cache = torch.zeros((1, 4, 1, 1), dtype=torch.float32)
    v_cache = torch.tensor([[[[2.0]], [[4.0]], [[8.0]], [[16.0]]]])

    actual = _reference()(
        q,
        k_cache,
        v_cache,
        torch.tensor([[0, -1], [0, -1]], dtype=torch.int32),
        torch.tensor([1, 4], dtype=torch.int32),
    )

    torch.testing.assert_close(actual[:, 0, 0], torch.tensor([2.0, 7.5]))


def test_reference_maps_contiguous_query_groups_to_each_kv_head() -> None:
    q = torch.zeros((1, 8, 1), dtype=torch.float32)
    k_cache = torch.zeros((1, 1, 2, 1), dtype=torch.float32)
    v_cache = torch.tensor([[[[3.0], [9.0]]]])

    actual = _reference()(
        q,
        k_cache,
        v_cache,
        torch.tensor([[0, -1]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    torch.testing.assert_close(
        actual[0, :, 0], torch.tensor([3.0, 3.0, 3.0, 3.0, 9.0, 9.0, 9.0, 9.0])
    )


def test_reference_never_dereferences_minus_one_padding() -> None:
    q = torch.zeros((1, 1, 1), dtype=torch.float32)
    k_cache = torch.zeros((2, 4, 1, 1), dtype=torch.float32)
    v_cache = torch.full_like(k_cache, 100.0)
    v_cache[0, 0] = 2.0

    actual = _reference()(
        q,
        k_cache,
        v_cache,
        torch.tensor([[0, -1]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    torch.testing.assert_close(actual, torch.full_like(actual, 2.0))


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=("fp16", "bf16"),
)
def test_reference_keeps_softmax_and_value_accumulation_in_fp32(
    dtype: torch.dtype,
) -> None:
    case = build_accumulation_stress_case(dtype)

    actual = _reference()(
        case.q,
        case.k_cache,
        case.v_cache,
        case.block_tables,
        case.context_lens,
    )
    expected = dense_sdpa_oracle(case)
    rounded_probability_output = rounded_probability_flash_output(case)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(rounded_probability_output, actual)


def test_reference_rejects_non_integral_gqa_ratio() -> None:
    q = torch.zeros((1, 3, 4))
    cache = torch.zeros((1, 4, 2, 4))

    with pytest.raises(ValueError, match="multiple"):
        _reference()(
            q,
            cache,
            cache,
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        )


def test_reference_rejects_float64_outside_the_spec() -> None:
    q = torch.zeros((1, 1, 4), dtype=torch.float64)
    cache = torch.zeros((1, 4, 1, 4), dtype=torch.float64)

    with pytest.raises(TypeError, match="fp16, bf16, and fp32"):
        _reference()(
            q,
            cache,
            cache,
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        )


def test_reference_requires_int32_metadata_from_the_spec() -> None:
    q = torch.zeros((1, 1, 4))
    cache = torch.zeros((1, 4, 1, 4))

    with pytest.raises(TypeError, match="int32"):
        _reference()(
            q,
            cache,
            cache,
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
        )


def test_reference_rejects_padding_inside_the_context() -> None:
    q = torch.zeros((1, 1, 4))
    cache = torch.zeros((1, 4, 1, 4))

    with pytest.raises(ValueError, match="-1"):
        _reference()(
            q,
            cache,
            cache,
            torch.tensor([[0, -1]], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
        )
