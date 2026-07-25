import pytest
import torch

from autotree_core.kv import (
    KVInvariantError,
    KVPoolConfig,
    PagedKVPool,
    gather_branch_kv,
)


def make_pool(
    *,
    capacity: int = 3,
    page_size: int = 4,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 3,
) -> PagedKVPool:
    return PagedKVPool(
        KVPoolConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            capacity=capacity,
            page_size=page_size,
        )
    )


def test_empty_gather_returns_contiguous_typed_tensors() -> None:
    pool = make_pool()

    k, v = gather_branch_kv(pool, [], 0, layer=1)

    expected_shape = (0, pool.config.num_kv_heads, pool.config.head_dim)
    assert k.shape == expected_shape
    assert v.shape == expected_shape
    assert k.dtype is pool.config.dtype
    assert v.dtype is pool.config.dtype
    assert k.device == pool.config.device
    assert v.device == pool.config.device
    assert k.is_contiguous()
    assert v.is_contiguous()


def test_gather_uses_torch_pages_and_excludes_padded_tail_slots() -> None:
    pool = make_pool(page_size=4)
    first_page = pool.alloc_page()
    second_page = pool.alloc_page()
    pool._set_page_length(first_page, 4)
    pool._set_page_length(second_page, 2)

    expected_by_layer: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in range(pool.config.num_layers):
        values = torch.arange(
            layer * 100,
            layer * 100 + 6 * pool.config.num_kv_heads * pool.config.head_dim,
            dtype=pool.config.dtype,
        ).reshape(6, pool.config.num_kv_heads, pool.config.head_dim)
        expected_k = values.clone()
        expected_v = values + 1_000
        pool.k_cache[layer][first_page].copy_(expected_k[:4])
        pool.v_cache[layer][first_page].copy_(expected_v[:4])
        pool.k_cache[layer][second_page, :2].copy_(expected_k[4:])
        pool.v_cache[layer][second_page, :2].copy_(expected_v[4:])
        pool.k_cache[layer][second_page, 2:].fill_(99_999)
        pool.v_cache[layer][second_page, 2:].fill_(88_888)
        expected_by_layer.append((expected_k, expected_v))

    for layer, (expected_k, expected_v) in enumerate(expected_by_layer):
        actual_k, actual_v = gather_branch_kv(
            pool,
            [first_page, second_page],
            6,
            layer=layer,
        )
        assert torch.equal(actual_k, expected_k)
        assert torch.equal(actual_v, expected_v)
        assert actual_k.is_contiguous()
        assert actual_v.is_contiguous()
        assert 99_999 not in actual_k
        assert 88_888 not in actual_v


def test_gather_is_detached_from_cache_storage() -> None:
    pool = make_pool(capacity=1)
    page_id = pool.alloc_page()
    pool._set_page_length(page_id, 1)
    pool.k_cache[0][page_id, 0].fill_(3)
    pool.v_cache[0][page_id, 0].fill_(4)

    gathered_k, gathered_v = gather_branch_kv(pool, [page_id], 1)
    gathered_k.fill_(30)
    gathered_v.fill_(40)

    assert torch.all(pool.k_cache[0][page_id, 0] == 3)
    assert torch.all(pool.v_cache[0][page_id, 0] == 4)


def test_page_valid_length_resets_when_page_is_reclaimed_and_reallocated() -> None:
    pool = make_pool(capacity=1)
    page_id = pool.alloc_page()
    pool._set_page_length(page_id, 3)
    assert pool._page_length(page_id) == 3

    pool.free(page_id)
    assert pool.alloc_page() == page_id
    assert pool._page_length(page_id) == 0


@pytest.mark.parametrize("layer", [-1, 2, True, 1.5])
def test_gather_rejects_invalid_layer(layer: object) -> None:
    pool = make_pool(num_layers=2)

    with pytest.raises((TypeError, ValueError)):
        gather_branch_kv(pool, [], 0, layer=layer)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_tokens", [-1, True, 1.5])
def test_gather_rejects_invalid_token_count(num_tokens: object) -> None:
    pool = make_pool()

    with pytest.raises((TypeError, ValueError)):
        gather_branch_kv(pool, [], num_tokens)  # type: ignore[arg-type]


def test_gather_rejects_table_that_does_not_exactly_cover_tokens() -> None:
    pool = make_pool()
    page_id = pool.alloc_page()
    pool._set_page_length(page_id, 4)

    with pytest.raises(KVInvariantError, match="block table"):
        gather_branch_kv(pool, [], 1)
    with pytest.raises(KVInvariantError, match="block table"):
        gather_branch_kv(pool, [page_id, page_id], 1)


def test_gather_rejects_unallocated_or_short_pages() -> None:
    pool = make_pool(capacity=2)
    allocated_page = pool.alloc_page()
    pool._set_page_length(allocated_page, 1)

    with pytest.raises(KVInvariantError, match="not allocated"):
        gather_branch_kv(pool, [1], 1)
    with pytest.raises(KVInvariantError, match="valid token"):
        gather_branch_kv(pool, [allocated_page], 2)


def test_gather_rejects_page_length_beyond_branch_tail() -> None:
    pool = make_pool(capacity=1)
    page_id = pool.alloc_page()
    pool._set_page_length(page_id, 2)

    with pytest.raises(KVInvariantError, match="valid token"):
        gather_branch_kv(pool, [page_id], 1)
