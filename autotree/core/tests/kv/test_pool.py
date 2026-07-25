from dataclasses import FrozenInstanceError

import pytest
import torch

from autotree_core.kv import (
    PAGE_SIZE,
    KVCapacityError,
    KVInvariantError,
    KVPoolConfig,
    KVStats,
    PagedKVPool,
)


def make_config(**overrides: object) -> KVPoolConfig:
    values = {
        "num_layers": 2,
        "num_kv_heads": 3,
        "head_dim": 4,
        "capacity": 3,
        "page_size": 5,
    }
    values.update(overrides)
    return KVPoolConfig(**values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cache_shapes_and_supported_dtypes(dtype: torch.dtype) -> None:
    config = make_config(dtype=dtype)
    pool = PagedKVPool(config)

    assert len(pool.k_cache) == config.num_layers
    assert len(pool.v_cache) == config.num_layers
    for cache in (*pool.k_cache, *pool.v_cache):
        assert cache.shape == (
            config.capacity,
            config.page_size,
            config.num_kv_heads,
            config.head_dim,
        )
        assert cache.dtype is dtype
        assert cache.device.type == "cpu"
        assert torch.count_nonzero(cache).item() == 0


def test_config_defaults_and_is_frozen_and_slotted() -> None:
    config = KVPoolConfig(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        capacity=2,
    )

    assert PAGE_SIZE == 16
    assert config.page_size == PAGE_SIZE
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.capacity = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_layers", 0),
        ("num_kv_heads", -1),
        ("head_dim", 0),
        ("capacity", 0),
        ("page_size", -1),
        ("num_layers", 1.5),
        ("capacity", True),
    ],
)
def test_config_rejects_invalid_dimensions(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        make_config(**{field: value})


@pytest.mark.parametrize("dtype", [torch.float64, torch.int8, "float32", [], {}])
def test_config_rejects_unsupported_dtypes(dtype: object) -> None:
    with pytest.raises(ValueError, match="dtype"):
        make_config(dtype=dtype)


def test_config_rejects_invalid_device() -> None:
    with pytest.raises(ValueError, match="device"):
        make_config(device="not-a-device")


def test_alloc_retain_free_and_deterministic_reuse() -> None:
    pool = PagedKVPool(make_config(capacity=3))

    assert [pool.alloc_page() for _ in range(3)] == [0, 1, 2]
    assert [pool.refcount(page_id) for page_id in range(3)] == [1, 1, 1]
    assert pool.used_pages == 3
    assert pool.available_pages == 0

    pool._retain(1)
    assert pool.refcount(1) == 2
    pool.free(1)
    assert pool.refcount(1) == 1
    assert pool.used_pages == 3

    pool.free(1)
    assert pool.refcount(1) == 0
    assert pool.used_pages == 2
    assert pool.available_pages == 1
    assert pool.alloc_page() == 1
    assert pool.refcount(1) == 1


def test_recycled_pages_are_zeroed_across_every_layer() -> None:
    pool = PagedKVPool(make_config(num_layers=2, capacity=1))
    page_id = pool.alloc_page()
    for cache in (*pool.k_cache, *pool.v_cache):
        cache[page_id].fill_(7)

    pool.free(page_id)
    assert pool.alloc_page() == page_id
    for cache in (*pool.k_cache, *pool.v_cache):
        assert torch.count_nonzero(cache[page_id]).item() == 0


def test_capacity_error_is_typed_and_allocation_is_atomic() -> None:
    pool = PagedKVPool(make_config(capacity=2))
    pool.alloc_page()
    pool.alloc_page()
    pool.k_cache[0][0].fill_(2)
    pool.v_cache[1][1].fill_(3)
    stats_before = pool.stats
    refcounts_before = [pool.refcount(page_id) for page_id in range(2)]
    cache_before = [cache.clone() for cache in (*pool.k_cache, *pool.v_cache)]

    with pytest.raises(KVCapacityError) as error:
        pool.alloc_page()

    assert error.value.required_pages == 1
    assert error.value.available_pages == 0
    assert pool.stats == stats_before
    assert [pool.refcount(page_id) for page_id in range(2)] == refcounts_before
    for actual, expected in zip((*pool.k_cache, *pool.v_cache), cache_before):
        assert torch.equal(actual, expected)


def test_invalid_page_operations_and_double_free_raise_invariant_error() -> None:
    pool = PagedKVPool(make_config(capacity=2))

    for invalid_page_id in (-1, 2, True, "0"):
        with pytest.raises(KVInvariantError):
            pool.refcount(invalid_page_id)  # type: ignore[arg-type]

    with pytest.raises(KVInvariantError):
        pool._retain(0)
    with pytest.raises(KVInvariantError):
        pool.free(0)

    page_id = pool.alloc_page()
    pool.free(page_id)
    with pytest.raises(KVInvariantError):
        pool.free(page_id)


def test_exact_pool_accounting_and_stats_snapshot() -> None:
    pool = PagedKVPool(make_config(capacity=2, page_size=4))

    assert pool.stats == KVStats(
        used_pages=0,
        available_pages=2,
        physical_tokens=0,
        logical_tokens=0,
        kv_reuse_ratio=0.0,
    )

    first_page = pool.alloc_page()
    assert pool.used_pages == 1
    assert pool.available_pages == 1
    assert pool.physical_tokens == 4
    assert pool.logical_tokens == 0
    assert pool.kv_reuse_ratio == 0.0

    pool._adjust_logical_tokens(6)
    assert pool.logical_tokens == 6
    assert pool.kv_reuse_ratio == 1.5

    pool._retain(first_page)
    assert pool.used_pages == 1
    pool.alloc_page()
    assert pool.stats == KVStats(
        used_pages=2,
        available_pages=0,
        physical_tokens=8,
        logical_tokens=6,
        kv_reuse_ratio=0.75,
    )

    pool._adjust_logical_tokens(-2)
    stats_before = pool.stats
    with pytest.raises(KVInvariantError):
        pool._adjust_logical_tokens(-5)
    assert pool.stats == stats_before
