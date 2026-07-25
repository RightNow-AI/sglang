import gc
import hashlib
import weakref

import pytest
import torch

from autotree_core.kv import KVInvariantError, KVPoolConfig, PagedKVPool, TreeState


def make_tree(
    *,
    capacity: int = 8,
    page_size: int = 2,
    num_layers: int = 2,
    num_kv_heads: int = 1,
    head_dim: int = 2,
    dtype: torch.dtype = torch.float32,
) -> tuple[PagedKVPool, TreeState]:
    pool = PagedKVPool(
        KVPoolConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            capacity=capacity,
            page_size=page_size,
            dtype=dtype,
        )
    )
    return pool, TreeState(pool)


def make_tokens(
    pool: PagedKVPool,
    num_tokens: int,
    *,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = pool.config
    shape = (
        config.num_layers,
        num_tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    count = config.num_layers * num_tokens * config.num_kv_heads * config.head_dim
    k = torch.arange(offset, offset + count, dtype=torch.float32).to(config.dtype)
    k = k.reshape(shape)
    return k, (k + 1_000).clone()


def append_independent_pages(
    tree: TreeState,
    first_k: torch.Tensor,
    first_v: torch.Tensor,
    second_k: torch.Tensor | None = None,
    second_v: torch.Tensor | None = None,
) -> int:
    child_id = tree.fork(tree.root_id)
    tree.append_tokens(tree.root_id, first_k, first_v)
    tree.append_tokens(
        child_id,
        first_k if second_k is None else second_k,
        first_v if second_v is None else second_v,
    )
    return child_id


def raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8).cpu().clone()


def snapshot_streams(
    pool: PagedKVPool,
    tree: TreeState,
    branch_ids: list[int],
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    return {
        branch_id: [
            tuple(raw_bytes(tensor) for tensor in tree.gather(branch_id, layer=layer))
            for layer in range(pool.config.num_layers)
        ]
        for branch_id in branch_ids
    }


def assert_streams_equal(
    actual: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    expected: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
) -> None:
    assert actual.keys() == expected.keys()
    for branch_id in actual:
        assert len(actual[branch_id]) == len(expected[branch_id])
        for actual_layer, expected_layer in zip(
            actual[branch_id], expected[branch_id], strict=True
        ):
            for actual_tensor, expected_tensor in zip(
                actual_layer, expected_layer, strict=True
            ):
                assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize("entrypoint", ["tree", "pool"])
def test_identical_full_pages_merge_byte_exactly_through_both_entrypoints(
    entrypoint: str,
) -> None:
    pool, tree = make_tree()
    k, v = make_tokens(pool, pool.config.page_size)
    child_id = append_independent_pages(tree, k, v)
    branch_ids = [tree.root_id, child_id]
    before = snapshot_streams(pool, tree, branch_ids)
    original_pages = [
        tree.get_branch(branch_id).block_table[0] for branch_id in branch_ids
    ]

    freed = tree.dedup_scan() if entrypoint == "tree" else pool.dedup_scan()

    canonical_page = min(original_pages)
    assert freed == 1
    assert [tree.get_branch(branch_id).block_table for branch_id in branch_ids] == [
        [canonical_page],
        [canonical_page],
    ]
    assert pool.refcount(canonical_page) == 2
    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (1, 2, 4)
    assert tree.kv_reuse_ratio == 2.0
    assert_streams_equal(snapshot_streams(pool, tree, branch_ids), before)


def test_identical_partial_pages_never_merge() -> None:
    pool, tree = make_tree(page_size=3)
    k, v = make_tokens(pool, 2)
    child_id = append_independent_pages(tree, k, v)
    tables_before = tree.branches

    assert tree.dedup_scan() == 0

    assert tree.branches == tables_before
    assert tree.used_pages == 2
    assert [
        pool.refcount(tree.get_branch(branch_id).block_table[0])
        for branch_id in (tree.root_id, child_id)
    ] == [1, 1]


@pytest.mark.parametrize(
    ("cache_kind", "layer"),
    [("k", 0), ("v", 0), ("k", 1), ("v", 1)],
)
def test_difference_in_any_layer_k_or_v_prevents_merge(
    cache_kind: str,
    layer: int,
) -> None:
    pool, tree = make_tree(num_layers=2)
    first_k, first_v = make_tokens(pool, pool.config.page_size)
    second_k = first_k.clone()
    second_v = first_v.clone()
    target = second_k if cache_kind == "k" else second_v
    target[layer, 0, 0, 0] += 1
    child_id = append_independent_pages(
        tree,
        first_k,
        first_v,
        second_k,
        second_v,
    )

    assert tree.dedup_scan() == 0

    assert (
        tree.get_branch(tree.root_id).block_table
        != tree.get_branch(child_id).block_table
    )
    assert tree.used_pages == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_supported_dtype_pages_merge_without_changing_bits(dtype: torch.dtype) -> None:
    pool, tree = make_tree(dtype=dtype)
    k, v = make_tokens(pool, pool.config.page_size)
    child_id = append_independent_pages(tree, k, v)
    branch_ids = [tree.root_id, child_id]
    before = snapshot_streams(pool, tree, branch_ids)

    assert tree.dedup_scan() == 1

    assert_streams_equal(snapshot_streams(pool, tree, branch_ids), before)


def float32_page_from_bits(bits: int) -> torch.Tensor:
    signed_bits = bits if bits < 2**31 else bits - 2**32
    return (
        torch.tensor([signed_bits], dtype=torch.int32)
        .view(torch.float32)
        .reshape(1, 1, 1, 1)
    )


@pytest.mark.parametrize(
    ("first_bits", "second_bits", "expected_freed"),
    [
        (0x00000000, 0x80000000, 0),
        (0x7FC00001, 0x7FC00002, 0),
        (0x7FC00001, 0x7FC00001, 1),
    ],
)
def test_raw_float_bits_control_identity(
    first_bits: int,
    second_bits: int,
    expected_freed: int,
) -> None:
    pool, tree = make_tree(
        page_size=1,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
    )
    first_k = float32_page_from_bits(first_bits)
    second_k = float32_page_from_bits(second_bits)
    v = torch.ones_like(first_k)
    child_id = append_independent_pages(tree, first_k, v, second_k, v)
    branch_ids = [tree.root_id, child_id]
    before = snapshot_streams(pool, tree, branch_ids)

    assert tree.dedup_scan() == expected_freed

    assert tree.used_pages == 2 - expected_freed
    assert_streams_equal(snapshot_streams(pool, tree, branch_ids), before)


def test_hash_collision_only_merges_exact_raw_byte_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, tree = make_tree(page_size=2, num_layers=1)
    matching_child = tree.fork(tree.root_id)
    distinct_child = tree.fork(tree.root_id)
    matching_k, matching_v = make_tokens(pool, 2)
    distinct_k, distinct_v = make_tokens(pool, 2, offset=100)
    tree.append_tokens(tree.root_id, matching_k, matching_v)
    tree.append_tokens(matching_child, matching_k, matching_v)
    tree.append_tokens(distinct_child, distinct_k, distinct_v)

    class CollidingDigest:
        def digest(self) -> bytes:
            return b"forced-collision"

    monkeypatch.setattr(hashlib, "sha256", lambda _content: CollidingDigest())

    assert tree.dedup_scan() == 1

    root_page = tree.get_branch(tree.root_id).block_table[0]
    assert tree.get_branch(matching_child).block_table == [root_page]
    assert tree.get_branch(distinct_child).block_table != [root_page]
    assert pool.refcount(root_page) == 2
    assert tree.used_pages == 2


def test_repeated_duplicates_preserve_refcounts_and_lifecycle() -> None:
    pool, tree = make_tree(capacity=8, page_size=2, num_layers=1)
    child_id = tree.fork(tree.root_id)
    page_k, page_v = make_tokens(pool, 2)
    repeated_k = torch.cat((page_k, page_k), dim=1)
    repeated_v = torch.cat((page_v, page_v), dim=1)
    tree.append_tokens(tree.root_id, repeated_k, repeated_v)
    tree.append_tokens(child_id, repeated_k, repeated_v)
    branch_ids = [tree.root_id, child_id]
    before = snapshot_streams(pool, tree, branch_ids)

    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (4, 8, 8)
    assert tree.kv_reuse_ratio == 1.0
    assert tree.dedup_scan() == 3

    canonical_page = tree.get_branch(tree.root_id).block_table[0]
    assert tree.get_branch(tree.root_id).block_table == [canonical_page, canonical_page]
    assert tree.get_branch(child_id).block_table == [canonical_page, canonical_page]
    assert pool.refcount(canonical_page) == 4
    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (1, 2, 8)
    assert tree.kv_reuse_ratio == 4.0
    assert_streams_equal(snapshot_streams(pool, tree, branch_ids), before)

    assert pool.dedup_scan() == 0
    assert pool.refcount(canonical_page) == 4
    fork_id = tree.fork(tree.root_id)
    assert pool.refcount(canonical_page) == 6
    assert tree.logical_tokens == 12
    tree.prune(fork_id)
    assert pool.refcount(canonical_page) == 4
    tree.prune(child_id)
    assert pool.refcount(canonical_page) == 2
    tree.prune(tree.root_id)
    assert pool.refcount(canonical_page) == 0
    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (0, 0, 0)
    assert tree.kv_reuse_ratio == 0.0


def test_corrupt_refcount_is_rejected_before_any_dedup_mutation() -> None:
    pool, tree = make_tree()
    k, v = make_tokens(pool, pool.config.page_size)
    child_id = append_independent_pages(tree, k, v)
    corrupt_page = tree.get_branch(child_id).block_table[0]
    pool._retain(corrupt_page)
    branches_before = tree.branches
    stats_before = pool.stats
    refcounts_before = [
        pool.refcount(page_id) for page_id in range(pool.config.capacity)
    ]
    cache_before = [raw_bytes(cache) for cache in (*pool.k_cache, *pool.v_cache)]

    with pytest.raises(KVInvariantError, match="refcount"):
        tree.dedup_scan()

    assert tree.branches == branches_before
    assert pool.stats == stats_before
    assert [pool.refcount(page_id) for page_id in range(pool.config.capacity)] == (
        refcounts_before
    )
    for actual, expected in zip(
        (*pool.k_cache, *pool.v_cache), cache_before, strict=True
    ):
        assert torch.equal(raw_bytes(actual), expected)


def test_dedup_reclaim_skips_zero_hook_and_reallocation_zeroes_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, tree = make_tree()
    k, v = make_tokens(pool, pool.config.page_size)
    child_id = append_independent_pages(tree, k, v)
    original_pages = [
        tree.get_branch(branch_id).block_table[0]
        for branch_id in (tree.root_id, child_id)
    ]
    canonical_page = min(original_pages)
    duplicate_page = max(original_pages)
    original_zero_page = pool._zero_page

    def fail_if_zeroed(_page_id: int) -> None:
        raise RuntimeError("dedup reclaim attempted device zeroing")

    monkeypatch.setattr(pool, "_zero_page", fail_if_zeroed)

    assert tree.dedup_scan() == 1
    assert tree.get_branch(tree.root_id).block_table == [canonical_page]
    assert tree.get_branch(child_id).block_table == [canonical_page]
    assert [pool.refcount(page_id) for page_id in original_pages] == [2, 0]
    assert pool.used_pages == 1

    monkeypatch.setattr(pool, "_zero_page", original_zero_page)
    assert pool.alloc_page() == duplicate_page
    for cache in (*pool.k_cache, *pool.v_cache):
        assert torch.count_nonzero(cache[duplicate_page]).item() == 0


def test_pool_binding_is_required_unique_and_starts_empty() -> None:
    config = KVPoolConfig(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        capacity=2,
        page_size=1,
    )
    unbound_pool = PagedKVPool(config)
    with pytest.raises(KVInvariantError, match="not bound"):
        unbound_pool.dedup_scan()

    tree = TreeState(unbound_pool)
    with pytest.raises(KVInvariantError, match="already bound"):
        TreeState(unbound_pool)
    assert tree.branches

    nonempty_pool = PagedKVPool(config)
    nonempty_pool.alloc_page()
    with pytest.raises(KVInvariantError, match="empty"):
        TreeState(nonempty_pool)


def test_pool_binding_is_weak_and_only_empty_stale_pool_can_rebind() -> None:
    pool, tree = make_tree()
    tree_reference = weakref.ref(tree)
    del tree
    gc.collect()

    assert tree_reference() is None
    with pytest.raises(KVInvariantError, match="not bound"):
        pool.dedup_scan()
    replacement = TreeState(pool)
    assert pool.dedup_scan() == 0
    del replacement

    nonempty_pool, nonempty_tree = make_tree()
    k, v = make_tokens(nonempty_pool, nonempty_pool.config.page_size)
    nonempty_tree.append_tokens(nonempty_tree.root_id, k, v)
    nonempty_tree_reference = weakref.ref(nonempty_tree)
    del nonempty_tree
    gc.collect()

    assert nonempty_tree_reference() is None
    with pytest.raises(KVInvariantError, match="not bound"):
        nonempty_pool.dedup_scan()
    with pytest.raises(KVInvariantError, match="empty"):
        TreeState(nonempty_pool)
