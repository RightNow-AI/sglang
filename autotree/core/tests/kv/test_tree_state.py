import pytest
import torch

from autotree_core.kv import (
    Branch,
    BranchHasChildrenError,
    KVCapacityError,
    KVPoolConfig,
    PagedKVPool,
    TreeState,
)


def make_tree(
    *,
    capacity: int = 8,
    page_size: int = 2,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 3,
) -> tuple[PagedKVPool, TreeState]:
    pool = PagedKVPool(
        KVPoolConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            capacity=capacity,
            page_size=page_size,
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
    k = torch.arange(
        offset,
        offset + count,
        dtype=config.dtype,
        device=config.device,
    ).reshape(shape)
    v = (k + 10_000).clone()
    return k, v


def clone_caches(pool: PagedKVPool) -> list[torch.Tensor]:
    return [cache.clone() for cache in (*pool.k_cache, *pool.v_cache)]


def assert_caches_equal(
    pool: PagedKVPool,
    expected: list[torch.Tensor],
) -> None:
    for actual, saved in zip((*pool.k_cache, *pool.v_cache), expected):
        assert torch.equal(actual, saved)


def test_root_and_branch_access_are_detached_slotted_snapshots() -> None:
    _, tree = make_tree()

    assert tree.root_id == 0
    assert tree.get_branch(tree.root_id) == Branch(
        branch_id=0,
        parent_id=None,
        num_tokens=0,
        block_table=[],
    )
    assert not hasattr(tree.get_branch(0), "__dict__")

    root_snapshot = tree.get_branch(0)
    root_snapshot.num_tokens = 99
    root_snapshot.block_table.append(7)
    branches_snapshot = tree.branches
    branches_snapshot[0].block_table.append(8)
    branches_snapshot.clear()

    assert tree.get_branch(0) == Branch(0, None, 0, [])
    with pytest.raises(KeyError):
        tree.get_branch(999)


def test_append_tokens_spans_pages_and_gathers_every_layer_exactly() -> None:
    pool, tree = make_tree(page_size=2)
    k, v = make_tokens(pool, 5)

    tree.append_tokens(tree.root_id, k, v)

    root = tree.get_branch(tree.root_id)
    assert root.num_tokens == 5
    assert len(root.block_table) == 3
    assert [pool._page_length(page_id) for page_id in root.block_table] == [2, 2, 1]
    assert tree.used_pages == 3
    assert tree.physical_tokens == 6
    assert tree.logical_tokens == 5
    assert tree.kv_reuse_ratio == pytest.approx(5 / 6)
    for layer in range(pool.config.num_layers):
        actual_k, actual_v = tree.gather(tree.root_id, layer=layer)
        assert torch.equal(actual_k, k[layer])
        assert torch.equal(actual_v, v[layer])


def test_append_token_accepts_one_token_without_a_batch_axis() -> None:
    pool, tree = make_tree()
    k, v = make_tokens(pool, 1)

    tree.append_token(tree.root_id, k[:, 0], v[:, 0])

    actual_k, actual_v = tree.gather(tree.root_id, layer=1)
    assert torch.equal(actual_k, k[1])
    assert torch.equal(actual_v, v[1])


def test_fork_shares_pages_and_only_increases_logical_accounting() -> None:
    pool, tree = make_tree(page_size=2)
    k, v = make_tokens(pool, 3)
    tree.append_tokens(tree.root_id, k, v)
    root_before = tree.get_branch(tree.root_id)
    caches_before = clone_caches(pool)

    child_id = tree.fork(tree.root_id)

    child = tree.get_branch(child_id)
    assert child.parent_id == tree.root_id
    assert child.num_tokens == root_before.num_tokens
    assert child.block_table == root_before.block_table
    assert tree.used_pages == 2
    assert tree.physical_tokens == 4
    assert tree.logical_tokens == 6
    assert tree.kv_reuse_ratio == pytest.approx(1.5)
    assert [pool.refcount(page_id) for page_id in child.block_table] == [2, 2]
    assert_caches_equal(pool, caches_before)


def test_two_siblings_diverge_with_partial_tail_copy_on_write() -> None:
    pool, tree = make_tree(page_size=4)
    prefix_k, prefix_v = make_tokens(pool, 1)
    tree.append_tokens(tree.root_id, prefix_k, prefix_v)
    left_id = tree.fork(tree.root_id)
    right_id = tree.fork(tree.root_id)
    left_k, left_v = make_tokens(pool, 2, offset=100)
    right_k, right_v = make_tokens(pool, 2, offset=500)

    tree.append_tokens(left_id, left_k, left_v)
    tree.append_tokens(right_id, right_k, right_v)

    root = tree.get_branch(tree.root_id)
    left = tree.get_branch(left_id)
    right = tree.get_branch(right_id)
    assert len({root.block_table[-1], left.block_table[-1], right.block_table[-1]}) == 3
    assert [
        pool.refcount(page_id)
        for page_id in (
            root.block_table[-1],
            left.block_table[-1],
            right.block_table[-1],
        )
    ] == [1, 1, 1]
    assert tree.used_pages == 3
    assert tree.logical_tokens == 7

    for layer in range(pool.config.num_layers):
        root_actual = tree.gather(tree.root_id, layer=layer)
        left_actual = tree.gather(left_id, layer=layer)
        right_actual = tree.gather(right_id, layer=layer)
        assert torch.equal(root_actual[0], prefix_k[layer])
        assert torch.equal(root_actual[1], prefix_v[layer])
        assert torch.equal(left_actual[0], torch.cat((prefix_k[layer], left_k[layer])))
        assert torch.equal(left_actual[1], torch.cat((prefix_v[layer], left_v[layer])))
        assert torch.equal(
            right_actual[0], torch.cat((prefix_k[layer], right_k[layer]))
        )
        assert torch.equal(
            right_actual[1], torch.cat((prefix_v[layer], right_v[layer]))
        )


def test_append_after_full_shared_page_allocates_boundary_without_cow() -> None:
    pool, tree = make_tree(page_size=2)
    prefix_k, prefix_v = make_tokens(pool, 2)
    tree.append_tokens(tree.root_id, prefix_k, prefix_v)
    child_id = tree.fork(tree.root_id)
    child_k, child_v = make_tokens(pool, 1, offset=100)

    tree.append_token(child_id, child_k[:, 0], child_v[:, 0])

    root = tree.get_branch(tree.root_id)
    child = tree.get_branch(child_id)
    assert root.block_table == child.block_table[:1]
    assert len(child.block_table) == 2
    assert pool.refcount(root.block_table[0]) == 2
    assert pool.refcount(child.block_table[1]) == 1
    assert tree.used_pages == 2
    assert tree.logical_tokens == 5
    child_actual = tree.gather(child_id, layer=0)
    assert torch.equal(child_actual[0], torch.cat((prefix_k[0], child_k[0])))
    assert torch.equal(child_actual[1], torch.cat((prefix_v[0], child_v[0])))


def test_prune_with_live_children_is_atomic_and_reports_sorted_ids() -> None:
    pool, tree = make_tree()
    k, v = make_tokens(pool, 3)
    tree.append_tokens(tree.root_id, k, v)
    first_child = tree.fork(tree.root_id)
    second_child = tree.fork(tree.root_id)
    stats_before = pool.stats
    branches_before = tree.branches
    refcounts_before = [
        pool.refcount(page_id) for page_id in range(pool.config.capacity)
    ]
    caches_before = clone_caches(pool)

    with pytest.raises(BranchHasChildrenError) as error:
        tree.prune(tree.root_id)

    assert error.value.branch_id == tree.root_id
    assert error.value.live_child_ids == sorted([first_child, second_child])
    assert pool.stats == stats_before
    assert tree.branches == branches_before
    assert [pool.refcount(page_id) for page_id in range(pool.config.capacity)] == (
        refcounts_before
    )
    assert_caches_equal(pool, caches_before)


def test_leaf_prune_reclaims_pages_at_last_reference_and_root_can_be_pruned() -> None:
    pool, tree = make_tree(page_size=2)
    k, v = make_tokens(pool, 3)
    tree.append_tokens(tree.root_id, k, v)
    child_id = tree.fork(tree.root_id)

    tree.prune(child_id)
    assert tree.used_pages == 2
    assert tree.logical_tokens == 3
    assert all(
        pool.refcount(page_id) == 1 for page_id in tree.get_branch(0).block_table
    )

    tree.prune(tree.root_id)
    assert tree.branches == {}
    assert tree.used_pages == 0
    assert tree.physical_tokens == 0
    assert tree.logical_tokens == 0
    assert tree.kv_reuse_ratio == 0.0
    with pytest.raises(KeyError):
        tree.get_branch(tree.root_id)


def test_accounting_tracks_append_fork_cow_and_prune_exactly() -> None:
    pool, tree = make_tree(page_size=2)
    initial_k, initial_v = make_tokens(pool, 3)
    tree.append_tokens(tree.root_id, initial_k, initial_v)
    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (2, 4, 3)
    assert tree.kv_reuse_ratio == pytest.approx(3 / 4)

    child_id = tree.fork(tree.root_id)
    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (2, 4, 6)
    assert tree.kv_reuse_ratio == pytest.approx(1.5)

    child_k, child_v = make_tokens(pool, 1, offset=100)
    tree.append_token(child_id, child_k[:, 0], child_v[:, 0])
    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (3, 6, 7)
    assert tree.kv_reuse_ratio == pytest.approx(7 / 6)

    tree.prune(child_id)
    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (2, 4, 3)
    tree.prune(tree.root_id)
    assert (tree.used_pages, tree.physical_tokens, tree.logical_tokens) == (0, 0, 0)


def test_empty_append_is_noop_and_empty_root_gathers_empty_tensors() -> None:
    pool, tree = make_tree()
    empty_k, empty_v = make_tokens(pool, 0)
    stats_before = pool.stats

    tree.append_tokens(tree.root_id, empty_k, empty_v)

    assert pool.stats == stats_before
    assert tree.get_branch(tree.root_id) == Branch(0, None, 0, [])
    actual_k, actual_v = tree.gather(tree.root_id, layer=1)
    expected_shape = (0, pool.config.num_kv_heads, pool.config.head_dim)
    assert actual_k.shape == expected_shape
    assert actual_v.shape == expected_shape
    assert actual_k.dtype is pool.config.dtype
    assert actual_v.dtype is pool.config.dtype
    assert actual_k.device == pool.config.device
    assert actual_v.device == pool.config.device


def test_append_validation_and_unknown_branches_never_mutate_state() -> None:
    pool, tree = make_tree()
    valid_k, valid_v = make_tokens(pool, 1)
    stats_before = pool.stats
    branches_before = tree.branches
    caches_before = clone_caches(pool)

    invalid_calls = [
        lambda: tree.append_token(0, valid_k, valid_v[:, 0]),
        lambda: tree.append_token(0, valid_k[:, 0], valid_v[:, 0, :, :2]),
        lambda: tree.append_tokens(0, valid_k, valid_v.to(torch.float64)),
        lambda: tree.append_tokens(0, valid_k, valid_v.to(device="meta")),
        lambda: tree.append_tokens(0, valid_k[:, :0], valid_v),
    ]
    for invalid_call in invalid_calls:
        with pytest.raises((TypeError, ValueError)):
            invalid_call()
        assert pool.stats == stats_before
        assert tree.branches == branches_before
        assert_caches_equal(pool, caches_before)

    for operation in (
        lambda: tree.get_branch(99),
        lambda: tree.fork(99),
        lambda: tree.prune(99),
        lambda: tree.append_token(99, valid_k[:, 0], valid_v[:, 0]),
        lambda: tree.append_tokens(99, valid_k, valid_v),
        lambda: tree.gather(99),
    ):
        with pytest.raises(KeyError):
            operation()


def test_sparse_append_is_rejected_before_any_mutation() -> None:
    pool, tree = make_tree()
    dense_k, dense_v = make_tokens(pool, 1)
    sparse_k = dense_k.to_sparse()
    sparse_v = dense_v.to_sparse()
    stats_before = pool.stats
    branches_before = tree.branches
    caches_before = clone_caches(pool)

    with pytest.raises(ValueError, match="strided"):
        tree.append_tokens(tree.root_id, sparse_k, sparse_v)

    assert pool.stats == stats_before
    assert tree.branches == branches_before
    assert [pool.refcount(page_id) for page_id in range(pool.config.capacity)] == (
        [0] * pool.config.capacity
    )
    assert_caches_equal(pool, caches_before)


def test_shared_partial_tail_cow_oom_is_atomic() -> None:
    pool, tree = make_tree(capacity=1, page_size=4)
    prefix_k, prefix_v = make_tokens(pool, 1)
    tree.append_tokens(tree.root_id, prefix_k, prefix_v)
    child_id = tree.fork(tree.root_id)
    child_k, child_v = make_tokens(pool, 1, offset=100)
    stats_before = pool.stats
    branches_before = tree.branches
    refcounts_before = [pool.refcount(0)]
    caches_before = clone_caches(pool)

    with pytest.raises(KVCapacityError) as error:
        tree.append_token(child_id, child_k[:, 0], child_v[:, 0])

    assert error.value.required_pages == 1
    assert error.value.available_pages == 0
    assert pool.stats == stats_before
    assert tree.branches == branches_before
    assert [pool.refcount(0)] == refcounts_before
    assert_caches_equal(pool, caches_before)


def test_multi_page_batch_oom_is_atomic_and_reports_total_requirement() -> None:
    pool, tree = make_tree(capacity=2, page_size=2)
    k, v = make_tokens(pool, 5)
    stats_before = pool.stats
    branches_before = tree.branches
    caches_before = clone_caches(pool)

    with pytest.raises(KVCapacityError) as error:
        tree.append_tokens(tree.root_id, k, v)

    assert error.value.required_pages == 3
    assert error.value.available_pages == 2
    assert pool.stats == stats_before
    assert tree.branches == branches_before
    assert [pool.refcount(page_id) for page_id in range(2)] == [0, 0]
    assert_caches_equal(pool, caches_before)


def test_preflight_counts_cow_and_all_boundary_pages_together() -> None:
    pool, tree = make_tree(capacity=2, page_size=2)
    prefix_k, prefix_v = make_tokens(pool, 1)
    tree.append_tokens(tree.root_id, prefix_k, prefix_v)
    child_id = tree.fork(tree.root_id)
    new_k, new_v = make_tokens(pool, 4, offset=100)
    stats_before = pool.stats
    branches_before = tree.branches
    caches_before = clone_caches(pool)

    with pytest.raises(KVCapacityError) as error:
        tree.append_tokens(child_id, new_k, new_v)

    assert error.value.required_pages == 3
    assert error.value.available_pages == 1
    assert pool.stats == stats_before
    assert tree.branches == branches_before
    assert pool.refcount(0) == 2
    assert pool.refcount(1) == 0
    assert_caches_equal(pool, caches_before)
