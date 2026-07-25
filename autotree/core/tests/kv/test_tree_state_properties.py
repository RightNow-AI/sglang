from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import pytest
import torch
from hypothesis import settings, strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from autotree_core.kv import (
    BranchHasChildrenError,
    KVPoolConfig,
    KVStats,
    PagedKVPool,
    TreeState,
)

NUM_LAYERS = 2
NUM_KV_HEADS = 1
HEAD_DIM = 1
PAGE_SIZE = 2
CAPACITY = 64

# Signed int32 encodings of float32 +0.0, -0.0, and 1.0. Viewing these as
# float32 preserves the exact payload bits, including the sign bit of zero.
FLOAT32_BIT_CODES = (0, -(2**31), 0x3F800000)
FLOAT32_BIT_CODE = st.sampled_from(FLOAT32_BIT_CODES)

LayerCodes = tuple[tuple[int, ...], ...]
PayloadCodes = tuple[LayerCodes, LayerCodes]


@st.composite
def payload_codes(
    draw: st.DrawFn,
    *,
    min_tokens: int,
    max_tokens: int,
) -> PayloadCodes:
    num_tokens = draw(st.integers(min_value=min_tokens, max_value=max_tokens))
    k_codes = tuple(
        tuple(draw(FLOAT32_BIT_CODE) for _ in range(num_tokens))
        for _ in range(NUM_LAYERS)
    )
    v_codes = tuple(
        tuple(draw(FLOAT32_BIT_CODE) for _ in range(num_tokens))
        for _ in range(NUM_LAYERS)
    )
    return k_codes, v_codes


SINGLE_TOKEN_PAYLOAD = payload_codes(min_tokens=1, max_tokens=1)
MULTI_TOKEN_PAYLOAD = payload_codes(min_tokens=2, max_tokens=3)
FULL_PAGE_PAYLOAD = payload_codes(min_tokens=PAGE_SIZE, max_tokens=PAGE_SIZE)


def tensor_from_codes(codes: LayerCodes) -> torch.Tensor:
    num_tokens = len(codes[0])
    encoded = torch.tensor(
        [code for layer_codes in codes for code in layer_codes],
        dtype=torch.int32,
    )
    return encoded.view(torch.float32).reshape(
        NUM_LAYERS,
        num_tokens,
        NUM_KV_HEADS,
        HEAD_DIM,
    )


def tensors_from_payload(payload: PayloadCodes) -> tuple[torch.Tensor, torch.Tensor]:
    k_codes, v_codes = payload
    return tensor_from_codes(k_codes), tensor_from_codes(v_codes)


def raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8).cpu().clone()


def byte_string(tensor: torch.Tensor) -> bytes:
    return raw_bytes(tensor).numpy().tobytes()


@dataclass(slots=True)
class DenseBranch:
    parent_id: int | None
    k: torch.Tensor
    v: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return self.k.shape[1]


class TreeStateProperties(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.pool = PagedKVPool(
            KVPoolConfig(
                num_layers=NUM_LAYERS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM,
                capacity=CAPACITY,
                page_size=PAGE_SIZE,
                dtype=torch.float32,
                device="cpu",
            )
        )
        self.tree = TreeState(self.pool)
        empty = torch.empty(
            (NUM_LAYERS, 0, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float32,
        )
        self.mirror = {
            self.tree.root_id: DenseBranch(
                parent_id=None,
                k=empty.clone(),
                v=empty.clone(),
            )
        }
        self.next_branch_id = self.tree.root_id + 1
        self.positive_dedup_transitions = 0

    @initialize(
        payload=FULL_PAGE_PAYLOAD,
        entrypoint=st.sampled_from(("tree", "pool")),
    )
    def require_positive_duplicate_dedup_transition(
        self,
        payload: PayloadCodes,
        entrypoint: str,
    ) -> None:
        """Start every example by creating and reclaiming a duplicate page."""
        k, v = tensors_from_payload(payload)
        self.tree.append_tokens(self.tree.root_id, k, v)
        self.append_to_mirror(self.tree.root_id, k, v)
        self.tree.append_tokens(self.tree.root_id, k, v)
        self.append_to_mirror(self.tree.root_id, k, v)
        streams_before = tuple(
            tuple(raw_bytes(tensor) for tensor in self.tree.gather(0, layer=layer))
            for layer in range(NUM_LAYERS)
        )
        used_pages_before = self.pool.used_pages

        freed = (
            self.tree.dedup_scan() if entrypoint == "tree" else self.pool.dedup_scan()
        )

        assert used_pages_before == 2
        assert freed == 1
        assert self.pool.used_pages == 1
        assert self.pool.refcount(self.tree.get_branch(0).block_table[0]) == 2
        streams_after = tuple(
            tuple(raw_bytes(tensor) for tensor in self.tree.gather(0, layer=layer))
            for layer in range(NUM_LAYERS)
        )
        for actual_layer, expected_layer in zip(
            streams_after,
            streams_before,
            strict=True,
        ):
            for actual, expected in zip(actual_layer, expected_layer, strict=True):
                assert torch.equal(actual, expected)
        self.positive_dedup_transitions += 1

    def draw_branch_id(self, data: st.DataObject, *, label: str) -> int:
        return data.draw(st.sampled_from(tuple(sorted(self.mirror))), label=label)

    def append_to_mirror(
        self,
        branch_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        branch = self.mirror[branch_id]
        branch.k = torch.cat((branch.k, k.clone()), dim=1)
        branch.v = torch.cat((branch.v, v.clone()), dim=1)

    def child_ids(self) -> set[int]:
        return {
            branch.parent_id
            for branch in self.mirror.values()
            if branch.parent_id is not None
        }

    def legal_leaf_ids(self) -> tuple[int, ...]:
        nonleaf_ids = self.child_ids()
        return tuple(
            branch_id
            for branch_id in sorted(self.mirror)
            if branch_id not in nonleaf_ids
            and not (branch_id == self.tree.root_id and len(self.mirror) == 1)
        )

    def nonleaf_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.child_ids()))

    def aligned_branch_ids(self) -> tuple[int, ...]:
        return tuple(
            branch_id
            for branch_id, branch in sorted(self.mirror.items())
            if branch.num_tokens % PAGE_SIZE == 0
        )

    def full_page_sources(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (branch_id, page_index)
            for branch_id, branch in sorted(self.mirror.items())
            for page_index in range(branch.num_tokens // PAGE_SIZE)
        )

    def page_occurrences(self) -> Counter[int]:
        return Counter(
            page_id
            for branch in self.tree.branches.values()
            for page_id in branch.block_table
        )

    def partial_occurrence_ids(self) -> dict[tuple[int, int], int]:
        branches = self.tree.branches
        return {
            (branch_id, branch.num_tokens // PAGE_SIZE): branches[
                branch_id
            ].block_table[-1]
            for branch_id, branch in self.mirror.items()
            if branch.num_tokens % PAGE_SIZE
        }

    def full_signature_groups(
        self,
    ) -> dict[tuple[bytes, ...], list[tuple[int, int]]]:
        groups: defaultdict[tuple[bytes, ...], list[tuple[int, int]]] = defaultdict(
            list
        )
        for branch_id, branch in self.mirror.items():
            for page_index in range(branch.num_tokens // PAGE_SIZE):
                page_start = page_index * PAGE_SIZE
                page_end = page_start + PAGE_SIZE
                signature = tuple(
                    byte_string(cache[layer, page_start:page_end])
                    for layer in range(NUM_LAYERS)
                    for cache in (branch.k, branch.v)
                )
                groups[signature].append((branch_id, page_index))
        return dict(groups)

    @rule(data=st.data(), payload=SINGLE_TOKEN_PAYLOAD)
    def append_token(self, data: st.DataObject, payload: PayloadCodes) -> None:
        branch_id = self.draw_branch_id(data, label="append_token_branch")
        k, v = tensors_from_payload(payload)

        self.tree.append_token(branch_id, k[:, 0], v[:, 0])
        self.append_to_mirror(branch_id, k, v)

    @rule(data=st.data(), payload=MULTI_TOKEN_PAYLOAD)
    def append_tokens(self, data: st.DataObject, payload: PayloadCodes) -> None:
        branch_id = self.draw_branch_id(data, label="append_tokens_branch")
        k, v = tensors_from_payload(payload)

        self.tree.append_tokens(branch_id, k, v)
        self.append_to_mirror(branch_id, k, v)

    @rule(data=st.data())
    def fork(self, data: st.DataObject) -> None:
        source_id = self.draw_branch_id(data, label="fork_source")
        source = self.mirror[source_id]

        branch_id = self.tree.fork(source_id)

        assert branch_id == self.next_branch_id
        self.mirror[branch_id] = DenseBranch(
            parent_id=source_id,
            k=source.k.clone(),
            v=source.v.clone(),
        )
        self.next_branch_id += 1

    @precondition(lambda self: bool(self.legal_leaf_ids()))
    @rule(data=st.data())
    def prune_leaf(self, data: st.DataObject) -> None:
        branch_id = data.draw(
            st.sampled_from(self.legal_leaf_ids()),
            label="legal_leaf",
        )

        self.tree.prune(branch_id)
        del self.mirror[branch_id]

    @precondition(lambda self: bool(self.nonleaf_ids()))
    @rule(data=st.data())
    def reject_nonleaf_prune_atomically(self, data: st.DataObject) -> None:
        branch_id = data.draw(
            st.sampled_from(self.nonleaf_ids()),
            label="illegal_nonleaf",
        )
        branches_before = self.tree.branches
        stats_before = self.pool.stats
        refcounts_before = tuple(
            self.pool.refcount(page_id) for page_id in range(CAPACITY)
        )
        lengths_before = tuple(self.pool._valid_lengths)
        free_pages_before = tuple(self.pool._free_pages)
        caches_before = tuple(
            raw_bytes(cache) for cache in (*self.pool.k_cache, *self.pool.v_cache)
        )

        with pytest.raises(BranchHasChildrenError):
            self.tree.prune(branch_id)

        assert self.tree.branches == branches_before
        assert self.pool.stats == stats_before
        assert (
            tuple(self.pool.refcount(page_id) for page_id in range(CAPACITY))
            == refcounts_before
        )
        assert tuple(self.pool._valid_lengths) == lengths_before
        assert tuple(self.pool._free_pages) == free_pages_before
        for actual, expected in zip(
            (*self.pool.k_cache, *self.pool.v_cache),
            caches_before,
            strict=True,
        ):
            assert torch.equal(raw_bytes(actual), expected)

    @precondition(
        lambda self: bool(self.full_page_sources()) and bool(self.aligned_branch_ids())
    )
    @rule(data=st.data())
    def append_existing_full_page(self, data: st.DataObject) -> None:
        source_id, source_page_index = data.draw(
            st.sampled_from(self.full_page_sources()),
            label="existing_page_source",
        )
        target_id = data.draw(
            st.sampled_from(self.aligned_branch_ids()),
            label="aligned_page_target",
        )
        source = self.mirror[source_id]
        page_start = source_page_index * PAGE_SIZE
        page_end = page_start + PAGE_SIZE
        page_k = source.k[:, page_start:page_end].clone()
        page_v = source.v[:, page_start:page_end].clone()
        occurrence_ids_before = set(self.page_occurrences())
        used_pages_before = self.pool.used_pages

        self.tree.append_tokens(target_id, page_k, page_v)

        new_page_id = self.tree.get_branch(target_id).block_table[-1]
        assert new_page_id not in occurrence_ids_before
        assert self.pool.used_pages == used_pages_before + 1
        self.append_to_mirror(target_id, page_k, page_v)

    @rule(entrypoint=st.sampled_from(("tree", "pool")))
    def dedup_scan(self, entrypoint: str) -> None:
        partial_ids_before = self.partial_occurrence_ids()
        signature_groups = self.full_signature_groups()
        used_pages_before = self.pool.used_pages

        freed = (
            self.tree.dedup_scan() if entrypoint == "tree" else self.pool.dedup_scan()
        )

        assert freed == used_pages_before - self.pool.used_pages
        branches_after = self.tree.branches
        for occurrence, page_id in partial_ids_before.items():
            branch_id, page_index = occurrence
            assert branches_after[branch_id].block_table[page_index] == page_id
        for occurrences in signature_groups.values():
            physical_ids = {
                branches_after[branch_id].block_table[page_index]
                for branch_id, page_index in occurrences
            }
            assert len(physical_ids) == 1

    @invariant()
    def tree_matches_independent_dense_mirror(self) -> None:
        assert self.positive_dedup_transitions == 1
        branches = self.tree.branches
        assert set(branches) == set(self.mirror)

        occurrences: Counter[int] = Counter()
        coverage_by_page: defaultdict[int, set[int]] = defaultdict(set)
        for branch_id, expected in self.mirror.items():
            actual = branches[branch_id]
            assert actual.branch_id == branch_id
            assert actual.parent_id == expected.parent_id
            assert actual.num_tokens == expected.num_tokens
            assert expected.k.shape == (
                NUM_LAYERS,
                expected.num_tokens,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            assert expected.v.shape == expected.k.shape
            expected_page_count = (expected.num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
            assert len(actual.block_table) == expected_page_count

            for layer in range(NUM_LAYERS):
                actual_k, actual_v = self.tree.gather(branch_id, layer=layer)
                assert actual_k.is_contiguous()
                assert actual_v.is_contiguous()
                assert torch.equal(raw_bytes(actual_k), raw_bytes(expected.k[layer]))
                assert torch.equal(raw_bytes(actual_v), raw_bytes(expected.v[layer]))

            occurrences.update(actual.block_table)
            for page_index, page_id in enumerate(actual.block_table):
                coverage_by_page[page_id].add(
                    min(
                        PAGE_SIZE,
                        expected.num_tokens - page_index * PAGE_SIZE,
                    )
                )

        for page_id in range(CAPACITY):
            assert self.pool.refcount(page_id) == occurrences[page_id]

        expected_used_pages = len(occurrences)
        expected_logical_tokens = sum(
            branch.num_tokens for branch in self.mirror.values()
        )
        expected_physical_tokens = expected_used_pages * PAGE_SIZE
        expected_ratio = (
            expected_logical_tokens / expected_physical_tokens
            if expected_physical_tokens
            else 0.0
        )
        expected_available_pages = CAPACITY - expected_used_pages

        assert self.pool.used_pages == expected_used_pages
        assert self.tree.used_pages == expected_used_pages
        assert self.pool.available_pages == expected_available_pages
        assert self.pool.logical_tokens == expected_logical_tokens
        assert self.tree.logical_tokens == expected_logical_tokens
        assert self.pool.physical_tokens == expected_physical_tokens
        assert self.tree.physical_tokens == expected_physical_tokens
        assert self.pool.kv_reuse_ratio == expected_ratio
        assert self.tree.kv_reuse_ratio == expected_ratio
        assert self.pool.stats == KVStats(
            used_pages=expected_used_pages,
            available_pages=expected_available_pages,
            physical_tokens=expected_physical_tokens,
            logical_tokens=expected_logical_tokens,
            kv_reuse_ratio=expected_ratio,
        )

        for page_id, coverage_lengths in coverage_by_page.items():
            assert len(coverage_lengths) == 1, (
                f"page {page_id} has contradictory coverage {coverage_lengths}"
            )
            assert self.pool._page_length(page_id) == next(iter(coverage_lengths))


TestTreeStateProperties = TreeStateProperties.TestCase
TestTreeStateProperties.settings = settings(
    max_examples=50,
    stateful_step_count=25,
    deadline=None,
)
