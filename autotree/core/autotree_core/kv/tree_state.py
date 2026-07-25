"""Branch lifecycle and copy-on-write appends for paged KV state."""

from collections import Counter
from dataclasses import dataclass, field

import torch

from .dedup import _discover_page_redirects
from .errors import BranchHasChildrenError, KVInvariantError
from .gather import gather_branch_kv
from .pool import PagedKVPool


@dataclass(slots=True)
class Branch:
    """Detached snapshot of one live branch."""

    branch_id: int
    parent_id: int | None
    num_tokens: int
    block_table: list[int]


@dataclass(slots=True)
class _BranchRecord:
    branch_id: int
    parent_id: int | None
    num_tokens: int
    block_table: list[int]
    child_ids: set[int] = field(default_factory=set)


class TreeState:
    """Own branch topology and logical views over a paged KV pool."""

    __slots__ = ("_branches", "_next_branch_id", "_pool", "__weakref__")

    def __init__(self, pool: PagedKVPool) -> None:
        if not isinstance(pool, PagedKVPool):
            raise TypeError("pool must be a PagedKVPool")
        self._pool = pool
        self._branches = {
            self.root_id: _BranchRecord(
                branch_id=self.root_id,
                parent_id=None,
                num_tokens=0,
                block_table=[],
            )
        }
        self._next_branch_id = self.root_id + 1
        self._pool._bind_tree_state(self)

    @property
    def root_id(self) -> int:
        """Stable id assigned to the initial root branch."""
        return 0

    @property
    def branches(self) -> dict[int, Branch]:
        """Return a fully detached snapshot of every live branch."""
        return {
            branch_id: self._snapshot(record)
            for branch_id, record in self._branches.items()
        }

    @property
    def used_pages(self) -> int:
        """Delegate physical page accounting to the pool."""
        return self._pool.used_pages

    @property
    def logical_tokens(self) -> int:
        """Delegate logical token accounting to the pool."""
        return self._pool.logical_tokens

    @property
    def physical_tokens(self) -> int:
        """Delegate physical token-slot accounting to the pool."""
        return self._pool.physical_tokens

    @property
    def kv_reuse_ratio(self) -> float:
        """Delegate logical-to-physical reuse accounting to the pool."""
        return self._pool.kv_reuse_ratio

    def get_branch(self, branch_id: int) -> Branch:
        """Return a detached snapshot of one live branch."""
        return self._snapshot(self._get_record(branch_id))

    def dedup_scan(self) -> int:
        """Merge byte-identical full pages and return physical pages freed."""
        records = tuple(
            sorted(self._branches.values(), key=lambda record: record.branch_id)
        )
        page_occurrences: Counter[int] = Counter()
        for record in records:
            self._validate_branch_coverage(record)
            page_occurrences.update(record.block_table)

        for page_id in sorted(page_occurrences):
            expected_refcount = page_occurrences[page_id]
            actual_refcount = self._pool.refcount(page_id)
            if actual_refcount != expected_refcount:
                raise KVInvariantError(
                    f"page {page_id} refcount {actual_refcount} does not match "
                    f"{expected_refcount} live block-table occurrence(s)"
                )

        redirects = _discover_page_redirects(self._pool, page_occurrences)
        if not redirects:
            return 0

        for duplicate_page_id in sorted(redirects):
            canonical_page_id = redirects[duplicate_page_id]
            for _ in range(page_occurrences[duplicate_page_id]):
                self._pool._retain(canonical_page_id)

        for record in records:
            record.block_table[:] = [
                redirects.get(page_id, page_id) for page_id in record.block_table
            ]

        for duplicate_page_id in sorted(redirects):
            for _ in range(page_occurrences[duplicate_page_id]):
                self._pool.free(duplicate_page_id)

        return len(redirects)

    def append_token(
        self,
        branch_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Append one token shaped ``[layers, kv_heads, head_dim]``."""
        record = self._get_record(branch_id)
        self._validate_tensor_pair(k, v)
        expected_shape = (
            self._pool.config.num_layers,
            self._pool.config.num_kv_heads,
            self._pool.config.head_dim,
        )
        if tuple(k.shape) != expected_shape:
            raise ValueError(
                f"append_token tensors must have shape {expected_shape}, "
                f"got {tuple(k.shape)}"
            )
        self._append_validated(record, k.unsqueeze(1), v.unsqueeze(1))

    def append_tokens(
        self,
        branch_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Append tokens shaped ``[layers, tokens, kv_heads, head_dim]``."""
        record = self._get_record(branch_id)
        self._validate_tensor_pair(k, v)
        expected_fixed_dimensions = (
            self._pool.config.num_layers,
            self._pool.config.num_kv_heads,
            self._pool.config.head_dim,
        )
        if (
            k.ndim != 4
            or (
                k.shape[0],
                k.shape[2],
                k.shape[3],
            )
            != expected_fixed_dimensions
        ):
            raise ValueError(
                "append_tokens tensors must have shape "
                "[num_layers, num_new_tokens, num_kv_heads, head_dim]"
            )
        self._append_validated(record, k, v)

    def fork(self, branch_id: int) -> int:
        """Create a child that shares every covered page occurrence."""
        source = self._get_record(branch_id)
        self._validate_branch_coverage(source)
        new_branch_id = self._next_branch_id
        for page_id in source.block_table:
            self._pool._retain(page_id)
        self._pool._adjust_logical_tokens(source.num_tokens)
        self._branches[new_branch_id] = _BranchRecord(
            branch_id=new_branch_id,
            parent_id=source.branch_id,
            num_tokens=source.num_tokens,
            block_table=source.block_table.copy(),
        )
        source.child_ids.add(new_branch_id)
        self._next_branch_id += 1
        return new_branch_id

    def prune(self, branch_id: int) -> None:
        """Remove a leaf branch and immediately release its page references."""
        record = self._get_record(branch_id)
        if record.child_ids:
            raise BranchHasChildrenError(record.branch_id, record.child_ids)

        self._validate_branch_coverage(record)
        for page_id in record.block_table:
            self._pool.free(page_id)
        self._pool._adjust_logical_tokens(-record.num_tokens)
        if record.parent_id is not None:
            self._branches[record.parent_id].child_ids.remove(record.branch_id)
        del self._branches[record.branch_id]

    def gather(
        self,
        branch_id: int,
        layer: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize one layer of a branch's complete token path."""
        record = self._get_record(branch_id)
        return gather_branch_kv(
            self._pool,
            record.block_table,
            record.num_tokens,
            layer=layer,
        )

    def _append_validated(
        self,
        record: _BranchRecord,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        num_new_tokens = k.shape[1]
        if num_new_tokens == 0:
            return

        self._validate_branch_coverage(record)
        page_size = self._pool.config.page_size
        current_page_count = len(record.block_table)
        final_page_count = (
            record.num_tokens + num_new_tokens + page_size - 1
        ) // page_size
        boundary_pages = final_page_count - current_page_count
        has_partial_tail = record.num_tokens % page_size != 0
        needs_cow = has_partial_tail and (
            self._pool.refcount(record.block_table[-1]) > 1
        )
        required_pages = boundary_pages + int(needs_cow)
        self._pool._require_capacity(required_pages)

        if needs_cow:
            shared_page_id = record.block_table[-1]
            copied_page_id = self._pool._copy_page(shared_page_id)
            record.block_table[-1] = copied_page_id
            self._pool.free(shared_page_id)

        input_offset = 0
        while input_offset < num_new_tokens:
            page_offset = record.num_tokens % page_size
            if page_offset == 0:
                page_id = self._pool.alloc_page()
                record.block_table.append(page_id)
            else:
                page_id = record.block_table[-1]

            write_count = min(
                page_size - page_offset,
                num_new_tokens - input_offset,
            )
            input_end = input_offset + write_count
            page_end = page_offset + write_count
            with torch.no_grad():
                for layer in range(self._pool.config.num_layers):
                    self._pool.k_cache[layer][page_id, page_offset:page_end].copy_(
                        k[layer, input_offset:input_end]
                    )
                    self._pool.v_cache[layer][page_id, page_offset:page_end].copy_(
                        v[layer, input_offset:input_end]
                    )
            self._pool._set_page_length(page_id, page_end)
            record.num_tokens += write_count
            input_offset = input_end

        self._pool._adjust_logical_tokens(num_new_tokens)

    def _validate_tensor_pair(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError("k and v must be torch.Tensor instances")
        if tuple(k.shape) != tuple(v.shape):
            raise ValueError(
                f"k and v must have exactly matching shapes, got "
                f"{tuple(k.shape)} and {tuple(v.shape)}"
            )
        if k.layout is not torch.strided or v.layout is not torch.strided:
            raise ValueError("k and v must use the strided tensor layout")
        if (
            k.dtype is not self._pool.config.dtype
            or v.dtype is not self._pool.config.dtype
        ):
            raise ValueError(f"k and v must have dtype {self._pool.config.dtype}")
        if k.device != self._pool.config.device or v.device != self._pool.config.device:
            raise ValueError(f"k and v must be on device {self._pool.config.device}")

    def _validate_branch_coverage(self, record: _BranchRecord) -> None:
        page_size = self._pool.config.page_size
        required_pages = (record.num_tokens + page_size - 1) // page_size
        if len(record.block_table) != required_pages:
            raise KVInvariantError(
                f"branch {record.branch_id} block table does not cover its tokens"
            )
        for page_index, page_id in enumerate(record.block_table):
            required_length = min(
                page_size,
                record.num_tokens - page_index * page_size,
            )
            if self._pool._page_length(page_id) != required_length:
                raise KVInvariantError(
                    f"branch {record.branch_id} page {page_id} has invalid length"
                )

    def _get_record(self, branch_id: int) -> _BranchRecord:
        if isinstance(branch_id, bool) or not isinstance(branch_id, int):
            raise KeyError(branch_id)
        try:
            return self._branches[branch_id]
        except KeyError:
            raise KeyError(branch_id) from None

    @staticmethod
    def _snapshot(record: _BranchRecord) -> Branch:
        return Branch(
            branch_id=record.branch_id,
            parent_id=record.parent_id,
            num_tokens=record.num_tokens,
            block_table=record.block_table.copy(),
        )
