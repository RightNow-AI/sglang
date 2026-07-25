"""Fixed-capacity paged storage for per-layer key and value tensors."""

from __future__ import annotations

import heapq
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .config import KVPoolConfig
from .errors import KVCapacityError, KVInvariantError

if TYPE_CHECKING:
    from .tree_state import TreeState


@dataclass(frozen=True, slots=True)
class KVStats:
    """An immutable snapshot of KV pool accounting."""

    used_pages: int
    available_pages: int
    physical_tokens: int
    logical_tokens: int
    kv_reuse_ratio: float


class PagedKVPool:
    """Own fixed-size K/V tensors and recycle reference-counted pages."""

    def __init__(self, config: KVPoolConfig) -> None:
        self.config = config
        cache_shape = (
            config.capacity,
            config.page_size,
            config.num_kv_heads,
            config.head_dim,
        )
        tensor_options = {"dtype": config.dtype, "device": config.device}
        self.k_cache = tuple(
            torch.zeros(cache_shape, **tensor_options) for _ in range(config.num_layers)
        )
        self.v_cache = tuple(
            torch.zeros(cache_shape, **tensor_options) for _ in range(config.num_layers)
        )
        self._refcounts = [0] * config.capacity
        self._valid_lengths = [0] * config.capacity
        self._free_pages = list(range(config.capacity))
        self._logical_tokens = 0
        self._bound_tree_state: weakref.ReferenceType[TreeState] | None = None

    @property
    def used_pages(self) -> int:
        """Number of currently allocated physical pages."""
        return self.config.capacity - len(self._free_pages)

    @property
    def available_pages(self) -> int:
        """Number of pages available for immediate allocation."""
        return len(self._free_pages)

    @property
    def physical_tokens(self) -> int:
        """Physical token slots consumed by allocated pages."""
        return self.used_pages * self.config.page_size

    @property
    def logical_tokens(self) -> int:
        """Logical tokens represented by the future tree state."""
        return self._logical_tokens

    @property
    def kv_reuse_ratio(self) -> float:
        """Logical tokens divided by occupied physical token slots."""
        physical_tokens = self.physical_tokens
        if physical_tokens == 0:
            return 0.0
        return self.logical_tokens / physical_tokens

    @property
    def stats(self) -> KVStats:
        """Return a consistent snapshot of current pool accounting."""
        return KVStats(
            used_pages=self.used_pages,
            available_pages=self.available_pages,
            physical_tokens=self.physical_tokens,
            logical_tokens=self.logical_tokens,
            kv_reuse_ratio=self.kv_reuse_ratio,
        )

    def dedup_scan(self) -> int:
        """Deduplicate full pages owned by this pool's bound tree state."""
        tree_state = (
            self._bound_tree_state() if self._bound_tree_state is not None else None
        )
        if tree_state is None:
            raise KVInvariantError("KV pool is not bound to a TreeState")
        return tree_state.dedup_scan()

    def alloc_page(self) -> int:
        """Allocate the lowest available page with a reference count of one."""
        if not self._free_pages:
            raise KVCapacityError(required_pages=1, available_pages=0)

        page_id = self._free_pages[0]
        if self._refcounts[page_id] != 0:
            raise KVInvariantError(
                f"free page {page_id} has refcount {self._refcounts[page_id]}"
            )

        page_id = heapq.heappop(self._free_pages)
        self._zero_page(page_id)
        self._valid_lengths[page_id] = 0
        self._refcounts[page_id] = 1
        return page_id

    def free(self, page_id: int) -> None:
        """Release one reference and recycle the page immediately at zero."""
        self._require_allocated(page_id)
        if self._refcounts[page_id] > 1:
            self._refcounts[page_id] -= 1
            return

        self._valid_lengths[page_id] = 0
        self._refcounts[page_id] = 0
        heapq.heappush(self._free_pages, page_id)

    def refcount(self, page_id: int) -> int:
        """Return the current reference count for a page."""
        self._validate_page_id(page_id)
        return self._refcounts[page_id]

    def _retain(self, page_id: int) -> None:
        """Add a reference to an allocated page for a future tree branch."""
        self._require_allocated(page_id)
        self._refcounts[page_id] += 1

    def _adjust_logical_tokens(self, delta: int) -> None:
        """Adjust logical-token accounting without allowing it below zero."""
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise KVInvariantError("logical token delta must be an integer")
        new_value = self._logical_tokens + delta
        if new_value < 0:
            raise KVInvariantError("logical token count cannot be negative")
        self._logical_tokens = new_value

    def _bind_tree_state(self, tree_state: TreeState) -> None:
        """Bind the pool to its sole tree state before either owns pages."""
        if self._bound_tree_state is not None and self._bound_tree_state() is not None:
            raise KVInvariantError("KV pool is already bound to a TreeState")
        if (
            self.used_pages != 0
            or self.logical_tokens != 0
            or any(self._refcounts)
            or any(self._valid_lengths)
        ):
            raise KVInvariantError("TreeState requires an empty KV pool")
        self._bound_tree_state = weakref.ref(tree_state)

    def _require_capacity(self, required_pages: int) -> None:
        """Validate that a multi-page operation can allocate atomically."""
        if (
            isinstance(required_pages, bool)
            or not isinstance(required_pages, int)
            or required_pages < 0
        ):
            raise KVInvariantError("required page count must be a non-negative integer")
        if required_pages > self.available_pages:
            raise KVCapacityError(
                required_pages=required_pages,
                available_pages=self.available_pages,
            )

    def _copy_page(self, page_id: int) -> int:
        """Allocate and byte-copy one page across every K/V layer."""
        self._require_allocated(page_id)
        copied_page_id = self.alloc_page()
        try:
            for cache in (*self.k_cache, *self.v_cache):
                cache[copied_page_id].copy_(cache[page_id])
            self._valid_lengths[copied_page_id] = self._valid_lengths[page_id]
        except Exception:
            self.free(copied_page_id)
            raise
        return copied_page_id

    def _page_length(self, page_id: int) -> int:
        """Return the number of valid token slots in an allocated page."""
        self._require_allocated(page_id)
        return self._valid_lengths[page_id]

    def _set_page_length(self, page_id: int, length: int) -> None:
        """Set the valid-token length for an allocated page."""
        self._require_allocated(page_id)
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or length < 0
            or length > self.config.page_size
        ):
            raise KVInvariantError(
                f"page length must be between 0 and {self.config.page_size}"
            )
        self._valid_lengths[page_id] = length

    def _validate_page_id(self, page_id: int) -> None:
        if (
            isinstance(page_id, bool)
            or not isinstance(page_id, int)
            or page_id < 0
            or page_id >= self.config.capacity
        ):
            raise KVInvariantError(f"invalid page id: {page_id!r}")

    def _require_allocated(self, page_id: int) -> None:
        self._validate_page_id(page_id)
        if self._refcounts[page_id] == 0:
            raise KVInvariantError(f"page {page_id} is not allocated")

    def _zero_page(self, page_id: int) -> None:
        for cache in (*self.k_cache, *self.v_cache):
            cache[page_id].zero_()
