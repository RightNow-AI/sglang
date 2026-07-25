"""Public API for AutoTree's paged KV cache."""

from .config import PAGE_SIZE, KVPoolConfig
from .errors import (
    BranchHasChildrenError,
    KVCapacityError,
    KVError,
    KVInvariantError,
)
from .gather import gather_branch_kv
from .pool import KVStats, PagedKVPool
from .tree_state import Branch, TreeState

__all__ = [
    "PAGE_SIZE",
    "Branch",
    "BranchHasChildrenError",
    "KVCapacityError",
    "KVError",
    "KVInvariantError",
    "KVPoolConfig",
    "KVStats",
    "PagedKVPool",
    "TreeState",
    "gather_branch_kv",
]
