"""autotree-core: Tree-KV engine for LLM test-time compute."""

from .kv import (
    PAGE_SIZE,
    Branch,
    BranchHasChildrenError,
    KVCapacityError,
    KVError,
    KVInvariantError,
    KVPoolConfig,
    KVStats,
    PagedKVPool,
    TreeState,
    gather_branch_kv,
)

__version__ = "0.1.0"

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
    "__version__",
    "gather_branch_kv",
]
