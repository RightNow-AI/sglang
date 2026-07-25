# autotree-core

`autotree-core` is AutoTree's device-agnostic Tree-KV data plane. It stores
per-layer key/value tensors in fixed-size pages and provides copy-on-write
branching, content-addressed full-page deduplication, immediate reclaim on
prune, exact memory accounting, and a pure-PyTorch gather path.

The package is CPU-testable on Windows and supports `torch.float32`,
`torch.float16`, and `torch.bfloat16`. Tree-attention kernels are maintained
separately and consume the block tables and gathered tensor shapes produced
here.

## Install for development

From this directory:

```powershell
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Basic lifecycle

```python
import torch

from autotree_core import KVPoolConfig, PagedKVPool, TreeState

config = KVPoolConfig(
    num_layers=2,
    num_kv_heads=4,
    head_dim=64,
    capacity=128,
    page_size=16,
    dtype=torch.float32,
    device="cpu",
)
pool = PagedKVPool(config)
tree = TreeState(pool)

# Shape: [layers, new_tokens, kv_heads, head_dim]
k = torch.randn(2, 8, 4, 64)
v = torch.randn_like(k)
tree.append_tokens(tree.root_id, k, v)

child_id = tree.fork(tree.root_id)
child_k = torch.randn(2, 1, 4, 64)
child_v = torch.randn_like(child_k)
tree.append_token(child_id, child_k[:, 0], child_v[:, 0])

# Shape for one layer: [num_tokens, kv_heads, head_dim]
materialized_k, materialized_v = tree.gather(child_id, layer=0)

# Full, byte-identical pages are redirected to one physical page.
pages_freed = tree.dedup_scan()

# Only leaves may be pruned; references reaching zero are reclaimed now.
tree.prune(child_id)
```

Forking shares every covered page and increments its reference count without
copying page tensors. Appending to a shared partial tail performs copy-on-write
before mutation, so sibling branches keep byte-identical views of their
prefixes. Appending after a shared full page allocates a new boundary page.

`dedup_scan()` considers only full pages. It hashes raw K/V bytes across every
layer, confirms byte equality within each hash bucket, rewrites branch block
tables to the canonical page, and releases duplicate references.

## Accounting

Accounting is available on either `PagedKVPool` or `TreeState`:

- `used_pages`: allocated physical pages.
- `physical_tokens`: `used_pages * page_size` token slots.
- `logical_tokens`: total tokens across all live branches.
- `kv_reuse_ratio`: `logical_tokens / physical_tokens`, or `0.0` for an empty
  pool.
- `pool.stats`: an immutable `KVStats` snapshot including available pages.

Capacity is fixed. Operations that cannot allocate all required pages raise
`KVCapacityError` before changing branch tables, refcounts, accounting, or
cache data. Other invariant violations raise `KVInvariantError`. Pruning a
branch with live children raises `BranchHasChildrenError`.

## Public API

The Tree-KV API is importable from either `autotree_core` or
`autotree_core.kv`:

```python
from autotree_core import (
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
```

## Test

```powershell
uv run pytest tests/kv -q
```
