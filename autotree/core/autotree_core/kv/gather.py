"""Pure-PyTorch materialization of one branch's paged KV path."""

import torch

from .errors import KVInvariantError
from .pool import PagedKVPool


def gather_branch_kv(
    pool: PagedKVPool,
    block_table: list[int],
    num_tokens: int,
    layer: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one layer of a branch without padded tail slots."""
    if not isinstance(pool, PagedKVPool):
        raise TypeError("pool must be a PagedKVPool")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError("layer must be an integer")
    if layer < 0 or layer >= pool.config.num_layers:
        raise ValueError(f"layer must be between 0 and {pool.config.num_layers - 1}")
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
        raise TypeError("num_tokens must be an integer")
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")

    page_size = pool.config.page_size
    required_pages = (num_tokens + page_size - 1) // page_size
    try:
        table_length = len(block_table)
    except TypeError as error:
        raise TypeError("block_table must be a sized sequence of page ids") from error
    if table_length != required_pages:
        raise KVInvariantError(
            "block table does not exactly cover the requested tokens: "
            f"expected {required_pages} page(s), got {table_length}"
        )

    for page_index, page_id in enumerate(block_table):
        pool._require_allocated(page_id)
        required_length = min(page_size, num_tokens - page_index * page_size)
        valid_length = pool._page_length(page_id)
        if valid_length != required_length:
            raise KVInvariantError(
                f"page {page_id} has {valid_length} valid token(s), "
                f"but exactly {required_length} are required"
            )

    output_shape = (0, pool.config.num_kv_heads, pool.config.head_dim)
    if num_tokens == 0:
        return (
            pool.k_cache[layer].new_empty(output_shape),
            pool.v_cache[layer].new_empty(output_shape),
        )

    page_ids = torch.tensor(
        block_table,
        dtype=torch.long,
        device=pool.config.device,
    )
    gathered_k = torch.index_select(pool.k_cache[layer], 0, page_ids)
    gathered_v = torch.index_select(pool.v_cache[layer], 0, page_ids)
    token_shape = (-1, pool.config.num_kv_heads, pool.config.head_dim)
    return (
        gathered_k.reshape(token_shape)[:num_tokens].contiguous(),
        gathered_v.reshape(token_shape)[:num_tokens].contiguous(),
    )
