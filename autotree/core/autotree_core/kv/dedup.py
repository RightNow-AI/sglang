"""Private content-addressed page deduplication helpers."""

import hashlib
from collections.abc import Iterator, Mapping

import torch

from .pool import PagedKVPool


def _discover_page_redirects(
    pool: PagedKVPool,
    page_occurrences: Mapping[int, int],
) -> dict[int, int]:
    """Plan duplicate-to-canonical redirects without mutating the pool."""
    candidate_page_ids = sorted(
        page_id
        for page_id in page_occurrences
        if pool._page_length(page_id) == pool.config.page_size
    )
    digest_buckets: dict[bytes, list[int]] = {}
    redirects: dict[int, int] = {}

    for page_id in candidate_page_ids:
        bucket = digest_buckets.setdefault(_page_digest(pool, page_id), [])
        for canonical_page_id in bucket:
            if _pages_equal(pool, page_id, canonical_page_id):
                redirects[page_id] = canonical_page_id
                break
        else:
            bucket.append(page_id)

    return redirects


def _page_digest(pool: PagedKVPool, page_id: int) -> bytes:
    with torch.no_grad():
        aggregate_page = torch.stack(tuple(_page_byte_chunks(pool, page_id)))
        host_page = aggregate_page.cpu()
    host_view = memoryview(host_page.numpy()).cast("B")
    return hashlib.sha256(host_view).digest()


def _pages_equal(
    pool: PagedKVPool,
    first_page_id: int,
    second_page_id: int,
) -> bool:
    with torch.no_grad():
        first_chunks = _page_byte_chunks(pool, first_page_id)
        second_chunks = _page_byte_chunks(pool, second_page_id)
        return all(
            torch.equal(first, second)
            for first, second in zip(first_chunks, second_chunks, strict=True)
        )


def _page_byte_chunks(pool: PagedKVPool, page_id: int) -> Iterator[torch.Tensor]:
    for layer in range(pool.config.num_layers):
        for cache in (pool.k_cache[layer], pool.v_cache[layer]):
            yield cache[page_id].detach().contiguous().view(torch.uint8)
