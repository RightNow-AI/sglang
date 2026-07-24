"""Shared-prefix metadata for tree decode batches."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SharedPrefixGroup:
    rids: list[str]
    shared_len: int
    branch_ids: list[int]


def disable_shared_prefix_for_cuda_graph_replay(forward_batch: Any) -> None:
    """Force CUDA graph replay onto the stock decode attention path.

    Decode graphs capture Python control flow with a dummy ForwardBatch that
    has no request ID list. The shared-prefix branch therefore cannot be part
    of the captured graph. Clear the runtime-only descriptor explicitly so it
    cannot be mistaken for an active optimization during graph replay.
    """
    forward_batch.shared_prefix_groups = None
