"""Shared-prefix metadata for tree decode batches."""

from dataclasses import dataclass


@dataclass
class SharedPrefixGroup:
    rids: list[str]
    shared_len: int
    branch_ids: list[int]
