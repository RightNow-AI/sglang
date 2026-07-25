"""Typed errors raised by the Tree-KV cache pool."""

from collections.abc import Iterable


class KVError(RuntimeError):
    """Base class for KV cache errors."""


class KVCapacityError(KVError):
    """Raised when the pool cannot satisfy a page allocation."""

    def __init__(self, required_pages: int, available_pages: int) -> None:
        self.required_pages = required_pages
        self.available_pages = available_pages
        super().__init__(
            "KV pool capacity exhausted: "
            f"required {required_pages} page(s), "
            f"but only {available_pages} available"
        )


class KVInvariantError(KVError):
    """Raised when an operation would violate pool invariants."""


class BranchHasChildrenError(KVError):
    """Raised when attempting to prune a branch with live children."""

    def __init__(self, branch_id: int, live_child_ids: Iterable[int]) -> None:
        self.branch_id = branch_id
        self.live_child_ids = sorted(live_child_ids)
        super().__init__(f"branch {branch_id} has live children: {self.live_child_ids}")
