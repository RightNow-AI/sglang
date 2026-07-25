"""Branch-aware attention kernels for AutoTree's paged Tree-KV layout."""

from .dispatch import tree_attention_decode
from .reference import reference_tree_attention_decode
from .triton_kernel import TRITON_AVAILABLE, triton_tree_attention_decode


__all__ = [
    "TRITON_AVAILABLE",
    "reference_tree_attention_decode",
    "tree_attention_decode",
    "triton_tree_attention_decode",
]
