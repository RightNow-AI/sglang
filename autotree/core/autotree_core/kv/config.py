"""Configuration for the paged KV cache pool."""

from dataclasses import dataclass

import torch

PAGE_SIZE = 16

_SUPPORTED_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})


@dataclass(frozen=True, slots=True)
class KVPoolConfig:
    """Shape, capacity, and placement of a fixed-size paged KV cache."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    capacity: int
    page_size: int = PAGE_SIZE
    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        for field_name in (
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "capacity",
            "page_size",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        if (
            not isinstance(self.dtype, torch.dtype)
            or self.dtype not in _SUPPORTED_DTYPES
        ):
            raise ValueError(
                "dtype must be torch.float32, torch.float16, or torch.bfloat16"
            )

        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"device is invalid: {self.device!r}") from error
        object.__setattr__(self, "device", device)
