"""Configuration for HuggingFace-backed Tree-KV model execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ModelExecutorConfig:
    """Model-loading and Tree-KV allocation settings."""

    model_id: str = "gpt2"
    revision: str | None = None
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32
    page_size: int = 4
    capacity_pages: int = 64
    trust_remote_code: bool = False
    local_files_only: bool = False
    attn_implementation: str = "eager"

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.revision is not None and not isinstance(self.revision, str):
            raise TypeError("revision must be a string or None")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be torch.float16, torch.bfloat16, or fp32")
        for field_name in ("page_size", "capacity_pages"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"device is invalid: {self.device!r}") from error
        if device.type == "cuda" and device.index is None and torch.cuda.is_available():
            # Canonicalize to an indexed device: model tensors report cuda:0,
            # and torch.device("cuda") != torch.device("cuda:0") in comparisons.
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)


__all__ = ["ModelExecutorConfig"]
