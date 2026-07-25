"""Decode benchmark for branch-count and context-length sweeps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
import time
from typing import Sequence

import torch

from .dispatch import tree_attention_decode
from .triton_kernel import TRITON_AVAILABLE


@dataclass(frozen=True)
class BenchmarkResult:
    branches: int
    context_len: int
    backend: str
    tokens_per_second: float


def _make_inputs(
    *,
    branches: int,
    context_len: int,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if branches < 1 or context_len < 1:
        raise ValueError("branches and context_len must be positive")
    pages_per_branch = ceil(context_len / page_size)
    shared_pages = max(pages_per_branch - 1, 0)
    num_pages = shared_pages + branches
    generator = torch.Generator().manual_seed(branches * 10_000 + context_len)
    cache_shape = (num_pages, page_size, num_kv_heads, head_dim)
    k_cache = torch.randn(cache_shape, generator=generator).to(
        dtype=dtype, device=device
    )
    v_cache = torch.randn(cache_shape, generator=generator).to(
        dtype=dtype, device=device
    )
    q = torch.randn((branches, num_q_heads, head_dim), generator=generator).to(
        dtype=dtype, device=device
    )
    block_tables = torch.full(
        (branches, pages_per_branch + 1), -1, dtype=torch.int32, device=device
    )
    if shared_pages:
        shared_ids = torch.arange(shared_pages, dtype=torch.int32, device=device)
        block_tables[:, :shared_pages] = shared_ids
    block_tables[:, pages_per_branch - 1] = torch.arange(
        shared_pages,
        shared_pages + branches,
        dtype=torch.int32,
        device=device,
    )
    context_lens = torch.full(
        (branches,), context_len, dtype=torch.int32, device=device
    )
    return q, k_cache, v_cache, block_tables, context_lens


def run_benchmark(
    *,
    branch_counts: Sequence[int],
    context_lengths: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    page_size: int = 16,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    warmup: int = 10,
    iterations: int = 100,
) -> list[BenchmarkResult]:
    """Run a decode-token throughput sweep and return one result per shape."""
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations positive")

    results: list[BenchmarkResult] = []
    for branches in branch_counts:
        for context_len in context_lengths:
            inputs = _make_inputs(
                branches=branches,
                context_len=context_len,
                page_size=page_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(warmup):
                tree_attention_decode(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            started = time.perf_counter()
            output = None
            for _ in range(iterations):
                output = tree_attention_decode(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

            if output is None or output.shape != inputs[0].shape:
                raise RuntimeError(
                    "benchmark decode did not produce the expected output"
                )
            if not torch.isfinite(output).all().item():
                raise RuntimeError("benchmark decode produced non-finite output")
            backend = (
                "triton" if device.type == "cuda" and TRITON_AVAILABLE else "reference"
            )
            results.append(
                BenchmarkResult(
                    branches=branches,
                    context_len=context_len,
                    backend=backend,
                    tokens_per_second=branches * iterations / elapsed,
                )
            )
    return results


def _parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in value.split(","))
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError(
            "values must be comma-separated positive integers"
        )
    return parsed


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=_parse_int_list, default=(1, 4, 16, 32))
    parser.add_argument(
        "--contexts", type=_parse_int_list, default=(16, 128, 512, 2048)
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run a small end-to-end CPU sweep regardless of GPU availability",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        device = torch.device("cpu")
        dtype = torch.float32
        branches = (1, 4)
        contexts = (16, 17)
        warmup = 0
        iterations = 1
        num_q_heads, num_kv_heads, head_dim = 4, 1, 8
    else:
        requested_device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else args.device
        )
        if requested_device == "auto":
            requested_device = "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device(requested_device)
        dtype = _dtype_from_name(args.dtype)
        branches = args.branches
        contexts = args.contexts
        warmup = args.warmup
        iterations = args.iterations
        num_q_heads, num_kv_heads, head_dim = 32, 8, 128

    results = run_benchmark(
        branch_counts=branches,
        context_lengths=contexts,
        device=device,
        dtype=dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        warmup=warmup,
        iterations=iterations,
    )
    for result in results:
        print(
            f"backend={result.backend} branches={result.branches} "
            f"context={result.context_len} "
            f"tokens/sec={result.tokens_per_second:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BenchmarkResult", "main", "run_benchmark"]
