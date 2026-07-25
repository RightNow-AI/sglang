"""Command-line entry point for autotree-serve."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

import uvicorn

from .app import create_app
from .engine import DeterministicEngine
from .runner import DEFAULT_MAX_CONCURRENT_REQUESTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotree")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser(
        "serve",
        help="Run the OpenAI-compatible API server.",
        description=(
            "Run autotree-serve. 'deterministic' does not serve real model weights; "
            "it is a seeded toy generator. "
            "'treekv' loads a real HuggingFace model through the CPU Tree-KV demo engine."
        ),
    )
    serve.add_argument(
        "--model",
        default="gpt2",
        help=(
            "HuggingFace model identifier for --engine treekv (default: gpt2). "
            "The deterministic engine always exposes deterministic-demo."
        ),
    )
    serve.add_argument(
        "--engine",
        default="deterministic",
        help=(
            "Engine implementation. 'deterministic' is a seeded toy generator; "
            "'treekv' uses the Rust scheduler and real HuggingFace weights on CPU."
        ),
    )
    serve.add_argument(
        "--kv-pages",
        type=_positive_int,
        default=None,
        help=(
            "Explicit Tree-KV page limit for --engine treekv. By default it is "
            "derived from model context length and --kv-branch-headroom."
        ),
    )
    serve.add_argument(
        "--kv-branch-headroom",
        type=_branch_headroom,
        default=1.5,
        help=(
            "Multiplier applied to context-window pages when deriving the Tree-KV "
            "limit (default: 1.5)."
        ),
    )
    serve.add_argument(
        "--device",
        default="cpu",
        help="torch device for --engine treekv, e.g. cpu or cuda (default: cpu)",
    )
    serve.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="model dtype for --engine treekv (default: float32)",
    )
    serve.add_argument(
        "--max-concurrent-requests",
        type=_positive_int,
        default=DEFAULT_MAX_CONCURRENT_REQUESTS,
        help=(
            "Maximum generation requests allowed in flight before later requests "
            f"queue (default: {DEFAULT_MAX_CONCURRENT_REQUESTS})."
        ),
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.engine == "treekv":
        try:
            engine = _load_treekv_engine(
                args.model,
                kv_pages=args.kv_pages,
                kv_branch_headroom=args.kv_branch_headroom,
                device=args.device,
                dtype=args.dtype,
            )
        except Exception as error:
            print(
                f"Failed to load Tree-KV model {args.model!r}: {error}",
                file=sys.stderr,
            )
            raise SystemExit(2) from error
    elif args.engine == "deterministic":
        engine = DeterministicEngine()
    else:
        print(f"Unknown engine '{args.engine}'. No fallback was selected.", file=sys.stderr)
        raise SystemExit(2)

    uvicorn.run(
        create_app(
            engine,
            max_concurrent_requests=args.max_concurrent_requests,
        ),
        host=args.host,
        port=args.port,
    )


def _load_treekv_engine(
    model_id: str,
    *,
    kv_pages: int | None,
    kv_branch_headroom: float,
    device: str = "cpu",
    dtype: str = "float32",
):
    from autotree_core.engine import TreeKVEngine

    return TreeKVEngine(
        model_id=model_id,
        kv_pages=kv_pages,
        kv_branch_headroom=kv_branch_headroom,
        device=device,
        dtype=dtype,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _branch_headroom(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 1.0:
        raise argparse.ArgumentTypeError("must be finite and at least 1.0")
    return parsed


if __name__ == "__main__":
    main()
