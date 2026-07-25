# A100 GPU validation - 2026-07-19

First hardware validation of the AutoTree Tree-KV engine. Everything below was
measured, not simulated. Environment: Lambda Cloud `gpu_1x_a100_sxm4`
(A100 SXM4 40GB), driver 570.148.08 (CUDA 12.8), Python 3.12.13,
torch 2.11.0+cu128, triton 3.6.0, transformers 5.14.1.

> Driver note: torch 2.12+ ships CUDA 13 wheels only. On CUDA 12.8 drivers you
> must pin `torch==2.11.0+cu128` (`--index-url
> https://download.pytorch.org/whl/cu128`) or CUDA is silently unavailable.
> `uv run` re-syncs the environment back to `uv.lock` (which pins the CPU/cu130
> build); invoke `.venv/bin/python -m pytest` directly on GPU boxes.

## Gate results

| Gate | Result |
|---|---|
| Triton tree-attention kernel parity vs CPU reference | 106 passed, 1 skipped |
| Modeling parity, tiny + gpt2, CUDA float32 | 23 passed |
| Modeling parity, Qwen/Qwen3-8B, CUDA bfloat16 | 13 passed |
| Full core suite on GPU | 213 passed, 5 skipped |

Token-level greedy parity vs stock HuggingFace `generate` holds for Qwen3-8B
bf16 on CUDA. Logit-level guarantees are scoped by kernel path - see
"Numerical parity contract" in `tree-kv-spec.md`.

## GPU-only defects found and fixed by this run

1. `ModelExecutorConfig` kept bare `torch.device("cuda")`, which compares
   unequal to the `cuda:0` device real tensors report; the prefill KV guard
   rejected every valid CUDA execution. Fixed by canonicalizing to an indexed
   device at config time (`config.py`).
2. Parity tests built prompts and sampling generators on CPU regardless of
   executor device (hard `RuntimeError` under CUDA). Fixed by making the tests
   device-aware.
3. Logit/KV assertions assumed bitwise equality, which GPU GEMM batch
   non-invariance makes unsound. Replaced with the scoped contract
   (`_assert_matches` / `_assert_cross_kernel_matches` in `test_parity.py`).

Measured cross-kernel divergence (batched Triton tree-attention vs per-branch
decode), Qwen3-8B bf16: max abs diff 0.375 on a logit range of ~44
(rel-L2 2.6%), argmax identical, top-5 overlap 5/5 - reduced-precision
compounding, not a kernel defect.

## Tree-attention decode kernel microbenchmark

`python -m autotree_core.kernels.bench_decode --device cuda --dtype bfloat16
--branches 1,2,4,8,16,32 --contexts 128,512,2048 --iterations 200`

| branches | ctx 128 tok/s | ctx 512 tok/s | ctx 2048 tok/s |
|---|---|---|---|
| 1 | 17,042 | 12,457 | 3,325 |
| 2 | 33,888 | 24,830 | 6,630 |
| 4 | 68,365 | 49,619 | 13,226 |
| 8 | 136,578 | 98,894 | 26,423 |
| 16 | 265,660 | 127,566 | 33,781 |
| 32 | 539,264 | 208,326 | 54,619 |

Branch scaling at ctx 128 is 31.6x for 32 branches (near-linear): the kernel
makes additional reasoning branches nearly free. This is a kernel
microbenchmark on synthetic inputs, not an end-to-end serving benchmark; it
supports the mechanism claim, not the end-to-end 3-10x cost claim, which still
requires the full ThoughtBench protocol on real workloads.
