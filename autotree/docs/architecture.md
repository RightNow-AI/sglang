# Architecture

<!-- markdownlint-disable MD013 -->

AutoTree's current implementation is a CPU-testable vertical slice of the
larger GPU serving blueprint. The code is split by contracts so GPU execution
can replace reference paths without changing the tree model or wire surface.

| Component | Location | What exists today |
| --- | --- | --- |
| KV state | `core/autotree_core/kv/` | Fixed-size paged K/V storage, copy-on-write branch forks, leaf pruning, full-page content deduplication, gathers, and exact logical/physical accounting |
| Kernels | `core/autotree_core/kernels/` | Pure-PyTorch reference tree attention and dispatch to an import-guarded Triton decode kernel on supported systems |
| Scheduler | `scheduler/` | Rust branch tree, beam/best-first/MCTS policies, token budgets, deterministic RNG, pruning commands, and optional PyO3 bindings |
| Engine | `core/autotree_core/engine/` | Protocol events, a Hugging Face model executor, and the CPU TreeKV orchestration loop |
| Serve | `serve/` | CLI, FastAPI application, OpenAI-style chat/tree completions, SSE branch events, validation, and Prometheus metrics |
| SDK | `sdk/` | Typed HTTP/SSE client, trace validation, rollout batches, and GRPO/RLHF-shaped exports |
| ThoughtBench | `thoughtbench/` | Resumable fixture runner, graders, metrics, schema-validated results, and reports; bundled data is synthetic contract data only |

## Request flow

A tree request enters `autotree-serve`, is validated, and becomes an engine
generation request. The TreeKV engine tokenizes the prompt and runs real model
weights through the model executor. Scheduler events decide when branches
continue, fork, or terminate. Each live branch references its root-to-node KV
path through the paged pool. The server returns the winning completion and a
summary, or emits branch events over SSE while generation is in progress.

The present CPU path favors correctness and contract integration. The Triton
decode implementation is not validated by this Windows flow, and the current
server is not a production SGLang fork, distributed runtime, or large-model GPU
deployment.

## Source of truth

[`core/docs/tree-kv-spec.md`](../core/docs/tree-kv-spec.md) is the normative
contract for KV pages, tree state, tree-attention shapes, scheduler events, and
the CPU/GPU environment boundary. Implementations and documentation should be
reconciled to that specification rather than inventing a second tree model.

The project targets a 3-10x cost reduction at equal accuracy and a 5x gain in
rollout throughput. Both are hypotheses until reproducible benchmark evidence
exists at scale. Measured evidence so far is listed in the README and in
[`first-benchmark.md`](first-benchmark.md).
