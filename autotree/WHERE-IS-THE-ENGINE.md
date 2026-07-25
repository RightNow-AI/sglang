# Where the engine lives

Read this before writing code in this repository.

## The production engine is the SGLang fork, not this repo

    repo    : the AutoTree fork of SGLang
    path    : C:\Users\jaber\RightNow-Full\sglang-tree-integration
    branch  : main   (integration/cascade-v1 merged into it 2026-07-25)
    remote  : https://github.com/RightNow-AI/sglang.git  (PUBLIC, see the push rule below)
    engine  : python/sglang/srt/tree/          13 modules
    tests   : test/registered/unit/tree/       129 passing
    e2e gate: bench/e2e/acceptance.py          11/11 against a live server

All serving work goes there. It has real CUDA kernels, continuous batching,
RadixAttention, CUDA graphs, fp8 KV, and multi-GPU, because it is a fork of a
production engine.

## What THIS repo is for

Docs, benchmarks, the SDK, the paper, figures, the leaderboard, and the
measurement ledgers in AGENTS-GOALs. It also holds `core/`, `scheduler/` (Rust)
and `serve/`, which are the ORIGINAL STANDALONE PROTOTYPE.

The prototype is not the production line and cannot become it. It allocates a
`PagedKVPool` per request and runs HuggingFace eager forwards. Its value was
proving mechanisms cheaply before paying to build them in the fork, and it did
that job: execution-time consensus pruning and bounded concurrency were both
proven here first, then implemented independently in the fork.

## Why the two are not merged into one repository

They have no shared git history, so there is nothing to merge. Beyond that:

1. The fork is 7,420 tracked files against this repo's 280. Absorbing it makes
   this repo 26x larger for no benefit.
2. The fork's whole value is staying mergeable with fast-moving upstream SGLang.
   `git fetch origin && git merge origin/main` has to keep working. Folding
   7,420 upstream files into an unrelated repository destroys that permanently.
3. The two codebases share no code. The consensus algorithm exists in both,
   written separately against each codebase's shape: the fork version reuses
   `python/sglang/srt/tree/answers.py`, the prototype version uses its own
   extraction. Copying one into the other would not compile.

Keeping them separate is the correct architecture, not an accident.

## Push rule

The fork's remote is PUBLIC. Mechanism work stays local: commit in the fork,
do not push mechanism branches to `RightNow-AI/sglang`. This has been violated
once before (`bench/pruning-economics`) and had to be deleted remotely.

## Current state, 2026-07-25

Engine, all verified by running the tests rather than taken from a report:

- tree unit tests 91 -> 129
- e2e acceptance 11/11 against a live server (Lambda A10, Qwen2.5-1.5B, fork
  loaded via PYTHONPATH, confirmed as the fork by /v1/tree/memo/stats returning
  200 rather than 404)
- AutoTree OFF proven inert on the hot path, which is the base_url promise
- consensus pruning runs during generation, not only at winner selection
- fixed: branch misattribution across concurrent trees, adaptive width expanding
  once instead of laddering, verify.py discarding per-token values, parent hold
  generating unused tail tokens
- shared-read attention default flipped OFF, it was falsified as 1.4x slower
  than stock fa3 and was defaulting on

Position, measured and not estimated: we do NOT beat vLLM or SGLang on
throughput. Trunk sharing is 1.85x real but stock SGLang already delivers 1.89x,
so the engine-exclusive delta is 1.00x and it failed a preregistered 1.5x bar.
The defensible claim is tree semantics as a first-class primitive plus the 8.9x
gap between majority-vote selection (9,323 tokens per correct answer) and oracle
selection (1,052), which is free and exact in verifiable domains.

Full detail: `AGENTS-GOALs/2026-07-25-saturation-verdict-engine-thesis-dead.md`
and `AGENTS-GOALs/2026-07-25-kimi-k3-production-strategy.md`.
