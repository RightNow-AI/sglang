# AutoTree engine: tree execution on the SGLang runtime

Phase-1 goal: a single production-speed engine serving `/v1/tree/completions`
with fork-at-end-of-prefill, mid-flight pruning with instant page reclaim, and
hard per-tree token budgets, driven by the AutoTree Rust scheduler
(beam / best-first / MCTS). Mid-token forking and tree-mask decode kernels are
phase 2.

## Design anchors (from runtime recon, file:line verified)

1. Prefix sharing rides the radix cache. Fork = insert the parent sequence as
   a locked tree node (`cache_unfinished_req`, radix_cache.py:490), then N
   branch requests whose `prefix_indices` point at that node, protected by
   `inc_lock_ref` (radix_cache.py:594). No KV copies.
2. Never free shared pages directly. The allocator free list has no refcount
   (allocator/token.py:66, paged.py:261). Branch prune frees only its own
   suffix via the normal finish path (`release_kv_cache`, mem_cache/common.py:131)
   plus `dec_lock_ref`. Double-free of a shared page is silent corruption:
   every lifetime decision routes through the radix lock.
3. Fork and prune happen in the scheduler, not the tokenizer manager. The
   generated tokens live only in `Req.output_ids`. Branch spawn uses
   `_add_request_to_queue` (scheduler.py:2391); prune uses the `to_finish`
   flag exactly like `abort_request` (scheduler.py:4030).
4. Fork and prune land on batch boundaries only, to stay clear of the overlap
   scheduler's in-flight window (scheduler.py:1554, allocator backup/restore).
5. Per-tree budgets extend the `PrefillAdder` precedent (schedule_policy.py:441)
   with a decode-side counter keyed by tree id.
6. Policy decisions come from the `autotree_scheduler` Rust crate (PyO3):
   the engine feeds per-branch token events, the crate returns
   continue / kill / finalize commands under the tree budget.

## Phase 1 deliverables

- `TreeGenerateReqInput` and tree metadata plumbing (io_struct, Req fields)
- Scheduler tree module: parent prefill, branch spawn, prune, budget, winner
- `/v1/tree/completions`: protocol models, serving object, route, streaming
  envelope with per-branch events, per-branch usage aggregation
- CPU unit tests mirroring `test/registered/unit` patterns (simulated radix
  cache, mock allocators) for fork lock accounting, prune reclaim, budgets
- Conformance: the AutoTree repo e2e suite runs against this engine on a GPU
  box and must pass the same wire contract

## Phase 2 (not now)

- Fork at arbitrary decode token (page-aligned split of live sequences)
- Tree-mask batched decode borrowing EAGLE mask kernels (eagle_utils.py:140)
- Content-addressed dedup across sibling suffixes

## Linux validation, 2026-07-20 (Lambda A10, lmsysorg/sglang:latest container)

- tree suite: 7/7 pass on the fork-proper environment; 6/7 in the container
  overlay, where the one failure is a known base-version artifact (the test
  drives fork-HEAD `cache_finished_req(kv_len_to_handle=...)`, absent in the
  container's 0.5.15 base). Not a fork defect.
- tree_api suite: 7/7 pass in container (needs pytest-asyncio).
- No-breakage check: container radix suite 31 passed plus 61 subtests, zero
  failures with the fork overlaid.
- Engine boots and generates with fork code overlaid (Qwen2.5-0.5B, CUDA
  graphs captured, correct output).

Environment recipe that works: official container + additive overlay of
srt/tree and the two openai files; or fork-source install with
torch==2.11.0 (cu-matched build), transformers==5.12.1, sgl-kernel wheel
from sgl-project releases (arch-matched), flashinfer-python==0.6.14 plus
flashinfer-jit-cache==0.6.14 from flashinfer.ai/whl/<cu>, outlines==0.1.11,
numpy<2, gguf, pytest, pytest-asyncio.

Remaining for phase-1 runtime: scheduler.py splice of the committed bridge,
tokenizer-manager fan-in, endpoint smoke over HTTP, wire-contract conformance
with the AutoTree SDK.

## Runtime splice status, 2026-07-20 (GPU, container SGLang 0.5.15)

PROVEN on an A10: the /v1/tree/completions endpoint reaches the scheduler,
recognizes tree requests, and FORKS sibling branches mid-execution that SHARE
the parent's prefix KV via the radix cache. Log evidence, beam-4 request:

  [tree] request ... forked 3 sibling branches after prefill
  Prefill batch, #new-seq: 3, #new-token: 3, #cached-token: 123

Three branches prefilled with 3 new tokens total while sharing 123 cached
prefix tokens - mid-execution fork with prefix-KV reuse, running in the
production scheduler. This is the core mechanism.

REMAINING (focused hardening lane, not a quick patch): forked branches are
currently built by hand-constructing Req objects, which do not inherit the
field-type invariants SGLang's real intake path establishes (origin_input_ids
as list vs output_ids as array; several `origin_input_ids + output_ids` concat
sites across schedule_batch.py assume matching types set during init). The
correct fix is to spawn branches by routing a TokenizedGenerateReqInput per
branch (prompt = parent prompt + tokens-generated-so-far) through
handle_generate_request, so the real request-init path sets every invariant and
the radix cache shares the prefix automatically. Then: prune-emit via to_finish,
TreeResult over the customized_info channel, overlap-scheduler safety, SDK
conformance. The splice mechanics (dispatch entry, hooks, HTTP route,
serving fallback) are done and committed; only branch construction needs the
intake-path rewrite.
