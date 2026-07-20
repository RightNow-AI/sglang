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
