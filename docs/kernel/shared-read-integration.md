# Shared-read tree-attention: integration map

Goal: when a tree has B sibling branches sharing a long prefix of length S, make
the decode step read that shared prefix KV **once** instead of B times. Decode is
memory-bandwidth-bound, so this converts the per-step bill from `B*S + W` to
`S + W + B*u` (u = tiny per-branch suffix). Projected ceiling: ~4x at B=8 / ~7.7x
at B=16 at 100k context (bandwidth arithmetic).

## Verdict: this is a WIRING job, not a new kernel. Path (A) is open.

The cascade primitive (Hydragen-style "attend, return log-sum-exp, merge") is
already vendored and already used in this fork's prefill path. The decode path
does not yet use it, but every building block exists.

## Evidence (file:line, python/sglang/srt/layers/attention/flashinfer_backend.py)

- `from flashinfer.cascade import merge_state` (line 90) plus a Triton fallback
  `merge_state_triton` (line 92). `_safe_merge_state(...)` (lines 109-120) picks
  flashinfer vs Triton by a head-count safety check.
- The **prefill path already does a two-level cascade merge** (lines ~1385-1396):
  two `forward_return_lse(...)` passes (a ragged wrapper `o1,s1` and
  `prefill_wrapper_paged` `o2,s2`) merged by `o,_ = _safe_merge_state(o1,s1,o2,s2)`.
  This is exactly the shared/suffix decomposition math (log-sum-exp state merge),
  proven working here for the ragged+paged split.
- `forward_decode(...)` (line 1410) currently uses a single `decode_wrapper`
  (`self.forward_metadata.decode_wrappers[...]`). It does NOT split into
  shared-prefix + suffix. This is the one method to change.

## The integration (path A), concretely

1. **Group detection.** The tree runtime already forks siblings as ordinary Reqs
   that radix-share the parent's prefix (tree_runtime.py `_fork_branches`). The
   scheduler batches them into a decode step. Add a per-batch descriptor that
   marks "these rids share a prefix of length S" (a shared-prefix group). The
   runtime knows the fork point (k tokens) exactly.
2. **Two-level decode.** In `forward_decode`, when the batch carries a
   shared-prefix group, compute:
   - one **shared** attention pass: all branches' decode queries against the
     shared prefix KV, read once (a prefill/append wrapper with a single KV
     range and B query rows), returning `o_shared, s_shared` via
     `forward_return_lse`;
   - per-branch **suffix** attention over each branch's unique tokens, returning
     `o_suf, s_suf`;
   - `o, _ = _safe_merge_state(o_shared, s_shared, o_suf, s_suf)` (reuse the
     existing helper).
3. **Descriptor plumbing.** The shared-prefix group needs kv_indptr/qo_indptr
   describing the shared range once and the B suffix ranges. FlashInfer's
   cascade / MultiLevelCascadeAttentionWrapper is the canonical builder; the
   prefill path's paged/ragged wrappers are the working reference in-repo.

## First file to change

`python/sglang/srt/layers/attention/flashinfer_backend.py`, `forward_decode`
(line 1410) plus its metadata builder (the decode wrapper setup), driven by a
shared-prefix group id passed through `ForwardBatch`. The merge helper
`_safe_merge_state` is reused as-is.

## Correctness plan

Numerical parity: for a fixed prefix + B branches, the shared-read output must
equal per-branch attention within fp tolerance. Test at tiny scale on GPU:
build one batch two ways (per-branch decode vs shared+suffix+merge) and assert
max abs diff < 1e-2 (bf16). The merge math is associative log-sum-exp, so parity
should hold to numerical noise; the risk is descriptor/index bugs, not math.

## Three riskiest unknowns

1. **CUDA graph capture.** Decode runs under CUDA graphs; a variable
   shared-prefix group shape may not capture cleanly (batch-shape must be fixed
   per captured bucket). May need a dedicated captured path or graph-disable for
   tree batches. (The H100 crash we hit was a separate cuda-graph permission
   issue, unrelated but a reminder this path is graph-sensitive.)
2. **Grouping in continuous batching.** Sibling branches must land in the SAME
   decode step for the shared read to apply; the scheduler batches by
   availability, not by tree. Need to co-schedule a tree's live branches.
3. **Suffix length skew.** Branches diverge in length; the suffix wrapper must
   handle ragged per-branch suffixes (the paged wrapper already does ragged).

## Effort

Path A (reuse cascade merge in decode): the 2-4 week bet. Path B (custom Triton
shared-read): unnecessary given A. Path C (infeasible): ruled out - the merge
machinery is already here and proven in prefill.
