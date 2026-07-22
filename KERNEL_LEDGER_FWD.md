# Forward-decode shared-read ledger
# patch probe

## 2026-07-22 - inspection bank

- Scope confirmed: edit only `python/sglang/srt/layers/attention/flashinfer_backend.py`; this ledger is the only additional file.
- Stock `forward_decode` currently writes K/V, obtains the decode KV handle, then calls one `BatchDecodeWithPagedKVCacheWrapper.forward` over the whole batch. The falsy `shared_prefix_groups` branch must remain exactly that code.
- Existing prefill decomposition uses `BatchPrefillWithRaggedKVCacheWrapper.forward_return_lse`, `BatchPrefillWithPagedKVCacheWrapper.forward_return_lse`, then `_safe_merge_state`.
- Planned correctness-first decode implementation: lazily create two dedicated wrappers of class `BatchPrefillWithPagedKVCacheWrapper`, sharing the existing FlashInfer workspace. The shared wrapper will receive one KV segment per active shared-prefix group and multiple query rows per segment; the suffix wrapper will receive one KV segment per batch row.
- Shared descriptor shapes: `qo_indptr=[num_groups+1]`, whose deltas are active branch counts; `kv_indptr=[num_groups+1]`, whose deltas are each group's `shared_len`; `kv_indices=[sum(shared_len)]`; query tensor `[num_grouped_rows, tp_q_heads, head_dim]`.
- Suffix descriptor shapes: `qo_indptr=[batch+1]` with unit deltas; `kv_indptr=[batch+1]`, whose deltas are `seq_len-shared_len` for grouped rows and `seq_len` for ungrouped rows; `kv_indices=[sum(segment_lens)]`; query tensor `[batch, tp_q_heads, head_dim]`.
- Merge shapes: shared and gathered suffix outputs `[num_grouped_rows, tp_q_heads, head_dim]`; LSE tensors `[num_grouped_rows, tp_q_heads]`.
- CUDA graph caveat: the new tree path has variable group descriptors and is intended only with CUDA graph disabled.
- Remaining: implement the guarded helper, validate group-to-batch mapping and positive suffix lengths, run syntax/static checks, then record exact final line numbers.
- Current uncertainty: `SharedPrefixGroup.branch_ids` are stable tree branch IDs rather than guaranteed current batch row indices. Prefer `ForwardBatch.rids` for row resolution, with `branch_ids` only as a fallback when batch RIDs are unavailable.
- Progress: added the module-level CUDA-graph-disabled caveat; the earlier `# patch probe` line was only a patch-helper smoke test.
- Progress: added the top-of-`forward_decode` truthy-group guard; the entire falsy stock decode block below it is unchanged.
- Progress: added the helper front half: stock-equivalent KV write/cache selection, full-self-attention validation, RID-to-row resolution, group de-duplication, and positive shared/suffix validation.
- Progress: added both paged descriptors: shared uses one canonical request KV range per group with grouped query rows; suffix uses one `[shared_len:end)` range per grouped row and `[0:end)` for ungrouped rows.
- Progress: completed the two paged `forward_return_lse` calls and `_safe_merge_state`; grouped outputs are scattered back while ungrouped rows retain the suffix wrapper's full-range result.
- Verification: source AST/compile and structural checks pass, Ruff F-class checks pass, no pytest/GPU run; final source ranges are CUDA-graph note 13-14, guard 1422-1425, helper 1476-1741; biggest uncertainty is runtime behavior of FlashInfer 0.6.14 on the GPU box because that package is absent locally.
