# AutoTree CUDA Graph Compatibility

## Verdict

The scheduler-owned AutoTree path is compatible with decode CUDA graphs.
Branch creation, delayed fork triggers, token accounting, budget enforcement,
adaptive width, value pruning, majority selection, and finalization all run
outside the captured model forward.

The optional shared-prefix single-read attention path is not compatible with
the current decode graph capture. Graph replay safely uses stock decode
attention instead. This is the desired fallback because the shared-read path
was slower in measurement and is not required for AutoTree correctness.

## ForwardBatch evidence

There are two different `ForwardBatch` objects to distinguish:

1. The ordinary runtime batch is built by `ForwardBatch.init_new`. It populates
   the Python request ID list at
   `python/sglang/srt/model_executor/forward_batch_info.py:739`, then derives
   `shared_prefix_groups` from those IDs at lines 749 through 778.
2. The static capture batch is built directly by
   `DecodeCudaGraphRunner.capture_prepare` at
   `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:670`.
   Its constructor starts at line 784. It passes only the hashed GPU tensor
   `rids_int` at line 813 and never passes the Python `rids` field, whose
   dataclass default is `None` at
   `python/sglang/srt/model_executor/forward_batch_info.py:455`.

The replay-side attention metadata view is constructed at
`decode_cuda_graph_runner.py:129-183`. It also does not expose `rids` or
`shared_prefix_groups`. For the default full decode backend, replay at
`python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:123-130`
replays the already captured graph and returns its stored outputs. It does not
inject runtime Python fields into the captured model control flow.

Therefore the model forward captured for CUDA graph replay cannot select the
`shared_prefix_groups` attention branch. It selects normal decode attention.

## Explicit degradation behavior

`DecodeCudaGraphRunner.execute` now clears `shared_prefix_groups` before graph
replay at `decode_cuda_graph_runner.py:1213-1221`. The helper lives in
`python/sglang/srt/tree/shared_prefix.py`. This does not remove `rids` and does
not affect scheduler bookkeeping. It only makes the already existing stock
attention fallback explicit.

Eager forwards can still use the shared-read path unless it is disabled by the
environment. For performance measurements, keep it disabled everywhere with
`AUTOTREE_SHARED_READ=0` so an eager fallback batch cannot re-enable the
falsified optimization.

## What remains outside the graph

The graph boundary is the model runner forward. Sampling is invoked after
`model_runner.forward` in
`python/sglang/srt/managers/tp_worker.py:564-599`. Scheduler result processing
then consumes the sampled tokens and calls AutoTree hooks:

- `on_prefill_done` at
  `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:239`
- `on_token` at
  `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:718`
- `on_request_finished` at
  `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:725`

CUDA graph capture does not capture or replay these Python scheduler actions.

## Graph-enabled launch

Decode CUDA graphs are enabled by default on CUDA, but use the explicit backend
flag in measurement scripts so the intended mode is unambiguous:

```bash
AUTOTREE_SHARED_READ=0 python -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --port 30000 \
  --cuda-graph-backend-decode full
```

PowerShell equivalent:

```powershell
$env:AUTOTREE_SHARED_READ = "0"
python -m sglang.launch_server `
  --model-path <MODEL_PATH> `
  --port 30000 `
  --cuda-graph-backend-decode full
```

CUDA graphs must stay enabled in every AutoTree run script, configuration, and
measurement command. If a measurement can exceed the automatically selected
capture range, set
`--cuda-graph-max-bs-decode <MAX_LIVE_BATCH_SIZE>` high enough to cover it.

The server can still fall back to eager execution for a batch that violates a
graph admission constraint. Confirm the next measurement records CUDA graph
usage rather than assuming that the launch flag guarantees every step replayed
a graph.
