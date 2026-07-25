# Generation concurrency state analysis

`EngineRunner` previously held one `asyncio.Lock` for the full lifetime of every
Tree-KV generation. The lock covered much more than the state that can actually
be shared between requests.

## State reached by `generate()`

| State | Scope | Concurrency treatment |
| --- | --- | --- |
| Request messages, sampling parameters, stop sequences, tree policy, and token budget | Per request | Immutable request value. No lock required. |
| Worker thread and event queue in `EngineRunner` | Per request | Each generation creates its own thread, event loop, and queue. No lock required. |
| App `EventAccumulator`, response event list, stream id, audit token count, and SSE formatting state | Per request | Created inside one request or stream. No lock required. |
| Scheduler returned by `_scheduler_factory` | Per request | A new scheduler is constructed for each generation. Scheduler commands and branch state are not shared. |
| Torch sampling generator | Per request | A new seeded `torch.Generator` is constructed for each generation. |
| Parent maps, branch text, token counts, scores, active and terminal sets, and command deque | Per request | Local variables inside `TreeKVEngine.generate()`. No lock required. |
| `ModelExecution` logits and token id dictionaries | Per request | A new `ModelExecution` is returned by prefill for each request. |
| `PagedKVPool` K/V tensors, free-page heap, refcounts, valid lengths, and counters | Per request | `ModelExecutor.prefill()` constructs a new pool for every request. The pool is not global in the current implementation. |
| `TreeState` branches and branch id allocator | Per request | Constructed with the request's pool during prefill. No cross-request page ids or branch records exist. |
| Loaded HuggingFace model weights and buffers | Global per `ModelExecutor` | Read by every request. Model forwards must be coordinated with the mutable forest-forward adapter described below. |
| Model config, module-level `eager_attention_forward`, and CPU linear `forward` overrides | Global | Forest decode temporarily mutates all three. The existing `_FOREST_FORWARD_LOCK` protects forest calls from each other, but prefill and single-branch decode did not take it and could observe another request's temporary bindings. Every shared-model forward must take this narrow step lock. |
| Tokenizer object and any implementation-owned caches | Global per `TreeKVEngine` | Encode and decode calls use a narrow tokenizer lock. Generated text and token ids remain request-local. |
| Engine metadata, executor configuration, scheduler factory reference, and dedup interval | Global, immutable after construction | Safe to read concurrently. The scheduler factory must return an independent scheduler for each call. |
| Runner accepting flag, admitted count, in-flight count, queued count, semaphore, and drained event | Global per server | Mutated only on the server event loop. The semaphore bounds generation concurrency. |
| Prometheus collectors | Global per server | The client library synchronizes collector mutation. Generation admission metrics are updated by the runner on the server event loop. |

## Resulting lock boundary

The server-wide generation lock can be replaced by a bounded semaphore. Each
admitted Tree-KV request keeps its own worker and request state. The only model
critical section is one HuggingFace forward step, using the existing reentrant
forest-forward lock so normal forwards cannot overlap temporary forest bindings.
Tokenizer calls use a separate narrow lock.

The concurrency cap is also a memory bound. Because the current `PagedKVPool`
is allocated per request, total KV allocation grows with the number of in-flight
requests even though page bookkeeping cannot cross-contaminate requests.
