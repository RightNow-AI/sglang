# Tree-KV capacity behavior

For `autotree serve --engine treekv`, the default page limit is:

`ceil(model_context_tokens / page_size * kv_branch_headroom)`

The default `kv_branch_headroom` is `1.5`. `--kv-pages` overrides the derived
limit, and `--kv-branch-headroom` changes the multiplier.

The server admits up to eight generation requests at once by default. Set
`--max-concurrent-requests` to a positive integer to tune that bound. Requests
beyond the bound wait for a slot instead of entering the engine immediately.
The `in_flight_requests` and `queued_requests` gauges expose current admission
state, and `concurrency_rejections_total` counts requests rejected after the
runner has stopped admission during shutdown.

The current Tree-KV implementation creates one `PagedKVPool` per request, so
the concurrency limit also bounds the number of full request-local pools that
can exist at once. Lower the limit when model and KV allocations approach the
device memory budget.

Prompt admission and decode exhaustion are exposed as
`kv_capacity_exhausted` errors instead of server errors. A non-streaming
request receives HTTP 429. Once an SSE response has started, the server emits
an `error` event and closes the stream normally.

The engine does not prune branches on its own when allocation fails. The Rust
scheduler owns branch liveness and emits `Kill` commands; injecting an
engine-side kill would leave scheduler and KV state inconsistent. Capacity is
therefore recovered only through scheduler-issued pruning, otherwise the
request ends with the typed error above.
