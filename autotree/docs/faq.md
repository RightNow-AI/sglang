# Frequently asked questions

## What works now?

The repository can run a real GPT-2 CPU tree completion end to end. The path
includes a Hugging Face model executor, paged Tree-KV state, the Rust scheduler,
the HTTP server, and a returned tree summary. CPU unit and contract suites also
cover KV operations, reference attention, scheduling, serving, the SDK, and the
provenance-labeled ThoughtBench harness (fixture and real task sets).

The server also has a seeded deterministic engine for API tests. It does not
load model weights and should not be presented as a model-serving result.

## What needs a GPU?

Triton kernel execution and parity, CUDA-specific optimization, large-model
validation, throughput measurement, and the blueprint's cost/quality gates all
need a supported Linux GPU environment. The Windows CPU quickstart proves the
integration path, not GPU correctness or performance.

No current result proves the blueprint's 3-10x cost reduction or 5x rollout
throughput target. Those numbers are research hypotheses until the planned
benchmarks reproduce them.

## Is the API OpenAI compatible?

It implements a useful subset:

- `GET /v1/models` for the one model served by the process.
- `POST /v1/chat/completions` for non-streaming and SSE streaming chat output.
- `POST /v1/tree/completions` for AutoTree's required `tree` extension.
- `GET /metrics` for Prometheus text metrics.

The chat schema accepts common fields such as `model`, `messages`,
`temperature`, `top_p`, token limits, stop sequences, `seed`, `stream`, and
`stream_options`. Unsupported semantics such as multiple choices and tool calls
are rejected explicitly rather than ignored. This is not yet every endpoint or
feature of the OpenAI API.

## Can I point an OpenAI client at it?

Yes, for the implemented chat-completions subset, by using
`http://127.0.0.1:8000/v1` as the base URL and any non-empty local API key. The
tree-specific endpoint and branch-event types are available through the typed
client in `sdk/` or a direct HTTP request.

## Is there a playground?

Yes. With the server running, open `http://127.0.0.1:8000/playground`: it is a
fully offline page served by `autotree-serve` that streams the live tree via
SSE and visualizes branch growth, pruning, and merging in real time. The
HTTP/SSE API and SDK are the other live surfaces.

## Is ThoughtBench a benchmark result?

No. Its bundled tasks are tiny synthetic fixtures used to test runner,
resumption, grading, metrics, and report contracts. Result artifacts are
stamped to prevent them from being cited as AIME, GPQA, LiveCodeBench, cost, or
performance evidence.

## What does the Dockerfile provide?

It builds the Rust scheduler binding in a separate builder stage, installs the
CPU core and server into a Python 3.12 slim runtime, drops privileges, and runs
the GPT-2 TreeKV server on port 8000. Model weights are downloaded at first
container start unless an external Hugging Face cache is mounted.
