# TreeKV quickstart (CPU)

<!-- markdownlint-disable MD013 -->

This walkthrough starts AutoTree with real GPT-2 weights on CPU and returns a
live tree completion. It exercises the current repository implementation; it
does not demonstrate GPU kernels or establish a speed, cost, or quality claim.

## Prerequisites

- Windows PowerShell.
- `uv` 0.9 or newer.
- Rust and Cargo 1.93 or newer.
- Network access for Python dependencies and the first GPT-2 download.

Bare `python` is not used. Every Python operation goes through `uv` and the
repository-root `.venv`.

## 1. Create the environment and build the scheduler

From the repository root:

```powershell
uv venv --python 3.12 --clear
uvx maturin build --release --features python --manifest-path scheduler/Cargo.toml --out dist
$wheel = Get-ChildItem dist/autotree_scheduler-*.whl | Select-Object -First 1
uv pip install -e './core[engine]' -e ./serve $wheel.FullName
```

Expected output includes all three local distributions:

```text
+ autotree-core==0.1.0
+ autotree-scheduler==0.1.0
+ autotree-serve==0.1.0
```

The `engine` extra is required because TreeKV loads a Hugging Face model. The
Rust wheel supplies the scheduler's Python binding; the server does not
silently replace a missing TreeKV dependency with its deterministic test
engine.

## 2. Start the CPU server

```powershell
uv run --no-project autotree serve --engine treekv --model gpt2
```

The first run downloads GPT-2. Keep this terminal open. A successful startup
ends with:

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

This is a CPU demonstration with a small model. It is not the GPU serving
architecture described in the roadmap.

## 3. Request a tree completion

Open a second PowerShell window in the repository root:

```powershell
@'
{"model":"gpt2","messages":[{"role":"user","content":"Explain why shared prefixes matter."}],"max_tokens":4,"seed":7,"tree":{"policy":"beam","branches":3,"budget_tokens":12}}
'@ | curl.exe --silent --show-error --fail http://127.0.0.1:8000/v1/tree/completions --header "Content-Type: application/json" --data-binary '@-'
```

The tested request returned HTTP 200 with this body. The response ID,
timestamp, generated text, and floating-point scores can vary across runs and
dependency versions.

```json
{"id":"chatcmpl-68a6eca20db344f8a86e4fad4c8346b4","object":"chat.completion","created":1784411136,"model":"gpt2","choices":[{"index":0,"message":{"role":"assistant","content":" How to find out"},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":13,"completion_tokens":10,"total_tokens":23},"tree":{"policy":"beam","branch_count":4,"pruned_count":3,"merged_count":0,"winner_branch_id":"branch-1","tokens_spent_per_branch":{"branch-0":1,"branch-1":3,"branch-2":3,"branch-3":3},"final_scores":{"branch-0":-3.3197202682495117,"branch-1":-10.647787928581238,"branch-2":-13.065826058387756,"branch-3":-18.146788120269775},"scorer":null,"kv_reuse_ratio":3.5}}
```

`branches` configures the scheduler's fork width. The summary counts all
branches in the executed tree, including its root, so `branch_count` may be
larger than the requested fork width. `budget_tokens` is a hard tree budget;
`max_tokens` caps generated tokens for an individual path.

Stop the server with Ctrl+C.

## What just ran

1. `maturin` compiled the Rust scheduler with its Python feature and produced
   an ABI3 wheel for Python 3.12 or newer.
2. `autotree-core[engine]` installed the Tree-KV engine, PyTorch, Transformers,
   and Safetensors; `autotree-serve` installed the CLI and HTTP server.
3. The CLI loaded GPT-2 through the CPU model executor and connected it to the
   paged KV state and Rust scheduler.
4. `/v1/tree/completions` returned the winning text plus the tree execution
   summary.

## Troubleshooting

- `Failed to load Tree-KV CPU model`: confirm the scheduler wheel was installed
  into the root `.venv` and that the first model download can reach Hugging
  Face.
- Port 8000 already in use: stop the other process, or append `--port 8001` to
  the server invocation and use the same port in the request URL.
- PowerShell opens the Microsoft Store for `python`: ignore bare `python`; this
  guide intentionally uses `uv`.
- Triton is unavailable on Windows: expected for this CPU path. GPU kernel
  validation requires a supported Linux GPU environment.

The repository does not currently include the blueprint's web playground.
Use the HTTP response and streaming API as the live demo surfaces today.

## GPU variant

The same server runs real models on CUDA: install the cu128 torch build
(`torch==2.11.0+cu128` from `https://download.pytorch.org/whl/cu128` on
CUDA 12.8 drivers), then add `--device cuda --dtype bfloat16` to the serve
command. Hardware validation evidence lives in `core/docs/a100-validation.md`.
