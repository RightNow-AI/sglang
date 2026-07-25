# vLLM and SGLang compatibility notes

AutoTree currently implements an OpenAI-compatible subset at
`/v1/chat/completions` plus `/v1/tree/completions`. It does not implement a
vLLM- or SGLang-compatible command-line flag parser. Migration scripts must use
only the flags accepted by `autotree serve`:

| AutoTree flag | Meaning |
| --- | --- |
| `--model` | Hugging Face ID or local Transformers model directory |
| `--engine` | `treekv` for real CPU weights or `deterministic` for the seeded test engine |
| `--kv-pages` | Explicit TreeKV page limit; positive integer |
| `--kv-branch-headroom` | Multiplier used to derive the default page limit; finite and at least 1.0 |
| `--host` | HTTP bind address |
| `--port` | HTTP port |

There are no accepted equivalents today for vLLM tensor/pipeline parallelism,
GPU memory utilization, quantization flags, or SGLang distributed-node flags.
Passing an unknown flag fails argument parsing instead of silently ignoring it.

At the HTTP layer, common chat fields including `model`, `messages`,
`temperature`, `top_p`, token limits, stop sequences, `seed`, `stream`, and
`stream_options` are accepted. Multiple choices, tool calling, structured
response formats, logprobs, and other unsupported semantic fields return an
explicit `unsupported_feature` error. `core/docs/wire-spec.md` is the normative
tree endpoint contract.
