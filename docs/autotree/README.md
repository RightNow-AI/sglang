# AutoTree

AutoTree adds tree execution to this SGLang fork through
`POST /v1/tree/completions`. The parent prompt is prefetched, then the scheduler
forks sibling requests through SGLang's normal intake path. The siblings match
the prompt prefix through the radix cache and generate with separate sampling
seeds.

The current engine forks at the end of prefill, before output-token decode. It
does not fork at an arbitrary decode token.

During generation, AutoTree:

- accounts for output tokens across all branches against one tree budget;
- tracks running mean output-token log probability as a value proxy;
- can prune a trailing branch after warmup while retaining a minimum number of
  live branches;
- stops early when a strict majority of all branches has already produced the
  same parsed numeric answer;
- sends a token-aligned final snapshot through the parent request; and
- selects the returned answer by self-consistency when branch outputs contain
  parseable numeric answers, with mean log probability as the fallback.

The parent request is also branch `0`. The runtime holds it past its natural
EOS so it can carry the final tree snapshot back through the existing response
channel. Text after that natural EOS is scaffolding and is trimmed before the
answer is returned.

## Five-minute quickstart

Run these commands from an environment where this fork is installed and a GPU
is available.

### 1. Launch the server

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --port 30000
```

Wait for the health endpoint:

```bash
curl -f http://127.0.0.1:30000/health
```

### 2. Send a tree request

```bash
curl http://127.0.0.1:30000/v1/tree/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "What is 17 times 6? End with #### followed by the answer."
      }
    ],
    "max_tokens": 64,
    "temperature": 0.7,
    "seed": 7,
    "tree": {
      "policy": "beam",
      "branches": 4,
      "budget_tokens": 256,
      "scorer": null
    }
  }'
```

Use non-streaming requests with the current phase-1 runtime. The request schema
contains streaming fields, but the `tree_runtime.py` fallback returns ordinary
SGLang chunks rather than the structured tree events expected by the serving
stream adapter.

### 3. Read the response

The answer is in `choices[0].message.content`. Tree accounting is in `tree`:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "tree": {
    "policy": "beam",
    "branch_count": 4,
    "pruned_count": 0,
    "merged_count": 0,
    "winner_branch_id": "0",
    "tokens_spent_per_branch": {},
    "final_scores": {},
    "scorer": "mean_logprob",
    "kv_reuse_ratio": null
  }
}
```

The values above show the envelope, not a claimed result for the example. See
[the API reference](api.md) for exact field semantics.

## Container deployment

The fork can also be overlaid onto a stock SGLang container. The maintained
recovery path is described in [`tools/splice`](../../tools/splice/README.md),
with the host assumptions and commands in
[`redeploy_container.sh`](../../tools/splice/redeploy_container.sh). The script
recreates its container, checks GPU visibility, overlays the tree files,
applies the splice, checks imports, and launches the server.

## Measured so far

The engine has been exercised on model workloads and its conservative pruning
default reflects observed accuracy loss from more aggressive margins with the
current mean-logprob proxy. This repository documentation intentionally does
not publish benchmark figures. Benchmarks, methodology, and raw logs live on
the project website.

## Read next

- [API reference](api.md)
- [How it works](how-it-works.md)
- [Migrate from vLLM](migrate-from-vllm.md)
