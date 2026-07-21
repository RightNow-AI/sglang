# Migrate from vLLM

This fork keeps the OpenAI-style chat surface used by vLLM clients. Point the
same Python client at the SGLang server by changing `base_url`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
)
```

## Tree execution off

Keep using `client.chat.completions.create(...)`. Requests to
`/v1/chat/completions` follow the stock SGLang generation path. The tree hooks
are defensive and inactive when a request has no tree envelope.

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Write one sentence."}],
)
```

## Tree execution on

Tree execution uses the separate `/v1/tree/completions` route. The OpenAI
client's generated chat method cannot change that route, so use its generic
POST method and merge the tree mapping as `extra_body`:

```python
from openai.types.chat import ChatCompletion

extra_body = {"tree": {"policy": "beam", "branches": 4, "budget_tokens": 384}}

completion = client.post(
    "/tree/completions",
    cast_to=ChatCompletion,
    body={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "Solve 21 * 4. End with #### N."}],
    },
    options={"extra_json": extra_body},
)
```

Use non-streaming requests with the current phase-1 runtime. See the
[API reference](api.md#streaming-status) for the code-path limitation.

## Concept mapping

| vLLM concept | AutoTree or SGLang equivalent |
| --- | --- |
| `n > 1` independent completions | `tree.branches` under one tree request. The endpoint still requires `n: 1`. |
| Per-completion `max_tokens` | `max_tokens` or `max_completion_tokens` per branch, plus `tree.budget_tokens` across the tree. |
| Prefix caching | SGLang radix caching, with tree siblings submitted through normal intake so they match the shared prompt prefix. |
| Multiple returned candidates | One returned winner plus per-branch token counts and mean-logprob scores in the `tree` summary. |
| Client-side majority vote | Serving-side numeric-answer self-consistency, with mean-logprob fallback. |

The current runtime accepts `beam`, `best_first`, and `mcts` as policy labels,
but it does not switch among distinct scheduling algorithms based on that
field. Likewise, `tree.scorer` is not a plug-in scorer selector in this phase.

## Performance evidence

Benchmarks with raw logs: see the project website. This page makes no
performance comparison between vLLM and AutoTree.

## Container users

For an overlay onto a stock SGLang container, start with the maintained
[`tools/splice` deployment note](../../tools/splice/README.md) and inspect
[`redeploy_container.sh`](../../tools/splice/redeploy_container.sh) for its host
paths, container name, GPU check, and launch command.

## Related pages

- [AutoTree overview](README.md)
- [API reference](api.md)
- [How it works](how-it-works.md)
