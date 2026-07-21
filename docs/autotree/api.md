# Tree completions API

`POST /v1/tree/completions` accepts an OpenAI-style chat request plus a required
`tree` object. Send `Content-Type: application/json`.

The supported adoption path for the current `tree_runtime.py` integration is
non-streaming. See [Streaming status](#streaming-status).

## Request body

### Top-level fields

| Field | Type | Default | Constraints and behavior |
| --- | --- | --- | --- |
| `model` | string | required | Must contain at least one character. Passed to the normal SGLang chat prompt renderer. |
| `messages` | array of message objects | required | At least one message. |
| `tree` | object | required | See [Tree fields](#tree-fields). Unknown fields inside this object are rejected. |
| `stream` | boolean | `false` | The schema accepts it, but `true` is not supported end to end by the current phase-1 runtime fallback. |
| `stream_options` | object or null | `null` | Defines `include_usage`, a boolean defaulting to `false`. Extra keys are accepted. The tree request model does not require `stream=true` when this object is present. |
| `max_completion_tokens` | integer or null | `null` | From `1` through `4096`. Takes precedence over `max_tokens`. |
| `max_tokens` | integer or null | `null` | From `1` through `4096`. Used only when `max_completion_tokens` is null. |
| `temperature` | number | `1.0` | From `0.0` through `2.0`. |
| `top_p` | number | `1.0` | Greater than `0.0` and no greater than `1.0`. |
| `stop` | string, array of strings, or null | `null` | Forwarded to the normal chat request. |
| `n` | integer | `1` | Must equal `1`. Tree branches are controlled by `tree.branches`, not `n`. |
| `seed` | integer or null | `null` | A null seed resolves to `0`. Child branch `b` uses the resolved seed plus `b`. |
| `user` | string or null | `null` | Forwarded to the normal chat request. |

If both token-limit fields are null, the resolved per-branch generation limit
is `16`. `tree.budget_tokens` is separate: it counts output tokens across the
tree and triggers finalization when the accumulated count reaches or exceeds
the budget.

The top-level request model accepts unknown fields, but the tree adapter only
forwards the documented fields above. Do not rely on an unknown top-level field
affecting generation.

### Message fields

| Field | Type | Default | Constraints and behavior |
| --- | --- | --- | --- |
| `role` | string | required | Must contain at least one character. |
| `content` | string | required | Message text. |

Message objects accept extra fields and are passed to the normal SGLang chat
request renderer.

### Tree fields

| Field | Type | Default | Constraints and behavior |
| --- | --- | --- | --- |
| `policy` | string | required | One of `beam`, `best_first`, or `mcts`. The current runtime records and returns this value, but does not select different execution logic from it. |
| `branches` | integer | required | From `1` through `64`, including parent branch `0`. |
| `budget_tokens` | integer | required | From `1` through `1000000`. Counts generated output tokens across branches. |
| `scorer` | string or null | `null` | No name validation or scorer dispatch is implemented in the current runtime. See the response `scorer` rules below. |

## Complete curl example

```bash
curl http://127.0.0.1:30000/v1/tree/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Compute 125 divided by 5. End with #### followed by the answer."
      }
    ],
    "max_completion_tokens": 96,
    "temperature": 0.7,
    "top_p": 1.0,
    "seed": 11,
    "n": 1,
    "stream": false,
    "tree": {
      "policy": "beam",
      "branches": 4,
      "budget_tokens": 384,
      "scorer": null
    }
  }'
```

## Python OpenAI client example

The generated `client.chat.completions.create()` method always calls
`/v1/chat/completions`, so adding `tree` through that method's `extra_body`
would not reach this endpoint. Use the same `OpenAI` client with its generic
POST method. The SDK's generic request option named `extra_json` is how its
generated methods merge an `extra_body` mapping into the JSON body.

```python
from openai import OpenAI
from openai.types.chat import ChatCompletion

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
)

extra_body = {
    "tree": {
        "policy": "beam",
        "branches": 4,
        "budget_tokens": 384,
        "scorer": None,
    }
}

completion = client.post(
    "/tree/completions",
    cast_to=ChatCompletion,
    body={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "Compute 125 divided by 5. End with #### followed by the answer.",
            }
        ],
        "max_tokens": 96,
        "temperature": 0.7,
        "seed": 11,
    },
    options={"extra_json": extra_body},
)

print(completion.choices[0].message.content)
print(completion.model_extra["tree"])
```

`ChatCompletion` parses the OpenAI-compatible fields. The additional `tree`
object remains available through Pydantic's `model_extra` mapping.

## Non-streaming response

| Field | Type | Semantics |
| --- | --- | --- |
| `id` | string | Generated with a `chatcmpl-` prefix. |
| `object` | string | Always `chat.completion`. |
| `created` | integer | Unix timestamp in seconds. |
| `model` | string | The request's model value. |
| `choices` | array | Contains one choice. |
| `usage` | object | Token counts for the returned completion. |
| `tree` | object | Final tree summary. |

The only choice has:

| Field | Type | Semantics |
| --- | --- | --- |
| `index` | integer | Always `0`. |
| `message.role` | string | Always `assistant`. |
| `message.content` | string | Text of the selected branch, trimmed at its first EOS. |
| `logprobs` | null | Always null in this response model. |
| `finish_reason` | string | `stop` or `length`. |

`usage.prompt_tokens`, `usage.completion_tokens`, and `usage.total_tokens` are
non-negative integers. Completion usage describes the returned winner, not the
sum of work across branches. Per-branch work is reported in the tree summary.

### Tree summary

| Field | Type | Semantics |
| --- | --- | --- |
| `policy` | string | Policy value from the final runtime snapshot, or the request if no snapshot is available. |
| `branch_count` | integer | Number of branches represented by the runtime. |
| `pruned_count` | integer | Number of branches marked pruned. This includes value-pruned branches and active non-winners stopped during finalization. |
| `merged_count` | integer | `0` in the current runtime fallback. |
| `winner_branch_id` | string | Branch selected by the serving vote when a vote succeeds, otherwise the runtime leader from the final snapshot. The degraded no-snapshot path uses branch `0`. |
| `tokens_spent_per_branch` | object | Map from string branch ID to accounted output-token count. Empty if no runtime snapshot reached the response. |
| `final_scores` | object | Map from branch ID to running mean output-token log probability for branches with accounted tokens. Snapshot values are rounded to four decimal places. |
| `scorer` | string or null | `self_consistency` when numeric-answer voting selected a winner. Otherwise it is the requested scorer, or `mean_logprob` when a normal snapshot exists and no scorer was requested. It can be null on the degraded no-snapshot path. |
| `kv_reuse_ratio` | number or null | Null in this phase-1 response path because per-request KV reuse is not measured and surfaced through the runtime snapshot. |

Self-consistency extracts the last value following `####` when present,
otherwise the last number in each branch. It normalizes commas and integer-like
floats. The answer with the most branch votes wins. An answer-count tie is
broken by the highest mean-logprob branch among the tied answers. If no branch
contains a parseable number, selection falls back to the runtime leader.

## Streaming status

The protocol defines server-sent event models for `branch_started`, `token`,
`branch_pruned`, `branch_merged`, and `done`, followed by `data: [DONE]`.

| Event | Payload fields |
| --- | --- |
| `branch_started` | `branch_id` and nullable `parent_id`. |
| `token` | `branch_id`, zero-based `token_index`, token text, nullable `token_id`, and token `logprob`. |
| `branch_pruned` | `branch_id` and non-empty `reason`. |
| `branch_merged` | `branch_id` and `into_branch_id`. |
| `done` | Winner `branch_id`, winner `text`, `finish_reason`, `usage`, `counters`, and the final `tree` summary. |

The `done.counters` object contains non-negative `logical_tokens`,
`physical_tokens`, `useful_tokens`, `elapsed_seconds`, and `ttft_seconds`, plus
integer arrays `unique_tokens_per_step` and `branch_tokens_per_step`. If a
`TreeResult` has no counters, the serving adapter supplies zero values and
empty arrays.

However, the integrated phase-1 runtime returns ordinary SGLang generation
chunks. The non-streaming handler converts that ordinary result into a tree
response, while the streaming handler currently requires structured
`TreeBranchEvent` and `TreeResult` values and does not perform that conversion.

Use `stream: false` with this runtime. A streaming request can fail with
`Tree scheduler returned an invalid stream envelope.`

## Errors

- Requests without `Content-Type: application/json` return HTTP `400`.
- Schema violations return HTTP `400` with an `invalid_request_error`, the
  first validation message, a dotted `param`, and code `validation_error`.
- Examples include a missing or empty `model`, no messages, an unsupported
  policy, out-of-range branch or token counts, an extra field inside `tree`, or
  `n` other than `1`.
- Prompt rendering, sampling conversion, and tree parameter `ValueError`
  failures return HTTP `400` through the common OpenAI serving error envelope.
- An invalid non-streaming scheduler result returns HTTP `500` with type
  `InternalServerError`.
- Streaming envelope failures return an error response or terminate the stream,
  depending on whether the first chunk has already been emitted.

## Related pages

- [AutoTree overview](README.md)
- [How it works](how-it-works.md)
- [Migrate from vLLM](migrate-from-vllm.md)
