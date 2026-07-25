# AutoTree Tree Completions Wire Contract v1

Status: **NORMATIVE** for `autotree-serve`, `autotree-sdk`, and consumers of
`POST /v1/tree/completions`.

Contract version: **1.3.0**. The `/v1` path identifies this major wire version.
Breaking changes require a new major endpoint or an explicitly negotiated wire
version. Additive response fields may be introduced within v1; clients must not
infer semantics from fields that are not specified here.

Version 1.1.0 adds the terminal `error` stream event and the `[DONE]` sentinel.
Version 1.2.0 defines `token.logprob` as the unscaled model log probability,
independent of the sampling temperature and nucleus truncation, and defines
terminal branch scores on the scheduler's mean per-token path scale.
Version 1.3.0 adds the nullable `token.token_id` field and makes the default
seed resolution explicit.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## Request

The request body is JSON with these fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `model` | string | yes | Non-empty and equal to a model exposed by this serve process. |
| `messages` | array | yes | At least one `{role, content}` object. `role` is a non-empty string and `content` is a string. |
| `tree` | object | yes | Tree-search parameters described below. |
| `stream` | boolean | no | Defaults to `false`. |
| `max_completion_tokens` | integer | no | 1 through 4096. Takes precedence over `max_tokens`. |
| `max_tokens` | integer | no | 1 through 4096. Used only when `max_completion_tokens` is absent. |
| `temperature` | number | no | 0 through 2 inclusive; default 1. |
| `top_p` | number | no | Greater than 0 and at most 1; default 1. |
| `stop` | string or string array | no | Stop sequence or sequences. |
| `n` | integer | no | MUST equal 1; default 1. |
| `seed` | integer or null | no | Sampling seed. Omitted or `null` resolves to `0`. |
| `user` | string or null | no | Caller-provided user identifier. |
| `stream_options` | object or null | no | Accepted for OpenAI compatibility. Tree streams always report usage in `done`. |

If neither token-limit field is present, `max_tokens` resolves to 16.

The `tree` object has exactly these fields:

| Field | Type | Required | Contract |
|---|---|---:|---|
| `policy` | string | yes | One of `beam`, `best_first`, or `mcts`. |
| `branches` | integer | yes | 1 through 64. |
| `budget_tokens` | integer | yes | 1 through 1,000,000. |
| `scorer` | string or null | no | One of `logprob` or `self_consistency`; `null` uses log-probability selection. |

The following OpenAI chat fields are explicitly unsupported and MUST produce an
`unsupported_feature` error when present: `audio`, `frequency_penalty`,
`function_call`, `functions`, `logit_bias`, `logprobs`, `modalities`,
`parallel_tool_calls`, `prediction`, `presence_penalty`, `reasoning_effort`,
`response_format`, `tool_choice`, `tools`, `top_logprobs`, and
`web_search_options`. Other unknown top-level request fields are currently
accepted and ignored for compatibility; callers SHOULD NOT rely on them.

Errors use the OpenAI-shaped body:

```json
{
  "error": {
    "message": "human-readable detail",
    "type": "invalid_request_error",
    "param": "field-or-null",
    "code": "machine-readable-code-or-null"
  }
}
```

## Non-stream response

With `stream: false`, the server returns one JSON object:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "served-model-id",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "winning text"},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "total_tokens": 2
  },
  "tree": {}
}
```

`created` is a Unix timestamp in seconds. `choices` MUST contain exactly one
choice. `finish_reason` is `stop` or `length`. The `tree` object is required and
has the Tree Summary shape defined below. The winning `message.content` MUST be
the concatenation of tokens on the root-to-winner branch path.

## Streaming response

With `stream: true`, the response media type is `text/event-stream`. Each event
contains an SSE `event:` line equal to the JSON payload's `type`, followed by one
`data:` line containing a single JSON object and a blank line. The final JSON
event is followed by `data: [DONE]` and a blank line. Successful streams end with
`done`; failed streams end with `error`.

### `branch_started`

```json
{"type":"branch_started","branch_id":"b0","parent_id":null}
```

- `branch_id`: unique branch identifier.
- `parent_id`: parent branch identifier, or `null` for a root branch.

### `token`

```json
{"type":"token","branch_id":"b0","token_index":0,"token":"hello","token_id":15339,"logprob":-0.25}
```

- `branch_id`: branch that owns this token span.
- `token_index`: zero-based index within that branch's own emitted tokens.
- `token`: exact text span contributed by this event.
- `token_id`: sampled model vocabulary ID, or `null` only when the serving
  engine cannot provide an ID. Consumers MUST NOT infer an ID by retokenizing
  `token` text.
- `logprob`: finite natural-log probability of this sampled token under the
  model's raw full-vocabulary logits: `log_softmax(raw_logits)[token_id]`. It is
  independent of temperature and `top_p`, is not a branch score or placeholder,
  and is not special-cased to `0.0` for greedy sampling. A behavior-policy
  logprob, if added later, MUST use a separate explicitly named field.

### `branch_pruned`

```json
{"type":"branch_pruned","branch_id":"b1","reason":"beam_pruned"}
```

- `branch_id`: branch being terminalized.
- `reason`: non-empty machine-readable or stable human-readable prune reason.

### `branch_merged`

```json
{"type":"branch_merged","branch_id":"b2","into_branch_id":"b0"}
```

- `branch_id`: branch being terminalized.
- `into_branch_id`: distinct, already-started, still-live merge target.

### `error` (added in v1.1.0)

```json
{
  "type": "error",
  "error": {
    "message": "Tree-KV capacity is exhausted.",
    "type": "rate_limit_error",
    "param": "kv_pages",
    "code": "kv_capacity_exhausted"
  },
  "retry_after_seconds": 1
}
```

`error` is the terminal JSON event for a failed stream and occurs instead of
`done`. Its nested object uses the same `message`, `type`, `param`, and `code`
fields as an HTTP error body. `retry_after_seconds`, when present, is a
non-negative integer that tells callers the minimum backoff before retrying.
Capacity failures MUST use code `kv_capacity_exhausted`, param `kv_pages`, and a
`retry_after_seconds` value. If capacity exhaustion is known before response
headers are sent, the server MUST return HTTP 429 with the equivalent
`Retry-After` header instead of starting an SSE body. Mid-stream failures cannot
change HTTP headers and therefore communicate backoff in this event.

No branch-completion or usage reconciliation is required after `error`; partial
branch events are diagnostic only. SDKs MUST parse this event as a typed stream
failure and MUST NOT report it as an unknown event or a missing `done` event.

### `done`

```json
{
  "type": "done",
  "branch_id": "b0",
  "text": "hello",
  "finish_reason": "length",
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 3,
    "total_tokens": 4
  },
  "counters": {
    "logical_tokens": 6,
    "physical_tokens": 3,
    "useful_tokens": 1,
    "elapsed_seconds": 0.01,
    "ttft_seconds": 0.001
  },
  "tree": {}
}
```

- `branch_id`: winning branch identifier.
- `text`: complete root-to-winner text.
- `finish_reason`: `stop` or `length`.
- `usage`: Usage object defined below.
- `counters`: non-negative `logical_tokens`, `physical_tokens`, and
  `useful_tokens`; positive `elapsed_seconds`; non-negative `ttft_seconds`.
- `tree`: required Tree Summary object.

## Shared objects

### Usage

`prompt_tokens`, `completion_tokens`, and `total_tokens` are non-negative
integers. `total_tokens` MUST equal `prompt_tokens + completion_tokens`.
`completion_tokens` MUST equal the total number of emitted `token` events across
all branches, not only the winning branch.

### Tree Summary

The `tree` object in both the non-stream response and `done` contains every field
below:

| Field | Type | Contract |
|---|---|---|
| `policy` | string | Executed tree policy. |
| `branch_count` | integer | Number of `branch_started` events; at least 1. |
| `pruned_count` | integer | Number of `branch_pruned` events. |
| `merged_count` | integer | Number of `branch_merged` events. |
| `winner_branch_id` | string | MUST equal `done.branch_id`. |
| `tokens_spent_per_branch` | object | Every branch ID mapped to its count of owned `token` events. |
| `final_scores` | object | Every branch ID mapped to its finite mean per-token path score. This is branch-keyed, never positional. |
| `scorer` | string or null | Executed scorer identifier. |
| `kv_reuse_ratio` | number | `logical_tokens / physical_tokens`; finite and at least 1. |

`self_consistency` is a final selection rule, not a learned value model. It
groups finalized branches by extracted numeric answer (preferring the last
boxed answer, otherwise the last numeric token). The largest agreement group
wins; cumulative log-probability breaks group-size ties and selects the branch
within the winning group. Branches without an extractable answer are independent
singleton groups. `final_scores` remains the mean per-token path score mapping.

For a served tree completion, `physical_tokens` MUST be positive and
`kv_reuse_ratio` MUST reconcile exactly with the `done.counters` values within
normal floating-point tolerance. This ratio is a multiplier, not a percentage:
`1` means no reuse, while `5` means five logical token references per physical
token stored.

## Stream invariants

1. A branch MUST be started exactly once before any event references it.
2. A non-null `parent_id` MUST reference an already-started branch. The ordered
   parent chain defines the branch's root-to-leaf `branch_path`.
3. `token_index` MUST start at 0 independently for each branch and increase by
   exactly 1. Tokens MUST NOT arrive after that branch is terminal.
4. On a successful stream, every started branch MUST receive exactly one terminal
   event: the winner gets `done`; every other branch gets `branch_pruned` or
   `branch_merged`.
5. A successful stream MUST contain exactly one `done` as its final JSON event.
   A failed stream MUST contain exactly one `error` as its final JSON event and
   MUST NOT contain `done`. Both forms MUST then emit the `[DONE]` sentinel.
6. `done.text` MUST equal the concatenation of `token` text across the winner's
   complete root-to-winner path. SDK exports MUST use that same lineage rule so
   forked branches retain shared-prefix tokens.
7. Usage, branch counts, prune/merge counts, per-branch token counts,
   `final_scores`, winner identity, and KV reuse MUST reconcile with the events
   and counters as defined above.

Any server engine that violates these invariants MUST fail the request rather
than serialize a dishonest stream. SDK consumers MUST reject a fully consumed
stream that violates them.
