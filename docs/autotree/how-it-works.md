# How AutoTree works

AutoTree is attached to the normal SGLang tokenizer and scheduler path. The
parent is an ordinary generate request wrapped with tree parameters. Branches
are ordinary scheduler requests created after the parent's prefill completes.

## Request lifecycle

```text
POST /v1/tree/completions
          |
          v
validate request and render the normal chat prompt
          |
          v
tokenizer intake wraps the parent with tree parameters
          |
          v
normal scheduler intake and parent prefill
          |
          v
fork siblings through the real generate-request intake path
          |          shared prompt prefix through radix caching
          v
per-token accounting for every active branch
          |          tokens, total spend, running mean logprob
          v
periodic EMVPT check
          |          prune a sufficiently trailing eligible sibling
          v
periodic majority-lock check
          |          finalize if a strict majority is irreversible
          v
finalize on majority lock, sibling completion, or tree budget
          |
          v
attach token-aligned final snapshot to parent customized_info
          |
          v
serving layer decodes branch outputs and votes
          |
          v
OpenAI-style response plus tree summary
```

## Intake and fork

The API adapter first converts the tree request to the normal SGLang chat
request. It forces output log probabilities on because the runtime uses them
for value accounting. The tokenizer manager then sends a wrapped tokenized
parent to the scheduler.

The parent follows the normal scheduler intake path. When its prefill finishes,
the runtime creates branches `1` through `branches - 1`. Each child is a copy
of the original tokenized request with:

- a request ID suffixed with `#treeN`;
- a copied sampling-parameter object; and
- seed equal to the resolved request seed plus its branch number.

Each child is submitted through `scheduler.handle_generate_request()`. This is
the real intake path, so normal request invariants and radix-prefix matching are
used instead of constructing partial scheduler requests by hand. The shared
prompt prefix can therefore reuse radix-cached KV state.

This implementation forks once, after prefill. Arbitrary mid-decode forking and
tree-mask decode are not part of this runtime path.

## Parent lifetime and snapshots

Branch `0` is both a candidate and the response carrier. With multiple
branches, the runtime records its original `max_new_tokens` and `ignore_eos`,
then temporarily increases its token limit by `64` and ignores EOS. Sibling
branches retain the caller's original sampling settings.

The held parent is not allowed to vote until its natural EOS token appears.
For voting and response text, its output is trimmed at that first EOS. Tokens
after it are transport scaffolding, not part of the answer.

The runtime writes its state to `parent.req.customized_info["autotree"]`. The
channel is token-aligned: entries before the newest parent token are null and
the snapshot is placed at the newest token index so the output streamer carries
it with the next chunk. Finalization attaches a snapshot that includes every
available branch's output IDs. The serving layer uses the last real snapshot.

The runtime deliberately does not replace the parent's output-token array with
the winner's tokens. Post-hoc replacement would desynchronize KV allocator page
accounting. The serving layer reconstructs the winner's text from the snapshot
instead.

## Token budget and value accounting

Every token callback increments the branch token count and the tree-wide spent
count. A scalar log probability is added directly. A list, such as one emitted
by speculative decoding, is summed. If no log probability is passed to the
hook, the runtime reads newly accumulated values from the request's output-token
logprob list.

The tree budget is checked after token accounting. Reaching or exceeding
`tree.budget_tokens` finalizes the run. `max_tokens` or
`max_completion_tokens` remains the per-branch generation limit.

## EMVPT pruning

The current EMVPT value is a proxy, not a trained value prediction. For a
branch with generated tokens, it is:

```text
value = sum(output-token log probabilities) / accounted output tokens
```

At each configured interval, the runtime considers active branches that have
completed warmup. It finds the best mean log probability and prunes eligible
non-parent branches whose gap exceeds the margin, stopping when the configured
minimum number of branches remains. Branch `0` is never value-pruned because it
carries the wire response. If fewer than two branches are scored, or all scores
are zero, the check does not prune.

The environment controls are:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `AUTOTREE_VALUE_CHECK_INTERVAL` | `16` | Tree-wide accounted tokens between value and majority checks. |
| `AUTOTREE_VALUE_WARMUP_TOKENS` | `8` | Tokens a branch needs before it is eligible for value comparison. |
| `AUTOTREE_VALUE_MARGIN` | `0.8` | Required gap in mean log probability, in nats per token. |
| `AUTOTREE_VALUE_MIN_KEEP` | `2` | Minimum number of active branches retained by value pruning. |

The margin is intentionally conservative. Measurements observed that smaller
margins could remove minority branches that later supplied the correct answer.
Mean token log probability is too naive to separate branch value reliably at a
fine margin, so pruning that fires more often can cost answer accuracy. A
trained value head is the intended replacement for this proxy. Until then,
lower the margin only with workload-specific evidence or a stronger scorer.

The request's `tree.scorer` string does not replace this runtime proxy. The
current implementation stores the string for response accounting but does not
dispatch a scorer implementation by name.

## Majority lock

Majority lock uses parsed numeric final answers. It prefers the last number
following `####`; if that marker is absent, it uses the last number in the
text. A non-parent sibling is eligible only after it finishes. The held parent
is eligible only after its natural EOS appears.

For `N` total branches, a lock requires:

```text
matching finished votes >= floor(N / 2) + 1
```

This is safe with respect to the final majority vote because the matching
answer already owns more than half of every possible vote, not merely more than
half of the branches that happen to be finished. Every unfinished branch could
disagree and still could not outvote the locked answer. The runtime can stop
the remaining work without changing the majority result.

Majority lock does not claim that numeric answer extraction is suitable for
every task. Prompts that need voting should make the final numeric answer
unambiguous, for example by ending with `#### N`.

## Final selection

Runtime finalization first records the active branch with the highest mean
log probability, breaking an exact score tie toward the lower branch ID. It
marks other active branches for normal finish and attaches the output-bearing
snapshot.

The serving layer then performs self-consistency when it can decode parseable
numeric answers from the snapshot. The answer with the most votes wins. An
answer-count tie is broken by the highest mean-logprob branch among the tied
answers. If no numeric answer can be extracted, the runtime's mean-logprob
leader remains the winner.

## What the unit suite covers

The registered CPU unit suite under `test/registered/unit/tree/` covers:

- tree-budget accounting and per-branch spend in the standalone manager;
- shared radix-prefix lock lifetime and suffix-only reclaim on prune;
- scheduler bridge fork, prune, finalize, and result retrieval;
- answer extraction, majority voting, tie breaking, and empty-output handling;
- the final snapshot conversion used by the serving layer;
- mean-logprob pruning margin, warmup, minimum-keep behavior, and environment
  overrides;
- majority-lock threshold, parent-EOS gating, and disagreement behavior;
- token-aligned snapshot placement and the rule that finalization never mutates
  parent output IDs; and
- splice registration ordering and idempotence.

These are unit and simulated integration tests. This directory does not itself
provide a live GPU endpoint test, a performance benchmark, or a per-request KV
reuse measurement.

## Related pages

- [AutoTree overview](README.md)
- [API reference](api.md)
- [Migrate from vLLM](migrate-from-vllm.md)
