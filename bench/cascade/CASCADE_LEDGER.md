# Cascade Economics Ledger

This benchmark measures a small-model reasoning tree that conditionally
escalates to one greedy large-model completion. It compares that cascade with
large-model best-of-8 and large-model greedy baselines. The cost headline uses
explicit per-token weights, not raw token counts alone. The defaults are 1.0
cost unit per small-model token and 10.0 cost units per large-model token.
Those weights are an approximation supplied to the harness, not a measured
provider bill.

## Modes and request shapes

cascade sends one POST to the small server at /v1/tree/completions:

    {
      "model": "$SMALL_MODEL",
      "messages": [{"role": "user", "content": "<prompt><ANSWER_SUFFIX>"}],
      "max_tokens": 512,
      "temperature": 0.7,
      "seed": "<base seed + absolute item index>",
      "tree": {
        "policy": "beam",
        "branches": 8,
        "budget_tokens": 4096
      }
    }

The harness canonicalizes every non-null tree.branch_answers value and votes
over the valid answers. L is the leader count and V is the number of valid,
non-null voters. With the defaults, the small result is accepted only when
L >= 6 and V >= 4. Otherwise the harness sends one temperature-0 POST to
the large server at /v1/chat/completions. The accepted small answer is
extracted from choices[0].message.content. leader_matches_winner records
whether that answer equals the branch-answer leader. A mismatch is diagnostic
and does not itself force escalation or invalidate the item.

large_bo8 sends 8 POST requests to the large server at
/v1/chat/completions. The 8 requests for an item run in parallel. Sample seed
i is base seed + absolute item index * 1000 + i. The normalized extracted
answers are majority-voted, including null results, with ties resolved by the
lowest sample index.

large_greedy sends one temperature-0 POST to the large server. Its seed is
base seed + absolute item index.

All requests append this exact suffix to the prompt:

    \nSolve step by step, then give the final numeric answer on the last line as: Answer: <number>

## Tokens, cost, and validity

Small tokens are the sum of tree.tokens_spent_per_branch. Large tokens are
the sum of usage.completion_tokens for the large requests that actually ran.
Per-item and summary cost is:

    small_tokens * small_cost + large_tokens * large_cost

Each mode and seed summary reports items, correct count, accuracy, raw small
and large tokens, total cost units, cost per correct answer, mean wall time,
wall_s_total, wall_per_correct, error_count, error_n, and error_rate. It also
reports accuracy_excluding_errors and cost_per_correct_excluding_errors.
Cascade summaries additionally report escalation count and rate. Cost per
correct divides by max(correct_count, 1).

A successful response with zero reported completion tokens is invalid. A
missing or malformed token report is also invalid. Cascade additionally
requires a valid tree.branch_answers map before accepting the small result.
Request errors, timeouts, malformed responses, and unexpected item exceptions
are recorded in the JSONL and do not stop other items. Records with errors are
not counted correct, although any reported token use is retained. An
unexpected internal exception has unknown escalation state and is excluded
from the escalation-rate denominator.

## Answer normalization

Extraction uses the last literal Answer: marker and takes the first number
after it. If there is no marker, it uses the last number in the text. Dollar
signs, commas, percent signs, and whitespace are removed. Integers and
decimals are canonicalized, so 18.0 equals 18.

## Resume and output

Each completed item is appended immediately to --out-jsonl under a write
lock, then flushed and synced. Resume keys are (mode, seed, id). Existing
valid keys are skipped. Invalid or truncated lines and duplicate keys are
ignored with warnings. Duplicate records keep the first occurrence. The load
warning is:

    WARNING: MEASUREMENT INTEGRITY: dropped <n> duplicate record(s) by (mode, seed, id) while loading <path>

Final summarization independently applies the same first-record-wins dedupe and
prints:

    WARNING: MEASUREMENT INTEGRITY: dropped <n> duplicate record(s) by (mode, seed, id) during final summarization for mode=<mode> seed=<seed>

A missing final newline is repaired before appending. `--fresh` truncates
`--out-jsonl`, flushes it, and syncs it before any resume load or item write.

Use a fresh JSONL path whenever model, URL, data contents, branches, answer
threshold, token limit, temperature, or cost weights change. Those settings
are intentionally not part of the resume key. The final --out JSON records
the full run configuration and one summary per mode and seed.

## Two-server orchestrator commands

Replace the three placeholder values before running. The small tree server is
on port 30000 and the large chat server is on port 30001.

    python3 bench/cascade/measure_cascade.py --data "C:/absolute/path/gsm8k.jsonl" --small-model "SMALL_SERVED_MODEL" --large-model "LARGE_SERVED_MODEL" --small-url http://127.0.0.1:30000 --large-url http://127.0.0.1:30001 --seeds "0,1,2" --branches 8 --agree-threshold 6 --max-tokens 512 --temperature 0.7 --small-cost 1.0 --large-cost 10.0 --dry-run

    python3 bench/cascade/measure_cascade.py --mode cascade --data "C:/absolute/path/gsm8k.jsonl" --small-model "SMALL_SERVED_MODEL" --large-model "LARGE_SERVED_MODEL" --small-url http://127.0.0.1:30000 --large-url http://127.0.0.1:30001 --seeds "0,1,2" --branches 8 --agree-threshold 6 --max-tokens 512 --temperature 0.7 --small-cost 1.0 --large-cost 10.0 --timeout 300 --concurrency 4 --out-jsonl bench/cascade/cascade_items.jsonl --out bench/cascade/cascade.json

    python3 bench/cascade/measure_cascade.py --mode large_bo8 --data "C:/absolute/path/gsm8k.jsonl" --large-model "LARGE_SERVED_MODEL" --large-url http://127.0.0.1:30001 --seeds "0,1,2" --branches 8 --agree-threshold 6 --max-tokens 512 --temperature 0.7 --small-cost 1.0 --large-cost 10.0 --timeout 300 --concurrency 4 --out-jsonl bench/cascade/large_bo8_items.jsonl --out bench/cascade/large_bo8.json

    python3 bench/cascade/measure_cascade.py --mode large_greedy --data "C:/absolute/path/gsm8k.jsonl" --large-model "LARGE_SERVED_MODEL" --large-url http://127.0.0.1:30001 --seeds "0,1,2" --branches 8 --agree-threshold 6 --max-tokens 512 --temperature 0.7 --small-cost 1.0 --large-cost 10.0 --timeout 300 --concurrency 4 --out-jsonl bench/cascade/large_greedy_items.jsonl --out bench/cascade/large_greedy.json

    python3 bench/cascade/measure_cascade.py --compare bench/cascade/cascade.json bench/cascade/large_bo8.json bench/cascade/large_greedy.json

The comparison aggregates all seed rows. It prints both as-scored and
excluding-errors accuracy deltas as cascade minus baseline, raw token totals,
cost per correct, wall seconds per correct, baseline-over-cascade cost and wall
ratios, cascade escalation rate, and exact-number verdict lines. A ratio of
4.0 means the named baseline used four times the metric per correct answer in
that run. The verdict does not claim accuracy parity or generalize beyond the
recorded data and configuration.

## Measurement integrity hardening (2026-07-24)

The historical `correct` field and as-scored accuracy are unchanged. Errored
items remain incorrect for backward compatibility. The added excluding-errors
accuracy removes errored items from the denominator. The added excluding-errors
cost per correct also removes the cost of errored items from its numerator.

`--compare` prints this flag when arm error rates differ by more than 5
percentage points:

    FLAG: ERROR RATE IMBALANCE: <tree> error_rate=<rate> and <baseline> error_rate=<rate> differ by <points> pt; comparison is not apples to apples

Every summary records seeds_nominal and seeds_effective. For each nominal seed,
the harness sorts records by id, hashes the resulting `(id, extracted,
correct)` tuple list, and counts distinct hashes. It prints this flag when the
count collapses:

    FLAG: EFFECTIVE SEED COUNT: mode=<mode> seeds_nominal=<n> seeds_effective=<n>; nominal seeds are not independent evidence

Every summary also records wall_s_total and wall_per_correct. `--compare`
prints the wall ratio immediately after the cost ratio. When those ratios favor
different arms it prints:

    FLAG: COST/WALL REVERSAL: cost_per_correct favors <arm> but wall_per_correct favors <arm> in <tree> versus <baseline>

## CascadeTree target scoring upgrade (2026-07-23)

The fourth mode, cascade_score, keeps the cascade small-tree request,
branch_answers confidence gate, and confident acceptance path unchanged. On
escalation it ranks distinct normalized branch answers by vote count, then by
the best final_scores value among branches with that answer when available,
then lexicographically. It sends the top two candidates to the large server
for target scoring. If fewer than two candidates exist, scoring cannot choose
between alternatives and the mode uses the ordinary greedy fallback.

### Fork-confirmed native scoring contract

This fork exposes POST or PUT /generate and binds its JSON body directly to
GenerateReqInput at python/sglang/srt/entrypoints/http_server.py:842-848. The
current GenerateReqInput schema names text at
python/sglang/srt/managers/io_struct.py:162-163, sampling_params at
python/sglang/srt/managers/io_struct.py:196-197, return_logprob at
python/sglang/srt/managers/io_struct.py:198-199, and logprob_start_len at
python/sglang/srt/managers/io_struct.py:200-202. These are top-level request
fields; max_new_tokens belongs inside sampling_params.

The response builds meta_info.prompt_tokens at
python/sglang/srt/managers/tokenizer_manager.py:1956-1963 and adds logprob
metadata when return_logprob is true at
python/sglang/srt/managers/tokenizer_manager.py:1970-1976. The exact returned
array name is meta_info.input_token_logprobs at
python/sglang/srt/managers/tokenizer_manager.py:2223-2251. Fork tests use
max_new_tokens=0 for prompt scoring at
test/registered/core/test_srt_endpoint.py:228-235 and confirm that, with
logprob_start_len=0, prompt_tokens equals the returned input logprob array
length at test/registered/core/test_srt_endpoint.py:136-140.

The OpenAI-compatible completions adapter explicitly warns that prompt
logprobs are not computed there and directs callers to native /generate at
python/sglang/srt/entrypoints/openai/serving_completions.py:71-75. Therefore
cascade_score uses this native request for each candidate:

    {
      "text": "<prompt><ANSWER_SUFFIX>\nAnswer: <candidate>",
      "sampling_params": {"max_new_tokens": 0, "temperature": 0},
      "return_logprob": true,
      "logprob_start_len": 0
    }

### Candidate-span approximation and accounting

There is no tokenizer call in this offline harness, so it cannot calculate the
exact token boundary between the prefix and candidate. It estimates candidate
token count K as max(1, ceil(character_count(candidate) / 4)), takes the last K
entries from meta_info.input_token_logprobs, and uses their mean logprob as the
candidate span score. This is an explicit approximation, not an exact
tokenizer boundary. Each item records estimated_candidate_tokens and
raw_logprob_lengths so the GPU orchestrator can compare the estimate with the
real tokenization. Fork test code also states that the first input-token
logprob has no meaning at python/sglang/test/runners.py:812-818; taking the
candidate suffix avoids using that first entry for normal nonempty prompts.

Each successful scoring response contributes meta_info.prompt_tokens to
large_prefill_tokens. If prompt_tokens is absent but input_token_logprobs is
present, the harness uses that array length because logprob_start_len is zero
and records the source in scoring_prompt_token_sources. Cost is:

    small_tokens * small_cost
    + large_tokens * large_cost
    + large_prefill_tokens * large_prefill_cost

The new --large-prefill-cost defaults to 2.0 cost units per prompt token,
compared with the existing default 10.0 per large decode token. This encodes
the benchmark assumption that large-model prefill is five times cheaper per
token than large-model decode. It is an assumption, not a measured provider
bill.

Items record large_prefill_tokens, scored_candidates, span_scores,
raw_logprob_lengths, estimated_candidate_tokens, scoring_prompt_tokens,
scoring_prompt_token_sources, scoring_errors, and chosen_by. chosen_by is
vote for a confident small result, score for a successful two-candidate
comparison, or fallback for one greedy large chat completion. Scoring errors
are diagnostic and do not invalidate a successful fallback. If either scoring
call fails or lacks usable input logprobs, exactly one greedy large chat call
uses the same URL and body as cascade. A transport failure that returns neither
prompt_tokens nor input logprobs cannot be token-counted and contributes zero
known prefill tokens; scoring_errors preserves that uncertainty.

### Dry-run proof and commands

The no-network dry run on July 23, 2026 used a one-item prompt, branches=2,
and max_tokens=16. It printed both native scoring calls at
http://127.0.0.1:30001/generate with max_new_tokens=0,
return_logprob=true, logprob_start_len=0, and candidate lines ending in
Answer: <candidate_1> and Answer: <candidate_2>. It also printed the unchanged
greedy fallback at http://127.0.0.1:30001/v1/chat/completions.

    python3 bench/cascade/measure_cascade.py --data "C:/absolute/path/gsm8k.jsonl" --small-model "SMALL_SERVED_MODEL" --large-model "LARGE_SERVED_MODEL" --small-url http://127.0.0.1:30000 --large-url http://127.0.0.1:30001 --branches 8 --max-tokens 512 --large-prefill-cost 2.0 --dry-run

    python3 bench/cascade/measure_cascade.py --mode cascade_score --data "C:/absolute/path/gsm8k.jsonl" --small-model "SMALL_SERVED_MODEL" --large-model "LARGE_SERVED_MODEL" --small-url http://127.0.0.1:30000 --large-url http://127.0.0.1:30001 --seeds "0,1,2" --branches 8 --agree-threshold 6 --max-tokens 512 --temperature 0.7 --small-cost 1.0 --large-cost 10.0 --large-prefill-cost 2.0 --timeout 300 --concurrency 4 --out-jsonl bench/cascade/cascade_score_items.jsonl --out bench/cascade/cascade_score.json

    python3 bench/cascade/measure_cascade.py --compare bench/cascade/cascade_score.json bench/cascade/large_bo8.json bench/cascade/large_greedy.json

No GPU server or tokenizer was available in this lane, so native response
compatibility and the four-characters-per-token span estimate still require
orchestrator validation on the served large model.

# Learned escalation gate wiring

Date: 2026-07-23

Lane: `feat/gated-cascade`

## Scope

- Enhanced `bench/cascade/measure_cascade.py` in place.
- Added `--gate-model`, `--gate-threshold` with default `0.75`, and `--gate-selftest MODEL_JSON ITEMS_JSONL`.
- No non-stdlib runtime dependency was added.
- No commit, push, or pytest run was performed.

## Mirrored feature contract

The harness uses the exact ordered feature list declared in `bench/valuehead/train_gate.py:14-24`:

1. `leader_count`: validated maximum normalized vote count, then converted to float.
2. `voter_count`: count of non-null numeric branch answers, then converted to float.
3. `leader_share`: `leader_count / max(voter_count, 1)`.
4. `n_distinct_answers`: number of distinct normalized numeric answers, then converted to float.
5. `second_count`: second-highest normalized vote count or zero, then converted to float.
6. `margin`: `(leader_count - second_count) / max(voter_count, 1)`.
7. `null_fraction`: `(branches - voter_count) / branches`.
8. `leader_matches_winner`: `1.0` for true and `0.0` for false.
9. `small_tokens_per_branch`: `(small_tokens / branches) / 512.0`.

Parity sources:

- Canonical numeric answer normalization, including whitespace removal, currency/comma/percent removal, sign handling, leading-zero removal, and trailing fractional-zero removal: `bench/valuehead/train_gate.py:33-54`.
- Missing vote handling, numeric vote validation, count validation, feature order, and feature formulas: `bench/valuehead/train_gate.py:70-124`.
- Stored-standardizer application with zero standard deviation mapped to scale `1.0`: `bench/valuehead/train_gate.py:213-220`.
- Numerically stable sigmoid: `bench/valuehead/train_gate.py:223-228`.
- Model JSON field layout, ordered feature names, named means/stds/weights, intercept, and the `512.0` token scale: `bench/valuehead/train_gate.py:704-750`.
- Saved-model validation and named parameter loading: `bench/valuehead/train_gate.py:825-858`.

## Runtime behavior

- With `--gate-model`, `cascade` and `cascade_score` compute the mirrored features, standardize with stored means/stds, apply the logistic intercept and weights, and accept the small answer only when the small response is otherwise valid and `gate_p >= gate_threshold`.
- Empty or missing `branch_answers` auto-escalate with `gate_p: null` and `chosen_by: "no_votes"`.
- Malformed vote payloads do not crash the run. They auto-escalate with a null probability when the parity feature contract cannot be satisfied.
- Gated item records include `gate_p`, `gate_threshold`, and `gated: true`.
- Gated summary config includes `gate_model` as `{basename, sha256}` plus `gate_threshold`.
- Use a fresh `--out-jsonl` whenever the gate model or gate threshold changes because the existing resume key remains `(mode, seed, id)`.
- `--dry-run` is unchanged except for a top-level gate config when a model is set.
- Without `--gate-model`, the original fixed rule remains the only confidence path and no gate fields are serialized.
- Compare keeps its existing common comparability keys. `gate_model` and `gate_threshold` are additionally compared only when at least two compared documents are tree modes (`cascade` or `cascade_score`), so large baselines tolerate the new tree-only config keys.

## Gate selftest

No `AGENTS-GOALs` directory or lane sample records were reachable, so the selftest used five synthetic valid cascade records. Keys are sorted, feature and probability values use fixed 12-decimal strings, and each record is one JSON line.

```json
{"error":null,"feature_names":["leader_count","voter_count","leader_share","n_distinct_answers","second_count","margin","null_fraction","leader_matches_winner","small_tokens_per_branch"],"features":{"leader_count":"6.000000000000","leader_matches_winner":"1.000000000000","leader_share":"0.857142857143","margin":"0.714285714286","n_distinct_answers":"2.000000000000","null_fraction":"0.125000000000","second_count":"1.000000000000","small_tokens_per_branch":"0.500000000000","voter_count":"7.000000000000"},"gate_p":"0.000000352173","id":"sample-1","index":1,"line":1}
{"error":null,"feature_names":["leader_count","voter_count","leader_share","n_distinct_answers","second_count","margin","null_fraction","leader_matches_winner","small_tokens_per_branch"],"features":{"leader_count":"4.000000000000","leader_matches_winner":"0.000000000000","leader_share":"0.571428571429","margin":"0.142857142857","n_distinct_answers":"2.000000000000","null_fraction":"0.125000000000","second_count":"3.000000000000","small_tokens_per_branch":"0.250000000000","voter_count":"7.000000000000"},"gate_p":"0.000003197052","id":"sample-2","index":2,"line":2}
{"error":null,"feature_names":["leader_count","voter_count","leader_share","n_distinct_answers","second_count","margin","null_fraction","leader_matches_winner","small_tokens_per_branch"],"features":{"leader_count":"0.000000000000","leader_matches_winner":"0.000000000000","leader_share":"0.000000000000","margin":"0.000000000000","n_distinct_answers":"0.000000000000","null_fraction":"1.000000000000","second_count":"0.000000000000","small_tokens_per_branch":"0.000000000000","voter_count":"0.000000000000"},"gate_p":"0.370440466907","id":"sample-3","index":3,"line":3}
{"error":null,"feature_names":["leader_count","voter_count","leader_share","n_distinct_answers","second_count","margin","null_fraction","leader_matches_winner","small_tokens_per_branch"],"features":{"leader_count":"2.000000000000","leader_matches_winner":"1.000000000000","leader_share":"0.400000000000","margin":"0.000000000000","n_distinct_answers":"3.000000000000","null_fraction":"0.000000000000","second_count":"2.000000000000","small_tokens_per_branch":"0.500000000000","voter_count":"5.000000000000"},"gate_p":"0.000398595074","id":"sample-4","index":4,"line":4}
{"error":null,"feature_names":["leader_count","voter_count","leader_share","n_distinct_answers","second_count","margin","null_fraction","leader_matches_winner","small_tokens_per_branch"],"features":{"leader_count":"2.000000000000","leader_matches_winner":"0.000000000000","leader_share":"0.333333333333","margin":"0.166666666667","n_distinct_answers":"5.000000000000","null_fraction":"0.250000000000","second_count":"1.000000000000","small_tokens_per_branch":"0.750000000000","voter_count":"6.000000000000"},"gate_p":"0.000103751354","id":"sample-5","index":5,"line":5}
```

The printed features and probabilities were cross-checked against `train_gate.py`'s `require_cascade_features`, `load_model`, `standardize_rows`, and `predict_probabilities` implementations. Result: `PARITY_OK records=5 tolerance=5e-13`.

Additional focused checks:

- `EDGE_OK empty_votes malformed_votes baseline_rule tree_compare_keys`
- `NO_VOTES_OK cascade cascade_score chosen_by=no_votes gate_p=null`
- `BASELINE_IDENTICAL modes=2 confidence_paths=2 records_exact`
- Python AST parse passed.
- CLI help exposes all three gate arguments.
- `git diff --check` passed.

## Uncertainty and environment note

- The real small-tree and large-model endpoints were not called, so live service integration remains unverified.
- A failed Python `tempfile` attempt created an empty untracked directory named `.cascade-gate-_o3wsxqm` at the worktree root with an ACL that this sandbox cannot inspect or remove. It contains no fixture files and does not appear in the tracked diff. The synthetic fixture files used for the successful checks were removed.

## Outcome-level self-speculative search (2026-07-24)

Lane: `feat/spectree`

The `spec_tree` mode changes the target-model compute mode rather than merely
reducing generated token count. A cheap drafter produces complete reasoning
branches. The target scores every branch with teacher-forced prefill, selects
the highest target mean log probability among branches with extractable
answers, and decodes at most one greedy repair. The target never decodes N
reasoning chains. This is the intended contrast with `large_bo8`, which spends
N full target decodes on a memory-bandwidth-bound decode path.

### Same-family coverage fix

The previous cascade paired a Llama-3.1-8B drafter with a Qwen2.5-72B target.
Measured coverage showed that the cross-family drafter never proposed roughly
35 to 53 percent of the answers the target would produce. `spec_tree` therefore
defaults `--small-model` to `Qwen/Qwen2.5-7B-Instruct`, matching the Qwen2.5
family of the intended `Qwen/Qwen2.5-72B-Instruct-AWQ` target. An explicit
`--small-model` still overrides the default. Same-family drafting is the
coverage fix, not a cosmetic model choice.

### Draft, verify, select, and repair paths

Draft first sends `POST /v1/tree/completions` to the small server with the
configured branch count, temperature, and maximum tokens. If the response
contains N per-branch texts in `tree.branch_texts`,
`tree.branch_continuations`, `tree.branch_outputs`, or text-bearing
`tree.branches` entries, the record uses `draft_source: "tree"`. The current
tree response is allowed to expose only winner content plus
`tree.branch_answers`; when N texts are unavailable, the harness sends N
concurrent independent `POST /v1/chat/completions` requests to the same-family
small model and records `draft_source: "samples"`. Draft cost includes both the
initial tree attempt and sample fallback work when both ran.

Target verification prefers one request:

    POST /v1/tree/verify
    {
      "model": "$LARGE_MODEL",
      "prompt": "<prompt><ANSWER_SUFFIX>\n",
      "continuations": ["<draft_0>", "<draft_1>", "..."]
    }

The harness accepts per-continuation rows carrying `mean_logprob` and
`n_tokens`, or equivalent parallel score and token-count arrays. It records
`verify_path: "batched"` and preserves every target score in draft order.

Only an HTTP 404 switches verification to the native fallback. The fallback
first calls `POST /v1/tokenize` on the target with the exact raw verification
prompt. It then sends one concurrent `POST /generate` per continuation:

    {
      "text": "<prompt><ANSWER_SUFFIX>\n<continuation>",
      "sampling_params": {"max_new_tokens": 0, "temperature": 0},
      "return_logprob": true,
      "logprob_start_len": "<exact tokenized prompt count>"
    }

This removes the older four-characters-per-token boundary approximation. The
mean is taken over the returned continuation log probabilities, while the
full `meta_info.prompt_tokens` count for each request is charged as target
prefill. These records use `verify_path: "fallback"`.

Selection considers only drafts with extractable answers and usable target
scores. The highest target mean log probability wins, with the lower draft
index breaking an exact score tie. One temperature-0 target chat completion is
used when no draft has an extractable answer, the top two usable target scores
are within `--repair-margin` nats, or a draft or verifier failure prevents safe
selection. The default repair margin is `0.05`. Records state `repaired` and
`repair_reason`; `chosen_index` remains the best scored draft index when one
was available before repair.

### Cost accounting and records

Each item records the three cost-bearing token classes independently:

    draft_tokens * small_cost
    + verify_prefill_tokens * large_prefill_cost
    + repair_tokens * large_cost

The defaults remain `small_cost=1.0`, `large_prefill_cost=2.0`, and
`large_cost=10.0`. `draft_tokens` also populates `small_tokens`,
`verify_prefill_tokens` also populates `large_prefill_tokens`, and
`repair_tokens` also populates `large_tokens`, so summaries and cost per
correct remain comparable with `large_bo8`. Records additionally include
`draft_answers`, `target_scores`, `target_score_tokens`, `chosen_index`,
`repaired`, `verify_path`, `draft_source`, token-source breakdowns, and
nonfatal path diagnostics. Summary rows report repair count and repair rate.
For `spec_tree`, the shared summary table's `ESC_RATE` column displays that
repair rate.
`--compare` accepts `spec_tree` in the first, tree-mode position and compares
it with `large_bo8`, `large_greedy`, or both.

### Dry-run proof and uncertainty

The no-network dry run used an in-memory item, branches=2, max_tokens=16, the
default Qwen2.5 7B drafter, and a Qwen2.5 72B AWQ target. It printed the tree
draft body, both independent-sample fallback bodies, the single batched verify
body, the tokenize plus two native verification fallback bodies, and the one
greedy repair body. The native bodies showed
`sampling_params.max_new_tokens=0`, `return_logprob=true`, and
`logprob_start_len=<tokenized_prompt_count>`.

The `/v1/tree/verify` route is being added by another lane and is not present
in this worktree, so its live response envelope remains unverified here. The
parser accepts the expected `mean_logprob` and `n_tokens` forms, but the exact
merged route response still needs an orchestrator canary. No GPU endpoint was
called, no pytest command was run, and no commit or push was performed.
