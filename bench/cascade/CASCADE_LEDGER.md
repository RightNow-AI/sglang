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
and error count. Cascade summaries also report escalation count and rate.
Cost per correct divides by max(correct_count, 1).

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
ignored with warnings. A missing final newline is repaired before appending.

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

The comparison aggregates all seed rows. It prints accuracy deltas as cascade
minus baseline, raw token totals, cost per correct, baseline-over-cascade cost
ratios, cascade escalation rate, and exact-number verdict lines. A ratio of
4.0 means the named baseline cost four times as much per correct answer in
that run. The verdict does not claim accuracy parity or generalize beyond the
recorded data and configuration.

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
