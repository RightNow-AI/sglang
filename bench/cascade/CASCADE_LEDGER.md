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
