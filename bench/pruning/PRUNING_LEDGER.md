# Pruning Economics Ledger

This benchmark compares AutoTree server-side tree execution with sequential
best-of-n on the same server and model. The headline metric is generated tokens
per correct answer. It does not measure or modify the retired shared-read
kernel.

## Request shapes

Tree sends one POST /v1/tree/completions per item:

    {
      "model": "$MODEL",
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

Tree generated tokens are the sum of
tree.tokens_spent_per_branch.values(). The item record also captures
tree.pruned_count and tree.winner_branch_id.

Best-of-n sends 8 separate sequential POST /v1/chat/completions requests per
item. Each body has the same model, messages, max tokens, and temperature. Its
seed is base seed + absolute item index * 1000 + sample index. Generated tokens
are the sum of usage.completion_tokens across all 8 calls.

Both modes append this exact suffix to the prompt:

    \nSolve step by step, then give the final numeric answer on the last line as: Answer: <number>

Tree temperature must be greater than zero so branches receive sampling
diversity.

## Answer and voting rules

Extraction takes the last literal Answer: marker and parses the first number
after it. Dollar signs, commas, percent signs, and surrounding whitespace are
removed. Integers and decimals are canonicalized, so 18.0 equals 18. If there
is no marker, extraction uses the last number in the text. If parsing fails,
the extracted answer is null and wrong.

Best-of-n majority vote includes every extracted result, including null. Ties
go to the answer whose first occurrence has the lowest sample index. Any
request error, timeout, malformed successful response, or zero-token successful
response makes the whole item wrong. Reported tokens are still retained.

## Resume and output

Each completed item is appended immediately to --out-jsonl under a write lock,
then flushed and synced. Resume keys are (mode, seed, id). Existing keys are
skipped. A truncated final line is ignored and separated before new lines are
appended. The final --out JSON contains one summary per mode and seed.

Use a fresh JSONL path when changing model, branches, max tokens, temperature,
or data contents because the resume key intentionally contains only mode, seed,
and item id.

## Orchestrator commands

Set the served model name and absolute data path first:

    export MODEL='REPLACE_WITH_SERVED_MODEL_NAME'
    export DATA='/absolute/path/gsm8k_test.jsonl'

Print both exact request families without network access:

    python3 bench/pruning/measure_pruning_economics.py --data "$DATA" --model "$MODEL" --seeds "0,1,2" --branches 8 --max-tokens 512 --temperature 0.7 --dry-run

Run tree:

    python3 bench/pruning/measure_pruning_economics.py --mode tree --data "$DATA" --model "$MODEL" --base-url http://127.0.0.1:30000 --seeds "0,1,2" --branches 8 --max-tokens 512 --temperature 0.7 --timeout 300 --concurrency 4 --out-jsonl bench/pruning/tree_items.jsonl --out bench/pruning/tree.json

Run sequential best-of-n:

    python3 bench/pruning/measure_pruning_economics.py --mode bon --data "$DATA" --model "$MODEL" --base-url http://127.0.0.1:30000 --seeds "0,1,2" --branches 8 --max-tokens 512 --temperature 0.7 --timeout 300 --concurrency 4 --out-jsonl bench/pruning/bon_items.jsonl --out bench/pruning/bon.json

Compare:

    python3 bench/pruning/measure_pruning_economics.py --compare bench/pruning/tree.json bench/pruning/bon.json

The compare command reports tree accuracy minus best-of-n accuracy in
percentage points, best-of-n tokens divided by tree tokens, and best-of-n
tokens-per-correct divided by tree tokens-per-correct. A token ratio of at
least 1.5 with accuracy delta greater than -1.0 point is "pruning wins".
Ratios from 0.9 inclusive to 1.5 exclusive are "marginal". All other cases are
"no win". Comparison stops if the recorded run configs or aggregate item
counts differ.

## 2026-07-23 parallel best-of-n addendum

The best-of-n path remains sequential by default. Passing --bon-parallel fires
all branch requests for an item concurrently. Seeds remain base seed + absolute
item index * 1000 + sample index. Results are consumed in sample-index order,
so majority-vote ties and per-sample error ordering do not depend on completion
order. Generated tokens remain the sum of usage.completion_tokens across every
sample. The item wall time therefore measures parallel latency when the flag is
enabled.

Each newly written item record and the summary config include bon_parallel as a
boolean. Dry-run output includes the same field. Use a fresh JSONL output path
when changing bon_parallel because resume keys remain (mode, seed, id).

Comparison no longer treats concurrency as a comparability key because it is
measurement infrastructure rather than model economics. All other existing
comparability keys remain. When the best-of-n summary records bon_parallel as
true, comparison prints a NOTE that bon wall times are parallel.

Run the strong latency baseline with:

    python3 bench/pruning/measure_pruning_economics.py --mode bon --bon-parallel --data "$DATA" --model "$MODEL" --base-url http://127.0.0.1:30000 --seeds "0,1,2" --branches 8 --max-tokens 512 --temperature 0.7 --timeout 300 --concurrency 4 --out-jsonl bench/pruning/bon_parallel_items.jsonl --out bench/pruning/bon_parallel.json

## Publication figures

Render pooled multi-seed results with per-seed accuracy ranges:

    python3 bench/pruning/plot_pareto.py --summaries bench/pruning/tree.json bench/pruning/bon.json --label bench/pruning/tree.json="AutoTree pruning" --label bench/pruning/bon.json="Sequential best-of-n"

The command writes `pareto` and `tokens_per_correct` in PDF, SVG, and PNG
formats under `bench/pruning/figures` by default. Use `--demo` for a complete
synthetic-data rendering check without benchmark traffic.
