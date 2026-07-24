# Agentic repetition benchmark ledger

## Claim boundary

This benchmark measures exact reuse across a multi-call workload. It does not
claim that an engine makes a single novel problem 20x or 100x cheaper. On a
non-repeating workload, the memo hit rate is approximately zero and the memo
multiplier is 1x.

If a fraction `h` of calls are served from a memo, the ideal equal-cost,
zero-lookup-cost multiplier is:

```text
1 / (1 - h)
```

Examples:

```text
h = 0.80  ->   5x
h = 0.95  ->  20x
h = 0.99  -> 100x
```

The measured token ratio can differ because calls generate different numbers of
tokens. The measured wall ratio also includes memo hashing and lookup time. A
miss still pays the full server request cost.

## Trace construction

`build_agentic_trace.py` reads the existing MATH hard, AIME, and GSM8K JSONL
sets. It writes a JSONL header followed by call records with exactly these call
fields:

```text
call_id, task_id, step, prompt, gold
```

Controlled mode rounds `calls * repeat_rate` to the nearest feasible repeat
count. Every controlled repeat preserves `task_id`, `step`, `prompt`, and
`gold`, and its prompt is byte-identical to an earlier call. The header records
the requested rate, realized rate, strict repeat rate, source paths, source row
counts, and source SHA-256 hashes.

Natural mode creates several calls per workflow. Later workflows insert prior
solve prompts verbatim as recurring subproblems. Those calls can have a new
workflow `task_id` and `step`, but the memo-visible prompt bytes are identical.
Natural mode measures the realized rate instead of targeting `--repeat-rate`.

The generated hard-task JSONL is ignored by git. The builder first checks this
worktree's canonical `bench/tasks/data` paths. If they are absent, it selects a
sibling worktree copy and records that fallback in the trace header. Explicit
`--math-data`, `--aime-data`, and `--gsm8k-data` paths override discovery.

## Exact-match memo

The memo key is SHA-256 over canonical UTF-8 JSON containing:

```text
prompt + model + sampling parameters
```

Sampling parameters include endpoint mode, maximum tokens, temperature, seed,
and tree parameters when tree mode is selected. JSON keys are sorted and NaN is
rejected. There is no whitespace normalization, semantic matching, embedding
lookup, fuzzy matching, or cross-parameter reuse. Exact match is intentionally
strict so every hit is auditable and no accuracy loss is hidden behind an
approximate cache decision.

The hit rate is a workload property. The same model and server can show a high
hit rate on recurring subproblems and a zero hit rate on novel traffic. The
server does not create the repetition measured here.

## Scoring and cost

Prompts require a final line of `FINAL: <answer>`. Scoring supports exact
numeric values, simple fractions, and conservative normalized symbolic strings.
It does not claim full theorem-prover equivalence. This can undercount correct
MATH answers written in a different but equivalent form.

Chat generated tokens come from `usage.completion_tokens`. Tree generated
tokens prefer the sum of `tree.tokens_spent_per_branch` and fall back to
`usage.completion_tokens`. A token multiplier is unavailable if any real
request lacks a valid token report. Cost units are one unit per reported
generated token.

`--out-jsonl` is append-only and resumable by arm, seed, and `call_id`. The run
configuration hash includes the trace hash, model, endpoint, sampling settings,
seeds, and stub status. A different configuration is refused instead of mixed
into the same record file.

## Honesty gates

The tool always:

1. Prints the realized trace repetition rate.
2. Prints the memo hit rate beside every reported multiplier.
3. Prints the accuracy delta in percentage points.
4. Emits a loud warning when memo accuracy is more than 1 percentage point
   below no-memo accuracy.
5. States that the multiplier is a function of workload repetition, not of the
   engine.
6. States that novel non-repeating traffic has approximately zero hit rate and
   a 1x multiplier.
7. Labels `--stub` output as a plumbing test that must never be reported as
   model or engine performance.

## Exact commands

Build a controlled 80 percent repetition trace:

```text
python bench/agentic/build_agentic_trace.py --mode controlled --repeat-rate 0.80 --calls 100 --seed 0 --out bench/agentic/trace_r80.jsonl
```

Build a measured natural trace:

```text
python bench/agentic/build_agentic_trace.py --mode natural --calls 100 --natural-reuses 2 --seed 0 --out bench/agentic/trace_natural.jsonl
```

Inspect the exact live chat request and memo key without sending it:

```text
python bench/agentic/measure_memo.py --trace bench/agentic/trace_r80.jsonl --model MODEL --base-url http://127.0.0.1:30000 --mode chat --max-tokens 512 --temperature 0 --seeds 0 --dry-run
```

Replay both live chat arms:

```text
python bench/agentic/measure_memo.py --trace bench/agentic/trace_r80.jsonl --model MODEL --base-url http://127.0.0.1:30000 --mode chat --max-tokens 512 --temperature 0 --seeds 0 --arm both --out-jsonl bench/agentic/live_chat_records.jsonl --out bench/agentic/live_chat_summary.json
```

Replay both live tree arms:

```text
python bench/agentic/measure_memo.py --trace bench/agentic/trace_r80.jsonl --model MODEL --base-url http://127.0.0.1:30000 --mode tree --branches 8 --max-tokens 512 --temperature 0 --seeds 0 --arm both --out-jsonl bench/agentic/live_tree_records.jsonl --out bench/agentic/live_tree_summary.json
```

Run separate arms and compare their summary files:

```text
python bench/agentic/measure_memo.py --trace bench/agentic/trace_r80.jsonl --model MODEL --mode chat --arm nomemo --out-jsonl bench/agentic/nomemo_records.jsonl --out bench/agentic/nomemo.json
python bench/agentic/measure_memo.py --trace bench/agentic/trace_r80.jsonl --model MODEL --mode chat --arm memo --out-jsonl bench/agentic/memo_records.jsonl --out bench/agentic/memo.json
python bench/agentic/measure_memo.py --compare bench/agentic/nomemo.json bench/agentic/memo.json
```

Run the deterministic offline plumbing self-check with no network:

```text
python bench/agentic/measure_memo.py --trace bench/agentic/trace_r80.jsonl --model stub-model --mode chat --seeds 0 --stub --arm both --out-jsonl bench/agentic/stub_records.jsonl --out bench/agentic/stub_summary.json
```
