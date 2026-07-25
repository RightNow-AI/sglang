# Tool-as-drafter mechanical-span ledger

## Result on the existing archive

No real mechanical-span fraction is measurable from the persisted archive.

On 2026-07-25, the requested archive scan covered the two
`results-tputdiag` item files and every `*items*.jsonl` file in
`results-headline` and `results-largetree`: 27 files and 4,868 records. The
records contain extracted answers, gold answers, branch answer summaries,
sample answer summaries, token counts, costs, and timing. They do not contain
raw model completions or reasoning traces.

A broader check covered all 88 JSON and JSONL files under the six
`AutoTree/AGENTS-GOALs/results-*` directories, plus the throughput diagnostic
text logs. It found no full-text fields such as `trace_text`, `reasoning`,
`output_text`, `content`, `response`, or branch/sample trace collections. The
throughput logs contain server and decode metrics, not generated text.

Therefore:

- Real `m`: not available.
- Real bootstrap confidence interval: not available.
- Real tool reproduction rate: not available.
- Real model-wrong/tool-right rate: not available.
- Real implied multiplier: not computed.

Answer strings such as `extracted`, `sample_answers`, and `branch_answers` are
not treated as traces. Measuring them would make the denominator a final answer
rather than the generated reasoning path and would inflate or otherwise distort
`m`.

## Mechanism and cost arithmetic

Tool-as-drafter asks the model to decide what computation is needed, then lets
a deterministic symbolic or arithmetic tool emit only the computation's output
span. It relaxes token-distribution equivalence to outcome equivalence. The
model's setup tokens remain on the critical path.

For approximate trace length `L`, mechanical-output fraction `m`, and drafter
step-cost ratio `d` on the remaining spans, the requested idealized relative
cost is:

`cost = (1 - m) * (L * d + 1) / L + m * 0`

The reported speedup is:

`speedup = 1 / cost`

This is an upper bound. It assumes perfect span detection, correct routing,
zero tool latency, zero splice overhead, no extra model tokens to request the
tool call, and no loss from breaking the model's decode stream.

## What is counted

The setup side of every computation is REASONING. Only the deterministic output
side is eligible to be MECHANICAL.

Examples:

- In `12 * 37 = 444`, only `444` is counted mechanical. `12 * 37` is setup.
- In `3.5 km = 3500 m`, only `3500 m` is counted mechanical.
- In `2 * (x + 3) = 2*x + 6`, only `2*x + 6` is counted mechanical.
- In `Therefore the answer is 444`, only the repeated value `444` is counted.

This boundary is deliberately stricter than counting a whole equation. It
implements the honesty gate that the model must first select and set up the
computation.

## Approximate tokenization

Every token figure and every `m` is labeled approximate. For any text segment,
the estimator is:

`max(whitespace-delimited chunks, ceil(non-whitespace characters / 4))`

The mechanical numerator sums this approximation over non-overlapping detected
output spans. The denominator applies it to the full assistant trace. The
result is not a tokenizer-exact fraction and should not be presented as one.

## Detector variants

Both variants are designed to prefer false negatives over false positives.

### Conservative

1. Pure arithmetic: an explicit equation or result arrow with a numeric
   expression on the left, at least one arithmetic operator, and a numeric or
   simple LaTeX fraction on the right. The tool evaluates the left side and
   checks the stated right side.
2. Unit conversion: an explicit numeric conversion between whitelisted linear
   units in the same dimension. The whitelist covers common length, time,
   mass, and volume units. Temperature and context-dependent conversions are
   excluded.
3. Symbolic simplification: an algebraic equality near an explicit cue such as
   `simplify`, `expand`, `factor`, `reduce`, `evaluate`, or `compute`. SymPy
   checks exact symbolic equivalence.
4. Numeric restatement: a value after `therefore`, `thus`, `hence`, or `so`,
   with an explicit answer/value/result label or one-letter variable, within
   320 characters after a detected numeric computation. It is checked against
   the tool's result, not against the model's prior stated result.

### Liberal

The liberal detector includes every conservative rule and adds:

1. Bare algebraic equalities without a transformation cue, but only when SymPy
   proves the two sides are identical. A bare false equation is not labeled
   mechanical because it may be a condition or setup.
2. A wider conclusion-cue restatement pattern that does not require an explicit
   answer/value/result label.

The liberal number is a range endpoint, not a preferred claim. Neither variant
detects natural-language arithmetic, implicit multi-line derivations, calculus,
matrix operations, general theorem application, arbitrary LaTeX, or units
outside the whitelist. These omissions bias `m` downward.

## Verifiability

Each detected span is evaluated before it is counted:

- Arithmetic and unit conversions use exact numeric comparison.
- Symbolic simplifications use exact SymPy equivalence.
- Numeric restatements are compared with the nearest eligible prior tool result.

`tool exact` is the fraction of detected spans whose stated output matches the
tool result. `model wrong/tool right` is the fraction whose stated output does
not match the deterministic tool result. This is a span-level statement, not a
claim that the whole answer is wrong or that replacing the span alone always
repairs the final answer.

SymPy 1.14.0 was already installed in the current environment and was used for
the self-check. If SymPy is absent, the script uses its own stdlib AST rational
arithmetic evaluator and disables symbolic simplification detection rather than
guessing at equivalence.

## Confidence interval

The script groups all traces and spans by problem ID, samples problem IDs with
replacement, recomputes the aggregate token-weighted `m`, and reports the 2.5th
and 97.5th percentiles from 10,000 deterministic-seeded replicates. This is a
by-problem bootstrap, so multiple traces for one problem stay in the same
resampled cluster.

## Synthetic plumbing validation only

Because no real traces were persisted, `synthetic_traces.jsonl` exercises the
pipeline with correct and incorrect arithmetic, unit conversion, cued symbolic
simplification, liberal-only bare symbolic identity, restatement, and a
reasoning-only trace.

Command:

```powershell
py -3 bench\tooldraft\measure_mechanical_fraction.py bench\tooldraft\synthetic_traces.jsonl --synthetic --show-spans
```

Observed synthetic output:

| Detector | Problems | Traces | Mean `L~` | Mechanical/total approx tokens | Spans | `m~` | 95% problem-bootstrap CI | Tool exact | Model wrong/tool right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative | 7 | 7 | 18.4 | 13/129 | 9 | 10.08% | [4.76%, 14.73%] | 55.56% | 44.44% |
| Liberal | 7 | 7 | 18.4 | 14/129 | 10 | 10.85% | [6.11%, 14.88%] | 60.00% | 40.00% |

Synthetic idealized speedup upper bounds, using the measured synthetic mean
`L~ = 18.4`:

| Detector | `d=0.12` | `d=0.05` | `d=0` |
| --- | ---: | ---: | ---: |
| Conservative | 6.382x | 10.666x | 20.494x |
| Liberal | 6.437x | 10.759x | 20.672x |

These numbers validate parsing, evaluation, aggregation, bootstrapping, and the
cost formula only. They are not evidence about the real workload and must not
be quoted as the measured tool-as-drafter multiplier.

## Required capture on the next GPU run

Persist the raw assistant text before `extract_answer` or any normalization.
The simplest robust format is one JSONL record per generated trajectory:

```json
{"id":"math_hard-0001","mode":"large_bo8","seed":0,"trace_id":"sample-3","trace_text":"<exact raw assistant completion>","generated_tokens":641,"finish_reason":"stop","model":"<model id>"}
```

Equivalent compact record fields are also supported:

- Greedy: `trace_text`.
- Best-of-N: `sample_traces`, aligned with `sample_answers`.
- Tree: `branch_traces`, keyed by branch ID, plus `winner_branch_id`.

Capture requirements:

1. Store the complete assistant completion, not only the extracted answer.
2. Preserve newlines, Unicode math characters, and LaTeX exactly.
3. Store every sampled or surfaced branch trace, not only the winner.
4. Keep problem ID, mode, seed, model, sample or branch ID, generated token
   count, and finish reason next to the text.
5. Keep the prompt separate. `m` uses generated assistant text as its
   denominator, not prompt tokens.
6. Record truncation explicitly. A max-token-truncated trace should remain in
   the data but be identifiable for sensitivity analysis.
7. Do not reconstruct traces from final answers, branch votes, token counts, or
   server decode logs.

If an endpoint does not expose branch text, that run cannot support a
branch-level mechanical-span measurement. Capture should fail loudly or mark
the affected trajectory as missing rather than substituting its extracted
answer.

## Running on future traces

With the standard capture fields:

```powershell
py -3 bench\tooldraft\measure_mechanical_fraction.py path\to\results --show-spans
```

For another schema, pass one or more dot paths:

```powershell
py -3 bench\tooldraft\measure_mechanical_fraction.py traces.jsonl --text-field response.output_text
```

With no positional inputs, the script searches the requested throughput,
headline, and largetree item archives in a local `AGENTS-GOALs` directory and
in the sibling `AutoTree/AGENTS-GOALs` directory.
