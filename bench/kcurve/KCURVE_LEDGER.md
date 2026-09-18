# Accuracy-versus-k ledger

## Why this measurement now controls the thesis

The premise inverted after cost was recomputed from raw run records instead of
an API token price. On the prior measurements supplied to this lane, large
best-of-8 was already 1.19x cheaper per correct answer in GPU time than greedy
on MATH-hard and 1.45x cheaper on AIME. Eight parallel branches used only
1.14x to 1.17x the wall time while adding 15.0 percentage points of accuracy
on MATH-hard and 9.5 percentage points on AIME.

Those numbers are prior measured inputs to this lane. This worktree did not
rerun the live GPU measurements. Their implication is that more parallel
sampling can reduce compute time per correct answer on hard reasoning tasks.
The unanswered product question is where that benefit stops.

## Cost unit and K_STAR

For every k, the primary cost metric is:

```text
GPU-seconds per correct answer = mean wall seconds per item / accuracy
```

This is the required wall-time proxy for GPU cost. It is not a token-price
estimate and it is not a direct utilization integral from GPU telemetry.
Generated tokens are reported in a separate column because parallel branches
can batch below saturation, so token count does not map linearly to elapsed GPU
time. No token price is used by the k-curve summary or analyzer. The sweep
passes `--small-cost 0 --large-cost 0` only to disable the inherited synthetic
cost-unit fields in `measure_cascade.py`; the k-curve code never reads those
fields.

`K_STAR` is the measured k with the smallest point estimate of mean wall time
per item divided by accuracy. If two k values tie exactly, the smaller k wins.
The analyzer also performs a paired problem-cluster bootstrap. Problems are
resampled with replacement, the same sampled problem indices are used at every
k, and all three seed observations remain nested within each problem. It
prints a 95 percent interval for GPU-seconds per correct, the frequency with
which each k is selected, and a 95 percent interval over bootstrap K_STAR
selections.

The analyzer calls the curve still `FALLING` only when the largest measured k
has a strictly lower point-estimate GPU-seconds per correct than the preceding
k. It calls for extending the sweep when that largest k is also the global
point-estimate K_STAR. If the final segment falls but an earlier k remains the
global minimum, the shape is reported as non-monotonic instead of claiming the
optimum is beyond the sweep.

## Concurrency boundary

Wall time depends on concurrency. Every summary records the
`measure_cascade.py --concurrency` value, and the analyzer refuses to combine
summaries with different concurrency or run configuration. `large_bo8` runs k
branch requests in parallel inside each item worker, so the upper bound on
simultaneous branch requests is approximately `concurrency * k`.

K_STAR is valid only for the recorded concurrency, hardware, served model,
server configuration, and load regime. The batching interpretation applies
only while the server remains below saturation. A different concurrency or a
saturated server requires a separate sweep and a separate K_STAR.

## Locked problem set, resume, and honesty flags

`sweep_k_accuracy.py` shells out to
`bench/cascade/measure_cascade.py --mode large_bo8 --branches k`. It does not
copy answer extraction, equivalence, sampling, voting, request, or timing
logic. The wrapper records the absolute dataset path, full-file SHA-256,
offset, limit, model, endpoint, seeds, temperature, concurrency, and cascade
script hash. The same locked problem rows and three seeds are used at every k.

Each completed k writes `k_NNN_summary.json`. A rerun skips a k when that
summary exists and matches the requested configuration. An interrupted k
retains its raw item JSONL, so the underlying cascade runner resumes completed
problem-seed records. A run manifest prevents an interrupted raw JSONL from
being mixed with a changed dataset, runner, model, seed set, or concurrency.

At k=1, majority vote is a no-op and the single sampled answer is the result.
It is greedy-equivalent only in branch count. It still uses the configured
sampling temperature, default 0.7, rather than the separate `large_greedy`
mode's temperature-0 decoding.

If accuracy at k is below accuracy at k/2, the analyzer prints an explicit
`ACCURACY REGRESSION` flag and labels it as noise or a vote-implementation bug.
The point is never smoothed away. Errored observations remain incorrect, and
the analyzer prints an additional flag when any are present. It also preserves
the cascade runner's nominal-versus-effective seed evidence and flags any k
where the three nominal seeds collapse to fewer distinct result sets.

## Exact commands

From the repository root in PowerShell, replace the served model name if the
endpoint exposes a different identifier. Keep one output directory per task
and concurrency.

```powershell
$LARGE_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
$LARGE_URL = "http://127.0.0.1:30001"

py -3 bench/kcurve/sweep_k_accuracy.py --data bench/tasks/data/math_hard.jsonl --large-model $LARGE_MODEL --large-url $LARGE_URL --answer-mode math --k-ladder "1,2,4,8,16,32" --seeds "0,1,2" --concurrency 4 --max-tokens 512 --temperature 0.7 --timeout 300 --out-dir bench/kcurve/results/math_hard_c4

py -3 bench/kcurve/analyze_kcurve.py --summary-dir bench/kcurve/results/math_hard_c4 --bootstrap-samples 5000 --bootstrap-seed 1729

py -3 bench/kcurve/sweep_k_accuracy.py --data bench/tasks/data/aime.jsonl --large-model $LARGE_MODEL --large-url $LARGE_URL --answer-mode math --k-ladder "1,2,4,8,16,32" --seeds "0,1,2" --concurrency 4 --max-tokens 512 --temperature 0.7 --timeout 300 --out-dir bench/kcurve/results/aime_c4

py -3 bench/kcurve/analyze_kcurve.py --summary-dir bench/kcurve/results/aime_c4 --bootstrap-samples 5000 --bootstrap-seed 1729
```

Run the no-network synthetic self-check with:

```powershell
py -3 bench/kcurve/analyze_kcurve.py --self-check --bootstrap-samples 5000 --bootstrap-seed 1729
```

The self-check constructs two temporary sweeps, one with an interior
turnaround and one still falling at the largest k, prints both analyses, and
deletes the temporary directory.
