# ISO-Accuracy Pareto Sweep Ledger

## Why one point is not a cost claim

A single AutoTree point and a single best-of-8 point can show parity, a point win, or a point loss. They cannot establish how much cheaper either system is at the same accuracy because neither point reveals the surrounding accuracy-versus-cost tradeoff. An `N x cheaper` claim requires measured curves for both systems and a horizontal comparison at one shared accuracy.

This sweep measures both arms at the same `k` ladder:

- Baseline: `large_bo8` with `--branches k`
- Tree: `large_tree` with `--branches k`
- Default ladder: `1,2,4,8,16`

For `large_bo8`, `k=1` is the greedy-equivalent single-sample rung.

Each arm/k run has its own summary JSON and per-item JSONL. `sweep_k.py` skips a run when its summary JSON already exists, so an interrupted sweep can be resumed without repeating completed rungs. The measurement implementation remains in `bench/cascade/measure_cascade.py`; the sweep driver only invokes it.

## What the analysis does

For every arm/k summary, `iso_accuracy.py` pools seeds into accuracy, cost per problem, and cost per correct. It reads the referenced per-item JSONL to validate those totals and to obtain correctness outcomes for uncertainty estimation.

The headline target is the accuracy of the highest measured baseline `k`. The analysis removes measured tree points that are dominated by a lower-cost point with equal or higher accuracy, then linearly interpolates cost between the two adjacent measured tree Pareto points that bracket the target accuracy. The headline ratio is:

`baseline top-rung cost per problem / interpolated tree cost per problem at the same accuracy`

The reverse readout linearly interpolates tree accuracy at the baseline top-rung cost when that cost is bracketed by measured tree Pareto points.

Interpolation never extends beyond the measured tree accuracy or cost range. If the tree frontier does not bracket the baseline top-rung accuracy, the analysis prints `NO ISO-ACCURACY POINT EXISTS`, reports the accuracy gap, and does not produce a cost ratio. It likewise refuses to extrapolate the reverse matched-cost readout.

The confidence interval uses a paired hierarchical bootstrap. Problems are sampled with replacement, then seeds are sampled with replacement within each selected problem. The same problem and nested seed draws are used across every arm/k point. Each replicate recomputes the accuracies and the matched-accuracy interpolation while keeping the measured cost coordinates fixed. The report includes how many bootstrap replicates contained an in-range iso-accuracy point; the percentile interval is explicitly conditional on those valid replicates.

## Exact commands

```powershell
py -3 bench/pareto/sweep_k.py --data <data.jsonl> --large-model <served-large-model> --large-url http://127.0.0.1:30001 --seeds 0,1,2 --answer-mode numeric --max-tokens 512 --timeout 300 --large-cost 10 --small-cost 1 --ks 1,2,4,8,16 --out-dir bench/pareto/results
```

```powershell
py -3 bench/pareto/iso_accuracy.py --summaries "bench/pareto/results/*_summary.json"
```
