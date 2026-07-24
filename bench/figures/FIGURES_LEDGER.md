# Publication figures ledger

`make_figures.py` turns measured cascade, large-tree, and rollout summary JSONs
into four publication figures. It uses only the Python standard library and
matplotlib. Every rendered figure is written as PDF, SVG, and 300 dpi PNG.

## Figure inputs

### Figure 1: `cost_vs_accuracy`

Consumes one `config + summaries` document per system:

- `config.mode`
- `summaries[].mode`
- `summaries[].seed`
- `summaries[].items`
- `summaries[].correct_count`
- `summaries[].accuracy`
- `summaries[].total_cost_units`

The plotted point is pooled across seeds: total cost units divided by total
items on x, and total correct divided by total items on y. The horizontal and
vertical whiskers span the minimum and maximum per-seed values. A summary whose
mode is `large_bo8` supplies the vote@8 horizontal accuracy guide.

### Figure 2: `cost_per_correct_bars`

Consumes `summaries[].correct_count` and `summaries[].total_cost_units`, plus
`config.mode` and the display label. Cost per correct is computed as aggregate
cost across seeds divided by aggregate correct answers. It is not the mean of
per-seed ratios. If aggregate correct count is zero, the figure is skipped
because the ratio is undefined. The lowest bar uses a darker shade of the same
color, with no red/green pass/fail semantics.

### Figure 3: `capture_vs_cost`

Consumes the same pooled accuracy, correct-count, item-count, and cost fields as
Figures 1 and 2. Modes `large_bo8` and `large_greedy` identify the two baselines.
All other modes are candidate systems.

For candidate `S`:

```text
K = cost_per_correct(S) / cost_per_correct(vote@8)
capture = (accuracy(S) - accuracy(greedy))
          / (accuracy(vote@8) - accuracy(greedy))
```

Capture is plotted as a percent. The shaded preregistered target is exactly
`K <= 0.35` and `capture >= 50%`. Values are not clipped into the box. If
vote@8 does not outperform greedy, capture is undefined and the figure is
skipped.

### Figure 4: `rollout_ess`

Consumes one `config + summary` rollout document per arm:

- `config.mode`
- `summary.generated_tokens_per_rollout`
- `summary.effective_diversity`

The raw bar is `generated_tokens_per_rollout`. The adjusted bar is derived only
from stored summary values:

```text
adjusted_tokens_per_effective_sample =
    generated_tokens_per_rollout / effective_diversity
```

`effective_diversity` is the current rollout harness proxy: mean distinct final
answers divided by rollout count. The chart calls this an ESS adjustment so the
raw and diversity-adjusted token costs remain visually inseparable, but it does
not claim an independently estimated statistical effective sample size. A zero
or missing diversity value makes the adjusted quantity undefined, so the
figure is skipped.

## Legend rule

Scatter points are identified only through legend-keyed series. Do not add
floating point annotations. Clustered points can place an offset annotation
next to a neighboring point and silently assign the wrong system name. That
mislabeling failure has already occurred on real clustered results. Legends
keep point identity tied to marker shape and color instead of screen position.

## Regeneration

Run the complete synthetic pipeline:

```powershell
py -3.14 bench/figures/make_figures.py --demo --out-dir bench/figures/out
```

Regenerate from all summary JSONs under the AGENTS-GOALs result directories:

```powershell
py -3.14 bench/figures/make_figures.py --summaries "AGENTS-GOALs/**/results/**/*.json" --out-dir bench/figures/out
```

Override ambiguous file-stem labels with repeatable path-specific labels:

```powershell
py -3.14 bench/figures/make_figures.py --summaries "AGENTS-GOALs/**/results/**/*.json" --label "AGENTS-GOALs/example/results/large_bo8.json=vote@8" --label "AGENTS-GOALs/example/results/large_tree.json=AutoTree" --out-dir bench/figures/out
```

The script prints every written absolute path. A valid JSON that lacks a
figure's required fields does not receive invented defaults. The affected
figure is skipped with a `SKIP <figure>: <reason>` message.
