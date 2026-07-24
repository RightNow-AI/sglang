# Branch Value Model Ledger

`train_branch_value.py` trains a branch-level probability of correctness from
the per-item JSONL written by `bench/cascade/measure_cascade.py`. It creates one
training row for every non-null string in `branch_answers`. The label is 1 when
that branch answer matches `gold`, and 0 otherwise. With `--answer-mode math`,
matching and sibling answer grouping use `bench/tasks/math_equiv.py`.

All seeds for the same problem `id` stay in one cross-validation fold. This
also keeps every branch from that problem out of the corresponding training
fold. Reported AUC, log-loss, accuracy, and calibration are out-of-fold.

## Feature sets

Feature Set A is preferred and engine-usable:

1. `mean_logprob`
2. `tokens_norm`, defined as branch token count divided by 512

Set A is selected by `--feature-set auto` only when every answered branch has
both values. The loader accepts explicit per-branch mean maps named
`branch_mean_logprobs`, `mean_logprobs_per_branch`,
`mean_logprob_per_branch`, or `mean_logprobs`. It also accepts the tree wire
fields `final_scores` plus `tokens_spent_per_branch`, in which case it computes
`mean_logprob = final_score / token_count`. These maps may be top-level or
inside `tree`. `branch_metrics[branch_id]` entries with `mean_logprob` and
`tokens` or `token_count` are also accepted.

Feature Set B is analysis-only:

1. `answer_agreement`: branches in the same answer-equivalence class divided
   by the number of non-null branch voters
2. `is_leader`: 1 for any branch tied for the largest answer class, otherwise 0
3. `n_distinct_answers_norm`: distinct answer classes divided by voters

Set B needs final sibling answers, so the engine cannot compute it on a branch
in isolation during generation. If any answered branch lacks Set A inputs,
auto mode uses Set B for all rows instead of mixing feature schemas or silently
dropping branches. Use `--feature-set a` to require complete Set A data and
fail if it is incomplete. Use `--feature-set b` for an explicit analysis-only
run.

## Train-on-final, apply-on-partial caveat

Even Set A has a distribution shift. Training rows use each branch's final
mean logprob and final token count. An engine value decision made during
generation uses partial mean logprob and partial token count. A favorable
offline result is therefore necessary but not sufficient for a safe live
pruning or early-stop policy. A live integration must validate the model at
the exact partial-generation checkpoints where it will be applied.

## Model schema

The emitted JSON has exactly this deployment-facing shape:

```json
{
  "type": "branch_value_logistic",
  "feature_names": ["mean_logprob", "tokens_norm"],
  "weights": [0.0, 0.0],
  "bias": 0.0,
  "standardization": {
    "mean": [0.0, 0.0],
    "std": [1.0, 1.0]
  },
  "engine_usable": true
}
```

For Feature Set B, `feature_names` contains its three names and
`engine_usable` is false. Zero standard deviations are allowed and use a scale
of 1 at prediction time. The weights operate on standardized features. The
bias is not regularized.

## Decisive comparison

The report compares the learned model's OOF AUC with raw `mean_logprob` used as
a single-feature ranker on the exact same branch rows and folds. Higher raw
mean logprob is treated as more likely correct. The predefined meaningful
margin is 0.02 AUC.

If the learned model improves by less than 0.02, the report starts with a
prominent `RETIRE THIS LEVER` finding. If raw mean logprob is not available for
every evaluated row, the report says that the decisive comparison is
inconclusive rather than comparing different subsets. This usually happens
when auto mode falls back to Set B on older cascade JSONLs that only retained
`branch_answers` and aggregate `small_tokens`.

## Commands

Run the synthetic grouped holdout. It asserts OOF AUC greater than 0.8:

```powershell
python bench/valuehead/train_branch_value.py --demo
```

On this Windows machine, if `python` resolves to the Microsoft Store alias,
use the installed interpreter directly:

```powershell
& 'C:\Users\jaber\AppData\Local\Python\bin\python3.exe' bench/valuehead/train_branch_value.py --demo
```

From this worktree, the shared result directories currently live under the
sibling `../AutoTree/AGENTS-GOALs`. Train on every largetree JSONL with:

```powershell
$pythonExe = 'C:\Users\jaber\AppData\Local\Python\bin\python3.exe'
$goalRoot = (Resolve-Path ../AutoTree/AGENTS-GOALs).Path
$largeTreeItems = (Get-ChildItem (Join-Path $goalRoot results-largetree) -Recurse -Filter *.jsonl).FullName
& $pythonExe bench/valuehead/train_branch_value.py `
  --items $largeTreeItems `
  --answer-mode math `
  --feature-set auto `
  --out-model bench/valuehead/largetree-branch-value.json `
  --out-report bench/valuehead/largetree-branch-value.txt
```

Train on every costbattle JSONL with:

```powershell
$pythonExe = 'C:\Users\jaber\AppData\Local\Python\bin\python3.exe'
$goalRoot = (Resolve-Path ../AutoTree/AGENTS-GOALs).Path
$costBattleItems = (Get-ChildItem (Join-Path $goalRoot results-costbattle) -Recurse -Filter *.jsonl).FullName
& $pythonExe bench/valuehead/train_branch_value.py `
  --items $costBattleItems `
  --answer-mode math `
  --feature-set auto `
  --out-model bench/valuehead/costbattle-branch-value.json `
  --out-report bench/valuehead/costbattle-branch-value.txt
```

Use `--answer-mode numeric` instead when those runs used the cascade numeric
answer contract. Do not mix math and numeric runs in one training invocation.

Apply a model and write one output row per answered branch. Each output row
contains the branch features and appended `p_correct`. A Set A model emits a
null probability plus `missing_feature_set_a_inputs` for a branch whose live
features are absent.

```powershell
& $pythonExe bench/valuehead/train_branch_value.py `
  --predict bench/valuehead/largetree-branch-value.json `
  --items $largeTreeItems `
  --answer-mode math `
  --out-predictions bench/valuehead/largetree-branch-predictions.jsonl
```
