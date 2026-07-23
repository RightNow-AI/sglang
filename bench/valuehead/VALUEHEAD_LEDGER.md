# Escalation-gate calibrator ledger

## Scope

- Branch: `bench/value-head`
- Implementation: `bench/valuehead/train_gate.py`
- Runtime: Python standard library only
- Inputs: one or more cascade item JSONL files from `bench/cascade/measure_cascade.py`
- Outputs: model JSON, text report, and optional scored JSONL

## Label contract

The target is whether the small tree's winner answer is correct. The trainer copies the cascade harness's canonical-decimal normalization and applies it independently to `small_winner_answer` and `gold`.

The label is 1 only when the normalized small answer is non-null and exactly equals normalized gold. A null or non-numeric small answer is incorrect. The record's `correct` field is never used as the training label because, on escalated records, it describes the large model's final answer.

## Features

The saved model uses these ordered features:

1. `leader_count`
2. `voter_count`
3. `leader_share`
4. `n_distinct_answers`
5. `second_count`
6. `margin`
7. `null_fraction`
8. `leader_matches_winner`
9. `small_tokens_per_branch`, already divided by 512

Means and population standard deviations are saved in the model JSON. A zero standard deviation uses a scale of 1 only while transforming.

## Training and reporting

- Logistic regression is implemented directly with `math`, deterministic zero initialization, L2 regularization, and a fixed learning-rate schedule.
- K-fold metrics are computed from held-out predictions. Standardization is fit on each fold's training rows only.
- The report includes per-fold and overall accuracy, rank-based AUC, log-loss, ten equal-width calibration bins, threshold analysis from 0.05 through 0.95, and the fixed `leader_count >= 6` comparison.
- Decision analysis uses observed small labels for accepted rows. For projected escalations it uses the aggregate final-answer accuracy from originally escalated, error-free input rows. This is an explicit counterfactual assumption because large answers were not recorded for originally accepted rows.
- Projected cost includes every row's recorded small tokens and the observed mean large tokens from originally escalated rows. Missing `large_tokens` values use `--assumed-large-tokens`.
- Input records whose `error` is not null are skipped during training. Predict mode preserves them and writes `p_small_correct: null`.
- Reports print sample counts and show a LOW-DATA banner when usable n is below 100.

## Verified commands

```text
py -3 -m py_compile bench/valuehead/train_gate.py
py -3 bench/valuehead/train_gate.py --help
py -3 bench/valuehead/train_gate.py --demo
```

Verified demo result on 2026-07-23 with default seed 1729:

- Synthetic records: n=500
- Held-out out-of-fold AUC: 0.958829
- Assertion: passed because 0.958829 > 0.8
- Overall out-of-fold accuracy: 449/500, 89.8%
- No server, GPU, sklearn, torch, or pytest was used

A file-mode smoke test trained on 39 usable synthetic rows plus one skipped error row, wrote a model with all nine features, and scored 40 output rows as 39 numeric probabilities plus one null probability.

## Known limits

- Threshold economics are projections, not observed counterfactual cascade runs.
- Calibration and threshold choice can be unstable on small or distribution-shifted inputs; the report exposes n and calibration bins rather than claiming production calibration.
- No real recorded benchmark JSONL was supplied in this lane, so no real-benchmark accuracy or cost claim is recorded here.
