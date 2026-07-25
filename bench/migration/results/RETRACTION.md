# The 3.04x migration claim is retracted, killed by the n=1 control

Date: 2026-07-26
Status: MEASURED and RETRACTED before publication.

## The claim I nearly made

The migration bench measured, live on an A10 with Qwen2.5-1.5B over 40 locked
GSM8K items:

| arm | accuracy | tok/item | tok/correct |
|---|---|---|---|
| vllm_style, n=4 plus client voting | 72.5% | 1,299 | 1,792 |
| autotree one call | 55.0% | 324 | **590** |

3.04x cheaper per correct answer. It was about to be the headline.

## The control that kills it

The comparison baseline was wrong. Best-of-4-with-voting is not what a
cost-conscious user runs; `n=1` is. Computed from the Phase 2 branch records,
same model, same locked set:

| arm | accuracy | tok/correct |
|---|---|---|
| **single sample, n=1** | **66.1%** | **455** |
| autotree one call | 55.0% | 590 |
| vllm n=4 plus voting | 72.5% | 1,792 |

**Single sampling dominates the tree on BOTH axes**: 455 versus 590 tokens per
correct answer, and 66.1 versus 55.0 percent accuracy. Cheaper and better.

The 3.04x was real arithmetic against a baseline nobody would pick. Against the
simplest possible baseline, one request with n=1, AutoTree's early termination
does not reach the cost frontier. Not branching does.

## Why the tree loses here

Early termination cuts token spend hard (324 versus 1,299 per item) but it stops
branches before they finish, so accuracy falls further than cost does. The ratio
moves the wrong way. That is a property of this operating point, not a bug: the
knob works, it just does not land anywhere useful on GSM8K with a 1.5B.

## What would have to be true for a migration claim

A tree operating point that beats n=1 on tokens per correct answer. That means
accuracy must fall slower than tokens do, which requires the pruning decision to
remove branches that were going to be wrong. Today it prunes on agreement, and
agreement is already what majority vote uses, so it removes branches that were
going to be right about as often.

None of the three selection results from Phase 2 change this: every gold-free
selector measured between 0.93x and 1.07x of majority vote.

## Process note

This is the fourth promising number today that died under its own control, and
the second in this file that I produced myself. The first version of the
migration bench was also a strawman that inflated AutoTree by 40pp until its own
equivalence check caught it. Both were caught before publication, which is the
only reason the honesty gates are worth having. Raw per-item records for every
arm are committed under `bench/migration/results/` and `bench/verifier/results/`.

## How to reproduce the control

    python bench/verifier/value_signal_probe.py --branches <phase2 branches.jsonl>

The branch-level correct rate and per-branch token count are printed there.
Single-sample tokens per correct answer is (total_tokens / k) / branch_rate.
