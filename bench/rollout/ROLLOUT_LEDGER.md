# Rollout Forest estimator ledger

## Claim contract

The tree arm uses a benchmark-side Horvitz-Thompson survival sample over the
trajectory surface returned by `/v1/tree/completions`. Branches inferred to be
kept by the tree policy have inclusion probability 1. Other surfaced branches
receive independent, deterministic-seeded Bernoulli survival with probability
at least `--survival-floor`, which defaults to 0.1. Every trajectory record
contains its inclusion probability and inverse-probability weight. A surviving
trajectory with inclusion probability pi has weight `1 / pi`.

The survival floor is required for unbiasedness. If a branch can have inclusion
probability zero, no finite importance weight can recover its contribution to
the full-N estimator. A zero-floor run therefore cannot support a Rollout Forest
multiplier claim. The harness also refuses the claim when the endpoint does not
surface a complete N-branch trajectory population, because aggregate prune and
merge counts are not enough to reconstruct missing branch propensities.

## Effective sample size

For surviving weights w_i, the harness reports Kish effective sample size:

`ESS = (sum_i w_i)^2 / sum_i (w_i^2)`

It also reports `ESS / N` over all intended trajectories and:

`tokens_per_effective_sample = total_generated_tokens / ESS`

The only claimable multiplier is the ESS-adjusted token ratio:

`(independent tokens / independent ESS) / (tree tokens / tree ESS)`

Raw rollouts per second, raw trajectory count, and the unadjusted token ratio
are diagnostics only. They are not Rollout Forest multiplier claims.

## Answer-distribution equivalence

The harness forms inverse-probability-weighted answer distributions for the
tree and independent arms. It pairs records by prompt ID and seed, computes TV
within each pair, and reports the mean paired distance. This is also the TV of
the joint prompt-answer distributions when prompts receive equal mass. It does
not pool unrelated answer strings across different questions for the claim
gate. For distributions P and Q, total variation distance is:

`TV(P, Q) = 0.5 * sum_a |P(a) - Q(a)|`

The compare verdict refuses a win if the ESS-adjusted token ratio is at most
1.2x, if answer TV distance exceeds 0.25, or if either estimator is incomplete.
Passing these benchmark gates supports a like-for-like answer-distribution
claim for the measured workload. It does not establish full trainer equivalence.

## Remaining validation

Full validation remains future work: compare the weighted Rollout Forest
gradient estimator with the full-N independent estimator on a real trainer and
measure gradient-estimator cosine similarity. Until that experiment is run,
the harness must not claim end-to-end RL training equivalence.
