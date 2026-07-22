# Shared-read performance ledger

This file is append-only. Add new dated entries; do not rewrite prior results.

## 2026-07-22 - Measurement harness

- Request shape: non-streaming `POST /v1/tree/completions`, one long user prompt,
  `tree.branches=B`, `tree.budget_tokens=B*gen_tokens`, and no `fork_at_text`.
  Omitting `fork_at_text` deliberately selects the post-prefill fork, making the
  padded prompt the shared prefix for every branch.
- Sustained decode: the prompt asks for a long free-form harbor continuation and
  explicitly excludes digits, equations, measurements, dates, numbered lists,
  and short answers. This avoids the numeric-answer majority lock that can end a
  tree after only a few tokens.
- Invalid-run guard: the warmup and every measured repetition must return the
  requested branch count, one token count per branch, and at least `100*B`
  accounted decode tokens in `tree.tokens_spent_per_branch`. Any HTTP, JSON,
  envelope, accounting, or threshold failure emits null metrics, sets
  `valid=false`, prints `INVALID`, and exits non-zero.
- Aggregation: discard one warmup, then report the median measured wall time and
  an observed median per-request total decode-token count (`median_low` for an
  even number of reps, so the harness never invents a fractional token). Derived throughput is
  `median_wall_s*1000/total_decode_tokens` ms/token and its tokens/s reciprocal.
  Shared-prefix length is an explicit approximation using four UTF-8 bytes per
  token because the dependency-free client intentionally does not load a tokenizer.
  The `--mode` value labels the result only; the orchestrator owns server boot
  with `AUTOTREE_SHARED_READ=1` or `0`.
