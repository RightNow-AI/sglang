# Shared-Read Performance Sweep Ledger

Entries are append-only.

## 2026-07-23 - Initial shared-read sweep runner

- Added `sweep_shared_read.py`, a pure-standard-library client for `POST /v1/tree/completions`.
- The request uses prefill-fork mode by omitting `fork_at_text`; the padded prompt is therefore the shared prefix for every branch.
- Fixed-length decode is requested with `ignore_eos: true`, deterministic sampling, and a free-form continuation prompt that avoids numeric-answer majority locking.
- The runner checks `tree.tokens_spent_per_branch` after every request. It prints `WARN` and marks the row invalid when total decode tokens differ from `branches * gen_tokens`, so early termination is never presented as an apples-to-apples result.
- Repetition zero is a discarded warmup. The reported wall time is the median of the requested measured repetitions.
- The runner does not import SGLang, start a server, or change engine code.