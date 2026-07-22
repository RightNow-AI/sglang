# GPU shared-read verification ledger

## 2026-07-22 - start

- Lane: `gpuverify`; writable checkout: `C:\Users\jaber\RightNow-Full\sglang-tree-gpuverify` (`feat/kernel-verify`).
- The brief named sibling `C:\Users\jaber\RightNow-Full\sglang-tree-kernel`, but this session grants write access only to the verification checkout, so deliverables are being created here.
- Scope locked to `bench/kernel/verify_shared_read.py` plus this append-only ledger. No engine edits, pytest, commit, push, or live-server execution.
- Grounded endpoint: `POST /v1/tree/completions`; request tree parameters include `branches`, `budget_tokens`, and `fork_at_text`.
- Honesty rule: the current non-streaming wire schema may omit per-branch output token IDs. The verifier will preserve IDs from recognized response trace shapes when available; otherwise it records `null` and fails the correctness gate instead of inferring IDs.

## 2026-07-22 - request and capture scaffolding

- Added the standalone stdlib client skeleton and conditional CLI for correctness, compare, measurement, and dry-run modes.
- Added eight fixed greedy requests with `fork_at_text=FORK:`, `branches=4`, stable medium filler, per-task seeds, and exact request bodies retained in output.
- Added B=4/8/16 long-prefix request construction and response extraction that accepts explicit branch token traces but never reconstructs missing IDs.
- Patch-tool note: the direct patch helper rejected writes in this Windows session; edits continued through the same Codex apply-patch engine invoked locally. Temporary patch probes were removed.

## 2026-07-22 - comparison and effect paths

- Added strict per-task comparison: request body equality, UTF-8 winner bytes, complete branch ID sets, and exact per-branch token-list equality.
- Added measurement accounting from `tree.tokens_spent_per_branch`, with optional newly appended `gen throughput (token/s)` log samples as a fallback.
- Added per-B wall time, aggregate tok/s, throughput source, and B=16/B=8 ratio. Missing throughput stays `null` and now makes the measurement verdict fail.
- Initial verification: Python 3.14 compile passed; CLI help passed; correctness dry-run produced 8 B=4 greedy requests; measurement dry-run produced B=4,8,16 and honored repeats/mode stamps.

## 2026-07-22 - review correction

- CodeRabbit CLI was unavailable, so no external review ran.
- Local second pass tightened measurement integrity: wall-clock aggregate tok/s is accepted only when `branch_count` equals the requested B and `tokens_spent_per_branch` contains exactly branch IDs 0 through B-1.
- This prevents a partial tree trace from being reported as a complete aggregate throughput measurement.
- The B=16/B=8 ratio is emitted only when both rows use the same throughput source, avoiding a wall-timing versus server-log mixed ratio.

## 2026-07-22 - final verification

- `py -3` in-memory compile passed for the 727-line script; no pytest was run.
- Correctness on/off dry-runs each emitted 8 requests; serialized request bodies were byte-for-byte equivalent after excluding the mode-only envelope. All bodies used temperature 0, B=4, and `fork_at_text=FORK:`.
- Mocked HTTP flows passed for complete on/off capture, strict compare, B=4/8/16 measurement, and server-log throughput parsing.
- Negative mocked flows passed: omitted branch token IDs were written as `null` and correctness exited FAIL; missing aggregate token accounting produced a `null` ratio and measurement exited FAIL.
- Final scope remains exactly the verifier script plus this ledger. No engine code, tests, commits, pushes, or live GPU server state were touched.
