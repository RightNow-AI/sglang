# Roadmap

AutoTree's thesis in one line: on chat serving we tie vLLM and always will, so
we do not sell that. On branching generation at long context - RL rollouts,
agentic search, plan-then-explore - a tree engine reads a shared prefix once
where per-request engines read it per branch. That is the whole bet.

Every claim below is tagged **[shipped]** (measured, in this release),
**[in progress]**, or **[planned]** (projected, not yet measured).

## v0.2 - Tree execution engine  [shipped]

- Mid-generation KV fork with tenant-isolated radix reuse.
- `/v1/tree/completions`, self-consistency vote, majority-locked early stop,
  per-branch trace, disconnect cleanup, fan-out cap.
- n=1 parity with stock SGLang (byte-identical), concurrent-load soak passed.
- Honest positioning: parity on vanilla best-of-n; the value is the capability
  and long-context branching, not a vanilla-throughput or accuracy headline.

## v0.3 - The single-read kernel  [in progress]

The lever that turns parity into dominance. Decode is memory-bandwidth-bound:
today B sibling branches read a shared prefix's KV B times per step; the kernel
reads it once (Hydragen / FlashInfer cascade style), reusing the `merge_state`
primitive already proven in the prefill path.

- Foundations proven: the decomposition is numerically identical to full
  attention (`test_shared_read_decomposition.py`); decode at 100k is
  ~2x KV-read-bound (the regime where the kernel pays); the integration is a
  wiring job in `forward_decode` (see `docs/kernel/shared-read-integration.md`).
- Target claim **[planned]**: >=3x end-to-end and >=6x lower KV bandwidth at
  100k context / B=8, >=5x at B=16, iso-accuracy, Nsight-verified, versus
  vLLM's best emulation (`n=B` single request, prefix cache warm). Larger at
  70B and 1M-context where KV dwarfs weights.
- Refuse-to-publish: any speedup <2x, any strawman baseline, cold-prefill
  results hidden, or accuracy drop >0.5pp.

## v0.4 - RL rollouts  [planned]

Long-context RL rollouts (G>=32 at 32k+ context) are the beachhead where the
kernel's win compounds. The verl adapters exist; this release makes AutoTree the
default rollout backend and demonstrates rollouts/hr/GPU at matched reward.

## Beyond

- Value-head-in-the-loop scheduling trained on production tree traces (the moat
  that strengthens with time) - start logging traces now.
- Deep search (beam / MCTS) at the engine level with transposition reuse - the
  keynote demo, not a near-term market.

## Non-goals

- Beating vLLM on chat serving or i.i.d. best-of-n. We are at parity there by
  construction and do not claim otherwise.
