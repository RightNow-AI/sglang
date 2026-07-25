# The 1000x hunt: honest ceiling is ~3x, and the engine owns 1.3-1.7x of it

Date: 2026-07-26
Status: 9-agent adversarial hunt, PARTIAL. Four of eight agents failed on session
limits, including three of the four refutations, so three survivors were never
killed-tested. Those are marked unvetted and should be treated as unproven.

## Why 1000x is not available

Conservation. On a fixed model and a fixed novel query you cannot produce a
72B-quality answer for a thousandth of a 72B forward pass, because the answer was
not computed. Any large factor must come from NOT doing the work.

Coverage then caps every do-less-work mechanism by Amdahl:

    S = 1 / ((1 - c) + c/R)

c is the fraction of traffic served cheaply, R is how much cheaper. At c=0.5,
R=10 that is 1.82x. At c=0.9 it is 5.3x. **1000x needs R=1000, which is a 39 MB
model.** No arrangement of routing, caching or scheduling turns a 19.5x on half
your traffic into more than 1.9x on the bill.

## Survivors, ranked

**A. Coverage routing on an engine-internal discriminator.** Read the layer-N
hidden state during the prefill you already compute, and decide before token 1
whether the query needs full thinking. Roughly 4 KFLOP against 144 GFLOP per
token, so it is free in a way no client-side router can match. Ceiling 2-3x
realistic, engine-exclusive wedge about 1.35x. Killed if probe AUC on the label
"does full thinking change the final answer" is below ~0.85. That is a different
label and feature space from the 0.72-0.76 we measured for per-branch selection,
so it is genuinely untested. The only survivor whose refutation agent actually
ran.

**B. Paused-session KV parking.** Claimed 4.8x users per GPU at a 21 percent duty
cycle. UNVETTED and suspicious: SGLang HiCache and vLLM LMCache already ship
hierarchical KV offload. This has the exact shape of the trunk-sharing trap,
where we measured a real 1.85x and a 1.00x delta against stock. A two-day test
against stock-with-HiCache decides it before any building.

**C. Occupancy-aware branch admission.** 1.2-1.45x, and worth exactly 0 at
saturation, which is where any customer worth having runs. Sell as free accuracy
on a GPU you already bought, never as cost reduction.

**D. Zero-advantage rollout abort. I VERIFIED THIS AND THE WORKFLOW WAS WRONG.**
GRPO gives an all-same-reward group an advantage of exactly 0, so those tokens
buy no gradient. The workflow claimed 83.3 percent of GSM8K groups are
degenerate. Measured on our own records: 41.3 percent (GSM8K), 59.2 percent
(MATH). It also treated detection as free. Split by what agreement can actually
see:

| | zero-advantage | agreement-detectable | realistic ceiling |
|---|---|---|---|
| GSM8K | 41.3% | 32.7% | **1.49x** |
| MATH L4-5 | 59.2% | **10.0%** | **1.11x** |

On MATH, 49.2 percent of groups are ALL WRONG: every sample agrees on the same
wrong answer. Agreement cannot separate confidently-right from confidently-wrong
without the gold answer, so most of the waste is invisible to the only signal we
have. Same wall as Phase 2, Phase 3, the migration claim and the memo result,
now reached from a fifth direction.

**E. Bitwise determinism.** The only item that genuinely requires owning the
kernels, which is the one asset this team actually built. Sell it for MORE, not
less: regulated replay, eval vendors, RL teams who otherwise cannot reproduce a
training bug. Killed if vLLM ships batch-invariant mode as a flag first.

## Graveyard, so these are not re-proposed

- Trunk sharing: 1.85x real, 1.00x engine-exclusive
- Shared-read kernel: 1.4x slower than stock fa3
- Content-addressed KV dedup at arbitrary offsets: RoPE forbids it
- Position-independent KV reuse: capped 1.05x, prefill is 1.5% of a reasoning request
- Precompute at ingest: prompt caching, already shipped everywhere
- Exact-key memo: 0.0% on agentic, definitionally not empirically
- Tool-as-drafter: 1.05x, mechanical fraction 0.087 not the guessed 0.75
- Speculation generally: M*/B = 1.13x at serving batch 32
- Idle-time background compute: buys a 1.55x latency regression
- Branching itself: 4-6x WORSE than single sampling per correct answer, even with
  a perfect oracle selector

## What I would actually do

Test A then B, both cheap, both before building anything.

A needs a probe trained on hidden states with a preregistered out-of-fold AUC
gate at 0.85. B needs one two-day benchmark against stock-with-HiCache at a fixed
duty cycle.

If A clears its gate, the honest product is a 2-3x cost router with a 1.35x
engine wedge. That is a real business and it is not 1000x. If A fails, the engine
story is finished and what remains is E, sold as a premium rather than a
discount.
