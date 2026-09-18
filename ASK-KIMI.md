# Paste this to Kimi K3

I am building AutoTree, a tree-execution serving engine as a fork of SGLang.
Working directory: C:\Users\jaber\RightNow-Full\sglang-tree-integration
Branch: integration/cascade-v1

Be brutally quantitative. Label every guess [GUESS]. Do not flatter me and do not
chain unvalidated multipliers, I have been burned by that three times.

## WHAT THE ENGINE IS

A fork of SGLang that executes reasoning as a TREE instead of independent chains:
fork a request mid-generation, run k branches that share the trunk's KV, prune
branches that diverge from the emerging consensus, enforce a token budget across
the whole tree, and select a winner. Exposed as an OpenAI-compatible server plus
a /v1/tree/completions extension. With no tree params it behaves exactly like
stock SGLang, so adoption is a base_url swap.

## STATE TODAY (all measured, not estimated)

Engine hardening, verified by running the tests, not self-reported:
- tree unit tests 91 -> 129
- end to end acceptance gate 11/11 PASS against a live server (Lambda A10,
  Qwen2.5-1.5B, fork loaded via PYTHONPATH). Checks: stock chat + streaming
  compatibility, malformed tree params rejected 4xx across six shapes, three
  abrupt mid-stream socket disconnects survived, mixed stock+tree concurrent
  load, token budget enforced, still healthy after every hostile probe
- AutoTree OFF proven inert on the hot path (the scheduler calls the hooks
  unconditionally, so this is the base_url promise)
- consensus pruning now runs DURING generation, not only at winner selection
- fixed: branch misattribution across concurrent trees, adaptive width expanding
  once instead of laddering, verify.py discarding per-token values, parent hold
  generating unused tail tokens
- shared-read attention default flipped OFF (it was falsified as 1.4x SLOWER
  than stock fa3 and was defaulting ON)

## MEASURED NEGATIVES, DO NOT CONTRADICT WITHOUT ARITHMETIC

H100 SXM5, Qwen2.5-72B-AWQ, CUDA graphs ON, 2026-07-25:
- KV wall is 13 concurrent sequences at ~8k context (103,948 KV tokens fp16).
  fp8_e5m2 doubles it exactly: 207,896 tokens, 27 seats, peak decode 396 -> 574
  tok/s. fp8 is SLOWER at batch 1 (41.5 vs 51.0) and only pays at the wall.
- CUDA graphs worth 1.4x at batch 1 (51 vs 35.5 tok/s).
- TRUNK SHARING, the decisive one. 48 sequences, ~7500-token shared trunk:
  unique prompts 14 admitted / occupancy 1.00 / 420 tok/s;
  shared trunk 48 admitted / 0.54 / 1014 tok/s (1.85x KV saved);
  stock SGLang n=8 48 admitted / 0.53 / 1011 tok/s (1.89x).
  RadixAttention ALREADY captures the entire benefit. Our engine-exclusive
  delta over a client that re-prompts is 1.00x, and 1.17x over a naive n=8
  client. This FAILED a preregistered 1.5x bar.
- MATH-500 level 4-5, n=60: greedy 48.3% @733 tok/item (1517 tok/correct);
  single sample T=0.7 53.3% @718 (1346 tok/correct, the cheapest arm);
  best-of-8 61.7% accuracy with 68.3% COVERAGE @5749 (9323 tok/correct).
  Greedy beats best-of-8 by 4.45x on correct answers per GPU-hour.
- Selection gap 6.7pp. Majority vote costs 9323 tok/correct, oracle selection
  costs 1052. An 8.9x gap sits between them. A PERFECT scorer is worth only
  1.28x over plain single sampling on this workload.
- Also falsified: tool-as-drafter (mechanical token fraction measured 0.087 on
  GSM8K calculator-annotation ground truth versus a guessed 0.75, giving 1.05x);
  speculation ceiling at serving batch B is about 1.13x at B=32.
- mean-logprob does NOT predict correctness (failed three separate ways).
  Branch AGREEMENT does (91.3% on GSM8K).
- Content-addressed KV dedup at arbitrary offsets is impossible for RoPE models
  because KV is position-dependent, so the "KV CDN" idea is dead on physics.

## THE HONEST POSITION I HAVE LANDED ON

We do NOT beat vLLM or SGLang on throughput and I have stopped claiming it.
What we have that nobody ships is tree semantics as a first-class primitive, and
one quantified opening: in VERIFIABLE domains (code with a test suite, SQL with
execution checks, math with a symbolic checker) selection is free and exact, so
you sit at the oracle row: 1.28x cheaper per correct answer AND +15pp accuracy
versus single sampling. There, branching is the cheap option rather than a 6.9x
premium.

## MY TWO QUESTIONS

1. HOW IS IT GOING? Given the state and the negatives above, judge this project
   honestly. Is "the engine that makes test-time compute pay in verifiable
   domains" a real position or a consolation prize? What would you need to see
   measured in the next two weeks to believe it, and what would make you tell me
   to stop? Name the specific number that would change your mind either way.

2. GIVE ME TWO IDEAS. Not seven. Two concrete things to build next, ranked, each
   with: the mechanism, the arithmetic for what it is worth if it works, what
   kills it, how many engineer-days, and whether a competent client could
   replicate it in under a week (if yes it is not a moat, say so). At least one
   must be buildable on this fork in under two weeks. Prefer ideas that exploit
   the 8.9x selection gap or the verifiable-domain shape, since those are the
   only quantified openings I have left. If you think both of my current
   directions are wrong, say that instead and give me your two.
