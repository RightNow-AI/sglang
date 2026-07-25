# First real ThoughtBench measurement (2026-07-20)

Hardware: one H100 PCIe 80GB. Model: Qwen/Qwen3-8B in bfloat16, served by
`autotree serve --engine treekv --device cuda --dtype bfloat16`.
Tasks: 25 numeric MATH-500 problems, difficulty levels 1 to 3, real provenance,
3 seeds. Result files with full sample-level data live in
`thoughtbench/results/`. The task file ships next to them
(`math500-subset25.jsonl`, MATH dataset, MIT license).

| Arm | acc@1 | acc@4 | tokens per correct | KV reuse |
|---|---|---|---|---|
| Sequential best-of-4, 640 tokens per sample | 30.7% | 56.0% | 3,159 | n/a |
| Tree beam-8, 1,920 budget tokens | 21.3% | n/a | 5,011 | **8.79x** |

## What this establishes

- **The mechanism works end to end.** 8.79x KV reuse on real workloads:
  branches share prefix KV exactly as the kernel-level A100 results predicted.
- **A deliberate negative result.** Log-probability winner selection with
  roughly 240 tokens per branch loses to independent sampling on tasks the
  model can already solve. The pluggable value scorer exists to replace that
  selection rule. Any value-guided configuration must beat these numbers.
- **AIME is out of reach at reference-engine speed.** At roughly 3.4 tokens
  per second per stream, AIME-scale thinking budgets truncate every sample.
  Frontier-difficulty rows require production-rate serving.

Protocol, configs, and per-sample data are in the results JSONs. Every file
carries a schema-enforced provenance stamp: fixture data cannot claim to be a
benchmark result, and real data is labeled real. GPU cost of this run was
about 28 USD (8.5 hours at 3.29 USD per hour).
