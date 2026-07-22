# Changelog

All notable changes to the AutoTree tree-execution additions to SGLang.
This project follows semantic versioning for the AutoTree layer; the underlying
SGLang fork tracks upstream separately (see docs/PROVENANCE.md).

## [0.2.0] - 2026-07-22

First public tree-execution release. AutoTree adds one primitive to SGLang:
fork a running generation at token granularity and decode the branches as a
tree, exposed through an OpenAI-compatible endpoint.

### Added
- **Tree execution engine** (`python/sglang/srt/tree/`): mid-generation KV fork
  through the production scheduler intake path; sibling branches radix-share the
  parent prefix.
- **`/v1/tree/completions`** OpenAI-compatible endpoint with a `tree` request
  object (`policy`, `branches`, `budget_tokens`, `scorer`, `fork_at_text`) and a
  per-branch trace in the response (`tokens_spent_per_branch`, `final_scores`,
  `winner_branch_id`, `pruned_count`, `scorer`).
- **Mid-generation delimiter fork** (`fork_at_text`): the parent decodes until a
  delimiter appears in its output, then forks siblings that reuse the parent's
  generated KV under a tenant-isolated, fork-local radix namespace. Caches that
  cannot guarantee that isolation refuse the fork rather than fall back to an
  unsafe path.
- **Self-consistency winner selection**: branches are voted by extracted final
  answer at natural finish; the winning branch's text is returned.
- **Majority-locked early termination**: once finished branches agree beyond
  outvoting, the tree finalizes immediately and reclaims remaining tokens
  (safe by construction with respect to the vote).
- **Marginal-value (EMVPT) branch pruning**: per-branch mean-logprob value proxy
  with an env-tunable margin (default conservative; the naive proxy is measured
  too weak for aggressive pruning - see docs/autotree/how-it-works.md).
- **Robustness**: client-disconnect cleanup that frees all branches, and a hard
  per-tree fan-out cap (`AUTOTREE_MAX_BRANCHES`, default 64).
- **RL rollout adapters** for verl (`integrations/verl_autotree/`).
- **Documentation** (`docs/autotree/`): quickstart, API reference, mechanism
  page, and a migrate-from-vLLM guide.
- **Release hygiene**: NOTICE, PROVENANCE, SECURITY, REPRODUCE.
- **CI**: tree CPU test workflow (`.github/workflows/tree-tests.yml`).

### Verified
- 42 CPU unit tests for the tree runtime and serving.
- On GPU (Llama-3.1-8B): n=1 parity 20/20 byte-identical vs stock SGLang;
  delimiter fork reuses generated KV (measured shared-token reuse);
  concurrent-load soak with delimiter forks, zero scheduler exceptions.

### Honest notes
- On i.i.d. best-of-n, AutoTree is at parity with vLLM (`n>1` already shares
  prompt KV). The measured edge is the mid-generation-fork capability and
  long-context branching. No accuracy-headline claim: GSM8K tree-vs-single is
  within noise. Every benchmark carries its regime; numbers live with raw logs.
