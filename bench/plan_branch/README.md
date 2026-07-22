# AutoTree plan-then-branch benchmark

This standalone harness compares AutoTree with the strongest practical n=B
baseline on the workload AutoTree is meant to improve: a long shared retrieved
prefix, one generated plan, B candidate branches, and a final vote.

## Arms

- **autotree_tree:** one POST /v1/tree/completions request with
  fork_at_text="</plan>", branches=B, an explicit plan/branch/vote budget, and
  scorer="model_vote". The server-generated plan KV can be shared by siblings.
- **vllm_n:** the best-case warm emulation, not a straw baseline. It first calls
  POST /v1/chat/completions with n=1 for the plan, appends that generated plan
  to the same long prefix, calls again with n=B, then makes one compact vote
  call. Declare whether prefix caching was actually on.
- **sglang_fork:** optional; it runs the identical best-case two-phase baseline
  against a separate SGLang-compatible endpoint.

The synthetic retrieval task contains a known answer. Accuracy is exact match
on the normalized FINAL: answer line, while pairwise answer agreement is
reported separately. This makes the 0.5 percentage-point iso-accuracy guard an
actual accuracy check rather than a proxy.

## Dry run (no network)

The dry run prints the complete JSON bodies for all three arms and exits zero.
Because phase 2 and the vote depend on model output, dry-run mode uses clearly
identified deterministic mock plan/candidate outputs so downstream bodies are
exact rather than schematic.

    py -3 bench/plan_branch/plan_branch_bench.py --dry-run

The default dry run uses approximately 8k whitespace tokens and B=4. Supported
campaign values are 8192,32768,100000 context tokens and 4,8,16 branches. The
padding is deliberately described as approximate: without a model-specific
tokenizer the client does not fabricate an exact model-token count.

## Run a campaign

    py -3 bench/plan_branch/plan_branch_bench.py --autotree-base-url http://autotree-host:30000 --vllm-base-url http://vllm-host:8000 --sglang-base-url http://sglang-host:30001 --include-sglang --model your-model --context-lens 8192,32768,100000 --branches 4,8,16 --plan-tokens 128 --branch-tokens 256 --vote-tokens 64 --reps 3 --seeds 1,2,3 --vllm-prefix-caching on --sglang-prefix-caching on --gpu-cost-per-hour 2.50 --output bench/plan_branch/results.json

OPENAI_API_KEY is used when present; --api-key overrides it. Rep 0 is the cold
observation. It remains in the JSON, but only reps 1 and later enter the primary
warm comparison. Run at least two reps; the default of three gives one cold and
two warm measurements.

The GPU accounting assumes one occupied GPU per endpoint. It reports measured
wall-time-derived GPU-hours per 1,000 trees and, when --gpu-cost-per-hour is
supplied, USD cost per 1,000 trees. Completion-token metrics come only from
response usage.completion_tokens; absent usage is null, never estimated or
replaced with zero.

## Output and publication guards

The result JSON contains meta, every trial, and a grouped summary with
mean/median metrics, warm winner and margin, AutoTree-vs-vLLM speedup, accuracy
delta, and agreement. The summary refuses publication when:

- AutoTree speedup over vLLM is missing or below 2x
  ("not dominant, do not headline").
- Any included baseline does not have prefix caching declared on.
- Successful cold and warm observations are not both present.
- Accuracy delta is missing or its absolute value exceeds 0.5 percentage
  points.

Failed requests are preserved as error trials with metric fields set to null.
To print the small Markdown table again from a saved result:

    py -3 bench/plan_branch/plan_branch_bench.py --table-from bench/plan_branch/results.json

## Local gates

    py -3 -m py_compile bench/plan_branch/plan_branch_bench.py
    py -3 -m unittest bench.plan_branch.test_plan_branch_bench
    py -3 bench/plan_branch/plan_branch_bench.py --dry-run
