# autotree-sdk

`autotree-sdk` is the typed Python client and RL rollout-trace package for
AutoTree's OpenAI-compatible API. It does not run models or provide serving.

## Install for development

```powershell
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest -q
```

The distribution name in this lane is `autotree-sdk`; the import module is
`autotree_sdk`.

## First-class tree completions

```python
from autotree_sdk import TreeClient
with TreeClient("http://localhost:8000") as client:
    result = client.tree_completions(
        messages=[{"role": "user", "content": "Solve the problem."}],
        model="Qwen3-32B",
        branches=4,
        budget_tokens=1024,
        temperature=0.2,
    )
print(result.text, result.branches[result.winner_branch_id].final_score)
```

`tree_completions(...)` targets `/v1/tree/completions`, builds the exact tree
request envelope, and returns a typed `TreeCompletion`. Each entry in
`result.branches` contains its token spend and final score. Stock SGLang
servers that return 404 raise `TreeNotSupportedError` with guidance to use the
AutoTree fork.

`TreeClient.completions(...)` targets `/v1/chat/completions` and passes the
same optional `tree` extension. Streaming methods decode SSE across arbitrary
network chunks, ignore comment keep-alives, recognize `[DONE]`, and yield typed
events. Token events require the wire contract's numeric `logprob`. POST
requests are not silently retried.

Consumers must exhaust a streaming iterator for end-of-stream invariants to be
checked. A fully consumed stream rejects unknown/non-live branch tokens,
non-increasing token indices, missing `done`, mismatched usage, and inconsistent
tree counts with a `TraceInvariantError` whose `violation` attribute is stable.

## RL rollout entry point

```python
from autotree_sdk import rollout

batch = rollout(
    ["Prove that ...", "Write a correct implementation of ..."],
    k=8,
    policy="best_first",
    budget_tokens=4000,
    scorer="reward-head-v1",
    seed=17,
    base_url="http://localhost:8000",
    model="Qwen3-32B",
)

grpo_samples = batch.to_grpo_samples()
preference_pairs = batch.to_rlhf_pairs()
```

`rollout` makes one streamed tree request per prompt and returns a
`RolloutBatch`. Every `RolloutTree` contains its prompt, server usage and tree
summary, plus `RolloutBranch` records with:

- `branch_id`, `parent_id`, and `branch_path`: stable tree provenance.
- `tokens` and `completion`: exact streamed text fragments and their joined text.
- `token_ids`: position-aligned optional model vocabulary IDs. Engines that have
  the sampled ID emit it directly; values remain `None` only when an engine
  cannot provide one. The SDK never invents IDs by retokenizing text.
- `token_indices`: the server-provided positions, preserved independently from
  unavailable tokenizer IDs.
- `token_logprobs` and `cumulative_logprob`: position-aligned logprobs and their
  sum.
- `status`, `pruned`, `prune_reason`, and `merged_into`: terminal trace state.

### GRPO records

`to_grpo_samples()` returns flat dictionaries. Each field has the following
meaning:

- `prompt`: the original string or chat-message list.
- `completion`: root-to-branch streamed token text, including every shared
  prefix segment before the branch's own tokens.
- `token_ids`: position-aligned model vocabulary IDs, with `None` only for
  engines that cannot provide the sampled ID.
- `token_indices`: branch-local server positions for the root-to-branch token
  events, in path order.
- `token_logprobs`: server logprob for every root-to-branch token event.
- `cumulative_logprob`: sum of `token_logprobs`.
- `branch_path`: root-to-current branch ID path.
- `branch_id` / `parent_id`: direct branch identity and fork parent.
- `prompt_index`: group key for the `k` samples from one prompt.
- `status`: `completed`, `pruned`, or `merged` after trace assembly.
- `pruned` / `prune_reason`: rejection state and server reason.
- `merged_into`: destination branch ID for a merged branch.

Pruned branches are included by default so GRPO groups retain rejected
alternatives. Pass `include_pruned=False` to export only live-at-`done`
branches. Merged diagnostics remain opt-in.

### RLHF preference records

`to_rlhf_pairs()` compares all unequal-scored terminal branch pairs within each
prompt. `prompt` and `prompt_index` identify the group; `chosen` and `rejected`
are complete GRPO-shaped sample dictionaries; `chosen_score` and
`rejected_score` preserve the server scores used for ordering. Pruned branches
are included by default because early rejection is useful preference data;
merged branches are excluded.

The wire contract requires `final_scores` to be keyed by branch ID. The typed
client rejects positional arrays while parsing and raises `ExportError` if an
export candidate is missing its branch score instead of guessing identity.
