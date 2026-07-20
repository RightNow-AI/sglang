# verl AutoTree rollout adapter

This package lets a verl deployment select the AutoTree-enabled SGLang rollout
worker with `rollout.name: autotree`. It is intentionally small: SGLang keeps
ownership of server lifecycle and CUDA-IPC weight synchronization, while the
HTTP server adapter routes requests carrying a `sampling_params.tree` block to
the engine's local tree-generation flow.

## Install

Use an image containing the AutoTree SGLang engine build, then install this
package from the SGLang checkout:

```bash
pip install -e integrations/verl_autotree
```

The Python package itself can be imported without verl, SGLang, Ray, PyTorch,
or CUDA. Those dependencies are resolved only when verl loads an adapter class.

## Patch the verl registries

Run the idempotent patcher against the verl checkout:

```bash
python -m verl_autotree.patch_verl /path/to/verl
```

It inserts exactly these two registry lines:

```python
("autotree", "async"): "verl_autotree.adapters.AutoTreeServerAdapter",
RolloutReplicaRegistry.register("autotree", lambda: __import__("verl_autotree.adapters", fromlist=["AutoTreeReplica"]).AutoTreeReplica)
```

The script deliberately does not edit SGLang-only control-flow checks. For verl
at recon commit `6a6242f`, review and mirror the `sglang` behavior for
`autotree` at:

- `verl/checkpoint_engine/base.py:306`
- `verl/workers/rollout/replica.py:394`
- `verl/workers/rollout/llm_server.py:520`
- `verl/workers/rollout/replica.py:327` (reuse the `_load_sglang` vLLM-mock setup)

Line numbers are pinned to that commit and may move in a newer verl checkout.

## Configure rollout

The normal verl rollout selection and sample count are unchanged:

```yaml
rollout:
  name: autotree
  n: 4
```

Tree execution is selected per request by adding this block to the request's
sampling parameters:

```yaml
tree:
  policy: best_first
  branches: 4
  budget_tokens: 1024
  scorer: logprob
```

`policy`, `branches`, and `budget_tokens` are required. `scorer` is optional.
Requests without a `tree` block defer unchanged to verl's SGLang server path.

## Row-count contract

verl expands `rollout.n: k` into `k` requests before calling the backend. Each
request must therefore return one row. Internal branch pruning must return that
request's best-effort winner row; the backend must never return fewer than the
`k` requested rows. AutoTree branches are internal to a request and do not
replace verl's upstream `n` expansion.

## Current status and seam

Weight synchronization is inherited unchanged from verl's SGLang
`ServerAdapter`. Tree generation requires the AutoTree engine build providing
`sglang.srt.tree.TreeGenerateReqInput`, `TreeParams`, and `TreeResult`; a stock
engine without that feature raises a clear `NotImplementedError` naming the
missing `/v1/tree/completions` engine feature. The adapter calls the local
`tokenizer_manager.generate_request` tree flow and maps its winning branch to a
single verl `TokenOutput`. Winner log probabilities are forwarded when the
engine result exposes them; the current tree result contract does not require
them.

The four pinned SGLang-only call sites listed above remain an operator patch
until verl accepts `autotree` as a built-in backend.

## CPU-only tests

The adapter tests use fake verl/SGLang base classes and require neither CUDA nor
the frameworks themselves:

```bash
python -m pytest integrations/verl_autotree/tests -q
```
