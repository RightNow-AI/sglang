# Provenance

AutoTree is a fork of [SGLang](https://github.com/sgl-project/sglang), the
open-source serving engine, licensed under Apache-2.0. AutoTree adds one
capability to SGLang: tree execution, the ability to fork a running generation
at token granularity and decode the branches as a single batched tree.

## Upstream

- Upstream project: `sgl-project/sglang` (Apache-2.0)
- Fork base commit: `3d82dacd580a2313064dd4efce482b9c51ee8698`
- Fork remote: `RightNow-AI/sglang`, branch `tree-engine`

To see exactly what AutoTree changed relative to the base:

```
git diff 3d82dacd580a2313064dd4efce482b9c51ee8698..tree-engine
```

## AutoTree-added paths

These paths are new in AutoTree and do not exist upstream:

- `python/sglang/srt/tree/` — the tree runtime (fork, value pricing,
  majority-lock, self-consistency vote, snapshot channel)
- `python/sglang/srt/entrypoints/openai/serving_tree.py` — the
  `/v1/tree/completions` handler
- `python/sglang/srt/entrypoints/openai/protocol_tree.py` — tree request and
  response models
- `test/registered/unit/tree/` — the tree CPU test suite
- `tools/splice/` — tooling to apply the tree engine onto a stock SGLang
  container image
- `docs/autotree/` — user-facing documentation
- `integrations/verl_autotree/` — the RL rollout backend adapters
- `.github/workflows/tree-tests.yml` — CI for the tree suite

AutoTree also inserts guarded hooks into a small number of upstream files
(the scheduler, batch result processor, tokenizer manager, and HTTP server).
Each hook is inert when no tree request is in flight, so behavior is identical
to stock SGLang on the non-tree path. See `docs/autotree/how-it-works.md`.

## Upstream relationship

AutoTree tracks upstream SGLang and intends to propose the tree-execution
primitive upstream. A weekly upstream-sync check runs in CI
(`.github/workflows/upstream-sync-check.yml`). This is a friendly fork: the
goal is to contribute the primitive back, not to diverge.
