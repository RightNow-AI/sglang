# Contributing

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) and Rust 1.93 or newer.

```bash
git clone https://github.com/RightNow-AI/autotree
cd autotree
./scripts/verify-local.sh
```

The verify script builds the Rust scheduler wheel and runs every suite,
including the server tested against the real engine. It must pass before and
after your change. On Windows use `scripts/verify-local.ps1`.

## Rules

- Every bug fix ships a test that fails against the broken code.
- Every feature ships tests and documentation in the same PR.
- Do not overstate results. Benchmark artifacts carry provenance stamps;
  fixture data must never be presented as a measured result.
- The Tree-KV contract in `core/docs/tree-kv-spec.md` is normative. If your
  change alters engine or scheduler semantics, update the spec in the same PR.
- Keep commits small and messages direct.

## Reporting issues

Open a GitHub issue with reproduction steps. For crashes, include the full
traceback and your platform, Python, torch, and driver versions.
