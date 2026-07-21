# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to security@rightnowai.co.
Do not open a public issue for a security report. We aim to acknowledge within
three business days and will keep you updated on remediation.

Include, where possible: affected version or commit, a description of the
issue, and steps to reproduce.

## Scope

This policy covers the AutoTree tree-execution additions to SGLang: the tree
runtime (`python/sglang/srt/tree/`), the tree serving endpoint, and the splice
tooling. Vulnerabilities in upstream SGLang unrelated to these additions should
be reported to the upstream project.

## Known areas under active hardening

- **Tenant isolation of in-flight tree KV.** The mid-generation fork inserts a
  running request's generated KV into the radix cache so sibling branches can
  reuse it. This insertion is scoped to the tree's own branches by a fork-local
  radix namespace; cache backends that cannot guarantee that scoping refuse the
  fork rather than falling back to an unsafe path. This isolation is covered by
  a unit test (`test_identical_tree_prompts_and_plain_request_are_cache_isolated`)
  and is validated on GPU before release. Reports probing cross-request
  contamination in tree mode are especially welcome.
