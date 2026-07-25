# AutoTree figures

Regenerate the complete Phase-4 figure set from the repository root:

```console
uv run python figures/make_figures.py \
  --out out/
```

The entry point bootstraps the isolated `figures/pyproject.toml` environment when
the caller is not already inside it. Every output is written as PDF and SVG at
300 dpi, followed by `manifest.json` with SHA-256 hashes for the bundle, every
referenced ThoughtBench result, and every generated artifact.

## Honesty contract

The pipeline refuses a bundle or ThoughtBench result without a non-empty
provenance `kind` and `source`. Any panel whose source has
`"kind": "fixture"` receives a visible diagonal `FIXTURE DATA` watermark in
both output formats. Renderers never synthesize missing measurements.

`all-figures.fixture.json` references compact `thoughtbench.results.v1`
publication-layout fixtures for four model slots, three serving-system slots,
the six requested branching factors, and throughput baselines. They are
explicitly synthetic and exist to exercise the final chart structure; they are
not outputs from those named models or serving systems. Error bars are one
standard deviation over three fixture seeds. Regenerate these committed fixture
documents after changing their schema with
`uv run python fixtures/build_publication_fixtures.py`.

Three renderers are production implementations backed by explicitly synthetic
contract fixtures until their measurements exist:

- `autotree.kv-reuse-sweep.v1`: `depths`, `branching_factors`, and a matching
  two-dimensional `values` matrix of logical/physical token multipliers `>= 1`.
- `autotree.branch-trace.v1`: one rooted list of nodes with stable `id`, parent
  reference, value estimate, prune flag, and optional display label.
- `autotree.thinking-effort-sweep.v1`: model name and named series of effort
  points, each carrying the measured value from every seed.

Replace those supplemental objects with real depth/branch sweeps, branch
traces, and Inkling results without changing renderer code.

## Tests

```console
uv run --project figures pytest figures/tests
```

The suite pins provenance refusal, fixture watermarking, all seven renderers,
manifest hashes, and byte-for-byte deterministic outputs.
