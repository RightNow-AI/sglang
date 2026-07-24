# Benchmark suite ledger

## Scope

This substrate adds GPQA Diamond so cost-per-correct measurements can cover a
science reasoning domain in addition to math. It also adds strict
multiple-choice scoring and content-addressed task manifests. All tools use
only the Python standard library and emit no timestamps.

## GPQA Diamond builder

`build_gpqa.py` reads `Idavidrein/gpqa`, config `gpqa_diamond`, split `train`,
through the Hugging Face datasets-server rows API. It requires exactly 198
rows, paginates in stable `row_idx` order, and writes atomically to:

```text
bench/tasks/data/gpqa_diamond.jsonl
```

The Hugging Face repository is public metadata with gated data access. Accept
the dataset access terms before building. The builder checks `HF_TOKEN`,
`HUGGINGFACE_HUB_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, and the standard cached
Hugging Face token paths without printing credentials.

Run from the repository root:

```text
python bench/tasks/build_gpqa.py
```

The first returned row defines the active field mapping. Exact and normalized
aliases cover `Question`, `Correct Answer`, `Incorrect Answer 1` through
`Incorrect Answer 3`, and optional `Subdomain` or `Domain`. A successful build
prints the actual field names used.

For every question, the correct answer and three distractors are shuffled with
a local random generator seeded by the SHA-256 hash of the question text. The
output is reproducible and independent of process hash randomization. Rows use
stable ids from `gpqa-0001` through `gpqa-0198` and this shape:

```json
{"id":"gpqa-0001","prompt":"<question>\n\nA. <option>\nB. <option>\nC. <option>\nD. <option>\n\nAnswer with the letter A, B, C, or D.","gold":"A","meta":{"set":"gpqa_diamond","domain":"<source domain when present>"}}
```

## Multiple-choice equivalence

`mc_equiv.py` exports:

```text
extract_choice(text) -> A | B | C | D | None
is_choice_correct(pred, gold) -> bool
```

Extraction prefers the last case-insensitive `Answer: <letter>` marker. If no
such marker exists, it accepts a trailing standalone A through D, with optional
trailing punctuation. Gold values are strict A through D letters.

Run the direct assertion suite without pytest:

```text
python bench/tasks/test_mc_equiv.py
```

The cascade harness still needs a future `--answer-mode mc` integration before
GPQA can be scored there. This lane does not modify
`bench/cascade/measure_cascade.py`.

## Locked task manifests

Create a manifest after generating the task JSONLs:

```text
python bench/tasks/lock_manifest.py bench/tasks/data/math_hard.jsonl bench/tasks/data/aime.jsonl bench/tasks/data/gpqa_diamond.jsonl
```

The default output is `bench/tasks/data/task_manifest.json`. Use `--output` to
choose another location. Each file entry contains a path relative to the
manifest, the SHA-256 of the exact file bytes, the JSONL line count, and the
ordered complete id list:

```json
{
  "manifest_version": 1,
  "files": [
    {
      "path": "gpqa_diamond.jsonl",
      "sha256": "<64 lowercase hex characters>",
      "line_count": 198,
      "ids": ["gpqa-0001", "gpqa-0002"]
    }
  ]
}
```

The abbreviated id list above only illustrates the schema. A real manifest
contains every id. Creation rejects blank lines, malformed JSON, missing ids,
duplicate ids, and duplicate input paths.

Before each evaluation round, verify the locked files:

```text
python bench/tasks/lock_manifest.py --verify bench/tasks/data/task_manifest.json
```

Verification resolves paths relative to the manifest, recomputes all three
properties, prints `OK` for unchanged files, and exits nonzero with `DRIFT`
details for missing or changed files. Keep the accepted manifest with the
evaluation artifacts so every round names the exact task bytes it used.
