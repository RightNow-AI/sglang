# Hard-task benchmark ledger

## Scope

This directory builds benchmark inputs where a saturated GSM8K score is not a
useful discriminator. The two sets are MATH-500 levels 4 and 5 and the full
AIME 1983-2024 collection.

All implementation is Python standard library code. Generated rows have stable
source ordering, stable one-based four-digit ids, and no timestamps.

## Sources and schemas

MATH-500 uses `HuggingFaceH4/MATH-500`, config `default`, split `test`. The
dataset-server schema is `problem`, `solution`, `answer`, `subject`, `level`,
and `unique_id`. Only rows whose parsed level is 4 or 5 are emitted.

The requested AIME name is `qq8933/AIME_1983_2024`. Hugging Face currently
resolves that repository name to `di-zhang-fdu/AIME_1983_2024`. The active
dataset-server source uses config `default`, split `train`, with fields `ID`,
`Year`, `Problem Number`, `Question`, `Answer`, and `Part`. The builder follows
the Hub repository redirect and retains the current repository as a fallback.
Question, answer, year, and id lookups also use case-insensitive aliases.

## Build

From the repository root:

```text
python bench/tasks/build_hard_tasks.py
```

The script paginates the Hugging Face datasets-server rows API with 100 rows
per request and writes atomically to:

```text
bench/tasks/data/math_hard.jsonl
bench/tasks/data/aime.jsonl
```

Every row has string `id`, `prompt`, and `gold` fields plus deterministic
metadata. MATH answers intentionally remain LaTeX. AIME answers are validated
as integer strings in the inclusive range 0 through 999.

The source row `2022-II-8` lists `080 or 081 (both were accepted)`. The output
format has one gold string, so the builder deterministically uses the first
source-listed accepted answer, `080`, and retains the row.

## Evaluation boundary

The existing pruning and cascade loaders validate every gold answer as numeric.
They therefore cannot load `math_hard.jsonl` yet. This lane does not weaken
that validation or alter the LaTeX golds. A later harness integration should
use `math_equiv.py` for extraction and equivalence.

`math_equiv.py` exports:

```text
normalize_answer(s: str) -> str
is_equiv(a: str, b: str) -> bool
extract_final_answer(text: str) -> str | None
```

String normalization follows Hendrycks MATH-style rules. Numeric fallback is
limited to integers, decimals, and simple fractions, compared with an absolute
tolerance of 1e-6.

## Verification

Run the direct assertion suite without pytest:

```text
python bench/tasks/test_math_equiv.py
```

The direct assertion suite passes 52 assertions. The generated datasets contain
262 `math_hard` rows and 933 `aime` rows. A complete local validation confirms
sequential ids, string `id`/`prompt`/`gold` values, level filtering, AIME answer
ranges, and the expected metadata keys.
