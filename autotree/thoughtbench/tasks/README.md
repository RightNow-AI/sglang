# Bundled task data

## GSM8K test split

`gsm8k_test.jsonl` contains all 1,319 records from the official GSM8K test
split. It was converted from OpenAI's
[`test.jsonl`](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl)
into ThoughtBench's strict `id`, `prompt`, and `gold` schema. Each gold value is
the numeric answer after the source record's final `####` marker, normalized by
removing thousands separators. The downloaded source file had SHA-256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

`gsm8k_subset.jsonl` retains the earlier 50-record subset for quick examples.

Standard citation:

> Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun,
> Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
> Nakano, Christopher Hesse, and John Schulman. "Training Verifiers to Solve
> Math Word Problems." arXiv:2110.14168, 2021.

The upstream `openai/grade-school-math` repository and GSM8K data are released
under the [MIT License](https://github.com/openai/grade-school-math/blob/master/LICENSE).
The converted task file preserves that upstream dataset attribution and license
notice; ThoughtBench's own code remains under its repository license.