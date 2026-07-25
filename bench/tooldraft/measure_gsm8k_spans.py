#!/usr/bin/env python3
"""Ground-truth measurement of mechanical-span structure in GSM8K gold solutions.

GSM8K solutions carry inline calculator annotations of the form <<expr=result>>.
Every annotation is a human label saying: "the text immediately following this
point is the deterministic output of evaluating expr". That makes GSM8K the only
corpus where mechanical spans are labeled by construction rather than by a
heuristic detector we wrote ourselves.

We measure three things the tool-as-drafter speedup depends on:

  f      fraction of emitted tokens that a zero-cost tool could have drafted
  E[a]   mean length of a contiguous draftable span, in tokens
  dist   the full span-length distribution, because the mean hides the shape

Definitions, from strictest to most generous. The strict one is the one the
mechanism actually gets, because the model must still emit the setup ("16-3-4=")
itself before a tool knows what to evaluate.

  STRICT   only the result text emitted after the annotation
  PLUS     result text plus the trailing unit/noun phrase to end of clause
  ORACLE   result text plus everything to end of line, i.e. assume the tool can
           also predict how the model finishes the sentence

Token counts use a real HF tokenizer when one is importable, otherwise a
character-based approximation stated explicitly in the output.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ANNOT = re.compile(r"<<([^>]*?)>>")


def load_tokenizer():
    """Return (encode_fn, name). Falls back to a stated approximation."""
    for repo in ("Qwen/Qwen2.5-7B-Instruct", "gpt2"):
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(repo)
            return (lambda s: len(tok.encode(s, add_special_tokens=False)), repo)
        except Exception:
            continue

    def approx(s: str) -> int:
        # ~3.6 chars/token is the usual English rate for BPE tokenizers; digits
        # tokenize denser (~1 token per 1-3 digits) so this UNDERCOUNTS numeric
        # spans, which biases the result in the mechanism's favour. Stated so
        # the bias direction is auditable.
        if not s:
            return 0
        return max(1, round(len(s) / 3.6))

    return (approx, "APPROX(chars/3.6)")


def analyze(answer: str, ntok):
    """Return (total_tokens, spans_strict, spans_plus, spans_oracle).

    Each spans_* is a list of token lengths for the draftable spans under that
    definition. Annotations themselves are stripped: they are not emitted text.
    """
    # Drop the final "#### N" answer line: it is scaffolding, not reasoning.
    body = answer.split("####")[0]

    emitted = ANNOT.sub("", body)
    total = ntok(emitted)

    spans_strict, spans_plus, spans_oracle = [], [], []

    # Walk annotations in order over the ORIGINAL text so we know where each
    # result begins in the emitted stream.
    pos = 0
    for m in ANNOT.finditer(body):
        inner = m.group(1)
        if "=" not in inner:
            continue
        result = inner.rsplit("=", 1)[1].strip()
        if not result:
            continue

        tail = body[m.end():]

        # STRICT: the model emits the result verbatim right after the annotation.
        # Confirm it actually does, then measure exactly that text.
        if tail.startswith(result):
            spans_strict.append(ntok(result))
            after = tail[len(result):]
        else:
            # Rare formatting drift (e.g. "$<<9*2=18>>18"). Match leading numeric.
            mm = re.match(r"[-+]?[\d,]*\.?\d+(?:/\d+)?", tail)
            if not mm:
                continue
            spans_strict.append(ntok(mm.group(0)))
            after = tail[mm.end():]

        # PLUS: result plus the rest of the clause (to comma, period, or newline).
        clause = re.match(r"[^,.\n]*", after).group(0)
        spans_plus.append(spans_strict[-1] + ntok(clause))

        # ORACLE: result plus everything to end of line.
        line = re.match(r"[^\n]*", after).group(0)
        spans_oracle.append(spans_strict[-1] + ntok(line))

        pos = m.end()

    return total, spans_strict, spans_plus, spans_oracle


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return xs[i]


def blended_speedup(f, ea, s_res=1.0, verify_cost=1.1):
    """relative_cost = (1-f)/s_res + f*verify_cost/ea ; speedup = 1/cost."""
    if ea <= 0:
        return 1.0
    cost = (1 - f) / s_res + f * verify_cost / ea
    return 1.0 / cost if cost > 0 else float("inf")


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "gsm8k_test.jsonl")
    ntok, tokname = load_tokenizer()

    tot_tokens = 0
    all_strict, all_plus, all_oracle = [], [], []
    per_item_f_strict = []
    n_items = 0
    n_annots = 0

    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        total, st, pl, orc = analyze(rec["answer"], ntok)
        if total == 0:
            continue
        n_items += 1
        n_annots += len(st)
        tot_tokens += total
        all_strict += st
        all_plus += pl
        all_oracle += orc
        per_item_f_strict.append(sum(st) / total)

    print("=" * 74)
    print("GSM8K GOLD SOLUTIONS - GROUND-TRUTH MECHANICAL SPAN STRUCTURE")
    print("=" * 74)
    print(f"tokenizer            : {tokname}")
    print(f"items                : {n_items}")
    print(f"total emitted tokens : {tot_tokens}")
    print(f"calculator annots    : {n_annots}  ({n_annots / n_items:.2f} per item)")
    print(f"mean tokens per item : {tot_tokens / n_items:.1f}")
    print()

    rows = [
        ("STRICT  (result only)", all_strict),
        ("PLUS    (+ clause)", all_plus),
        ("ORACLE  (+ rest of line)", all_oracle),
    ]
    print(f"{'definition':<26} {'f':>7} {'E[a]':>7} {'med':>5} {'p90':>5} {'max':>5}")
    print("-" * 74)
    fs = {}
    for name, spans in rows:
        f = sum(spans) / tot_tokens
        ea = statistics.mean(spans) if spans else 0
        fs[name.split()[0]] = (f, ea)
        print(f"{name:<26} {f:>7.3f} {ea:>7.2f} {pct(spans,50):>5} "
              f"{pct(spans,90):>5} {pct(spans,100):>5}")
    print()

    print("SPAN LENGTH HISTOGRAM (strict)")
    hist = {}
    for x in all_strict:
        hist[x] = hist.get(x, 0) + 1
    for k in sorted(hist):
        bar = "#" * max(1, round(60 * hist[k] / len(all_strict)))
        print(f"  {k:>3} tok  {hist[k]:>6}  {100*hist[k]/len(all_strict):>5.1f}%  {bar}")
    print()

    print("=" * 74)
    print("IMPLIED BLENDED SPEEDUP  (relative_cost = (1-f)/s_res + f*1.1/E[a])")
    print("=" * 74)
    for name, (f, ea) in fs.items():
        for s_res, lbl in ((1.0, "no residual speedup"), (2.0, "residual 2x")):
            print(f"  {name:<8} f={f:.3f} E[a]={ea:.2f}  {lbl:<20} "
                  f"-> {blended_speedup(f, ea, s_res):.2f}x")
    print()

    print("WHAT WOULD BE NEEDED (solve for f at fixed E[a], s_res=1)")
    print(f"  {'target':>7}  " + "  ".join(f"E[a]={e:<4}" for e in (2, 4, 8, 16, 32)))
    for target in (3, 10, 50):
        cells = []
        for ea in (2, 4, 8, 16, 32):
            # 1/target = (1-f) + f*1.1/ea  ->  f = (1 - 1/target)/(1 - 1.1/ea)
            denom = 1 - 1.1 / ea
            f_req = (1 - 1 / target) / denom if denom > 0 else float("inf")
            cells.append("IMPOSSIBLE" if f_req > 1 else f"f={f_req:.3f}  ")
        print(f"  {target:>6}x  " + "  ".join(f"{c:<10}" for c in cells))


if __name__ == "__main__":
    main()
