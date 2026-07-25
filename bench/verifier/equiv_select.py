#!/usr/bin/env python3
"""Does equivalence-aware voting beat plain string majority? Offline, no GPU.

The Phase 2 run showed a well-formedness verifier is worthless: 1.02x on GSM8K
and 0.93x on MATH, i.e. worse than doing nothing. But it also showed a real
15-16pp gap between majority vote and coverage, so the votes ARE there and the
selector is failing to find them.

One concrete, gold-free reason it fails on MATH: symbolically equivalent answers
do not match as strings. "1/2", "0.5" and "\\frac{1}{2}" are the same answer and
vote as three different candidates, splitting a majority that should have won.
Plain string voting cannot see that.

This scores an equivalence-aware selector against plain majority over the SAME
committed branches, so the comparison needs no server and no new tokens. It
requires no knowledge of the gold answer, so unlike the ORACLE arm it is
something we could actually ship.

Run:
    python bench/verifier/equiv_select.py --branches <branches.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from fractions import Fraction
from pathlib import Path

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    HAVE_SYMPY = True
except Exception:
    sympy = None
    parse_latex = None
    HAVE_SYMPY = False


def basic_norm(a):
    """The normalizer the engine ships today: string level only."""
    if a is None:
        return None
    s = str(a).strip().rstrip(".").replace(" ", "").replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
    try:
        return str(float(s)) if re.fullmatch(r"-?\d+(\.\d+)?", s) else s.lower()
    except ValueError:
        return s.lower()


def canonical(a):
    """Equivalence-aware key: map symbolically equal answers to one string.

    Deliberately conservative and deterministic. Anything it cannot canonicalize
    falls back to the string normalizer, so this can only merge candidates that
    are provably equal, never split ones that already matched.
    """
    if a is None:
        return None
    s = basic_norm(a)
    if s is None:
        return None

    # plain rational, including a/b and \frac{a}{b}
    m = re.fullmatch(r"-?\d+/\d+", s)
    if m:
        try:
            return str(float(Fraction(s)))
        except (ValueError, ZeroDivisionError):
            return s
    m = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            v = Fraction(int(m.group(1)), int(m.group(2)))
            return str(float(-v if s.startswith("-") else v))
        except (ValueError, ZeroDivisionError):
            return s
    # percentages
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\\?%", s)
    if m:
        return str(float(m.group(1)) / 100)
    # trailing zeros: 2.0 and 2 are the same answer
    try:
        return str(float(s))
    except ValueError:
        pass

    if HAVE_SYMPY:
        try:
            expr = parse_latex(str(a)) if "\\" in str(a) else sympy.sympify(s)
            val = sympy.nsimplify(expr)
            f = float(val.evalf())
            return str(f)
        except Exception:
            pass
    return s


def vote(answers, keyfn):
    cand = [keyfn(a) for a in answers if a is not None]
    cand = [c for c in cand if c is not None]
    if not cand:
        return None
    # deterministic: highest count, then lexicographically smallest key
    counts = collections.Counter(cand)
    top = max(counts.values())
    return sorted(k for k, v in counts.items() if v == top)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True, nargs="+")
    args = ap.parse_args()

    print(f"sympy available: {HAVE_SYMPY}")
    for path in args.branches:
        recs = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
        tokens = sum(r["gen_tokens"] for r in recs)
        rows = {}
        for name, keyfn in (("majority (string)", basic_norm),
                            ("majority (equivalence-aware)", canonical)):
            correct = 0
            for r in recs:
                pick = vote(r["answers"], keyfn)
                if pick is not None and pick == keyfn(r["gold"]):
                    correct += 1
            rows[name] = (correct, tokens / correct if correct else None)
        # coverage under each key, the ceiling any selector could reach
        cov_s = sum(1 for r in recs
                    if basic_norm(r["gold"]) in {basic_norm(a) for a in r["answers"] if a})
        cov_c = sum(1 for r in recs
                    if canonical(r["gold"]) in {canonical(a) for a in r["answers"] if a})

        n = len(recs)
        print(f"\n=== {Path(path).name}  n={n}  tokens={tokens} ===")
        print(f"{'selector':<32} {'acc':>7} {'correct':>8} {'tok/correct':>12}")
        print("-" * 64)
        for name, (c, tpc) in rows.items():
            print(f"{name:<32} {c/n:>7.1%} {c:>8} "
                  f"{(f'{tpc:.1f}' if tpc else 'n/a'):>12}")
        base = rows["majority (string)"][1]
        new = rows["majority (equivalence-aware)"][1]
        if base and new:
            print(f"\n  equivalence-aware vs string majority: {base/new:.3f}x")
        print(f"  coverage under string key      : {cov_s/n:.1%}")
        print(f"  coverage under equivalence key : {cov_c/n:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
