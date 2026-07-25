#!/usr/bin/env python3
"""A REAL gold-free verifier: execute the arithmetic the branch claims.

Phase 2 measured a well-formedness verifier (useless) and an oracle verifier
(1.20-1.45x but it knows the answer). This is the middle case the goal actually
names: "code exec, math equivalence". It needs no gold answer and it checks
something genuinely true or false.

Mechanism: a reasoning trace states its own arithmetic, for example
"16 - 3 - 4 = 9". Every such claim is independently checkable by evaluating the
left side. A branch containing a wrong step has demonstrably erred, so it can be
rejected without knowing the right answer. This is exactly the verifiable-domain
argument, applied to the arithmetic a math trace exposes about itself.

Scored offline against the committed branches, so tokens are identical to the
majority-vote baseline and selection is the only variable.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from fractions import Fraction
from pathlib import Path

# "a op b = c" and "a op b op c = d", allowing $ , and spaces around tokens
STEP = re.compile(
    r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*/x×])\s*(-?[\d,]+(?:\.\d+)?)\s*=\s*(-?[\d,]+(?:\.\d+)?)"
)


def _num(s):
    try:
        return Fraction(str(s).replace(",", "").replace("$", ""))
    except (ValueError, ZeroDivisionError):
        return None


def arithmetic_errors(text, tol=Fraction(1, 100)):
    """Return (n_checked, n_wrong) for the arithmetic the branch asserts."""
    checked = wrong = 0
    for a, op, b, c in STEP.findall(text or ""):
        x, y, z = _num(a), _num(b), _num(c)
        if x is None or y is None or z is None:
            continue
        try:
            if op in "+":
                v = x + y
            elif op == "-":
                v = x - y
            elif op in "*x×":
                v = x * y
            elif op == "/":
                if y == 0:
                    continue
                v = x / y
            else:
                continue
        except (ZeroDivisionError, ValueError):
            continue
        checked += 1
        if abs(v - z) > tol * max(1, abs(z)):
            wrong += 1
    return checked, wrong


def basic_norm(a):
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


def vote(keys):
    keys = [k for k in keys if k is not None]
    if not keys:
        return None
    counts = collections.Counter(keys)
    top = max(counts.values())
    return sorted(k for k, v in counts.items() if v == top)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True, nargs="+")
    args = ap.parse_args()

    for path in args.branches:
        recs = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
        tokens = sum(r["gen_tokens"] for r in recs)
        n = len(recs)

        maj = arith = 0
        rejected_all = 0
        tot_checked = tot_wrong = 0
        clean_correct = clean_total = dirty_correct = dirty_total = 0

        for r in recs:
            gold = basic_norm(r["gold"])
            keys = [basic_norm(a) for a in r["answers"]]
            maj += vote(keys) == gold

            approved = []
            for text, key in zip(r["texts"], keys):
                checked, wrong = arithmetic_errors(text)
                tot_checked += checked
                tot_wrong += wrong
                if key is None:
                    continue
                if wrong == 0:
                    approved.append(key)
                    clean_total += 1
                    clean_correct += key == gold
                else:
                    dirty_total += 1
                    dirty_correct += key == gold
            if approved:
                arith += vote(approved) == gold
            else:
                rejected_all += 1
                arith += vote(keys) == gold      # degrade, never fail

        print(f"=== {Path(path).name}  n={n} ===")
        print(f"  arithmetic claims checked      : {tot_checked} ({tot_wrong} wrong, "
              f"{tot_wrong/tot_checked:.1%})" if tot_checked else "  no arithmetic found")
        if clean_total and dirty_total:
            print(f"  branch correct | NO arith error : {clean_correct/clean_total:.1%} "
                  f"(n={clean_total})")
            print(f"  branch correct | HAS arith error: {dirty_correct/dirty_total:.1%} "
                  f"(n={dirty_total})")
            print(f"  -> the check separates branches by "
                  f"{clean_correct/clean_total - dirty_correct/dirty_total:+.1%}")
        print(f"  all branches rejected on         : {rejected_all} items (fell back)")
        print(f"  majority vote                    : {maj:>4} correct, "
              f"{tokens/maj:.0f} tok/correct")
        print(f"  arithmetic-verified selection    : {arith:>4} correct, "
              f"{tokens/arith:.0f} tok/correct")
        if maj and arith:
            print(f"  LIFT vs majority                 : {(tokens/maj)/(tokens/arith):.3f}x\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
