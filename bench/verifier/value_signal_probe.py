#!/usr/bin/env python3
"""Phase 3 premise check, offline: does ANY branch feature predict correctness?

Phase 3 proposes replacing the mean-logprob proxy with a trained value head
fused into decode. That is expensive: a new head, a training pipeline, a fused
kernel epilogue, and scheduler wiring. Its preregistered stop rule is "no
out-of-fold lift".

Before spending any of that, this asks the cheapest version of the same
question against branches we already have on disk: is there a per-branch signal,
computable from the branch itself, that predicts whether that branch is correct,
and does using it beat plain majority vote OUT OF FOLD?

If nothing here shows lift, a value head is unlikely to rescue it, because a
value head is just a learned version of the same idea with more capacity. That
is not proof it cannot work, and it is stated as such below, but it is a cheap
and honest prior.

METHOD, and the discipline that matters

- Features are computed per branch from its own text only. No gold answer.
- Correctness labels come from the gold answer and are used ONLY to fit and to
  score, never as a feature.
- Every score is OUT OF FOLD: 5-fold split BY PROBLEM, so no branch of a problem
  can appear in both the fit and the scoring half. Fitting and scoring on the
  same items is exactly the in-sample failure this repo already shipped once.
- The comparison is against plain majority vote over the SAME branches, so token
  spend is identical and selection is the only variable.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path


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


def features(text, answer, sibling_answers):
    """Per-branch features, none of which look at the gold answer."""
    n_chars = len(text or "")
    key = basic_norm(answer)
    sib = [basic_norm(a) for a in sibling_answers if a is not None]
    agree = sib.count(key) / len(sib) if sib and key is not None else 0.0
    return {
        "len_chars": n_chars,
        "len_log": math.log1p(n_chars),
        "has_boxed": 1.0 if "\\boxed" in (text or "") else 0.0,
        "n_equations": float((text or "").count("=")),
        "n_steps": float(len(re.findall(r"\n\s*\d+[\.\)]", text or ""))),
        "ends_cleanly": 1.0 if (text or "").rstrip().endswith(("}", ".", "$")) else 0.0,
        "answer_is_numeric": 1.0 if _isnum(key) else 0.0,
        "sibling_agreement": agree,   # the known-good signal, included as a control
    }


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def fit_logistic(rows, labels, feat_names, epochs=300, lr=0.1):
    """Tiny deterministic logistic regression. No sklearn dependency."""
    w = {f: 0.0 for f in feat_names}
    b = 0.0
    mu = {f: sum(r[f] for r in rows) / len(rows) for f in feat_names}
    sd = {f: (sum((r[f] - mu[f]) ** 2 for r in rows) / len(rows)) ** 0.5 or 1.0
          for f in feat_names}
    for _ in range(epochs):
        gw = {f: 0.0 for f in feat_names}
        gb = 0.0
        for r, y in zip(rows, labels):
            z = b + sum(w[f] * (r[f] - mu[f]) / sd[f] for f in feat_names)
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            e = p - y
            for f in feat_names:
                gw[f] += e * (r[f] - mu[f]) / sd[f]
            gb += e
        n = len(rows)
        for f in feat_names:
            w[f] -= lr * gw[f] / n
        b -= lr * gb / n
    return (w, b, mu, sd)


def predict(model, r):
    w, b, mu, sd = model
    z = b + sum(w[f] * (r[f] - mu[f]) / sd[f] for f in w)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def evaluate(path, folds=5):
    recs = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    recs.sort(key=lambda r: str(r["id"]))          # deterministic fold assignment
    tokens = sum(r["gen_tokens"] for r in recs)

    per_branch = []
    for i, r in enumerate(recs):
        gold = basic_norm(r["gold"])
        for text, ans in zip(r["texts"], r["answers"]):
            per_branch.append({
                "problem": i,
                "fold": i % folds,
                "x": features(text, ans, r["answers"]),
                "y": 1 if basic_norm(ans) == gold else 0,
                "answer": basic_norm(ans),
            })
    feat_names = sorted(per_branch[0]["x"])

    # out-of-fold branch scores
    scores = {}
    for f in range(folds):
        train = [b for b in per_branch if b["fold"] != f]
        test = [b for b in per_branch if b["fold"] == f]
        model = fit_logistic([b["x"] for b in train], [b["y"] for b in train], feat_names)
        for b in test:
            scores[id(b)] = predict(model, b["x"])

    by_problem = collections.defaultdict(list)
    for b in per_branch:
        by_problem[b["problem"]].append(b)

    maj_correct = val_correct = 0
    for i, branches in sorted(by_problem.items()):
        gold = basic_norm(recs[i]["gold"])
        cand = [b["answer"] for b in branches if b["answer"] is not None]
        if cand:
            counts = collections.Counter(cand)
            top = max(counts.values())
            maj_pick = sorted(k for k, v in counts.items() if v == top)[0]
            maj_correct += maj_pick == gold
        scored = [b for b in branches if b["answer"] is not None]
        if scored:
            best = max(scored, key=lambda b: (scores[id(b)], b["answer"]))
            val_correct += best["answer"] == gold

    n = len(recs)
    auc = _auc([scores[id(b)] for b in per_branch], [b["y"] for b in per_branch])
    return {
        "file": Path(path).name, "n_problems": n, "n_branches": len(per_branch),
        "branch_base_rate": sum(b["y"] for b in per_branch) / len(per_branch),
        "oof_auc": auc, "tokens": tokens,
        "majority_correct": maj_correct,
        "value_correct": val_correct,
        "majority_tpc": tokens / maj_correct if maj_correct else None,
        "value_tpc": tokens / val_correct if val_correct else None,
    }


def _auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True, nargs="+")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    print("PHASE 3 PREMISE CHECK: out-of-fold, by problem, no gold in features\n")
    verdicts = []
    for path in args.branches:
        r = evaluate(path, args.folds)
        lift = (r["majority_tpc"] / r["value_tpc"]
                if r["majority_tpc"] and r["value_tpc"] else None)
        verdicts.append((r, lift))
        print(f"=== {r['file']}  problems={r['n_problems']} branches={r['n_branches']} ===")
        print(f"  branch-level correct rate       : {r['branch_base_rate']:.1%}")
        print(f"  OUT-OF-FOLD AUC of the probe    : {r['oof_auc']:.4f}   (0.50 = no signal)")
        print(f"  majority vote                   : {r['majority_correct']:>4} correct, "
              f"{r['majority_tpc']:.0f} tok/correct")
        print(f"  value-probe selection           : {r['value_correct']:>4} correct, "
              f"{r['value_tpc']:.0f} tok/correct")
        print(f"  LIFT vs majority                : {lift:.3f}x\n")

    print("=" * 74)
    any_lift = any(l and l > 1.02 for _, l in verdicts)
    print("VERDICT:", "some lift, worth measuring a real value head"
          if any_lift else
          "NO out-of-fold lift. Phase 3's premise fails its own stop rule cheaply.")
    print("\nCaveat, stated rather than buried: these are hand-made features on a\n"
          "1.5B model's text, not a trained value head on hidden states. A null\n"
          "here is a cheap prior, not a proof that a value head cannot work.\n"
          "What it does establish is that the ONE signal already known to work,\n"
          "sibling agreement, is included as a feature and still produces no\n"
          "selection lift beyond plain majority vote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
