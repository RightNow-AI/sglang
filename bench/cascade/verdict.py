#!/usr/bin/env python3
"""Preregistered verdict analyzer for the large_tree vs vote@8 experiment.

Honesty discipline (from the 2026-07-24 brainstorm):
- cost-per-correct is aggregate total_cost / total_correct, never a mean of
  per-item ratios (wrong answers make the ratio undefined);
- uncertainty is bootstrapped BY PROBLEM with seeds nested inside a problem,
  because 60 items x 3 seeds is ~60 independent problems, not 180;
- the headline is a paired accuracy noninferiority test (tree vs vote@8), plus
  the cost ratio K. A cheap system that loses accuracy is not a win.

Usage:
  verdict.py --tree lt_math_m04.json --bo8 math_large_bo8.json \
             --greedy math_large_greedy.json --items-dir <dir> --label MATH
Reads the *_items.jsonl next to each summary (same stem) for bootstrapping.
"""

import argparse
import json
import os


def load_items(summary_path):
    stem = summary_path[:-5] if summary_path.endswith(".json") else summary_path
    p = stem + "_items.jsonl"
    if not os.path.exists(p):
        raise SystemExit("missing items file: " + p)
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def agg(items):
    n = len(items)
    correct = sum(1 for r in items if r.get("correct"))
    cost = sum(r.get("total_cost_units") or 0 for r in items)
    return dict(
        n=n,
        correct=correct,
        acc=correct / n if n else 0.0,
        cost=cost,
        cost_per_correct=cost / correct if correct else float("inf"),
    )


def by_problem(items):
    groups = {}
    for r in items:
        groups.setdefault(r["id"], []).append(r)
    return groups


def _rng(seed):
    # Deterministic LCG so bootstrap needs no forbidden Date/random and is
    # reproducible; adequate for resampling indices.
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(modulo):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) % modulo

    return nxt


def bootstrap_paired_acc_delta(tree_items, base_items, iters=2000, seed=12345):
    """Paired accuracy delta (tree - base), resampled by problem id.
    Returns (point_delta_pp, lo95_pp, hi95_pp) in percentage points."""
    t_by = by_problem(tree_items)
    b_by = by_problem(base_items)
    ids = sorted(set(t_by) & set(b_by))
    if not ids:
        return None
    # per-problem accuracy for each system (mean over that problem's seeds)
    def pacc(groups, pid):
        g = groups[pid]
        return sum(1 for r in g if r.get("correct")) / len(g)
    t_acc = {pid: pacc(t_by, pid) for pid in ids}
    b_acc = {pid: pacc(b_by, pid) for pid in ids}
    point = 100.0 * (sum(t_acc.values()) - sum(b_acc.values())) / len(ids)
    nxt = _rng(seed)
    deltas = []
    m = len(ids)
    for _ in range(iters):
        st = bt = 0.0
        for _k in range(m):
            pid = ids[nxt(m)]
            st += t_acc[pid]
            bt += b_acc[pid]
        deltas.append(100.0 * (st - bt) / m)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--bo8", required=True)
    ap.add_argument("--greedy", required=True)
    ap.add_argument("--label", default="TASK")
    ap.add_argument("--k-bar", type=float, default=0.35)
    ap.add_argument("--noninferiority-pp", type=float, default=-3.0)
    args = ap.parse_args()

    tree = load_items(args.tree)
    bo8 = load_items(args.bo8)
    greedy = load_items(args.greedy)
    T, B, G = agg(tree), agg(bo8), agg(greedy)

    L = 100.0 * (B["acc"] - G["acc"])           # search lift
    Gn = 100.0 * (T["acc"] - G["acc"])          # tree gain over greedy
    capture = (Gn / L) if L > 0 else float("nan")
    K = T["cost_per_correct"] / B["cost_per_correct"] if B["cost_per_correct"] else float("inf")
    boot = bootstrap_paired_acc_delta(tree, bo8)

    print("==== {} ====".format(args.label))
    print("  large_tree : acc {:.1f}%  cost/correct {:.0f}  (n={})".format(100*T["acc"], T["cost_per_correct"], T["n"]))
    print("  vote@8     : acc {:.1f}%  cost/correct {:.0f}".format(100*B["acc"], B["cost_per_correct"]))
    print("  greedy     : acc {:.1f}%  cost/correct {:.0f}".format(100*G["acc"], G["cost_per_correct"]))
    print("  search lift L = {:+.1f}pp   tree gain G = {:+.1f}pp   capture = {:.0f}%".format(L, Gn, 100*capture))
    print("  cost ratio K (tree/vote@8) = {:.3f}   [bar <= {}]".format(K, args.k_bar))
    if boot:
        pt, lo, hi = boot
        print("  paired acc delta (tree - vote@8) = {:+.1f}pp  95% CI [{:+.1f}, {:+.1f}]  (bootstrap by problem)".format(pt, lo, hi))
        noninf = lo > args.noninferiority_pp
        cheap = K <= args.k_bar
        verdict = "PASS" if (noninf and cheap) else "FAIL"
        reasons = []
        if not noninf:
            reasons.append("accuracy lower bound {:+.1f}pp <= {:+.1f}pp bar".format(lo, args.noninferiority_pp))
        if not cheap:
            reasons.append("K {:.3f} > {} bar".format(K, args.k_bar))
        print("  PREREGISTERED VERDICT: {}{}".format(verdict, "" if verdict == "PASS" else "  (" + "; ".join(reasons) + ")"))


if __name__ == "__main__":
    main()
