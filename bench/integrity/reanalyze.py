#!/usr/bin/env python3
"""Integrity-corrected re-analysis of AutoTree measurement records.

The 2026-07-24 adversarial audit found five analysis-layer defects that
corrupted reported numbers. This tool recomputes every metric from raw item
records with all five corrected, and refuses to emit a comparison it cannot
defend:

1. DEDUPE. Item files are appended, and several archived files contain two
   concatenated runs. We dedupe on (mode, seed, id), keeping the first
   occurrence (matching the harness's own resume semantics), and REPORT how
   many duplicates were dropped instead of silently averaging them.
2. ERRORS ARE NOT WRONG ANSWERS. The harness scores an errored item as
   incorrect. Comparing an arm with a 35 percent error rate against a
   zero-error baseline is not apples to apples. We report accuracy three
   ways: as-scored (errors counted wrong), excluding errored items, and the
   error rate itself, so the reader can see the spread.
3. WALL CLOCK BESIDE COST. Cost-units and wall-seconds disagreed about who
   won. Both are always reported; a cost win with a wall-clock loss is
   flagged, never hidden.
4. EFFECTIVE SEEDS. A deterministic arm (greedy, temperature 0) produces
   identical output across nominal seeds. We detect that and report the
   effective seed count so nobody claims 3 seeds of evidence from 1.
5. AGGREGATE, NOT MEAN-OF-RATIOS. Cost per correct is total cost over total
   correct, never an average of per-item ratios.
"""

import argparse
import collections
import json
import os


def load_records(path):
    """Load, dedupe by (mode, seed, id), and report duplicates."""
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    seen = {}
    dup = 0
    for r in rows:
        key = (r.get("mode"), r.get("seed"), r.get("id"))
        if key in seen:
            dup += 1
            continue
        seen[key] = r
    return list(seen.values()), len(rows), dup


def effective_seeds(records):
    """Count seeds whose record sets are not byte-identical to another seed."""
    by_seed = collections.defaultdict(list)
    for r in records:
        by_seed[r.get("seed")].append(r)
    sigs = {}
    for seed, rs in by_seed.items():
        rs_sorted = sorted(rs, key=lambda r: str(r.get("id")))
        sigs[seed] = json.dumps(
            [[r.get("id"), r.get("extracted"), r.get("correct")] for r in rs_sorted],
            sort_keys=True,
        )
    return len(set(sigs.values())), len(by_seed)


def analyze(path, label):
    records, raw_n, dup = load_records(path)
    if not records:
        return None
    n = len(records)
    errored = [r for r in records if r.get("error")]
    clean = [r for r in records if not r.get("error")]
    correct_as_scored = sum(1 for r in records if r.get("correct"))
    correct_clean = sum(1 for r in clean if r.get("correct"))
    cost = sum(r.get("total_cost_units") or 0 for r in records)
    cost_clean = sum(r.get("total_cost_units") or 0 for r in clean)
    wall = sum(r.get("wall_s") or 0 for r in records)
    eff_seeds, nom_seeds = effective_seeds(records)
    return {
        "label": label,
        "path": os.path.basename(path),
        "raw_rows": raw_n,
        "duplicates_dropped": dup,
        "n": n,
        "error_n": len(errored),
        "error_rate": len(errored) / n,
        "acc_as_scored": correct_as_scored / n,
        "acc_excluding_errors": (correct_clean / len(clean)) if clean else 0.0,
        "cost_per_correct_as_scored": cost / correct_as_scored if correct_as_scored else float("inf"),
        "cost_per_correct_clean": cost_clean / correct_clean if correct_clean else float("inf"),
        "wall_s_total": wall,
        "wall_per_correct": wall / correct_as_scored if correct_as_scored else float("inf"),
        "seeds_nominal": nom_seeds,
        "seeds_effective": eff_seeds,
    }


def fmt(a):
    flags = []
    if a["duplicates_dropped"]:
        flags.append("DUPES_DROPPED={}".format(a["duplicates_dropped"]))
    if a["error_rate"] > 0.05:
        flags.append("ERROR_RATE={:.0%}".format(a["error_rate"]))
    if a["seeds_effective"] < a["seeds_nominal"]:
        flags.append("EFFECTIVE_SEEDS={}/{}".format(a["seeds_effective"], a["seeds_nominal"]))
    return (
        "{label:<26} n={n:<4} acc={acc_as_scored:.3f} (excl-err {acc_excluding_errors:.3f})  "
        "cost/corr={cost_per_correct_as_scored:>9.0f}  wall/corr={wall_per_correct:>7.2f}s  {flags}"
    ).format(flags=" ".join(flags) or "clean", **a)


def compare(tree, base):
    print("\n--- COMPARISON (integrity-corrected) ---")
    dacc = 100 * (tree["acc_as_scored"] - base["acc_as_scored"])
    dacc_clean = 100 * (tree["acc_excluding_errors"] - base["acc_excluding_errors"])
    k_cost = tree["cost_per_correct_as_scored"] / base["cost_per_correct_as_scored"]
    k_wall = tree["wall_per_correct"] / base["wall_per_correct"]
    print("accuracy delta (as-scored):      {:+.1f}pp".format(dacc))
    print("accuracy delta (excluding errs): {:+.1f}pp".format(dacc_clean))
    print("cost ratio  K_cost (tree/base):  {:.3f}".format(k_cost))
    print("wall ratio  K_wall (tree/base):  {:.3f}".format(k_wall))
    if k_cost < 1.0 and k_wall > 1.0:
        print("FLAG: cost-units say tree WINS but wall-clock says tree LOSES. "
              "A cost-per-correct claim here is not defensible without stating both.")
    if tree["error_rate"] > 0.05 and base["error_rate"] < 0.01:
        print("FLAG: arm error rates differ materially ({:.0%} vs {:.0%}); the as-scored "
              "accuracy penalizes the tree for harness failures, not reasoning failures."
              .format(tree["error_rate"], base["error_rate"]))
    if min(tree["seeds_effective"], base["seeds_effective"]) < 2:
        print("FLAG: an arm has fewer than 2 effective seeds; treat its accuracy as a point "
              "estimate with no seed variance.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True,
                    help="item jsonl paths, optionally as label=path")
    ap.add_argument("--compare", nargs=2, metavar=("TREE_LABEL", "BASE_LABEL"))
    args = ap.parse_args()
    out = {}
    print("=== INTEGRITY-CORRECTED ANALYSIS ===")
    for spec in args.items:
        if "=" in spec and not os.path.exists(spec):
            label, path = spec.split("=", 1)
        else:
            label, path = os.path.basename(spec).replace("_items.jsonl", ""), spec
        a = analyze(path, label)
        if a:
            out[label] = a
            print(fmt(a))
        else:
            print("{}: NO RECORDS".format(label))
    if args.compare:
        t, b = args.compare
        if t in out and b in out:
            compare(out[t], out[b])
        else:
            print("compare labels not found: {} {}".format(t, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
