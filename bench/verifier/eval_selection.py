#!/usr/bin/env python3
"""Does verifier-gated selection beat majority vote per correct answer?

This is the experiment that decides Phase 2. It is designed around the single
mistake this project has made most often: comparing two arms that differ in
something other than the variable of interest.

THE DESIGN CHOICE THAT MAKES IT VALID

Both selectors run over the SAME generated branches. We generate k branches per
problem exactly once, persist them, and then apply every selector offline to
that identical set of tokens. So the token spend is byte-identical across arms
and the ONLY thing that varies is selection. A design that generated separately
per arm would let sampling noise and different token counts contaminate the
comparison, which is exactly how the "best-of-8 is already cheaper" claim went
wrong earlier: it compared arms at different GPU occupancy.

Consequence worth stating plainly: because tokens are shared, tokens-per-correct
differs between arms ONLY through the correct count. That is the honest framing.
Verifier selection cannot reduce tokens spent; it can only raise how many of the
already-paid-for tokens land on a correct answer.

LEAKAGE

This repo already shipped an in-sample gate eval with 149/150 train/eval
overlap. So the eval set is LOCKED here: sampled once under a fixed seed, its
ids and a content hash written to a manifest, and every later run asserts the
manifest still matches. A --check-leakage flag cross-references the eval ids
against any tuning set and refuses to run on overlap.

RAW RECORDS

Every item is written to JSONL with its branches, extracted answers, gold, and
per-selector verdicts. No summary number in this project is allowed without the
raw records behind it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUFFIX = "\n\nSolve step by step. Put your final answer in \\boxed{}."


# --------------------------------------------------------------------------
# answer handling
# --------------------------------------------------------------------------

def extract(text):
    """Last \\boxed{...}, brace aware, then #### N, then a trailing number."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx >= 0:
        i, depth, buf = idx + len("\\boxed{"), 1, []
        while i < len(text) and depth:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        if buf:
            return "".join(buf).strip()
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return m[-1] if m else None


def norm(a):
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


def as_number(a):
    try:
        return float(norm(a))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# selectors, all operating on the SAME branch set
# --------------------------------------------------------------------------

def select_majority(answers):
    """Today's behavior: plurality over normalized answers."""
    cand = [norm(a) for a in answers if a is not None]
    if not cand:
        return None
    return collections.Counter(cand).most_common(1)[0][0]


def select_first(answers):
    """Baseline floor: take branch 0, i.e. no selection at all."""
    for a in answers:
        if a is not None:
            return norm(a)
    return None


def select_verified(answers, verifier):
    """Verifier-gated: approve, then fall back to majority among approved.

    Mirrors the engine contract: selection among approved branches uses the
    existing winner rule, and when nothing is approved it degrades to plain
    majority rather than failing.
    """
    approved = [a for a in answers if a is not None and verifier(a)]
    if approved:
        return select_majority(approved), True
    return select_majority(answers), False


def make_numeric_verifier(gold):
    """The ORACLE verifier: exact numeric agreement with the gold answer.

    This is an upper bound, not a shippable verifier. It answers "what would a
    perfect cheap verifier buy", which is the question Phase 2 is asking. A real
    deployment substitutes a test suite, a symbolic checker, or SQL execution.
    Reported as ORACLE so it can never be mistaken for a shipped capability.
    """
    g = as_number(gold)
    gn = norm(gold)

    def verify(answer):
        a = as_number(answer)
        if a is not None and g is not None:
            return abs(a - g) <= 1e-6
        return norm(answer) == gn

    return verify


def make_wellformed_verifier(_gold):
    """A HONEST shippable verifier: the answer must parse as a number.

    Requires no knowledge of the gold answer, so it is deployable. It only
    filters malformed or truncated branches. Included to separate what a real
    verifier buys from what an oracle buys.
    """
    def verify(answer):
        return as_number(answer) is not None

    return verify


# --------------------------------------------------------------------------
# locked eval set
# --------------------------------------------------------------------------

def lock_manifest(items):
    payload = "\n".join(f"{it['id']}\t{norm(it['answer'])}" for it in items)
    return {
        "n": len(items),
        "ids": [it["id"] for it in items],
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def load_locked(data_path, manifest_path, n, seed):
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    mf = Path(manifest_path)
    if mf.exists():
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in manifest["ids"] if i not in by_id]
        if missing:
            raise SystemExit(f"locked ids missing from the data file: {missing[:3]}")
        items = [by_id[i] for i in manifest["ids"]]
        if lock_manifest(items)["sha256"] != manifest["sha256"]:
            raise SystemExit(
                "LOCKED EVAL SET CHANGED. The manifest hash does not match the "
                "data. Refusing to run: a moved eval set invalidates every "
                "number ever reported against it."
            )
        return items, manifest, False
    rng = random.Random(seed)
    items = rng.sample(rows, min(n, len(rows)))
    manifest = lock_manifest(items)
    manifest["seed"] = seed
    manifest["source"] = str(data_path)
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return items, manifest, True


def check_leakage(items, tuning_path):
    """Refuse to run if the eval set overlaps a set used for tuning."""
    p = Path(tuning_path)
    if not p.exists():
        return 0, []
    tune = set()
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            tune.add(json.loads(line).get("id"))
        except json.JSONDecodeError:
            continue
    overlap = sorted({it["id"] for it in items} & tune)
    return len(overlap), overlap[:5]


# --------------------------------------------------------------------------
# generation, once, shared by every selector
# --------------------------------------------------------------------------

def post(base, payload, timeout=1800):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def generate_branches(base, model, items, k, temp, max_tokens, conc, out_jsonl):
    def one(it):
        r = post(base, {
            "model": model,
            "messages": [{"role": "user", "content": it["question"] + SUFFIX}],
            "n": k, "temperature": temp, "max_tokens": max_tokens,
        })
        texts = [c["message"]["content"] for c in r["choices"]]
        rec = {
            "id": it["id"],
            "gold": it["answer"],
            "answers": [extract(t) for t in texts],
            "gen_tokens": r["usage"]["completion_tokens"],
            "texts": texts,
        }
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        recs = list(ex.map(one, items))
    elapsed = time.time() - t0
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    return recs, elapsed


def score(recs, selector_name, selector):
    correct = fell_back = 0
    total_tokens = sum(r["gen_tokens"] for r in recs)
    per_item = []
    for r in recs:
        gold = norm(r["gold"])
        if selector_name.startswith("verified"):
            pick, approved = selector(r["answers"])
            fell_back += 0 if approved else 1
        else:
            pick, approved = selector(r["answers"]), None
        ok = pick == gold
        correct += ok
        per_item.append({"id": r["id"], "pick": pick, "gold": gold,
                         "correct": ok, "approved": approved})
    return {
        "selector": selector_name,
        "n": len(recs),
        "correct": correct,
        "accuracy": round(correct / len(recs), 4) if recs else 0.0,
        "gen_tokens": total_tokens,
        "tokens_per_correct": round(total_tokens / correct, 1) if correct else None,
        "fell_back": fell_back,
        "per_item": per_item,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl with id/question/answer")
    ap.add_argument("--manifest", default="bench/verifier/locked_eval.json")
    ap.add_argument("--tuning-set", default="", help="jsonl to leakage-check against")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="bench/verifier/results")
    args = ap.parse_args()

    items, manifest, newly_locked = load_locked(
        args.data, args.manifest, args.n, args.seed)
    print(f"eval set: n={manifest['n']} sha256={manifest['sha256'][:16]} "
          f"{'(NEWLY LOCKED)' if newly_locked else '(locked, hash verified)'}")

    if args.tuning_set:
        n_overlap, sample = check_leakage(items, args.tuning_set)
        if n_overlap:
            raise SystemExit(
                f"LEAKAGE: {n_overlap} eval ids appear in {args.tuning_set} "
                f"(e.g. {sample}). Refusing to run."
            )
        print(f"leakage check against {args.tuning_set}: clean")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "branches.jsonl"

    print(f"generating k={args.k} branches per problem, ONCE, shared by every selector ...")
    recs, elapsed = generate_branches(
        args.base, args.model, items, args.k, args.temperature,
        args.max_tokens, args.concurrency, raw)
    print(f"  {len(recs)} items, {sum(r['gen_tokens'] for r in recs)} tokens, {elapsed:.1f}s")

    arms = [
        ("first_branch", select_first),
        ("majority_vote", select_majority),
        ("verified_wellformed", lambda ans: select_verified(
            ans, make_wellformed_verifier(None))),
    ]
    results = [score(recs, name, sel) for name, sel in arms]
    # the oracle arm needs the gold answer per item, so it is scored separately
    oracle = {"selector": "verified_ORACLE", "n": len(recs), "correct": 0,
              "gen_tokens": sum(r["gen_tokens"] for r in recs),
              "fell_back": 0, "per_item": []}
    for r in recs:
        pick, approved = select_verified(
            r["answers"], make_numeric_verifier(r["gold"]))
        ok = pick == norm(r["gold"])
        oracle["correct"] += ok
        oracle["fell_back"] += 0 if approved else 1
        oracle["per_item"].append({"id": r["id"], "pick": pick,
                                   "gold": norm(r["gold"]), "correct": ok,
                                   "approved": approved})
    oracle["accuracy"] = round(oracle["correct"] / len(recs), 4) if recs else 0.0
    oracle["tokens_per_correct"] = (
        round(oracle["gen_tokens"] / oracle["correct"], 1) if oracle["correct"] else None)
    results.append(oracle)

    summary = {
        "manifest": manifest, "config": vars(args),
        "wall_seconds": round(elapsed, 2),
        "arms": [{k: v for k, v in a.items() if k != "per_item"} for a in results],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (outdir / "per_item.json").write_text(
        json.dumps({a["selector"]: a["per_item"] for a in results}, indent=1),
        encoding="utf-8")

    print("\n" + "=" * 88)
    print("SELECTION ARMS OVER IDENTICAL GENERATED BRANCHES")
    print("=" * 88)
    print(f"{'selector':<22} {'acc':>7} {'correct':>8} {'tok/correct':>12} {'fellback':>9}")
    print("-" * 88)
    for a in results:
        print(f"{a['selector']:<22} {a['accuracy']:>7.1%} {a['correct']:>8} "
              f"{str(a['tokens_per_correct']):>12} {a['fell_back']:>9}")
    maj = next(a for a in results if a["selector"] == "majority_vote")
    for a in results:
        if a["selector"].startswith("verified") and a["tokens_per_correct"] and maj["tokens_per_correct"]:
            ratio = maj["tokens_per_correct"] / a["tokens_per_correct"]
            print(f"\n{a['selector']} vs majority_vote: {ratio:.2f}x cheaper per correct answer")
    print("\nTokens are IDENTICAL across arms by construction, so any difference "
          "comes only from\nselection quality. Raw records: " + str(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
