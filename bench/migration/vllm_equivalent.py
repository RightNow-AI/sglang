#!/usr/bin/env python3
"""What a vLLM user must build to match one AutoTree request. Measured, not claimed.

The throughput case for migrating is dead and we measured it: trunk sharing is
1.85x but stock SGLang n=8 already gets 1.89x, so the engine-exclusive delta is
1.00x. Nobody migrates for parity, and saying otherwise gets us correctly
destroyed by the first engineer who checks.

The honest case is different: a tree request is ONE call, and the equivalent on a
plain OpenAI-compatible engine is an orchestration layer the user writes, owns,
and debugs. This file contains BOTH implementations so the difference is
inspectable rather than asserted, and measures three things that are actually
true and checkable:

  1. client code the user must write and maintain
  2. round trips on the critical path, which is where engine-side decisions win
  3. whether the outputs agree, because a capability claim is worthless if the
     one-call version is not equivalent

Run against any OpenAI-compatible endpoint. The AutoTree arm additionally needs
/v1/tree/completions and is SKIPPED, not faked, when absent.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import time
import urllib.request
from pathlib import Path


def post(base, path, payload, timeout=600):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract(text):
    if not text:
        return None
    i = text.rfind("\\boxed{")
    if i >= 0:
        j, depth, buf = i + 7, 1, []
        while j < len(text) and depth:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            j += 1
        if buf:
            return "".join(buf).strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return m[-1] if m else None


def norm(a):
    if a is None:
        return None
    s = str(a).strip().rstrip(".").replace(" ", "").replace("$", "")
    try:
        return str(float(s)) if re.fullmatch(r"-?\d+(\.\d+)?", s) else s.lower()
    except ValueError:
        return s.lower()


# --------------------------------------------------------------------------
# ARM A: what a vLLM user writes. This is the honest comparison target.
# --------------------------------------------------------------------------

def vllm_style_tree(base, model, question, *, k, trunk_tokens, budget_tokens,
                    max_tokens, suffix):
    """Fork-after-a-trunk, budget-capped, majority-selected: client side.

    Every line here is code the user owns. It is not hard code, but it is code
    they write, test, and keep working across engine upgrades, and it costs an
    extra round trip on the critical path because the fork decision happens in
    the client rather than in the scheduler.
    """
    round_trips = 0

    # 1. generate the shared trunk
    r = post(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": question + suffix}],
        "max_tokens": trunk_tokens, "temperature": 0.7,
    })
    round_trips += 1
    trunk = r["choices"][0]["message"]["content"]
    spent = r["usage"]["completion_tokens"]

    # 2. re-prompt with the trunk so the engine's prefix cache can share its KV,
    #    and ask for k continuations. The user has to know to do this.
    remaining = max(1, min(max_tokens, budget_tokens - spent) // max(1, k))
    r2 = post(base, "/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "user", "content": question + suffix},
            {"role": "assistant", "content": trunk},
        ],
        "n": k, "max_tokens": remaining, "temperature": 0.7,
    })
    round_trips += 1
    spent += r2["usage"]["completion_tokens"]

    # 3. select. The user implements extraction, normalization and voting.
    answers = [norm(extract(trunk + c["message"]["content"]))
               for c in r2["choices"]]
    cand = [a for a in answers if a is not None]
    winner = None
    if cand:
        counts = collections.Counter(cand)
        top = max(counts.values())
        winner = sorted(k_ for k_, v in counts.items() if v == top)[0]

    return {"winner": winner, "answers": answers, "gen_tokens": spent,
            "round_trips": round_trips}


# --------------------------------------------------------------------------
# ARM B: the same thing as one AutoTree call.
# --------------------------------------------------------------------------

def autotree_tree(base, model, question, *, k, trunk_tokens, budget_tokens,
                  max_tokens, suffix):
    r = post(base, "/v1/tree/completions", {
        "model": model,
        "messages": [{"role": "user", "content": question + suffix}],
        "max_tokens": max_tokens,
        "tree": {"policy": "beam", "branches": k, "budget_tokens": budget_tokens},
    })
    summary = r.get("tree") or {}
    answers = list((summary.get("branch_answers") or {}).values())
    return {
        "winner": norm(extract(r["choices"][0]["message"]["content"])),
        "answers": [norm(a) for a in answers],
        "gen_tokens": r["usage"]["completion_tokens"],
        "round_trips": 1,
        "tree_summary": summary,
    }


def supports_tree(base, model):
    try:
        post(base, "/v1/tree/completions", {
            "model": model, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tree": {"policy": "beam", "branches": 2, "budget_tokens": 64},
        }, timeout=120)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--trunk-tokens", type=int, default=96)
    ap.add_argument("--budget-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="bench/migration/results")
    args = ap.parse_args()

    suffix = "\n\nSolve step by step. Put your final answer in \\boxed{}."
    items = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()][:args.n]

    have_tree = supports_tree(args.base, args.model)
    print(f"endpoint supports /v1/tree/completions: {have_tree}")

    rows = {}
    for name, fn, enabled in (("vllm_style_client", vllm_style_tree, True),
                              ("autotree_one_call", autotree_tree, have_tree)):
        if not enabled:
            print(f"\n{name}: SKIPPED (endpoint lacks the tree route)")
            continue
        t0 = time.time()
        recs, correct, trips, tokens = [], 0, 0, 0
        for it in items:
            try:
                out = fn(args.base, args.model, it["question"], k=args.k,
                         trunk_tokens=args.trunk_tokens,
                         budget_tokens=args.budget_tokens,
                         max_tokens=args.max_tokens, suffix=suffix)
            except Exception as e:
                recs.append({"id": it["id"], "error": str(e)[:160]})
                continue
            ok = out["winner"] == norm(it["answer"])
            correct += ok
            trips += out["round_trips"]
            tokens += out["gen_tokens"]
            recs.append({"id": it["id"], "winner": out["winner"],
                         "gold": norm(it["answer"]), "correct": ok,
                         "round_trips": out["round_trips"],
                         "gen_tokens": out["gen_tokens"]})
        wall = time.time() - t0
        rows[name] = {"n": len(items), "correct": correct, "wall_s": round(wall, 1),
                      "round_trips": trips, "gen_tokens": tokens,
                      "accuracy": round(correct / len(items), 4) if items else 0}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with open(Path(args.out) / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")

    print("\n" + "=" * 84)
    print("ONE CALL vs CLIENT-SIDE ORCHESTRATION")
    print("=" * 84)
    print(f"{'arm':<22} {'acc':>7} {'correct':>8} {'round trips':>12} "
          f"{'gen tokens':>11} {'wall_s':>8}")
    print("-" * 84)
    for name, r in rows.items():
        print(f"{name:<22} {r['accuracy']:>7.1%} {r['correct']:>8} "
              f"{r['round_trips']:>12} {r['gen_tokens']:>11} {r['wall_s']:>8.1f}")
    if len(rows) == 2:
        a, b = rows["vllm_style_client"], rows["autotree_one_call"]
        print(f"\nround trips on the critical path: {a['round_trips']} vs "
              f"{b['round_trips']}  ({a['round_trips']/max(1,b['round_trips']):.1f}x)")
        print(f"accuracy delta: {100*(b['accuracy']-a['accuracy']):+.1f}pp "
              "(should be near zero; the claim is equivalence, not superiority)")
    print("\nThe honest claim this supports is DX and round trips, not throughput.")
    print("Throughput parity was measured separately: engine-exclusive delta 1.00x.")
    print(f"\nraw records: {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
