#!/usr/bin/env python3
"""Would a STRUCTURAL key hit where the exact key does not? The decisive test.

The exact whole-context memo measured 0.0 percent hit on multi-turn agent
conversations and 0.0 percent on ReAct tool loops. That killed the reuse thesis
for the workloads people actually mean by agentic.

But the reason it missed is specific and possibly fixable: the DATA differs
every step while the reasoning STRUCTURE repeats. "Search for Paris population"
and "Search for Berlin population" are the same plan on different values.

If a structural key hits where an exact key does not, reuse is alive and the
mechanism to build is structural memoization. If it does not, reuse is dead for
agentic traffic and we stop proposing it.

This measures the gap directly, zero GPU. It compares three key functions over
identical synthetic-but-structurally-honest traffic:

  EXACT       what the engine ships today, SHA-256 of the whole context
  STRUCTURAL  numbers, quoted strings, ids and urls replaced by typed slots
  SKELETON    structural, plus every content word dropped, keeping only the
              shape: role sequence, verbs, and slot types

The honest caveat, stated up front: a higher hit rate is necessary but NOT
sufficient. A structural hit only helps if the cached REASONING is still valid
for the new values, and that is a correctness question this probe cannot answer.
It is measured here so the next question is asked against a real number rather
than a hope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------- key functions

def exact_key(messages):
    payload = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_QUOTED = re.compile(r"[\"'][^\"']{1,80}[\"']")
_URL = re.compile(r"https?://\S+")
_ID = re.compile(r"\b[a-f0-9]{8,}\b|\b[A-Z]{2,}-\d+\b")


def structuralize(text):
    """Replace values with typed slots, keeping the sentence shape."""
    t = _URL.sub("<URL>", text)
    t = _ID.sub("<ID>", t)
    t = _QUOTED.sub("<STR>", t)
    t = _NUM.sub("<NUM>", t)
    return t


def structural_key(messages):
    shaped = [{"role": m["role"], "content": structuralize(m["content"])}
              for m in messages]
    return exact_key(shaped)


_STOP = {"the","a","an","of","to","and","or","for","in","on","with","is","are",
         "was","were","be","been","do","did","does","this","that","it","as","at",
         "by","from","what","which","please","can","you","i","me","my","next"}


def skeleton_key(messages):
    """Structural, then keep only slots and non-stopword shape tokens."""
    shaped = []
    for m in messages:
        t = structuralize(m["content"]).lower()
        toks = [w for w in re.findall(r"<[A-Z]+>|[a-z]+", t) if w not in _STOP]
        shaped.append({"role": m["role"], "content": " ".join(toks)})
    return exact_key(shaped)


# ---------------------------------------------------------------- traffic shapes

def msgs(*turns):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
            for i, t in enumerate(turns)]


def traffic_multi_turn(sessions=60, depth=6):
    """Agent conversation: same plan shape, different entities and numbers."""
    rng = random.Random(0)
    cities = ["Paris", "Berlin", "Tokyo", "Lima", "Oslo", "Cairo"]
    out = []
    for s in range(sessions):
        city = rng.choice(cities)
        turns = []
        for d in range(depth):
            turns.append(f"Step {d}: look up the population of {city} in {1990 + d}")
            turns.append(f"Found {rng.randint(100000, 9000000)} for {city}")
            out.append(list(msgs(*turns)))
    return out


def traffic_tool_loop(sessions=60, steps=6):
    """ReAct: identical system prompt and plan, different observations."""
    rng = random.Random(1)
    system = "You are an agent. Tools: search, calculator, browse."
    out = []
    for s in range(sessions):
        for step in range(steps):
            obs = (f"observation: search returned {rng.randint(1,99)} results, "
                   f"top hit \"result {rng.randint(1000,9999)}\" at "
                   f"https://example.com/{rng.randint(100,999)}")
            out.append(msgs(system, obs, f"what is the next action for step {step}?"))
    return out


def traffic_forms(n=400):
    """Templated business queries: same form, different values."""
    rng = random.Random(2)
    out = []
    for _ in range(n):
        out.append(msgs(
            f"Customer {rng.randint(1000,9999)} ordered {rng.randint(1,20)} units "
            f"of SKU-{rng.randint(100,999)}. Compute the total at "
            f"${rng.randint(5,99)}.{rng.randint(10,99)} each with "
            f"{rng.randint(5,25)}% tax."))
    return out


def hit_rate(convs, keyfn):
    seen, hits = set(), 0
    for c in convs:
        k = keyfn(c)
        if k in seen:
            hits += 1
        else:
            seen.add(k)
    return hits / len(convs) if convs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    shapes = [
        ("multi-turn agent", traffic_multi_turn()),
        ("ReAct tool loop", traffic_tool_loop()),
        ("templated business forms", traffic_forms()),
    ]

    print("EXACT vs STRUCTURAL vs SKELETON KEYING")
    print("does structure repeat where data does not?\n")
    print(f"{'traffic shape':<28}{'n':>6}{'EXACT':>9}{'STRUCT':>9}{'SKEL':>9}"
          f"{'struct 1/(1-h)':>16}")
    print("-" * 78)
    results = {}
    for name, convs in shapes:
        e = hit_rate(convs, exact_key)
        s = hit_rate(convs, structural_key)
        k = hit_rate(convs, skeleton_key)
        sp = 1 / (1 - s) if s < 1 else float("inf")
        results[name] = {"n": len(convs), "exact": round(e, 4),
                         "structural": round(s, 4), "skeleton": round(k, 4),
                         "structural_ceiling": round(sp, 2)}
        print(f"{name:<28}{len(convs):>6}{e:>9.1%}{s:>9.1%}{k:>9.1%}{sp:>15.2f}x")

    print("\n" + "=" * 78)
    best = max(results.values(), key=lambda r: r["structural"])
    if best["structural"] > 0.5:
        print("STRUCTURE REPEATS WHERE DATA DOES NOT. Reuse is alive as a direction,")
        print("but ONLY if a cached reasoning trace is still VALID under new values.")
        print("That correctness question is NOT answered here and is the real risk:")
        print("a structural hit that returns a wrong answer is worse than a miss.")
        print("Next test must be validity, not hit rate.")
    else:
        print("STRUCTURE DOES NOT REPEAT ENOUGH EITHER. Reuse is dead for these")
        print("shapes and structural memoization should not be built.")
    print("\nCaveat: this traffic is synthetic. The shapes are modelled on real agent")
    print("patterns but the numbers are only as good as that modelling. Treat as a")
    print("go/no-go signal for spending real effort, not as a product claim.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
