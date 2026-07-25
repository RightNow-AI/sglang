#!/usr/bin/env python3
"""What memo hit rate does realistic traffic actually produce? Offline, no GPU.

Reuse is the only lever left with an unbounded ceiling: cost scales as 1/(1-h).
At h=0.8 that is 5x, at h=0.9 it is 10x. Everything else measured in this
project is capped near 1.2x or is already in the incumbents.

So h is the number the whole remaining thesis rests on, and it must be measured
rather than assumed. This runs the ENGINE'S ACTUAL canonical_key over synthetic
but structurally honest traffic and counts hits.

WHAT THE KEY ACTUALLY DOES, read from memo.py before writing this: it is a
SHA-256 over the fully normalized message list plus model plus a small set of
sampling params. It is EXACT WHOLE-CONTEXT matching, not prefix matching. Two
requests hit only if their entire conversation is identical after whitespace
normalization. That is much narrower than "agentic workloads repeat a lot" and
the patterns below are chosen to expose exactly where it does and does not fire.

Note also, and this matters for the RL story: seed is excluded from the key
unless seeded determinism is requested. So k rollouts of one prompt share a key.
A memo hit would return the SAME answer for every rollout, which destroys the
sample diversity GRPO depends on. Memoization and rollout diversity are in
direct conflict, and that is reported here rather than discovered later.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Load memo.py DIRECTLY by path. Importing it through the sglang package pulls
# in srt.utils.common, which imports the POSIX-only `resource` module and fails
# on Windows. memo.py itself is pure stdlib, so a direct load needs no shims and
# keeps this probe runnable anywhere.
import importlib.util as _ilu

_MEMO = Path(__file__).resolve().parents[2] / "python" / "sglang" / "srt" / "tree" / "memo.py"
_spec = _ilu.spec_from_file_location("_memo_under_test", _MEMO)
_mod = _ilu.module_from_spec(_spec)
sys.modules["_memo_under_test"] = _mod
_spec.loader.exec_module(_mod)
canonical_key = _mod.canonical_key

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SAMPLING = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 512}


def msgs(*turns):
    out = []
    for i, t in enumerate(turns):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return out


def rate(keys):
    seen, hits = set(), 0
    for k in keys:
        if k in seen:
            hits += 1
        else:
            seen.add(k)
    return hits / len(keys) if keys else 0.0


def pattern_repeated_queries(n=500, unique=50):
    """A FAQ-shaped workload: a small set of questions asked over and over."""
    rng = random.Random(0)
    qs = [f"What is the capital of country number {i}?" for i in range(unique)]
    return [canonical_key(msgs(rng.choice(qs)), MODEL, SAMPLING) for _ in range(n)]


def pattern_multi_turn(sessions=50, depth=8):
    """An agent conversation. Context GROWS every turn, so nothing repeats."""
    keys = []
    for s in range(sessions):
        turns = []
        for d in range(depth):
            turns.append(f"session {s} step {d}: do the next thing")
            turns.append(f"ok, did step {d}")
            keys.append(canonical_key(msgs(*turns), MODEL, SAMPLING))
    return keys


def pattern_tool_loop(sessions=50, steps=8):
    """A ReAct loop: same system prompt and tools, DIFFERENT observations."""
    keys = []
    system = "You are an agent. Tools: search, calculator, browse."
    for s in range(sessions):
        for step in range(steps):
            obs = f"observation {s}-{step}: result payload {s*step}"
            keys.append(canonical_key(
                msgs(system, obs, f"what next for task {s}?"), MODEL, SAMPLING))
    return keys


def pattern_rl_rollouts(prompts=50, group=8, epochs=4):
    """GRPO: the same prompts, G samples each, replayed across epochs."""
    keys = []
    for _ in range(epochs):
        for p in range(prompts):
            for _g in range(group):
                keys.append(canonical_key(
                    msgs(f"solve problem {p}"), MODEL, SAMPLING))
    return keys


def pattern_paraphrased(n=500, unique=50):
    """Same intent, different wording. Semantic reuse, not exact reuse."""
    rng = random.Random(1)
    forms = ["What is {}?", "Tell me about {}.", "Can you explain {}?",
             "I'd like to know about {}.", "{} - explain please"]
    topics = [f"topic {i}" for i in range(unique)]
    return [canonical_key(msgs(rng.choice(forms).format(rng.choice(topics))),
                          MODEL, SAMPLING) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    patterns = [
        ("repeated identical queries (FAQ)", pattern_repeated_queries,
         "classic cache hit; a Redis wrapper does this in 20 lines"),
        ("multi-turn agent conversation", pattern_multi_turn,
         "context grows each turn, so nothing ever repeats"),
        ("ReAct tool loop", pattern_tool_loop,
         "system prompt repeats but observations differ"),
        ("RL rollouts, G samples x epochs", pattern_rl_rollouts,
         "hits, but see the diversity warning below"),
        ("paraphrased same-intent queries", pattern_paraphrased,
         "MISLEADING: hits are exact-duplicate collisions, not semantics"),
    ]

    print("MEMO HIT RATE UNDER THE ENGINE'S ACTUAL canonical_key")
    print("exact whole-context SHA-256, not prefix matching\n")
    print(f"{'pattern':<36}{'n':>7}{'hit rate':>11}{'1/(1-h)':>10}   note")
    print("-" * 100)
    results = {}
    for name, fn, note in patterns:
        keys = fn()
        h = rate(keys)
        speedup = 1 / (1 - h) if h < 1 else float("inf")
        results[name] = {"n": len(keys), "hit_rate": round(h, 4),
                         "implied_speedup": round(speedup, 2)}
        print(f"{name:<36}{len(keys):>7}{h:>10.1%}{speedup:>10.2f}x   {note}")

    print("\n" + "=" * 100)
    print("READ THIS BEFORE BELIEVING ANY OF IT")
    print("=" * 100)
    print("1. Multi-turn and tool-loop are the workloads people MEAN by 'agentic',")
    print("   and exact whole-context keying cannot hit on either, because the")
    print("   context is different every single time.")
    print("2. The FAQ pattern hits, but that is ordinary response caching. A client")
    print("   gets it from Redis in about twenty lines and needs no engine change,")
    print("   so it is not a migration reason.")
    print("3. RL rollouts hit, and that is a PROBLEM not a win: seed is excluded")
    print("   from the key, so all G samples of a prompt share one key. Serving a")
    print("   memo hit would return the SAME answer G times and destroy the sample")
    print("   diversity GRPO depends on. Memoization and rollout diversity are in")
    print("   direct conflict.")
    print("4. The paraphrase row is MISLEADING and is kept only with this warning.")
    print("   5 forms x 50 topics is 250 unique strings drawn 500 times, so its")
    print("   hits are exact-duplicate collisions, NOT semantic matching. Exact")
    print("   keying cannot match a paraphrase at all. Real semantic reuse needs")
    print("   embedding keys, which is a different and unbuilt mechanism. I wrote")
    print("   the pattern badly and am recording that rather than deleting it.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
