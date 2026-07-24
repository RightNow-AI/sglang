#!/usr/bin/env python3
"""Rollout Forest throughput: AutoTree tree-rollouts vs independent best-of-N.

This is the venue where a large multiplier is mathematically available. RL
rollout generation uses large N per prompt, so an independent baseline spends N
full path-equivalents; a tree that shares the prompt prefix, shares a reasoning
trunk, and prunes dead tails can produce the same N trajectories for far fewer
generated tokens. Against best-of-8 inference the ceiling is only ~8x; at N=64
the ceiling is ~64x, which is why the headline number lives here.

HONESTY RULE BUILT IN: raw rollouts-per-second is meaningless alone. A tree can
"win" by emitting N highly correlated trajectories, which carry less training
signal than N independent ones. Every throughput number here is reported beside
an effective-diversity number (distinct final answers / N), and the comparison
prints a warning when the tree's diversity is materially below the independent
baseline's. Real validation (gradient-estimator cosine vs full-N) is future work
and is NOT claimed by this harness.

Modes:
  tree         one POST /v1/tree/completions with branches=N
  independent  N concurrent POST /v1/chat/completions (the vLLM/SGLang shape)
"""

import argparse
import collections
import concurrent.futures
import json
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

NUMBER_RE = re.compile(r"[-+]?\s*\$?\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)")
ANSWER_SUFFIX = (
    "\nSolve step by step, then give the final answer on the last line as: "
    "Answer: <answer>"
)


def extract_answer(text):
    """Loose final-answer extraction, used only to count DISTINCT trajectories
    (diversity), never to score correctness here."""
    if not isinstance(text, str) or not text:
        return None
    marker = text.rfind("Answer:")
    if marker >= 0:
        tail = text[marker + len("Answer:"):].strip().split("\n")[0].strip()
        return tail[:80] or None
    match = list(NUMBER_RE.finditer(text))
    return match[-1].group(0).strip() if match else None


def post_json(url, body, timeout):
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace")), None
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            detail = ""
        return None, "http_{}: {}".format(exc.code, detail)
    except Exception as exc:
        return None, "{}: {}".format(type(exc).__name__, str(exc)[:200])


def item_seed(base_seed, index):
    return base_seed + index * 1000


def tree_body(args, item, base_seed, index):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": item["prompt"] + ANSWER_SUFFIX}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": item_seed(base_seed, index),
        "tree": {
            "policy": "beam",
            "branches": args.n,
            "budget_tokens": args.n * args.max_tokens,
        },
    }


def sample_body(args, item, base_seed, index, sample_index):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": item["prompt"] + ANSWER_SUFFIX}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": item_seed(base_seed, index) + sample_index,
    }


def run_tree_item(args, item, base_seed, index):
    url = args.base_url.rstrip("/") + "/v1/tree/completions"
    started = time.perf_counter()
    payload, error = post_json(url, tree_body(args, item, base_seed, index), args.timeout)
    wall = time.perf_counter() - started
    tokens, answers, pruned = 0, [], None
    if isinstance(payload, dict):
        tree = payload.get("tree") or {}
        spent = tree.get("tokens_spent_per_branch")
        if isinstance(spent, dict):
            tokens = sum(int(v) for v in spent.values() if isinstance(v, int))
        pruned = tree.get("pruned_count")
        branch_answers = tree.get("branch_answers")
        if isinstance(branch_answers, dict):
            answers = [a for a in branch_answers.values() if a]
        if not answers:
            try:
                answers = [extract_answer(payload["choices"][0]["message"]["content"])]
            except Exception:
                answers = []
    if error is None and tokens == 0:
        error = "zero_tokens"
    return {
        "id": item["id"],
        "mode": "tree",
        "seed": base_seed,
        "n": args.n,
        "generated_tokens": tokens,
        "wall_s": round(wall, 6),
        "distinct_answers": len({a for a in answers if a}),
        "answers_seen": len([a for a in answers if a]),
        "pruned_count": pruned,
        "error": error,
    }


def run_independent_item(args, item, base_seed, index):
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    started = time.perf_counter()

    def one(sample_index):
        payload, error = post_json(
            url, sample_body(args, item, base_seed, index, sample_index), args.timeout
        )
        tokens, answer = 0, None
        if isinstance(payload, dict):
            usage = payload.get("usage") or {}
            tokens = int(usage.get("completion_tokens") or 0)
            try:
                answer = extract_answer(payload["choices"][0]["message"]["content"])
            except Exception:
                answer = None
        return tokens, answer, error

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as pool:
        results = list(pool.map(one, range(args.n)))
    wall = time.perf_counter() - started
    tokens = sum(r[0] for r in results)
    answers = [r[1] for r in results if r[1]]
    errors = [r[2] for r in results if r[2]]
    error = "; ".join(errors[:3]) if errors else (None if tokens else "zero_tokens")
    return {
        "id": item["id"],
        "mode": "independent",
        "seed": base_seed,
        "n": args.n,
        "generated_tokens": tokens,
        "wall_s": round(wall, 6),
        "distinct_answers": len(set(answers)),
        "answers_seen": len(answers),
        "pruned_count": None,
        "error": error,
    }


def summarize(records, n):
    ok = [r for r in records if not r.get("error")]
    if not ok:
        return {"items": len(records), "valid": 0}
    total_tokens = sum(r["generated_tokens"] for r in ok)
    total_wall = sum(r["wall_s"] for r in ok)
    rollouts = n * len(ok)
    return {
        "items": len(records),
        "valid": len(ok),
        "errors": len(records) - len(ok),
        "total_generated_tokens": total_tokens,
        "generated_tokens_per_rollout": total_tokens / rollouts,
        "total_wall_s": round(total_wall, 3),
        "rollouts_per_second": rollouts / total_wall if total_wall else 0.0,
        "mean_distinct_answers": statistics.mean(r["distinct_answers"] for r in ok),
        "effective_diversity": statistics.mean(
            r["distinct_answers"] / n for r in ok
        ),
    }


def load_items(path, limit):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def do_compare(tree_path, indep_path):
    tree = json.load(open(tree_path, encoding="utf-8"))
    indep = json.load(open(indep_path, encoding="utf-8"))
    t, i = tree["summary"], indep["summary"]
    tok_ratio = (
        i["generated_tokens_per_rollout"] / t["generated_tokens_per_rollout"]
        if t.get("generated_tokens_per_rollout")
        else float("inf")
    )
    wall_ratio = (
        i["rollouts_per_second"] and t["rollouts_per_second"] / i["rollouts_per_second"]
    )
    print("n (rollouts per prompt):      {}".format(tree["config"]["n"]))
    print("tree   tokens/rollout: {:.1f}   rollouts/s: {:.2f}   diversity: {:.2f}".format(
        t["generated_tokens_per_rollout"], t["rollouts_per_second"], t["effective_diversity"]))
    print("indep  tokens/rollout: {:.1f}   rollouts/s: {:.2f}   diversity: {:.2f}".format(
        i["generated_tokens_per_rollout"], i["rollouts_per_second"], i["effective_diversity"]))
    print("TOKEN-COST RATIO (indep/tree): {:.3f}x cheaper".format(tok_ratio))
    print("WALL RATIO (tree/indep):       {:.3f}x".format(wall_ratio or 0.0))
    rel_div = (
        t["effective_diversity"] / i["effective_diversity"]
        if i.get("effective_diversity")
        else 0.0
    )
    print("RELATIVE DIVERSITY (tree/indep): {:.0f}%".format(100 * rel_div))
    if rel_div < 0.7:
        print(
            "WARNING: tree diversity is materially below the independent baseline. "
            "Correlated rollouts carry less RL training signal; a token-cost win "
            "here is NOT a like-for-like win."
        )
    print(
        "verdict: tree is {:.2f}x cheaper in generated tokens at {:.0f}% relative diversity".format(
            tok_ratio, 100 * rel_div
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--model")
    ap.add_argument("--mode", choices=("tree", "independent"))
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out-jsonl")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("TREE_JSON", "INDEP_JSON"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.compare:
        do_compare(*args.compare)
        return 0
    if not args.data or not args.model:
        ap.error("--data and --model are required")
    items = load_items(args.data, args.limit)
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.dry_run:
        print(json.dumps({
            "tree": {"url": args.base_url + "/v1/tree/completions",
                     "body": tree_body(args, items[0], seeds[0], 0)},
            "independent_first_of_n": {"url": args.base_url + "/v1/chat/completions",
                                       "body": sample_body(args, items[0], seeds[0], 0, 0)},
        }, indent=2)[:2000])
        return 0
    if not args.mode:
        ap.error("--mode is required unless --compare or --dry-run")

    done = set()
    if args.out_jsonl and os.path.exists(args.out_jsonl):
        for line in open(args.out_jsonl, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["mode"], r["seed"], r["id"]))
            except Exception:
                pass
    records, lock = [], threading.Lock()
    handle = open(args.out_jsonl, "a", encoding="utf-8") if args.out_jsonl else None
    runner = run_tree_item if args.mode == "tree" else run_independent_item

    tasks = [(item, s, idx) for s in seeds for idx, item in enumerate(items)
             if (args.mode, s, item["id"]) not in done]
    print("resume: {} done, {} pending".format(len(done), len(tasks)))

    def work(task):
        item, seed, idx = task
        rec = runner(args, item, seed, idx)
        with lock:
            records.append(rec)
            if handle:
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, tasks))
    if handle:
        handle.close()
    # include previously-done records for the summary
    if args.out_jsonl and os.path.exists(args.out_jsonl):
        records = []
        for line in open(args.out_jsonl, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("mode") == args.mode:
                    records.append(r)
            except Exception:
                pass
    summary = summarize(records, args.n)
    out = {"config": vars(args), "summary": summary}
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
