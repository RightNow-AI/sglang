#!/usr/bin/env python3
"""Rollout Forest throughput: AutoTree tree-rollouts vs independent best-of-N.

This is the venue where a large multiplier is mathematically available. RL
rollout generation uses large N per prompt, so an independent baseline spends N
full path-equivalents; a tree that shares the prompt prefix, shares a reasoning
trunk, and prunes dead tails can produce the same N trajectories for far fewer
generated tokens. Against best-of-8 inference the ceiling is only ~8x; at N=64
the ceiling is ~64x, which is why the headline number lives here.

HONESTY RULE BUILT IN: raw rollouts-per-second is meaningless alone. The tree
arm uses randomized, nonzero inclusion probabilities and inverse-probability
weights. Kish effective sample size (ESS), weighted answer-distribution total
variation distance, and generated tokens per effective sample are the claim
gates. Without a positive survival floor the importance estimator is biased and
no multiplier is claimable. Real validation (gradient-estimator cosine vs
full-N) is future work and is NOT claimed by this harness.

Modes:
  tree         one POST /v1/tree/completions with branches=N
  independent  N concurrent POST /v1/chat/completions (the vLLM/SGLang shape)
"""

import argparse
import collections
import concurrent.futures
import json
import math
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
UNPARSED_ANSWER = "<unparsed>"


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


def deterministic_uniform(base_seed, index, branch_id):
    """Stable pseudo-random value in [0, 1) without process-global RNG state."""
    state = item_seed(base_seed, index) & 0xFFFFFFFFFFFFFFFF
    for byte in str(branch_id).encode("utf-8"):
        state ^= byte
        state = (state * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 12
    state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 27
    state = (state * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF
    return (state >> 11) / float(1 << 53)


def kish_ess(weights):
    """Kish ESS = (sum w)^2 / sum(w^2) for positive finite weights."""
    clean = [float(w) for w in weights if isinstance(w, (int, float)) and w > 0]
    if not clean or not all(math.isfinite(w) for w in clean):
        return 0.0
    total = sum(clean)
    squares = sum(w * w for w in clean)
    return total * total / squares if squares else 0.0


def weighted_answer_distribution(trajectories):
    """Normalized inverse-probability-weighted answer distribution."""
    counts = collections.defaultdict(float)
    for trajectory in trajectories:
        weight = trajectory.get("weight")
        if not isinstance(weight, (int, float)) or weight <= 0:
            continue
        answer = trajectory.get("answer")
        key = str(answer) if answer not in (None, "") else UNPARSED_ANSWER
        counts[key] += float(weight)
    total = sum(counts.values())
    if not total:
        return {}
    return {key: counts[key] / total for key in sorted(counts)}


def answer_distributions_by_item(records):
    """Per-prompt distributions, keyed by seed and item ID for paired TV."""
    distributions = {}
    for record in records:
        key = json.dumps(
            [record.get("seed"), record.get("id")], separators=(",", ":")
        )
        distributions[key] = weighted_answer_distribution(
            record.get("trajectories") or []
        )
    return distributions


def total_variation_distance(left, right):
    """TV(P, Q) = 0.5 * sum_x |P(x) - Q(x)|."""
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0)))
                     for k in keys)


def ordered_branch_ids(tree, n):
    """Return all surfaced branch IDs plus placeholders up to the intended N."""
    seen = set()
    ids = []
    raw_winner = tree.get("winner_branch_id")
    if raw_winner is not None:
        winner = str(raw_winner)
        seen.add(winner)
        ids.append(winner)
    for field in ("branch_answers", "tokens_spent_per_branch", "final_scores"):
        values = tree.get(field)
        if not isinstance(values, dict):
            continue
        for raw_id in values:
            branch_id = str(raw_id)
            if branch_id not in seen:
                seen.add(branch_id)
                ids.append(branch_id)
    target = max(n, int(tree.get("branch_count") or 0))
    if all(branch_id.isdigit() for branch_id in ids):
        for candidate in range(target):
            branch_id = str(candidate)
            if branch_id not in seen:
                seen.add(branch_id)
                ids.append(branch_id)
    while len(ids) < target:
        branch_id = "missing_{}".format(len(ids))
        seen.add(branch_id)
        ids.append(branch_id)
    return ids


def nonnegative_int(value):
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def tree_trajectories(tree, fallback_answer, n, survival_floor, base_seed, index):
    """Build a benchmark-side Horvitz-Thompson survival sample.

    The wire response exposes aggregate prune/merge counts, not branch states.
    We conservatively infer the policy-kept stratum from the winner and highest
    final scores. Remaining surfaced branches get independent Bernoulli survival
    at ``survival_floor``. A complete N-branch surface is required for the
    estimator to be claimable; placeholders are recorded but never weighted.
    """
    answers = tree.get("branch_answers")
    answers = {str(k): v for k, v in answers.items()} if isinstance(answers, dict) else {}
    spent = tree.get("tokens_spent_per_branch")
    spent = {str(k): v for k, v in spent.items()} if isinstance(spent, dict) else {}
    scores = tree.get("final_scores")
    scores = {str(k): v for k, v in scores.items()} if isinstance(scores, dict) else {}
    branch_ids = ordered_branch_ids(tree, n)
    winner = str(tree.get("winner_branch_id") or branch_ids[0])
    if fallback_answer and not any(answer for answer in answers.values()):
        answers[winner] = fallback_answer

    pruned = nonnegative_int(tree.get("pruned_count"))
    merged = nonnegative_int(tree.get("merged_count"))
    keep_count = max(1, min(len(branch_ids), len(branch_ids) - pruned - merged))

    def score_key(branch_id):
        value = scores.get(branch_id)
        score = float(value) if isinstance(value, (int, float)) else float("-inf")
        return (branch_id != winner, -score, branch_id)

    policy_kept = set(sorted(branch_ids, key=score_key)[:keep_count])
    trajectories = []
    for branch_id in branch_ids:
        available = branch_id in spent or branch_id in answers or branch_id in scores
        inclusion_prob = 1.0 if branch_id in policy_kept else survival_floor
        draw = deterministic_uniform(base_seed, index, branch_id)
        included = available and draw < inclusion_prob
        trajectories.append({
            "branch_id": branch_id,
            "answer": answers.get(branch_id),
            "generated_tokens": nonnegative_int(spent.get(branch_id)),
            "available": available,
            "policy_kept": branch_id in policy_kept,
            "survival_draw": draw,
            "included": included,
            "inclusion_prob": inclusion_prob,
            "weight": 1.0 / inclusion_prob if included else 0.0,
        })
    return trajectories


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
    tokens, answers, pruned, merged = 0, [], None, None
    trajectories = []
    if isinstance(payload, dict):
        tree = payload.get("tree") or {}
        spent = tree.get("tokens_spent_per_branch")
        if isinstance(spent, dict):
            tokens = sum(int(v) for v in spent.values() if isinstance(v, int))
        pruned = tree.get("pruned_count")
        merged = tree.get("merged_count")
        branch_answers = tree.get("branch_answers")
        if isinstance(branch_answers, dict):
            answers = [a for a in branch_answers.values() if a]
        fallback_answer = None
        if not answers:
            try:
                fallback_answer = extract_answer(
                    payload["choices"][0]["message"]["content"]
                )
                answers = [fallback_answer]
            except Exception:
                answers = []
        trajectories = tree_trajectories(
            tree, fallback_answer, args.n, args.survival_floor, base_seed, index
        )
    if error is None and tokens == 0:
        error = "zero_tokens"
    available = sum(t["available"] for t in trajectories)
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
        "merged_count": merged,
        "survival_floor": args.survival_floor,
        "trajectories": trajectories,
        "estimator_complete": available == args.n and len(trajectories) == args.n,
        "estimator_scope": "surfaced_tree_trajectories",
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
    trajectories = [
        {
            "sample_index": sample_index,
            "answer": answer,
            "generated_tokens": sample_tokens,
            "available": sample_error is None,
            "policy_kept": True,
            "included": sample_error is None,
            "inclusion_prob": 1.0,
            "weight": 1.0 if sample_error is None else 0.0,
        }
        for sample_index, (sample_tokens, answer, sample_error) in enumerate(results)
    ]
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
        "merged_count": None,
        "survival_floor": 1.0,
        "trajectories": trajectories,
        "estimator_complete": not errors and len(trajectories) == args.n,
        "estimator_scope": "independent_trajectories",
        "error": error,
    }


def summarize(records, n):
    ok = [r for r in records if not r.get("error")]
    if not ok:
        return {"items": len(records), "valid": 0}
    total_tokens = sum(r["generated_tokens"] for r in ok)
    total_wall = sum(r["wall_s"] for r in ok)
    rollouts = n * len(ok)
    estimator_records = [r for r in ok if isinstance(r.get("trajectories"), list)]
    trajectories = [
        trajectory
        for record in estimator_records
        for trajectory in record["trajectories"]
    ]
    weights = [trajectory.get("weight") for trajectory in trajectories]
    ess = kish_ess(weights) if len(estimator_records) == len(ok) else None
    ess_over_n = ess / rollouts if ess is not None and rollouts else None
    tokens_per_effective_sample = (
        total_tokens / ess if ess is not None and ess > 0 else None
    )
    distribution = weighted_answer_distribution(trajectories)
    item_distributions = answer_distributions_by_item(estimator_records)
    estimator_valid = (
        len(estimator_records) == len(ok)
        and all(record.get("estimator_complete") for record in estimator_records)
        and bool(distribution)
        and all(item_distributions.values())
    )
    summary = {
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
        "ess": ess,
        "ess_over_n": ess_over_n,
        "tokens_per_effective_sample": tokens_per_effective_sample,
        "surviving_weighted_trajectories": sum(
            1 for trajectory in trajectories if trajectory.get("weight", 0) > 0
        ),
        "answer_distribution": distribution,
        "answer_distributions_by_item": item_distributions,
        "estimator_valid": estimator_valid,
    }
    if not estimator_valid:
        summary["estimator_warning"] = (
            "ESS-adjusted claims require new-format records and a complete N-branch "
            "trajectory surface for every valid item."
        )
    return summary


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
    tree_effective_cost = t.get("tokens_per_effective_sample")
    indep_effective_cost = i.get("tokens_per_effective_sample")
    ess_adjusted_ratio = (
        indep_effective_cost / tree_effective_cost
        if isinstance(indep_effective_cost, (int, float))
        and isinstance(tree_effective_cost, (int, float))
        and tree_effective_cost > 0
        else None
    )
    tree_dist = t.get("answer_distribution") or {}
    indep_dist = i.get("answer_distribution") or {}
    tree_item_dists = t.get("answer_distributions_by_item") or {}
    indep_item_dists = i.get("answer_distributions_by_item") or {}
    paired_items = (
        sorted(tree_item_dists)
        if tree_item_dists and set(tree_item_dists) == set(indep_item_dists)
        else []
    )
    if paired_items:
        tv_values = [
            total_variation_distance(tree_item_dists[key], indep_item_dists[key])
            for key in paired_items
        ]
        tv = sum(tv_values) / len(tv_values)
        tv_scope = "mean over {} paired items".format(len(tv_values))
    else:
        tv = (
            total_variation_distance(tree_dist, indep_dist)
            if tree_dist and indep_dist
            else None
        )
        tv_scope = "pooled fallback"
    print("n (rollouts per prompt):      {}".format(tree["config"]["n"]))
    print("tree   tokens/rollout: {:.1f}   rollouts/s: {:.2f}   diversity: {:.2f}".format(
        t["generated_tokens_per_rollout"], t["rollouts_per_second"], t["effective_diversity"]))
    print("indep  tokens/rollout: {:.1f}   rollouts/s: {:.2f}   diversity: {:.2f}".format(
        i["generated_tokens_per_rollout"], i["rollouts_per_second"], i["effective_diversity"]))
    print("RAW TOKEN RATIO (indep/tree):  {:.3f}x".format(tok_ratio))
    print("TREE ESS/N:                    {}".format(
        "{:.3f}".format(t["ess_over_n"]) if isinstance(t.get("ess_over_n"), (int, float))
        else "unavailable"))
    print("INDEP ESS/N:                   {}".format(
        "{:.3f}".format(i["ess_over_n"]) if isinstance(i.get("ess_over_n"), (int, float))
        else "unavailable"))
    print("ESS-ADJUSTED TOKEN RATIO:       {}".format(
        "{:.3f}x".format(ess_adjusted_ratio) if ess_adjusted_ratio is not None
        else "unavailable"))
    print("TREE WEIGHTED ANSWERS (pooled): {}".format(json.dumps(tree_dist, sort_keys=True)))
    print("INDEP ANSWERS (pooled):         {}".format(json.dumps(indep_dist, sort_keys=True)))
    print("ANSWER TV DISTANCE:             {}".format(
        "{:.3f}".format(tv) if tv is not None else "unavailable"))
    print("ANSWER TV SCOPE:                {}".format(tv_scope))
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
    if tv is not None and tv > 0.25:
        print(
            "WARNING: weighted answer TV distance exceeds 0.25. The tree is not "
            "sampling equivalently enough for a like-for-like multiplier."
        )
    refusal_reasons = []
    if ess_adjusted_ratio is None:
        refusal_reasons.append("ESS-adjusted ratio is unavailable")
    elif ess_adjusted_ratio <= 1.2:
        refusal_reasons.append("ESS-adjusted ratio is not above 1.2x")
    if tv is None:
        refusal_reasons.append("answer TV distance is unavailable")
    elif tv > 0.25:
        refusal_reasons.append("answer TV distance exceeds 0.25")
    if not t.get("estimator_valid") or not i.get("estimator_valid"):
        refusal_reasons.append("one or both estimators are incomplete")
    if (tree_item_dists or indep_item_dists) and not paired_items:
        refusal_reasons.append("the arms do not contain the same prompt and seed pairs")
    if tree.get("config", {}).get("n") != indep.get("config", {}).get("n"):
        refusal_reasons.append("the arms use different N")
    if refusal_reasons:
        print("verdict: REFUSE CLAIM - " + "; ".join(refusal_reasons))
    else:
        print(
            "verdict: CLAIMABLE BENCHMARK WIN - {:.2f}x ESS-adjusted token ratio "
            "at TV {:.3f}; real-trainer gradient validation remains required".format(
                ess_adjusted_ratio, tv
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
    ap.add_argument(
        "--survival-floor",
        type=float,
        default=0.1,
        help=(
            "minimum randomized branch inclusion probability; must be in (0, 1]. "
            "A zero floor makes the importance estimator biased"
        ),
    )
    ap.add_argument("--out-jsonl")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("TREE_JSON", "INDEP_JSON"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not 0 < args.survival_floor <= 1:
        ap.error("--survival-floor must be in (0, 1]")
    if args.n < 1:
        ap.error("--n must be at least 1")

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
                same_estimator = (
                    r.get("mode") != "tree"
                    or r.get("survival_floor") == args.survival_floor
                )
                if same_estimator:
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
                same_estimator = (
                    r.get("mode") != "tree"
                    or r.get("survival_floor") == args.survival_floor
                )
                if r.get("mode") == args.mode and same_estimator:
                    records.append(r)
            except Exception:
                pass
    summary = summarize(records, args.n)
    out = {"config": vars(args), "summary": summary, "records": records}
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
        print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
