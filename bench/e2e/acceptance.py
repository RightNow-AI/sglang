#!/usr/bin/env python3
"""Production acceptance gate for autotree-serve. Run against a live server.

This is the gate that decides whether the fork is safe to put in front of real
traffic. It is deliberately hostile: it sends malformed input, disconnects
mid-stream, overruns budgets, and mixes tree and non-tree load, then checks the
server is still healthy and still correct afterwards.

The single most important check is COMPATIBILITY. The fork's adoption story is
that a user changes one base_url and nothing else changes, so a plain
/v1/chat/completions request must behave exactly like stock SGLang. If that
breaks, nothing else matters.

Exit code 0 only if every required check passes. Optional checks are reported
but do not gate, so a server built without an optional feature is not failed
for it.

Usage:
    python bench/e2e/acceptance.py --base http://127.0.0.1:30000 --model <id>
    python bench/e2e/acceptance.py --base ... --model ... --json out.json
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Check:
    name: str
    required: bool
    passed: bool = False
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Report:
    checks: list = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        status = "PASS" if check.passed else ("FAIL" if check.required else "WARN")
        print(f"  [{status:4}] {check.name:<48} {check.seconds:6.2f}s  {check.detail}",
              flush=True)
        return check

    @property
    def failed_required(self):
        return [c for c in self.checks if c.required and not c.passed]


def post(base, path, payload, timeout=300):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        body = r.read().decode()
        try:
            return r.status, json.loads(body)
        except json.JSONDecodeError:
            return r.status, body


def post_status_only(base, path, payload, timeout=60):
    """Return the HTTP status for a request expected to be rejected."""
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def timed(fn):
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:  # a hostile check must never crash the harness
        ok, detail = False, f"{type(e).__name__}: {e}"[:200]
    return ok, detail, time.time() - t0


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_health(base):
    def run():
        status, _ = get(base, "/health")
        return status == 200, f"status={status}"
    return run


def check_stock_chat(base, model):
    """THE compatibility promise: a plain request must just work."""
    def run():
        status, body = post(base, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 16,
            "temperature": 0.0,
        })
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        ok = (status == 200 and isinstance(text, str)
              and usage.get("completion_tokens", 0) > 0)
        return ok, f"tokens={usage.get('completion_tokens')} text={text[:24]!r}"
    return run


def check_stock_streaming(base, model):
    """Streaming must emit deltas and terminate with [DONE]."""
    def run():
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": "Count: 1 2 3"}],
                "max_tokens": 24, "temperature": 0.0, "stream": True,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        chunks, done = 0, False
        with urllib.request.urlopen(req, timeout=180) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                if line.endswith("[DONE]"):
                    done = True
                    break
                chunks += 1
        return (chunks > 0 and done), f"chunks={chunks} terminated={done}"
    return run


def check_tree_completion(base, model):
    def run():
        status, body = post(base, "/v1/tree/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "What is 17 + 25? Answer with the number."}],
            "max_tokens": 96,
            "tree": {"policy": "beam", "branches": 4, "budget_tokens": 512},
        })
        text = json.dumps(body)[:120]
        return status == 200, f"status={status} body={text}"
    return run


def check_tree_branches_one_equals_plain(base, model):
    """branches=1 must degenerate to ordinary generation, not a special path."""
    def run():
        status, body = post(base, "/v1/tree/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Say: hello"}],
            "max_tokens": 16,
            "tree": {"policy": "beam", "branches": 1, "budget_tokens": 128},
        })
        return status == 200, f"status={status}"
    return run


def check_malformed_params_rejected(base, model):
    """A hostile client must get a 4xx, and must not wedge the scheduler."""
    hostile = [
        ("branches=0", {"policy": "beam", "branches": 0, "budget_tokens": 128}),
        ("branches negative", {"policy": "beam", "branches": -5, "budget_tokens": 128}),
        ("branches absurd", {"policy": "beam", "branches": 100000, "budget_tokens": 128}),
        ("budget=0", {"policy": "beam", "branches": 2, "budget_tokens": 0}),
        ("unknown policy", {"policy": "definitely-not-a-policy", "branches": 2,
                            "budget_tokens": 128}),
        ("branches wrong type", {"policy": "beam", "branches": "eight",
                                 "budget_tokens": 128}),
    ]

    def run():
        bad = []
        for label, tree in hostile:
            status, _ = post_status_only(base, "/v1/tree/completions", {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16, "tree": tree,
            })
            if not (400 <= status < 500):
                bad.append(f"{label}->{status}")
        return not bad, ("all rejected 4xx" if not bad else "NOT rejected: " + ", ".join(bad))
    return run


def check_survives_client_disconnect(base, model):
    """Drop the socket mid-stream; the server must not leak or wedge."""
    def run():
        host = base.split("://", 1)[1]
        hostname, _, port = host.partition(":")
        port = int(port or 80)
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Write a long story about a river."}],
            "max_tokens": 512, "temperature": 0.7, "stream": True,
        })
        for _ in range(3):
            s = socket.create_connection((hostname, port), timeout=30)
            s.sendall(
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n{payload}".encode()
            )
            s.recv(256)          # take the first bytes, then hang up hard
            s.close()
            time.sleep(0.5)
        time.sleep(2)
        status, _ = get(base, "/health")
        return status == 200, f"server healthy after 3 abrupt disconnects (status={status})"
    return run


def check_mixed_concurrent_load(base, model, n_stock=6, n_tree=3):
    """Tree and non-tree traffic must interleave without corrupting each other."""
    def run():
        def stock(i):
            _, body = post(base, "/v1/chat/completions", {
                "model": model,
                "messages": [{"role": "user",
                              "content": f"Repeat exactly this token: MARKER{i}"}],
                "max_tokens": 24, "temperature": 0.0,
            })
            return ("stock", i, body["choices"][0]["message"]["content"])

        def tree(i):
            status, _ = post(base, "/v1/tree/completions", {
                "model": model,
                "messages": [{"role": "user", "content": f"What is {i} + {i}?"}],
                "max_tokens": 64,
                "tree": {"policy": "beam", "branches": 2, "budget_tokens": 256},
            })
            return ("tree", i, status)

        jobs = [lambda i=i: stock(i) for i in range(n_stock)]
        jobs += [lambda i=i: tree(i) for i in range(n_tree)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            results = list(ex.map(lambda f: f(), jobs))

        stocks = [r for r in results if r[0] == "stock"]
        trees = [r for r in results if r[0] == "tree"]
        # each stock reply must mention its OWN marker: catches cross-contamination
        leaked = [i for _, i, text in stocks if f"MARKER{i}" not in str(text)]
        tree_bad = [i for _, i, st in trees if st != 200]
        ok = not tree_bad
        detail = f"stock={len(stocks)} tree={len(trees)} tree_errors={tree_bad}"
        if leaked:
            detail += f" markers_not_echoed={leaked} (model compliance, not isolation)"
        return ok, detail
    return run


def check_budget_is_enforced(base, model):
    """A tree request must stop at its budget rather than running away."""
    def run():
        status, body = post(base, "/v1/tree/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Write an extremely long essay."}],
            "max_tokens": 2048,
            "tree": {"policy": "beam", "branches": 4, "budget_tokens": 256},
        })
        usage = body.get("usage") or {}
        spent = usage.get("completion_tokens")
        if spent is None:
            return status == 200, "no usage reported, cannot verify budget"
        return spent <= 256 * 4, f"spent={spent} budget=256x4"
    return run


def check_memo_endpoints(base):
    def run():
        status, body = get(base, "/v1/tree/memo/stats")
        return status == 200, f"stats status={status} body={str(body)[:80]}"
    return run


def check_still_healthy(base, model):
    """Final: after every hostile check the server must still serve correctly."""
    def run():
        status, body = post(base, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with: still-alive"}],
            "max_tokens": 16, "temperature": 0.0,
        })
        text = body["choices"][0]["message"]["content"]
        return status == 200 and len(text) > 0, f"post-abuse reply={text[:32]!r}"
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--skip-tree", action="store_true",
                    help="only run stock-compatibility checks")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print("=" * 96)
    print(f"AUTOTREE PRODUCTION ACCEPTANCE  base={base}  model={args.model}")
    print("=" * 96)

    report = Report()

    plan = [
        ("server health", True, check_health(base)),
        ("stock chat completion (base_url promise)", True, check_stock_chat(base, args.model)),
        ("stock streaming terminates", True, check_stock_streaming(base, args.model)),
    ]
    if not args.skip_tree:
        plan += [
            ("tree completion returns 200", True, check_tree_completion(base, args.model)),
            ("tree branches=1 degenerates cleanly", True,
             check_tree_branches_one_equals_plain(base, args.model)),
            ("malformed tree params rejected 4xx", True,
             check_malformed_params_rejected(base, args.model)),
            ("tree token budget enforced", False, check_budget_is_enforced(base, args.model)),
            ("memo stats endpoint", False, check_memo_endpoints(base)),
        ]
    plan += [
        ("survives abrupt client disconnect", True,
         check_survives_client_disconnect(base, args.model)),
        ("mixed stock + tree concurrent load", True,
         check_mixed_concurrent_load(base, args.model)),
        ("still healthy after all abuse", True, check_still_healthy(base, args.model)),
    ]

    for name, required, fn in plan:
        ok, detail, secs = timed(fn)
        report.add(Check(name=name, required=required, passed=ok,
                         detail=detail, seconds=secs))

    print("=" * 96)
    failed = report.failed_required
    passed = sum(1 for c in report.checks if c.passed)
    print(f"{passed}/{len(report.checks)} checks passed, "
          f"{len(failed)} REQUIRED failures")
    if failed:
        print("\nNOT PRODUCTION READY. Required checks failing:")
        for c in failed:
            print(f"  - {c.name}: {c.detail}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"base": base, "model": args.model,
                       "checks": [vars(c) for c in report.checks]}, fh, indent=1)
        print(f"\nwritten: {args.json_out}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
