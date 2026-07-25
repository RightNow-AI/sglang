"""/v1/tree/verify must return per-token values, not just the mean.

verify_branches computes a per-token logprob sequence and BranchScore carries
it, but the endpoint used to build its response from only mean_logprob,
sum_logprob, n_tokens and error, dropping token_logprobs at the HTTP boundary.

That mattered because mean-logprob has been measured USELESS for correctness in
this project. It failed three separate ways: as a pruning criterion, as an
early-stop signal, and as a vote weight. A caller trying to build a real
selector, which is the entire point of the verify endpoint, needs the per-token
sequence. Returning only the mean handed back the one number already known not
to work.

These tests pin the wire shape by reading the handler source, so they need
neither a running server nor the full sglang import chain.
"""

import re
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="per-commit-cpu")

REPO = Path(__file__).resolve().parents[4]
HTTP_SERVER = REPO / "python" / "sglang" / "srt" / "entrypoints" / "http_server.py"
VERIFY = REPO / "python" / "sglang" / "srt" / "tree" / "verify.py"


def handler_source():
    text = HTTP_SERVER.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"async def tree_verify\(.*?\n(?=\n@app)", text, re.S)
    assert match, "could not locate the tree_verify handler"
    return match.group(0)


def test_sources_are_present():
    """Guard the guard: bad paths would make the checks vacuously pass."""
    assert HTTP_SERVER.exists()
    assert VERIFY.exists()
    assert "tree_verify" in HTTP_SERVER.read_text(encoding="utf-8", errors="replace")


def test_verify_endpoint_returns_per_token_values():
    assert "token_logprobs" in handler_source(), (
        "/v1/tree/verify dropped token_logprobs; callers were left with only "
        "mean_logprob, which this project measured as useless for correctness"
    )


def test_verify_endpoint_still_returns_the_aggregates():
    """Adding per-token values must not remove what callers already used."""
    src = handler_source()
    for field in ("mean_logprob", "sum_logprob", "n_tokens", "error"):
        assert field in src, f"verify response lost {field}"


def test_branch_score_declares_token_logprobs():
    """The producer side must actually carry what the endpoint now returns."""
    text = VERIFY.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"class BranchScore\(TypedDict\):(.*?)(?=\n\n|\nclass |\n@)", text, re.S)
    assert match, "could not locate BranchScore"
    assert "token_logprobs" in match.group(1)


def test_scorer_populates_token_logprobs_not_only_the_mean():
    """_score_result must emit the per-token list it builds."""
    text = VERIFY.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"def _score_result\(.*?(?=\nasync def |\ndef )", text, re.S)
    assert match, "could not locate _score_result"
    body = match.group(0)
    assert '"token_logprobs": values' in body, (
        "_score_result computes per-token values then must return them"
    )
