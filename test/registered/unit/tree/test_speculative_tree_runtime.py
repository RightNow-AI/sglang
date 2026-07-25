import sys
import types
from types import SimpleNamespace

import pytest

from sglang.srt.tree import tree_runtime
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeReq:
    def __init__(self, rid: str, *, kv_committed_len: int = 0) -> None:
        self.rid = rid
        self.output_ids = []
        self.output_token_logprobs_val = []
        self.sampling_params = SimpleNamespace(max_new_tokens=32)
        self.customized_info = None
        self.to_finish = None
        self.kv_committed_len = kv_committed_len

    def finished(self) -> bool:
        return self.to_finish is not None


class PieceTokenizer:
    eos_token_id = 99

    def __init__(self, pieces=None) -> None:
        self.pieces = pieces or {}

    def decode(self, token_ids, skip_special_tokens=True) -> str:
        return "".join(self.pieces.get(token_id, str(token_id)) for token_id in token_ids)


def install_fake_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sglang.srt.managers.schedule_batch")
    module.FINISH_LENGTH = lambda *, length: ("length", length)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def make_run(
    *,
    branches: int,
    budget_tokens: int = 100,
    scorer=None,
    pieces=None,
):
    runtime = tree_runtime.SchedulerTreeRuntime(
        SimpleNamespace(tokenizer=PieceTokenizer(pieces))
    )
    run = tree_runtime._TreeRun(
        "parent",
        {
            "policy": "beam",
            "branches": branches,
            "budget_tokens": budget_tokens,
            "scorer": scorer,
            "consensus_warmup": 2,
            "consensus_interval": 1,
            "min_survivors": 2,
        },
    )
    for branch_id in range(branches):
        rid = "parent" if branch_id == 0 else f"child-{branch_id}"
        req = FakeReq(rid)
        runtime._register_branch(run, tree_runtime._BranchState(rid, branch_id, req))
    run.forked = branches > 1
    runtime.runs[run.parent_rid] = run
    return runtime, run


def test_spec_run_stops_committing_when_consensus_prunes_mid_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    pieces = {
        10: "Answer: ",
        11: "7 ",
        20: "Answer: ",
        21: "7 ",
        30: "Answer: ",
        31: "9 ",
        32: "must-not-commit",
    }
    runtime, run = make_run(
        branches=3,
        scorer="self_consistency",
        pieces=pieces,
    )

    runtime.commit_token_run(run.branches["0"].req, [10, 11], [-0.1, -0.1])
    runtime.commit_token_run(run.branches["1"].req, [20, 21], [-0.1, -0.1])
    outlier = run.branches["2"]
    accepted, accepted_logprobs = runtime.commit_token_run(
        outlier.req,
        [30, 31, 32],
        [-0.1, -0.1, -0.1],
    )

    assert accepted == [30, 31]
    assert accepted_logprobs == [-0.1, -0.1]
    assert outlier.req.output_ids == [30, 31]
    assert outlier.state == "pruned"
    assert outlier.req.to_finish == ("length", 2)


def test_spec_finalize_selects_scored_branch_instead_of_branch_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_run(branches=2, budget_tokens=3)

    runtime.commit_token_run(run.branches["0"].req, [10], [-5.0])
    runtime.commit_token_run(run.branches["1"].req, [20, 21], [-0.1, -0.1])

    assert run.finalized is True
    assert run.winner_branch_id == 1
    assert run.branches["1"].score == pytest.approx(-0.2)


def test_spec_chunk_accounting_matches_one_token_decode_steps() -> None:
    chunk_runtime, chunk_run = make_run(branches=1)
    step_runtime, step_run = make_run(branches=1)
    tokens = [4, 5, 6, 7]
    logprobs = [-0.4, -0.3, -0.2, -0.1]

    chunk_runtime.commit_token_run(
        chunk_run.branches["0"].req,
        tokens,
        logprobs,
    )
    for token_id, logprob in zip(tokens, logprobs):
        step_runtime.commit_token_run(
            step_run.branches["0"].req,
            [token_id],
            [logprob],
        )

    chunk_branch = chunk_run.branches["0"]
    step_branch = step_run.branches["0"]
    assert chunk_branch.req.output_ids == step_branch.req.output_ids == tokens
    assert chunk_branch.tokens == step_branch.tokens == len(tokens)
    assert chunk_branch.score == pytest.approx(step_branch.score)
    assert chunk_run.spent == step_run.spent == len(tokens)
    assert [event.__dict__ for event in chunk_run.events] == [
        event.__dict__ for event in step_run.events
    ]


def test_pruned_spec_suffix_rolls_back_logical_kv_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    pieces = {
        10: "Answer: ",
        11: "7 ",
        20: "Answer: ",
        21: "7 ",
        30: "Answer: ",
        31: "9 ",
        32: "must-not-commit",
    }
    runtime, run = make_run(
        branches=3,
        scorer="self_consistency",
        pieces=pieces,
    )
    runtime.commit_token_run(run.branches["0"].req, [10, 11], [-0.1, -0.1])
    runtime.commit_token_run(run.branches["1"].req, [20, 21], [-0.1, -0.1])

    outlier = run.branches["2"]
    prompt_kv_len = 10
    outlier.req.kv_committed_len = prompt_kv_len + 3
    accepted, _ = runtime.commit_token_run(
        outlier.req,
        [30, 31, 32],
        [-0.1, -0.1, -0.1],
        speculative=True,
    )

    assert accepted == [30, 31]
    assert outlier.req.kv_committed_len == prompt_kv_len + len(accepted)
    assert outlier.req.finished() is True


def test_spec_branch_rolled_back_when_sibling_already_finalized_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_run(branches=2, budget_tokens=1)
    parent = run.branches["0"].req
    sibling = run.branches["1"].req

    parent.kv_committed_len = 11
    sibling.kv_committed_len = 11
    runtime.commit_token_run(parent, [10], [-0.1], speculative=True)
    accepted, accepted_logprobs = runtime.commit_token_run(
        sibling,
        [20],
        [-0.2],
        speculative=True,
    )

    assert run.finalized is True
    assert accepted == []
    assert accepted_logprobs == []
    assert sibling.output_ids == []
    assert sibling.kv_committed_len == 10


def test_budget_trim_precedes_spec_accounting_and_keeps_kv_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_run(branches=1, budget_tokens=2)
    req = run.branches["0"].req
    prompt_kv_len = 10

    trimmed_tokens, trimmed_logprobs = runtime.trim_tokens_to_budget(
        req,
        [1, 2, 3],
        [-0.1, -0.2, -0.3],
    )
    req.kv_committed_len = prompt_kv_len + len(trimmed_tokens)
    accepted, accepted_logprobs = runtime.commit_token_run(
        req,
        trimmed_tokens,
        trimmed_logprobs,
        speculative=True,
    )

    assert accepted == [1, 2]
    assert accepted_logprobs == [-0.1, -0.2]
    assert req.output_ids == [1, 2]
    assert req.kv_committed_len == prompt_kv_len + len(req.output_ids)
    assert run.spent == 2
    assert run.finalized is True
