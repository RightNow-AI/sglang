from __future__ import annotations

import pickle
import sys
import types
from array import array
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.openai.protocol_tree import TreeParameters
from sglang.srt.tree import tree_runtime
from sglang.srt.tree.consensus import ConsensusConfig, consensus_scores
from sglang.srt.tree.params import TreeParams
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeReq:
    def __init__(self, rid: str, *, max_new_tokens: int = 4) -> None:
        self.rid = rid
        self.output_ids = array("l")
        self.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
        self.customized_info = None
        self.to_finish = None
        self._finished = False

    def finished(self) -> bool:
        return self._finished


class PieceTokenizer:
    eos_token_id = 99

    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def decode(self, token_ids, skip_special_tokens=True) -> str:
        return "".join(self.pieces[token_id] for token_id in token_ids)


SEQUENCES = {
    0: (10, 11, 12, 13),
    1: (20, 21, 22, 23),
    2: (30, 31, 32, 33),
}


def install_fake_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sglang.srt.managers.schedule_batch")
    module.FINISH_LENGTH = lambda *, length: ("length", length)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def make_runtime_case(
    *,
    consensus_enabled: bool,
    outlier_answer: str = "9 ",
) -> tuple[tree_runtime.SchedulerTreeRuntime, tree_runtime._TreeRun]:
    pieces = {
        10: "Answer: ",
        11: "7 ",
        12: "work ",
        13: "done",
        20: "Answer: ",
        21: "7 ",
        22: "work ",
        23: "done",
        30: "Answer: ",
        31: outlier_answer,
        32: "work ",
        33: "done",
    }
    runtime = tree_runtime.SchedulerTreeRuntime(
        SimpleNamespace(tokenizer=PieceTokenizer(pieces))
    )
    params = {
        "policy": "beam",
        "branches": 3,
        "budget_tokens": 100,
        "scorer": "self_consistency" if consensus_enabled else None,
        "consensus_warmup": 2,
        "consensus_interval": 1,
        "min_survivors": 2,
    }
    run = tree_runtime._TreeRun("parent", params)
    for branch_id in sorted(SEQUENCES):
        req = FakeReq(f"req-{branch_id}")
        branch = tree_runtime._BranchState(req.rid, branch_id, req)
        runtime._register_branch(run, branch)
    return runtime, run


def generate_rounds(
    runtime: tree_runtime.SchedulerTreeRuntime,
    run: tree_runtime._TreeRun,
    rounds: int,
    *,
    start: int = 0,
) -> None:
    for token_index in range(start, start + rounds):
        for branch_id in sorted(SEQUENCES):
            branch = run.branches[str(branch_id)]
            if branch.state != "active":
                continue
            token_id = SEQUENCES[branch_id][token_index]
            branch.req.output_ids.append(token_id)
            runtime._account_tokens(run, branch.req, [token_id], -0.1)


def test_consensus_answer_groups_score_by_live_branch_fraction() -> None:
    scores = consensus_scores(
        {
            3: "The answer is 9.",
            1: "Final answer: 1,000.",
            2: r"Therefore, \boxed{1000}.",
        },
        ConsensusConfig(min_survivors=2),
    )

    assert list(scores) == [1, 2, 3]
    assert scores == pytest.approx({1: 2 / 3, 2: 2 / 3, 3: 1 / 3})


def test_consensus_lexical_fallback_uses_trailing_overlap() -> None:
    scores = consensus_scores(
        {
            1: "old path alpha beta gamma",
            2: "new route alpha beta delta",
            3: "other route zeta eta theta",
        },
        ConsensusConfig(min_survivors=2, trailing_window=3),
    )

    assert scores == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 0.0})


def test_consensus_min_survivors_promotes_the_ranked_survivors() -> None:
    scores = consensus_scores(
        {
            1: "Answer: 7.",
            2: "Answer: 7.",
            3: "Answer: 9.",
            4: "Answer: 10.",
        },
        ConsensusConfig(min_survivors=3),
    )

    assert scores == pytest.approx({1: 0.5, 2: 0.5, 3: 0.5, 4: 0.25})


def test_consensus_tie_breaking_is_deterministic_by_branch_id() -> None:
    first = consensus_scores(
        {
            5: "Answer: 11.",
            2: "Answer: 7.",
            4: "Answer: 10.",
            1: "Answer: 7.",
            3: "Answer: 9.",
        },
        ConsensusConfig(min_survivors=3),
    )
    second = consensus_scores(
        {
            3: "Answer: 9.",
            1: "Answer: 7.",
            4: "Answer: 10.",
            2: "Answer: 7.",
            5: "Answer: 11.",
        },
        ConsensusConfig(min_survivors=3),
    )

    assert list(first) == list(second) == [1, 2, 3, 4, 5]
    assert first == second == pytest.approx(
        {1: 0.4, 2: 0.4, 3: 0.4, 4: 0.2, 5: 0.2}
    )


def test_consensus_tree_params_and_wire_schema_use_documented_defaults() -> None:
    params = TreeParams(scorer="self_consistency")
    params.validate()
    wire = TreeParameters(
        policy="beam",
        branches=3,
        budget_tokens=128,
        scorer="self_consistency",
    )
    transported = TreeParams(**wire.model_dump())
    transported.validate()

    assert params.consensus_warmup == wire.consensus_warmup == 64
    assert transported.consensus_warmup == 64
    assert params.consensus_interval == wire.consensus_interval == 32
    assert transported.consensus_interval == 32
    assert params.min_survivors == wire.min_survivors == 2
    assert transported.min_survivors == 2
    assert transported.to_runtime_dict()["consensus_warmup"] == 64


def test_consensus_kills_diverging_branch_before_branch_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_runtime_case(consensus_enabled=True)

    generate_rounds(runtime, run, 1)
    assert all(branch.state == "active" for branch in run.branches.values())

    generate_rounds(runtime, run, 1, start=1)

    outlier = run.branches["2"]
    assert outlier.state == "pruned"
    assert outlier.tokens == 2
    assert outlier.tokens < outlier.req.sampling_params.max_new_tokens
    assert outlier.req.to_finish == ("length", 2)


def test_consensus_spends_fewer_tokens_than_off_with_the_same_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    off_runtime, off_run = make_runtime_case(consensus_enabled=False)
    on_runtime, on_run = make_runtime_case(consensus_enabled=True)

    generate_rounds(off_runtime, off_run, 4)
    generate_rounds(on_runtime, on_run, 4)
    off_run.branches["0"].req.output_ids.append(99)
    on_run.branches["0"].req.output_ids.append(99)
    off_runtime._finalize(off_run, reason="test")
    on_runtime._finalize(on_run, reason="test")

    assert on_run.winner_branch_id == off_run.winner_branch_id == 0
    assert on_run.spent == 10
    assert off_run.spent == 12
    assert on_run.spent < off_run.spent


def test_consensus_unanimous_agreement_does_not_kill_any_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_runtime_case(
        consensus_enabled=True,
        outlier_answer="7 ",
    )

    generate_rounds(runtime, run, 4)

    assert run.pruned == 0
    assert run.spent == 12
    assert all(branch.state == "active" for branch in run.branches.values())


def test_consensus_off_is_byte_identical_with_explicit_default_knobs() -> None:
    left_runtime, left_run = make_runtime_case(consensus_enabled=False)
    right_runtime, right_run = make_runtime_case(consensus_enabled=False)
    for key in ("consensus_warmup", "consensus_interval", "min_survivors"):
        left_run.params.pop(key)

    generate_rounds(left_runtime, left_run, 2)
    generate_rounds(right_runtime, right_run, 2)
    left_runtime._attach_snapshot(left_run, include_outputs=True)
    right_runtime._attach_snapshot(right_run, include_outputs=True)

    left_bytes = pickle.dumps(
        left_run.branches["0"].req.customized_info,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    right_bytes = pickle.dumps(
        right_run.branches["0"].req.customized_info,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    assert right_bytes == left_bytes


def test_consensus_emits_profile_metric_for_killed_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_finish_reason(monkeypatch)
    increments = []
    monkeypatch.setattr(
        tree_runtime,
        "_autotree_profile_incr",
        lambda name, count=1: increments.append((name, count)),
    )
    runtime, run = make_runtime_case(consensus_enabled=True)

    generate_rounds(runtime, run, 2)

    assert increments == [("tree.consensus.branches_killed", 1)]
