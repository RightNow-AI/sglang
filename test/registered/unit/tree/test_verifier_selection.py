import importlib
import sys
import time
import types
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol_tree import (
    TreeParameters,
    TreeSummaryResponse,
)
from sglang.srt.tree import tree_runtime
from sglang.srt.tree.params import TreeParams, TreeSummary
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeReq:
    def __init__(self, output_ids):
        self.output_ids = array("l", output_ids)
        self.customized_info = None
        self.to_finish = None


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, decoded):
        self.decoded = decoded

    def decode(self, token_ids, skip_special_tokens=True):
        return self.decoded[tuple(token_ids)]


def install_fake_finish_reason(monkeypatch):
    module = types.ModuleType("sglang.srt.managers.schedule_batch")
    module.FINISH_LENGTH = lambda *, length: ("length", length)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def make_run(verifier):
    decoded = {
        (10,): "reasoning\n#### 7",
        (11,): "another route\n#### 7",
        (12,): "verified route\n#### 42.005",
    }
    runtime = tree_runtime.SchedulerTreeRuntime(
        SimpleNamespace(tokenizer=FakeTokenizer(decoded))
    )
    run = tree_runtime._TreeRun(
        "parent",
        {"policy": "beam", "budget_tokens": 64, "verifier": verifier},
    )
    means = (-0.1, -0.2, -0.9)
    for branch_id, mean in enumerate(means):
        req = FakeReq([10 + branch_id, 99])
        branch = tree_runtime._BranchState(f"req-{branch_id}", branch_id, req)
        branch.tokens = 1
        branch.score = mean
        run.branches[str(branch_id)] = branch
        run.branches_by_rid[branch.rid] = branch
    return runtime, run


def final_snapshot(run):
    return run.branches["0"].req.customized_info["autotree"][-1]


def test_regex_verifier_selects_approved_minority_branch(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_run({"type": "regex", "pattern": r"^42\.005$", "flags": ""})

    runtime._finalize(run, reason="test")

    assert run.winner_branch_id == 2
    assert final_snapshot(run)["verifier_approved_count"] == 1


def test_numeric_verifier_uses_absolute_tolerance(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_run({"type": "numeric", "equals": 42.0, "tolerance": 0.01})

    runtime._finalize(run, reason="test")

    assert run.winner_branch_id == 2
    assert final_snapshot(run)["verifier_fell_back"] is False


def test_reject_all_verifier_falls_back_to_legacy_majority(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_run({"type": "regex", "pattern": r"^999$", "flags": ""})

    runtime._finalize(run, reason="test")

    assert run.winner_branch_id == 0
    snapshot = final_snapshot(run)
    assert snapshot["verifier_used"] is True
    assert snapshot["verifier_approved_count"] == 0
    assert snapshot["verifier_fell_back"] is True


def test_no_verifier_keeps_legacy_majority_and_snapshot_shape(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    runtime, run = make_run(None)

    runtime._finalize(run, reason="test")

    assert run.winner_branch_id == 0
    snapshot = final_snapshot(run)
    assert "verifier_used" not in snapshot
    assert "verifier_approved_count" not in snapshot
    assert "verifier_fell_back" not in snapshot


def test_verifier_disables_majority_lock_and_preselection_pruning(monkeypatch):
    runtime, run = make_run({"type": "regex", "pattern": r"^42\.005$", "flags": ""})
    finalize = Mock()
    prune = Mock()
    monkeypatch.setattr(runtime, "_finalize", finalize)
    monkeypatch.setattr(runtime, "_prune_branch", prune)

    runtime._maybe_majority_lock(run)
    runtime._maybe_consensus_prune(run)
    runtime._maybe_value_prune(run)

    finalize.assert_not_called()
    prune.assert_not_called()


def test_callback_timeout_falls_back_without_raising(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    verifier_module = importlib.import_module("sglang.srt.tree.verifier")

    def slow_post(*_args, **_kwargs):
        time.sleep(0.25)
        return ["2"]

    monkeypatch.setattr(verifier_module, "_post_callback", slow_post)
    runtime, run = make_run(
        {
            "type": "callback",
            "url": "http://127.0.0.1:9/verify",
            "timeout_s": 0.01,
        }
    )

    started = time.monotonic()
    runtime._finalize(run, reason="test")
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert run.winner_branch_id == 0
    assert final_snapshot(run)["verifier_fell_back"] is True


def test_callback_verifier_selects_approved_branch(monkeypatch):
    install_fake_finish_reason(monkeypatch)
    verifier_module = importlib.import_module("sglang.srt.tree.verifier")
    monkeypatch.setattr(
        verifier_module,
        "_post_callback",
        lambda *_args, **_kwargs: ["2"],
    )
    runtime, run = make_run(
        {
            "type": "callback",
            "url": "https://verifier.example/check",
            "timeout_s": 0.1,
        }
    )

    runtime._finalize(run, reason="test")

    assert run.winner_branch_id == 2
    assert final_snapshot(run)["verifier_approved_count"] == 1
    assert final_snapshot(run)["verifier_fell_back"] is False


def test_callback_response_cap_fails_closed_after_one_attempt(monkeypatch):
    verifier_module = importlib.import_module("sglang.srt.tree.verifier")

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b"x" * (verifier_module.CALLBACK_MAX_RESPONSE_BYTES + 1)

    class FakeOpener:
        def __init__(self):
            self.calls = 0

        def open(self, *_args, **_kwargs):
            self.calls += 1
            return FakeResponse()

    opener = FakeOpener()
    monkeypatch.setattr(verifier_module, "build_opener", lambda *_args: opener)

    with pytest.raises(ValueError, match="too large"):
        verifier_module._post_callback(
            "https://verifier.example/check",
            [("0", "7")],
            0.1,
        )

    assert opener.calls == 1


def test_no_verifier_summary_serializes_byte_identically_to_legacy_shape():
    legacy = {
        "policy": "beam",
        "branch_count": 1,
        "pruned_count": 0,
        "merged_count": 0,
        "winner_branch_id": "0",
        "tokens_spent_per_branch": {"0": 1},
        "final_scores": {"0": -0.1},
        "scorer": None,
        "kv_reuse_ratio": None,
        "branch_answers": {"0": "7"},
        "served_from_memo": False,
        "memo_key": None,
    }
    summary = TreeSummaryResponse(
        **legacy,
        verifier_used=False,
        verifier_approved_count=0,
        verifier_fell_back=False,
    )

    assert summary.verifier_used is False
    assert summary.model_dump() == legacy
    assert summary.model_dump_json() == TreeSummaryResponse(**legacy).model_dump_json()


def test_no_verifier_request_serializes_byte_identically_to_legacy_shape():
    params = TreeParameters(policy="beam", branches=3, budget_tokens=64)
    legacy = {
        "policy": "beam",
        "branches": 3,
        "budget_tokens": 64,
        "scorer": None,
        "fork_at_text": None,
        "fork_at_entropy": None,
        "adaptive_width": None,
        "consensus_warmup": 64,
        "consensus_interval": 32,
        "min_survivors": 2,
    }

    assert params.model_dump() == legacy
    assert TreeParams(branches=3, budget_tokens=64).to_runtime_dict() == legacy


@pytest.mark.parametrize(
    ("verifier", "message"),
    [
        ({"type": "other"}, "regex"),
        ({"type": "regex", "flags": "i"}, "pattern"),
        (
            {"type": "numeric", "equals": 42.0, "tolerance": -0.1},
            "tolerance",
        ),
    ],
)
def test_malformed_verifier_blocks_are_rejected(verifier, message):
    with pytest.raises(ValidationError, match=message):
        TreeParameters(
            policy="beam",
            branches=3,
            budget_tokens=64,
            verifier=verifier,
        )


def test_all_verifier_wire_types_are_accepted():
    blocks = [
        {"type": "regex", "pattern": "^ok$", "flags": "i"},
        {"type": "numeric", "equals": 42.0, "tolerance": 1e-6},
        {
            "type": "callback",
            "url": "https://verifier.example/check",
            "timeout_s": 2.0,
        },
    ]

    for block in blocks:
        params = TreeParameters(
            policy="beam",
            branches=3,
            budget_tokens=64,
            verifier=block,
        )
        assert params.verifier.type == block["type"]


def test_tree_summary_dataclass_exposes_verifier_fields():
    summary = TreeSummary(
        policy="beam",
        branch_count=1,
        pruned_count=0,
        merged_count=0,
        winner_branch_id="0",
        tokens_spent_per_branch={"0": 1},
        final_scores={"0": -0.1},
        scorer=None,
        kv_reuse_ratio=None,
        verifier_used=True,
        verifier_approved_count=1,
        verifier_fell_back=False,
    )

    assert summary.to_dict()["verifier_approved_count"] == 1
