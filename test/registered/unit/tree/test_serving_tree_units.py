import sys
import types
from types import SimpleNamespace

import pytest


protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
protocol.ChatCompletionRequest = type("ChatCompletionRequest", (), {})
sys.modules[protocol.__name__] = protocol

serving_base = types.ModuleType("sglang.srt.entrypoints.openai.serving_base")
serving_base.OpenAIServingBase = type("OpenAIServingBase", (), {})
sys.modules[serving_base.__name__] = serving_base

from sglang.srt.tree.params import (  # noqa: E402
    TreeBranchEvent,
    TreeCounters,
    TreeGenerateReqInput,
    TreeParams,
    TreeResult,
    TreeSummary,
)

tree_package = sys.modules["sglang.srt.tree"]
tree_package.TreeBranchEvent = TreeBranchEvent
tree_package.TreeCounters = TreeCounters
tree_package.TreeGenerateReqInput = TreeGenerateReqInput
tree_package.TreeParams = TreeParams
tree_package.TreeResult = TreeResult
tree_package.TreeSummary = TreeSummary

from sglang.srt.entrypoints.openai.serving_tree import OpenAIServingTree  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def serving_without_init(tokenizer=None):
    serving = OpenAIServingTree.__new__(OpenAIServingTree)
    serving.tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)
    return serving


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\n#### 1,234.", "1234"),
        ("first 3, then -4.50", "-4.5"),
        ("the result is 42.", "42"),
        ("#### 7, despite a later note numbered 9", "7"),
        ("no numeric answer", None),
    ],
)
def test_extract_answer(text, expected):
    assert OpenAIServingTree._extract_answer(text) == expected


def test_self_consistency_vote_selects_majority_answer():
    serving = serving_without_init()
    branches = {
        "0": {"mean_logprob": -0.1},
        "1": {"mean_logprob": -0.4},
        "2": {"mean_logprob": -0.3},
    }
    texts = {"0": "#### 8", "1": "#### 9", "2": "#### 9"}

    assert serving._self_consistency_vote(branches, texts) == "2"


def test_self_consistency_vote_breaks_answer_tie_by_mean_logprob():
    serving = serving_without_init()
    branches = {
        "0": {"mean_logprob": -0.8},
        "1": {"mean_logprob": -0.2},
    }
    texts = {"0": "#### 8", "1": "#### 9"}

    assert serving._self_consistency_vote(branches, texts) == "1"


def test_self_consistency_vote_returns_none_without_answers():
    serving = serving_without_init()

    assert (
        serving._self_consistency_vote({"0": {"mean_logprob": -0.1}}, {"0": "unknown"})
        is None
    )


class FakeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, skip_special_tokens=True):
        return {
            (1,): "#### 7",
            (2,): "reasoning, answer #### 42",
            (3,): "another path gives 42",
        }[tuple(token_ids)]


def test_coerce_plain_result_parses_last_snapshot_and_votes():
    serving = serving_without_init(FakeTokenizer())
    snapshot = {
        "policy": "beam",
        "branch_count": 3,
        "pruned_count": 0,
        "winner_branch_id": "0",
        "branches": {
            "0": {"tokens": 1, "mean_logprob": -0.1, "output_ids": [1, 99]},
            "1": {"tokens": 1, "mean_logprob": -0.2, "output_ids": [2, 99]},
            "2": {"tokens": 1, "mean_logprob": -0.4, "output_ids": [3, 99]},
        },
    }
    plain = {
        "text": "parent fallback",
        "meta_info": {
            "prompt_tokens": 5,
            "completion_tokens": 9,
            "finish_reason": {"type": "length"},
            "output_ids": [1, 99],
            "autotree": [None, {"winner_branch_id": "stale"}, snapshot],
        },
    }
    adapted = SimpleNamespace(
        tree=SimpleNamespace(policy="beam", branches=3, budget_tokens=30, scorer=None)
    )

    result = serving._coerce_plain_result(plain, adapted)

    assert result.winner_text == "reasoning, answer #### 42"
    assert result.winner_token_ids == [2, 99]
    assert result.completion_tokens == 2
    assert result.finish_reason == "length"
    assert result.summary.winner_branch_id == "1"
    assert result.summary.scorer == "self_consistency"
    assert result.summary.tokens_spent_per_branch == {"0": 1, "1": 1, "2": 1}
