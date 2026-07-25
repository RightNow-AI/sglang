"""The verl adapter must handle what the engine actually yields.

Two bugs made the RL path fail on first real use, and neither was caught
because the verl integration had no tests at all:

1. The adapter asserted isinstance(result, TreeResult) and raised otherwise.
   The tokenizer manager yields PLAIN DICTS for tree requests; the serving
   layer consumes exactly that shape at serving_tree.py:309 via
   result.get("meta_info"). So the adapter raised on the first live call.

2. It read result.winner_log_probs, which does not exist on TreeResult. Because
   it used getattr with a None default, nothing raised: log_probs was silently
   always None. A trainer receiving no per-token logprobs cannot compute a
   policy gradient, so RL training would have been quietly wrong rather than
   loudly broken. That is the worse failure mode.

These pin the coercion against both real shapes. The adapter module only does
lazy imports inside its factory functions, so importing the helper here does
not require verl to be installed.
"""

import importlib.util
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="per-commit-cpu")

REPO = Path(__file__).resolve().parents[4]
ADAPTERS = REPO / "integrations" / "verl_autotree" / "verl_autotree" / "adapters.py"


def load_coerce():
    # The module must be registered in sys.modules BEFORE exec_module, or the
    # @dataclass decorator on _TreeBlock cannot resolve __module__ and raises.
    name = "_verl_adapters_under_test"
    if name in sys.modules:
        return sys.modules[name]._coerce_plain_result
    spec = importlib.util.spec_from_file_location(name, ADAPTERS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module._coerce_plain_result


def test_adapter_source_exists():
    """Guard the guard: a wrong path would make every test below vacuous."""
    assert ADAPTERS.exists(), f"verl adapter not found at {ADAPTERS}"


def test_plain_dict_yields_token_ids_and_logprobs():
    """The real engine shape: sglang triples of (logprob, token_id, text)."""
    coerce = load_coerce()
    result = {
        "meta_info": {
            "output_token_logprobs": [
                [-0.5, 101, "he"],
                [-1.25, 102, "llo"],
            ],
            "finish_reason": {"type": "stop"},
        },
        "tree": {"policy": "beam", "branch_count": 4},
    }
    token_ids, log_probs, finish, summary = coerce(result)

    assert token_ids == [101, 102]
    assert log_probs == [-0.5, -1.25], "logprobs must reach the trainer"
    assert finish == "stop"
    assert summary == {"policy": "beam", "branch_count": 4}


def test_plain_dict_without_logprobs_still_returns_token_ids():
    """return_logprob=False is legal; it must not crash or lose the tokens."""
    coerce = load_coerce()
    token_ids, log_probs, finish, _ = coerce(
        {"meta_info": {"finish_reason": {"type": "length"}}, "output_ids": [7, 8, 9]}
    )
    assert token_ids == [7, 8, 9]
    assert log_probs is None
    assert finish == "length"


def test_string_finish_reason_is_accepted():
    """finish_reason is a dict in some paths and a bare string in others."""
    coerce = load_coerce()
    _, _, finish, _ = coerce({"meta_info": {"finish_reason": "stop"}, "output_ids": []})
    assert finish == "stop"


def test_missing_meta_info_does_not_raise():
    coerce = load_coerce()
    token_ids, log_probs, finish, summary = coerce({})
    assert token_ids == []
    assert log_probs is None
    assert finish is None
    assert summary == {}


def test_treeresult_object_still_supported():
    """A future engine that yields an object must keep working."""
    coerce = load_coerce()

    class FakeSummary:
        def to_dict(self):
            return {"policy": "mcts"}

    class FakeTreeResult:
        winner_token_ids = [4, 5]
        finish_reason = "length"
        summary = FakeSummary()

    token_ids, log_probs, finish, summary = coerce(
        FakeTreeResult(), FakeTreeResult
    )
    assert token_ids == [4, 5]
    assert finish == "length"
    assert summary == {"policy": "mcts"}


def test_unsupported_envelope_raises_clearly():
    coerce = load_coerce()

    class Other:
        pass

    class Expected:
        pass

    try:
        coerce(Other(), Expected)
    except RuntimeError as exc:
        assert "unsupported envelope" in str(exc)
    else:
        raise AssertionError("an unknown envelope type must raise")
