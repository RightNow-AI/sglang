import asyncio
import sys
import types
from types import SimpleNamespace


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

from sglang.srt.entrypoints.openai.serving_tree import (  # noqa: E402
    OpenAIServingTree,
)
from sglang.srt.tree.memo import MemoStore  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeTokenizerManager:
    def __init__(self, result):
        self.result = result
        self.generate_calls = 0

    def generate_request(self, _adapted_request, _raw_request):
        self.generate_calls += 1

        async def generate():
            yield self.result

        return generate()


class MustNotBeConsultedStore:
    def __init__(self):
        self.get_calls = 0
        self.put_calls = 0

    def get(self, *_args, **_kwargs):
        self.get_calls += 1
        raise AssertionError("disabled memo store was consulted")

    def put(self, *_args, **_kwargs):
        self.put_calls += 1
        raise AssertionError("disabled memo store was written")


def make_result():
    return TreeResult(
        winner_text="reasoning\n#### 42",
        winner_token_ids=[10, 11],
        prompt_tokens=5,
        completion_tokens=9,
        summary=TreeSummary(
            policy="beam",
            branch_count=2,
            pruned_count=0,
            merged_count=0,
            winner_branch_id="1",
            tokens_spent_per_branch={"0": 4, "1": 5},
            final_scores={"0": -0.5, "1": -0.1},
            scorer="self_consistency",
            kv_reuse_ratio=0.5,
            branch_answers={"0": "41", "1": "42"},
        ),
    )


def make_request(
    *,
    model="model-a",
    temperature=0.2,
    context_version="ctx-1",
):
    return SimpleNamespace(
        model=model,
        temperature=temperature,
        top_p=0.9,
        resolved_max_tokens=128,
        seed=7,
        resolved_seed=7,
        context_version=context_version,
        tree=SimpleNamespace(policy="beam", branches=2, scorer=None),
    )


def make_adapted_request():
    return TreeGenerateReqInput(
        base=SimpleNamespace(text="rendered prompt", input_ids=None),
        tree=TreeParams(policy="beam", branches=2, budget_tokens=256),
    )


def make_serving(store):
    serving = OpenAIServingTree.__new__(OpenAIServingTree)
    serving.tokenizer_manager = FakeTokenizerManager(make_result())
    serving._memo_store = store
    return serving


async def handle(serving, request):
    return await serving._handle_non_streaming_request(
        make_adapted_request(), request, raw_request=object()
    )


def test_memo_unset_leaves_request_path_unchanged(monkeypatch):
    monkeypatch.delenv("AUTOTREE_MEMO", raising=False)
    store = MustNotBeConsultedStore()
    serving = make_serving(store)

    response = asyncio.run(handle(serving, make_request()))

    assert serving.tokenizer_manager.generate_calls == 1
    assert store.get_calls == 0
    assert store.put_calls == 0
    assert response.choices[0].message.content == "reasoning\n#### 42"
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 9
    assert response.tree.served_from_memo is False


def test_memo_miss_then_hit_skips_scheduler_and_reports_zero_tokens(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AUTOTREE_MEMO", "1")
    store = MemoStore(tmp_path / "memo.jsonl")
    serving = make_serving(store)
    request = make_request()

    first = asyncio.run(handle(serving, request))
    second = asyncio.run(handle(serving, request))

    assert first.tree.served_from_memo is False
    assert first.usage.total_tokens == 14
    assert serving.tokenizer_manager.generate_calls == 1
    assert second.choices[0].message.content == "reasoning\n#### 42"
    assert second.tree.served_from_memo is True
    assert second.tree.memo_key is not None
    assert len(second.tree.memo_key) == 64
    assert second.usage.prompt_tokens == 0
    assert second.usage.completion_tokens == 0
    assert second.usage.total_tokens == 0
    assert serving.memo_stats() == {
        "hits": 1,
        "misses": 1,
        "hit_rate": 0.5,
        "tokens_saved": 14,
        "agreement_rate": None,
    }


def test_temperature_model_and_context_version_each_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOTREE_MEMO", "1")
    store = MemoStore(tmp_path / "memo.jsonl")
    serving = make_serving(store)

    asyncio.run(handle(serving, make_request()))
    temperature_response = asyncio.run(
        handle(serving, make_request(temperature=0.3))
    )
    model_response = asyncio.run(handle(serving, make_request(model="model-b")))
    context_response = asyncio.run(
        handle(serving, make_request(context_version="ctx-2"))
    )

    assert serving.tokenizer_manager.generate_calls == 4
    assert temperature_response.tree.served_from_memo is False
    assert model_response.tree.served_from_memo is False
    assert context_response.tree.served_from_memo is False
    assert store.stats()["hits"] == 0
    assert store.stats()["misses"] == 4
    assert len(store) == 4


def test_memo_hit_does_not_replay_branch_statistics(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOTREE_MEMO", "1")
    serving = make_serving(MemoStore(tmp_path / "memo.jsonl"))
    request = make_request()

    asyncio.run(handle(serving, request))
    hit = asyncio.run(handle(serving, request))

    assert hit.tree.served_from_memo is True
    assert hit.tree.branch_answers == {}
    assert hit.tree.tokens_spent_per_branch == {}
    assert hit.tree.final_scores == {}
    assert hit.tree.winner_branch_id == "memo"

