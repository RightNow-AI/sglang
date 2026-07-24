import json
import threading

from sglang.srt.tree.memo import (
    VERIFY_ALWAYS,
    VERIFY_CHEAP,
    MemoStore,
    canonical_key,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_canonicalization_stability_and_answer_relevant_boundaries():
    prompt_a = "Solve this.  \r\n\r\n\r\nShow work.\t\r\n"
    prompt_b = "Solve this.\n\nShow work."
    base_params = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 128}

    base = canonical_key(prompt_a, "model-a", base_params, "ctx-1")
    assert base == canonical_key(prompt_b, "model-a", base_params, "ctx-1")
    assert base != canonical_key(prompt_b, "model-b", base_params, "ctx-1")
    assert base != canonical_key(
        prompt_b, "model-a", {**base_params, "temperature": 0.3}, "ctx-1"
    )
    assert base != canonical_key(prompt_b, "model-a", base_params, "ctx-2")


def test_hit_returns_stored_answer_and_increments_metrics(tmp_path):
    store = MemoStore(tmp_path / "memo.jsonl")
    key = canonical_key("2 + 2", "model-a", {}, "ctx-1")
    store.put(
        key,
        answer_text="The answer is 4.",
        extracted_answer="4",
        model="model-a",
        context_version="ctx-1",
        n_tokens_saved=37,
    )

    entry = store.get(key, model="model-a", context_version="ctx-1")

    assert entry is not None
    assert entry["answer_text"] == "The answer is 4."
    assert entry["hits"] == 1
    assert store.stats() == {
        "entries": 1,
        "hits": 1,
        "misses": 0,
        "hit_rate": 1.0,
        "tokens_saved_estimate": 37,
        "agreement_rate": None,
        "agreement_checks": 0,
        "agreements": 0,
        "disagreements": 0,
        "verification_needed": 0,
    }


def test_different_model_or_context_version_misses(tmp_path):
    store = MemoStore(tmp_path / "memo.jsonl", verification_policy=VERIFY_CHEAP)
    key = canonical_key("problem", "model-a", {}, "ctx-1")
    store.put(key, "answer", model="model-a", context_version="ctx-1")

    assert store.get(key, model="model-b", context_version="ctx-1") is None
    assert store.get(key, model="model-a", context_version="ctx-2") is None
    assert store.get(key, model="model-a", context_version="ctx-1") is not None
    assert store.stats()["misses"] == 2
    assert store.stats()["verification_needed"] == 1


def test_lru_eviction_respects_max_entries(tmp_path):
    store = MemoStore(tmp_path / "memo.jsonl", max_entries=2)
    store.put("key-a", "answer-a", model="model", context_version="ctx")
    store.put("key-b", "answer-b", model="model", context_version="ctx")
    assert store.get("key-a", model="model", context_version="ctx") is not None

    store.put("key-c", "answer-c", model="model", context_version="ctx")

    assert store.get("key-b", model="model", context_version="ctx") is None
    assert store.get("key-a", model="model", context_version="ctx") is not None
    assert store.get("key-c", model="model", context_version="ctx") is not None


def test_verify_always_records_agreement_and_disagreement(tmp_path):
    store = MemoStore(tmp_path / "memo.jsonl", verification_policy=VERIFY_ALWAYS)
    store.put("key", "answer", model="model", context_version="ctx")

    agreed = store.get(
        "key",
        model="model",
        context_version="ctx",
        verify_callback=lambda entry: entry["answer_text"] == "answer",
    )
    disagreed = store.get(
        "key",
        model="model",
        context_version="ctx",
        verify_callback=lambda _entry: False,
    )

    assert agreed is not None
    assert disagreed is None
    assert store.stats()["agreement_checks"] == 2
    assert store.stats()["agreements"] == 1
    assert store.stats()["disagreements"] == 1
    assert store.stats()["agreement_rate"] == 0.5


def test_concurrent_put_get_keeps_store_and_log_consistent(tmp_path):
    path = tmp_path / "memo.jsonl"
    store = MemoStore(path)
    errors = []

    def worker(worker_index):
        try:
            for item_index in range(20):
                key = f"{worker_index:02d}-{item_index:02d}"
                answer = f"answer-{key}"
                store.put(
                    key,
                    answer,
                    model="model",
                    context_version="ctx",
                    n_tokens_saved=1,
                )
                entry = store.get(key, model="model", context_version="ctx")
                assert entry is not None
                assert entry["answer_text"] == answer
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(store) == 160
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert records

    reopened = MemoStore(path)
    assert len(reopened) == 160
    assert reopened.stats()["hits"] == 160
