import json
from pathlib import Path

import httpx
import pytest

from thoughtbench.answers import extract_answer
from thoughtbench.cli import main
from thoughtbench.harness import (
    dry_run_requests,
    load_bench_tasks,
    load_harness_config,
    run_harness,
)
from thoughtbench.harness_schema import validate_benchmark_results


def test_bundled_task_sets_have_required_counts_and_gold_fields() -> None:
    root = Path(__file__).parents[1]
    gsm8k, _ = load_bench_tasks(root / "tasks" / "gsm8k_test.jsonl")
    subset, _ = load_bench_tasks(root / "tasks" / "gsm8k_subset.jsonl")
    math12, _ = load_bench_tasks(root / "tasks" / "math12.jsonl")

    assert len(gsm8k) == 1319
    assert len(subset) == 50
    assert len(math12) == 12
    assert len({task.id for task in gsm8k}) == 1319
    assert all(extract_answer(task.gold) == task.gold for task in gsm8k)
    assert {
        task.id: task.gold
        for task in (gsm8k[0], gsm8k[49], gsm8k[499], gsm8k[999], gsm8k[-1])
    } == {
        "gsm8k-test-0001": "18",
        "gsm8k-test-0050": "30",
        "gsm8k-test-0500": "10",
        "gsm8k-test-1000": "25",
        "gsm8k-test-1319": "14",
    }


@pytest.mark.parametrize(
    ("filename", "arm", "arm_parameters"),
    [
        ("release-accuracy-autotree-single.json", "single", {}),
        ("release-accuracy-vllm-bestof4.json", "best_of_n", {"n": 4}),
        (
            "release-accuracy-autotree-tree4.json",
            "tree",
            {"branches": 4, "budget_tokens": 2048, "policy": "beam"},
        ),
    ],
)
def test_release_accuracy_configs_validate(filename, arm, arm_parameters) -> None:
    root = Path(__file__).parents[1]
    config = load_harness_config(root / "configs" / filename)

    assert config.arm == arm
    assert config.task_file.resolve() == (root / "tasks" / "gsm8k_test.jsonl").resolve()
    assert config.seeds == [101, 102, 103]
    assert config.max_tokens == 512
    assert config.temperature == 0.7
    for field, expected in arm_parameters.items():
        assert getattr(config, field) == expected


@pytest.mark.parametrize(
    ("filename", "expected_url", "arm_body"),
    [
        (
            "release-accuracy-autotree-single.json",
            "http://autotree.example.invalid:8000/v1/chat/completions",
            {},
        ),
        (
            "release-accuracy-vllm-bestof4.json",
            "http://vllm.example.invalid:8000/v1/chat/completions",
            {"n": 4},
        ),
        (
            "release-accuracy-autotree-tree4.json",
            "http://autotree.example.invalid:8000/v1/tree/completions",
            {"tree": {"policy": "beam", "branches": 4, "budget_tokens": 2048}},
        ),
    ],
)
def test_release_accuracy_dry_run_request_bodies(filename, expected_url, arm_body) -> None:
    root = Path(__file__).parents[1]
    config = load_harness_config(root / "configs" / filename)

    requests = dry_run_requests(config, limit=1)

    assert len(requests) == 3
    assert [request["seed"] for request in requests] == [101, 102, 103]
    for request in requests:
        assert request["task_id"] == "gsm8k-test-0001"
        assert request["url"] == expected_url
        assert request["body"] == {
            "model": "replace-with-served-model",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast "
                        "every morning and bakes muffins for her friends every day with four. "
                        "She sells the remainder at the farmers' market daily for $2 per fresh "
                        "duck egg. How much in dollars does she make every day at the farmers' "
                        "market?"
                    ),
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "seed": request["seed"],
            **arm_body,
        }


def test_yaml_config_and_dry_run_print_exact_bodies_without_network(tmp_path, capsys) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "one", "prompt": "2+2?", "gold": "4"})
        + "\n"
        + json.dumps({"id": "two", "prompt": "3+3?", "gold": "6"})
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "label: dry",
                "engine_label: local",
                "model: model",
                "base_url: http://endpoint.test",
                "arm: single",
                "task_file: tasks.jsonl",
                "output_dir: results",
                "max_tokens: 8",
                "temperature: 0",
                "seeds: [7]",
                "concurrency: 1",
                "timeout: 5",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_harness_config(config)
    requests = dry_run_requests(loaded, limit=1)
    assert requests[0]["body"] == {
        "model": "model",
        "messages": [{"role": "user", "content": "2+2?"}],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 7,
    }

    assert main(["run", "--config", str(config), "--dry-run", "--limit", "1"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == requests[0]


def test_full_harness_run_with_mocked_transport_writes_valid_results(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "one", "prompt": "2+2?", "gold": "4"})
        + "\n"
        + json.dumps({"id": "two", "prompt": "3+3?", "gold": "6"})
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "label": "mock",
                "engine_label": "mock-engine",
                "model": "model",
                "base_url": "http://endpoint.test/v1",
                "arm": "best_of_n",
                "task_file": "tasks.jsonl",
                "output_dir": "results",
                "n": 3,
                "max_tokens": 8,
                "temperature": 0,
                "seeds": [7, 8],
                "concurrency": 1,
                "timeout": 5,
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Answer: 4"}},
                    {"message": {"content": "Answer: 3"}},
                    {"message": {"content": "#### 4"}},
                ],
                "usage": {"completion_tokens": 12},
            },
        )

    results, output = run_harness(
        load_harness_config(config_path),
        transport=httpx.MockTransport(handler),
        limit=1,
    )

    assert output.parent == tmp_path / "results"
    assert output.name.endswith("-mock.json")
    assert results.meta.base_url_redacted == "http://endpoint.test/v1"
    assert results.meta.params["task_count"] == 1
    assert results.meta.params["limit"] == 1
    assert results.summary.accuracy == 1
    assert results.summary.mean_tokens == 12
    assert results.summary.tokens_per_correct == 12
    assert len(results.tasks) == 2
    validate_benchmark_results(json.loads(output.read_text(encoding="utf-8")))
