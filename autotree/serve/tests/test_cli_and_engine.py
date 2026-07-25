from __future__ import annotations

import pytest

from autotree_serve import cli
from autotree_serve.app import _to_engine_request
from autotree_serve.cli import main
from autotree_serve.engine import (
    DeterministicEngine,
    GenerationDone,
    GenerationRequest,
    Message,
    TreeExecution,
)
from autotree_serve.schema import ChatCompletionRequest


def test_cli_help_is_honest_about_deterministic_engine(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())
    assert "seeded toy generator" in output
    assert "does not serve real model weights" in output
    assert "The deterministic engine always exposes deterministic-demo." in normalized_output
    assert "--kv-pages" in output
    assert "--kv-branch-headroom" in output
    assert "--max-concurrent-requests" in output


def test_consensus_tree_params_flow_to_engine_request() -> None:
    body = ChatCompletionRequest.model_validate(
        {
            "model": "toy",
            "messages": [{"role": "user", "content": "compare"}],
            "tree": {
                "policy": "beam",
                "branches": 4,
                "budget_tokens": 64,
                "scorer": "self_consistency",
                "consensus_interval": 8,
                "consensus_warmup": 16,
                "min_survivors": 3,
            },
        }
    )

    request = _to_engine_request(body)

    assert request.tree is not None
    assert request.tree.consensus_interval == 8
    assert request.tree.consensus_warmup == 16
    assert request.tree.min_survivors == 3


def test_treekv_engine_model_load_failure_is_honest(capsys, monkeypatch):
    def fail_load(_model_id, **_kwargs):
        raise OSError("weights unavailable")

    monkeypatch.setattr("autotree_serve.cli._load_treekv_engine", fail_load)
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--model", "demo", "--engine", "treekv"])

    assert exc.value.code != 0
    error = capsys.readouterr().err
    assert "Failed to load Tree-KV model 'demo'" in error
    assert "weights unavailable" in error


def test_cli_starts_treekv_server(monkeypatch):
    called = {}
    fake_engine = DeterministicEngine("gpt2")

    def fake_load(model_id, *, kv_pages, kv_branch_headroom, device="cpu", dtype="float32"):
        called.update(
            model_id=model_id,
            kv_pages=kv_pages,
            kv_branch_headroom=kv_branch_headroom,
        )
        return fake_engine

    monkeypatch.setattr("autotree_serve.cli._load_treekv_engine", fake_load)
    monkeypatch.setattr(
        "autotree_serve.cli.uvicorn.run",
        lambda app, *, host, port: called.update(app=app, host=host, port=port),
    )

    main(
        [
            "serve",
            "--engine",
            "treekv",
            "--kv-pages",
            "256",
            "--kv-branch-headroom",
            "2.0",
            "--max-concurrent-requests",
            "3",
        ]
    )

    assert called["app"].state.engine is fake_engine
    assert called["model_id"] == "gpt2"
    assert called["kv_pages"] == 256
    assert called["kv_branch_headroom"] == 2.0
    assert called["app"].state.engine_runner.max_concurrent_requests == 3
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8000


def test_cli_starts_deterministic_server(monkeypatch):
    called = {}

    def fake_run(app, *, host, port):
        called.update(app=app, host=host, port=port)

    monkeypatch.setattr("autotree_serve.cli.uvicorn.run", fake_run)
    main(
        [
            "serve",
            "--model",
            "toy-model",
            "--engine",
            "deterministic",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
        ]
    )

    assert called["host"] == "0.0.0.0"
    assert called["port"] == 8123
    assert called["app"].state.engine.model_metadata.id == "deterministic-demo"
    assert called["app"].state.engine.model_metadata.real_model_weights is False


async def test_deterministic_engine_repeats_seeded_event_stream():
    engine = DeterministicEngine("toy")
    request = GenerationRequest(
        model="toy",
        messages=(Message(role="user", content="repeat this"),),
        max_tokens=5,
        temperature=1.0,
        top_p=1.0,
        stop=(),
        seed=123,
        user=None,
        tree=TreeExecution(
            policy="beam",
            branches=3,
            budget_tokens=11,
            scorer=None,
        ),
    )

    first = [event async for event in engine.generate(request)]
    second = [event async for event in engine.generate(request)]
    first_done = next(event for event in first if isinstance(event, GenerationDone))
    second_done = next(event for event in second if isinstance(event, GenerationDone))

    assert first_done.text == second_done.text
    assert first_done.usage == second_done.usage
    assert first_done.tree_summary == second_done.tree_summary
    assert first_done.usage.completion_tokens == 11


async def test_tree_winner_has_generated_content_when_budget_is_narrow():
    engine = DeterministicEngine("toy")
    request = GenerationRequest(
        model="toy",
        messages=(Message(role="user", content="use the only generated token"),),
        max_tokens=16,
        temperature=1.0,
        top_p=1.0,
        stop=(),
        seed=1,
        user=None,
        tree=TreeExecution(
            policy="beam",
            branches=4,
            budget_tokens=1,
            scorer=None,
        ),
    )

    events = [event async for event in engine.generate(request)]
    done = next(event for event in events if isinstance(event, GenerationDone))

    assert done.text
    assert done.tree_summary is not None
    assert done.tree_summary.tokens_spent_per_branch[done.branch_id] == 1


def test_parser_accepts_device_and_dtype() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["serve", "--engine", "treekv", "--device", "cuda", "--dtype", "bfloat16"]
    )

    assert args.device == "cuda"
    assert args.dtype == "bfloat16"


def test_parser_device_defaults_stay_cpu_float32() -> None:
    args = cli.build_parser().parse_args(["serve"])

    assert args.device == "cpu"
    assert args.dtype == "float32"
    assert args.max_concurrent_requests == 8
