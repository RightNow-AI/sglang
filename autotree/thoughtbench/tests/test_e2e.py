from __future__ import annotations

from pathlib import Path
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from autotree_serve import create_app
from thoughtbench.models import (
    BudgetConfig,
    FIXTURE_NOTICE,
    FixtureProvenance,
    PricingConfig,
    RunConfig,
    TaskSetConfig,
    TreeConfig,
)
from thoughtbench.report import render_report
from thoughtbench.runner import PartialStore, partial_path_for, run_benchmark
from thoughtbench.schema import validate_results_payload


@pytest.fixture(scope="module")
def deterministic_server():
    app = create_app(model_id="autotree-deterministic")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="critical",
        access_log=False,
        timeout_keep_alive=1,
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/v1/models", timeout=0.2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.02)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("deterministic test server did not start")
    yield app, base_url
    server.should_exit = True
    thread.join(timeout=5)
    if thread.is_alive():
        server.force_exit = True
        thread.join(timeout=5)
    assert not thread.is_alive()


def _config(tmp_path: Path, base_url: str, mode: str) -> RunConfig:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "tasks.jsonl"
    tree = TreeConfig(policy="beam", branches=3, scorer="logprob") if mode == "tree" else None
    budget = BudgetConfig(
        name="tiny",
        max_tokens=4,
        tree_budget_tokens=12 if mode == "tree" else None,
    )
    return RunConfig(
        model="deterministic-demo",
        base_url=base_url,
        mode=mode,
        task_set=TaskSetConfig(
            name="contract-fixtures",
            path=fixture_path,
            provenance=FixtureProvenance(
                source="bundled synthetic fixtures",
                license="repository",
                notice=FIXTURE_NOTICE,
            ),
        ),
        output_path=tmp_path / f"{mode}.results.json",
        budgets=[budget],
        k_samples=1,
        seeds=[101, 202, 303],
        concurrency=2,
        tree=tree,
        pricing=PricingConfig(
            input_per_million_usd=0.1,
            output_per_million_usd=0.2,
        ),
    )


@pytest.mark.parametrize("mode", ["sequential", "tree"])
def test_deterministic_engine_results_validate_end_to_end(
    tmp_path, deterministic_server, mode
) -> None:
    _app, base_url = deterministic_server
    config = _config(tmp_path, base_url, mode)

    results = run_benchmark(config)

    validate_results_payload(results.model_dump(mode="json"))
    assert config.output_path.exists()
    assert not partial_path_for(config.output_path).exists()
    assert results.artifact_notice == FIXTURE_NOTICE
    assert results.benchmark_claims_allowed is False
    assert len(results.samples) == 6
    assert len(results.per_seed_metrics) == 3
    assert results.aggregate_metrics[0].seed_count == 3
    if mode == "sequential":
        assert all(sample.ttft_seconds is not None for sample in results.samples)
        assert all(sample.tree is None for sample in results.samples)
    else:
        assert all(sample.ttft_seconds is None for sample in results.samples)
        assert all(sample.tree is not None for sample in results.samples)
        assert all(sample.tree.branch_count == 3 for sample in results.samples if sample.tree)
        assert all(
            sample.kv_reuse_ratio is not None and sample.kv_reuse_ratio >= 1
            for sample in results.samples
        )
    report = render_report(config.output_path)
    assert FIXTURE_NOTICE in report
    assert mode in report


def test_runner_resumes_without_repeating_a_completed_sample(
    tmp_path, deterministic_server
) -> None:
    app, base_url = deterministic_server
    config = _config(tmp_path, base_url, "sequential")
    first = run_benchmark(config)
    store = PartialStore(partial_path_for(config.output_path), first.run_fingerprint)
    store.append(first.samples[0])
    counter = app.state.metrics.requests_total.labels(
        endpoint="/v1/chat/completions", status="200"
    )
    before = counter._value.get()

    resumed = run_benchmark(config)

    after = counter._value.get()
    assert after - before == 5
    assert len(resumed.samples) == 6
    assert resumed.samples[0] == first.samples[0]
