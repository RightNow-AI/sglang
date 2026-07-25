from __future__ import annotations

import math
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from autotree_sdk import TreeClient, rollout
from autotree_serve import create_app


@pytest.fixture(scope="module")
def deterministic_server() -> str:
    app = create_app(model_id="deterministic-demo")
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

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)
    if thread.is_alive():
        server.force_exit = True
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_real_serve_stream_is_consumable_by_real_sdk(
    deterministic_server: str,
) -> None:
    with TreeClient(deterministic_server) as client:
        batch = rollout(
            ["prove the wire contract"],
            3,
            budget_tokens=6,
            seed=17,
            model="deterministic-demo",
            client=client,
            max_tokens=2,
        )

    tree = batch.trees[0]
    assert tree.usage.completion_tokens == sum(
        len(branch.tokens) for branch in tree.branches
    )
    assert all(
        math.isfinite(logprob)
        for branch in tree.branches
        for logprob in branch.token_logprobs
    )
    assert all(
        branch.prune_reason
        for branch in tree.branches
        if branch.pruned
    )
