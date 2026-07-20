from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _install_fake_dependencies(monkeypatch):
    class ServerAdapter:
        pass

    class SGLangHttpServer:
        async def generate(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    class SGLangReplica:
        server_class = "sglang"

    class TokenOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    modules = {
        "verl": _module("verl"),
        "verl.workers": _module("verl.workers"),
        "verl.workers.rollout": _module("verl.workers.rollout"),
        "verl.workers.rollout.sglang_rollout": _module("verl.workers.rollout.sglang_rollout"),
        "verl.workers.rollout.sglang_rollout.sglang_rollout": _module(
            "verl.workers.rollout.sglang_rollout.sglang_rollout", ServerAdapter=ServerAdapter
        ),
        "verl.workers.rollout.sglang_rollout.async_sglang_server": _module(
            "verl.workers.rollout.sglang_rollout.async_sglang_server",
            SGLangHttpServer=SGLangHttpServer,
            SGLangReplica=SGLangReplica,
        ),
        "verl.workers.rollout.replica": _module(
            "verl.workers.rollout.replica", TokenOutput=TokenOutput
        ),
        "ray": _module("ray", remote=lambda cls: ("remote", cls)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _fresh_adapters(monkeypatch):
    _install_fake_dependencies(monkeypatch)
    sys.modules.pop("verl_autotree.adapters", None)
    package = importlib.import_module("verl_autotree")
    for name in package.__all__:
        package.__dict__.pop(name, None)
    return importlib.import_module("verl_autotree.adapters")


def test_generate_defers_to_sglang_without_tree(monkeypatch):
    adapters = _fresh_adapters(monkeypatch)
    server_class = adapters.AutoTreeHttpServer
    server = object.__new__(server_class)

    result = asyncio.run(server.generate([1, 2], {"temperature": 0.5}, "req-1"))

    assert result["args"] == ([1, 2], {"temperature": 0.5}, "req-1")
    assert result["kwargs"] == {
        "image_data": None,
        "video_data": None,
        "bootstrap_host": None,
        "bootstrap_port": None,
        "bootstrap_room": None,
    }


def test_tree_block_parsing_and_validation(monkeypatch):
    adapters = _fresh_adapters(monkeypatch)
    tree = adapters._extract_tree_block(
        {
            "tree": {
                "policy": "best_first",
                "branches": 4,
                "budget_tokens": 128,
                "scorer": "reward-v1",
            }
        }
    )

    assert tree.policy == "best_first"
    assert tree.branches == 4
    assert tree.budget_tokens == 128
    assert tree.scorer == "reward-v1"
    with pytest.raises(ValueError, match="missing required fields: budget_tokens"):
        adapters._extract_tree_block({"tree": {"policy": "beam", "branches": 2}})
    with pytest.raises(ValueError, match="branches must be a positive integer"):
        adapters._extract_tree_block(
            {"tree": {"policy": "beam", "branches": 0, "budget_tokens": 8}}
        )


def test_replica_replaces_sglang_server_actor(monkeypatch):
    adapters = _fresh_adapters(monkeypatch)
    replica = adapters.AutoTreeReplica()

    assert replica.server_class == ("remote", adapters.AutoTreeHttpServer)
