from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from autotree_core.modeling import ModelExecutor, ModelExecutorConfig


def _model_parameters() -> tuple[pytest.ParameterSet, ...]:
    configured = os.environ.get("AUTOTREE_TEST_MODEL_IDS")
    model_ids = (
        tuple(model_id.strip() for model_id in configured.split(",") if model_id.strip())
        if configured
        else ("tiny", "gpt2")
    )
    if not model_ids:
        raise ValueError("AUTOTREE_TEST_MODEL_IDS must contain at least one model id")
    return tuple(
        pytest.param(model_id, id=model_id.replace("/", "--"))
        for model_id in model_ids
    )


def _test_dtype() -> torch.dtype:
    name = os.environ.get("AUTOTREE_TEST_DTYPE", "float32")
    try:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name]
    except KeyError:
        raise ValueError(
            "AUTOTREE_TEST_DTYPE must be float32, float16, or bfloat16"
        ) from None


@dataclass(frozen=True, slots=True)
class ModelCase:
    name: str
    model_id: str
    executor: ModelExecutor
    prompt: tuple[int, ...]


def _write_tiny_model(path: Path) -> str:
    torch.manual_seed(314159)
    config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        use_cache=True,
    )
    model = GPT2LMHeadModel(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="session")
def tiny_model_id(tmp_path_factory: pytest.TempPathFactory) -> str:
    return _write_tiny_model(tmp_path_factory.mktemp("tiny-gpt2"))


@pytest.fixture(
    scope="session",
    params=_model_parameters(),
)
def model_case(request: pytest.FixtureRequest, tiny_model_id: str) -> ModelCase:
    name = str(request.param)
    model_id = tiny_model_id if name == "tiny" else name
    config = ModelExecutorConfig(
        model_id=model_id,
        device=os.environ.get("AUTOTREE_TEST_DEVICE", "cpu"),
        dtype=_test_dtype(),
        page_size=4,
        capacity_pages=32,
        local_files_only=name == "tiny",
    )
    try:
        executor = ModelExecutor(config)
    except OSError as error:
        pytest.fail(f"required model download/load failed for {model_id}: {error}")
    return ModelCase(
        name=name,
        model_id=model_id,
        executor=executor,
        prompt=(5, 6, 7, 8, 9, 10, 11, 12),
    )
