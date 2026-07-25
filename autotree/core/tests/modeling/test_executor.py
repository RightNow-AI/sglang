from __future__ import annotations

import pytest
import torch

from autotree_core.modeling import ModelExecutorConfig

from .conftest import ModelCase


def test_default_model_is_gpt2() -> None:
    config = ModelExecutorConfig()

    assert config.model_id == "gpt2"
    assert config.device == torch.device("cpu")
    assert config.dtype is torch.float32


def test_cpu_device_is_not_rewritten() -> None:
    config = ModelExecutorConfig(device="cpu")

    assert config.device == torch.device("cpu")
    assert config.device.index is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bare_cuda_device_canonicalizes_to_indexed_device() -> None:
    config = ModelExecutorConfig(device="cuda")

    assert config.device.type == "cuda"
    assert config.device.index is not None
    assert config.device == torch.zeros(1, device="cuda").device


def test_real_model_prefill_writes_every_layer_to_paged_kv(
    model_case: ModelCase,
) -> None:
    execution = model_case.executor.prefill(model_case.prompt)
    branch = execution.tree.get_branch(execution.root_id)

    assert branch.num_tokens == len(model_case.prompt)
    assert execution.token_ids(execution.root_id) == model_case.prompt
    assert len(branch.block_table) == 2
    assert execution.pool.used_pages == 2
    for layer in range(model_case.executor.num_layers):
        key, value = execution.gather_kv(execution.root_id, layer)
        assert key.shape == (
            len(model_case.prompt),
            model_case.executor.num_kv_heads,
            model_case.executor.head_dim,
        )
        assert value.shape == key.shape
        assert torch.count_nonzero(key).item() > 0
        assert torch.count_nonzero(value).item() > 0


def test_real_model_fork_cow_prune_and_dedup(model_case: ModelCase) -> None:
    executor = model_case.executor
    prompt = model_case.prompt[:3]

    divergent = executor.prefill(prompt)
    child_id = executor.fork(divergent, divergent.root_id)
    shared_page = divergent.tree.get_branch(divergent.root_id).block_table[0]
    assert divergent.tree.get_branch(child_id).block_table == [shared_page]
    assert divergent.pool.refcount(shared_page) == 2

    root_logits = divergent.next_logits(divergent.root_id)
    root_token, child_token = torch.topk(root_logits, 2).indices.tolist()
    executor.decode(divergent, divergent.root_id, root_token)
    executor.decode(divergent, child_id, child_token)
    root_page = divergent.tree.get_branch(divergent.root_id).block_table[0]
    child_page = divergent.tree.get_branch(child_id).block_table[0]
    assert root_page != child_page
    assert divergent.pool.used_pages == 2

    executor.prune(divergent, child_id)
    assert divergent.pool.used_pages == 1
    assert divergent.pool.refcount(root_page) == 1

    convergent = executor.prefill(prompt)
    child_id = executor.fork(convergent, convergent.root_id)
    token_id = int(torch.argmax(convergent.next_logits(convergent.root_id)).item())
    executor.decode(convergent, convergent.root_id, token_id)
    executor.decode(convergent, child_id, token_id)
    before = convergent.pool.used_pages

    freed = executor.deduplicate(convergent)

    assert freed == 1
    assert convergent.pool.used_pages == before - 1
    root_table = convergent.tree.get_branch(convergent.root_id).block_table
    child_table = convergent.tree.get_branch(child_id).block_table
    assert root_table == child_table
    assert convergent.pool.refcount(root_table[0]) == 2


def test_forked_real_model_reports_physical_kv_reuse(model_case: ModelCase) -> None:
    executor = model_case.executor
    execution = executor.prefill(model_case.prompt)

    executor.fork(execution, execution.root_id)

    assert execution.stats.logical_tokens == 2 * len(model_case.prompt)
    assert execution.stats.physical_tokens == len(model_case.prompt)
    assert execution.stats.physical_tokens < execution.stats.logical_tokens
    assert execution.stats.kv_reuse_ratio > 1.0
