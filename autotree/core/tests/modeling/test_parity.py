from __future__ import annotations

from collections import Counter
from importlib import import_module
from typing import Any

import torch

import autotree_core.modeling.executor as executor_module

from .conftest import ModelCase


def _stock_cache_pairs(
    cache: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache, strict=True))
    return tuple((layer[0], layer[1]) for layer in cache)


def _assert_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bitwise equality on CPU and for integer tensors; dtype-scaled closeness on
    CUDA floats, where GEMM reduction order is not batch-invariant."""
    if not actual.is_floating_point() or actual.device.type == "cpu":
        assert torch.equal(actual, expected), float(
            (actual.float() - expected.float()).abs().max()
        )
        return
    rtol, atol = {
        torch.float32: (1e-5, 1e-6),
        torch.bfloat16: (2e-2, 1e-3),
        torch.float16: (2e-3, 1e-4),
    }[actual.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _assert_cross_kernel_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare outputs produced by two different kernel paths (batched
    tree-attention vs per-branch decode). On CPU both dispatch to the same
    reference kernel, so the contract stays bitwise. On CUDA the paths use
    different GEMM/attention kernels whose reduction orders differ, and the
    error compounds across layers - elementwise closeness is not a sound
    contract there. Instead we require semantic equivalence: identical argmax,
    near-identical top-5, and bounded relative L2 error."""
    if actual.device.type == "cpu":
        assert torch.equal(actual, expected), float(
            (actual.float() - expected.float()).abs().max()
        )
        return
    a, b = actual.float(), expected.float()
    rel_l2 = float((a - b).norm() / b.norm().clamp_min(1e-12))
    limit = 0.03 if actual.dtype is torch.float32 else 0.06
    assert rel_l2 <= limit, f"relative L2 {rel_l2} exceeds {limit}"
    if actual.ndim == 1:
        assert int(a.argmax()) == int(b.argmax()), "argmax diverged across kernels"
        top_a = set(a.topk(5).indices.tolist())
        top_b = set(b.topk(5).indices.tolist())
        assert len(top_a & top_b) >= 4, f"top-5 overlap too low: {top_a & top_b}"


def test_paged_prefill_kv_is_bit_identical_to_stock_model_cache(
    model_case: ModelCase,
) -> None:
    executor = model_case.executor
    prompt = torch.tensor(
        [model_case.prompt[:6]], dtype=torch.long, device=executor.config.device
    )
    execution = executor.prefill(prompt)

    with torch.inference_mode():
        stock_output = executor.model(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            use_cache=True,
            return_dict=True,
        )

    for layer, (stock_k, stock_v) in enumerate(
        _stock_cache_pairs(stock_output.past_key_values)
    ):
        paged_k, paged_v = execution.gather_kv(execution.root_id, layer)
        _assert_matches(paged_k, stock_k[0].transpose(0, 1).contiguous())
        _assert_matches(paged_v, stock_v[0].transpose(0, 1).contiguous())


def test_tree_fork_has_bit_parity_with_sequential_execution(
    model_case: ModelCase,
) -> None:
    executor = model_case.executor
    sequential = executor.prefill(model_case.prompt[:6])
    tree = executor.prefill(model_case.prompt[:6])
    child_id = executor.fork(tree, tree.root_id)

    root_table = tree.tree.get_branch(tree.root_id).block_table
    child_table = tree.tree.get_branch(child_id).block_table
    assert root_table == child_table
    for layer in range(executor.num_layers):
        sequential_k, sequential_v = sequential.gather_kv(sequential.root_id, layer)
        child_k, child_v = tree.gather_kv(child_id, layer)
        _assert_matches(child_k, sequential_k)
        _assert_matches(child_v, sequential_v)

    for _ in range(3):
        sequential_logits = sequential.next_logits(sequential.root_id)
        tree_logits = tree.next_logits(child_id)
        _assert_matches(tree_logits, sequential_logits)
        token_id = int(torch.argmax(sequential_logits).item())

        sequential_step = executor.decode(sequential, sequential.root_id, token_id)
        tree_step = executor.decode(tree, child_id, token_id)

        _assert_matches(tree_step.logits, sequential_step.logits)
        for layer in range(executor.num_layers):
            sequential_k, sequential_v = sequential.gather_kv(sequential.root_id, layer)
            tree_k, tree_v = tree.gather_kv(child_id, layer)
            _assert_matches(tree_k, sequential_k)
            _assert_matches(tree_v, sequential_v)


def test_forest_batch_matches_old_per_branch_decode(
    model_case: ModelCase,
    monkeypatch,
) -> None:
    executor = model_case.executor
    prompt = model_case.prompt[:5]
    forest = executor.prefill(prompt)
    shorter_id = executor.fork(forest, forest.root_id)

    longer_baseline = executor.prefill(prompt)
    shorter_baseline = executor.prefill(prompt)
    setup_token = int(torch.argmax(forest.next_logits(forest.root_id)).item())
    executor.decode(forest, forest.root_id, setup_token)
    executor.decode(longer_baseline, longer_baseline.root_id, setup_token)

    longer_token = int(torch.argmax(forest.next_logits(forest.root_id)).item())
    shorter_token = int(torch.argmax(forest.next_logits(shorter_id)).item())
    longer_step = executor.decode(
        longer_baseline, longer_baseline.root_id, longer_token
    )
    shorter_step = executor.decode(
        shorter_baseline, shorter_baseline.root_id, shorter_token
    )

    forward_calls = 0
    dispatch_contexts: list[list[int]] = []
    original_forward = executor.model.forward
    original_dispatch = executor_module.tree_attention_decode
    original_attention_implementation = executor.model.config._attn_implementation

    def counted_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args, **kwargs)

    def counted_dispatch(*args, **kwargs):
        dispatch_contexts.append(args[4].tolist())
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(executor.model, "forward", counted_forward)
    monkeypatch.setattr(executor_module, "tree_attention_decode", counted_dispatch)
    batch_steps = executor.decode_batch(
        forest,
        [forest.root_id, shorter_id],
        [longer_token, shorter_token],
    )

    assert forward_calls == 1
    assert dispatch_contexts == [[7, 6]] * executor.num_layers
    assert (
        executor.model.config._attn_implementation == original_attention_implementation
    )
    _assert_cross_kernel_matches(batch_steps[0].logits, longer_step.logits)
    _assert_cross_kernel_matches(batch_steps[1].logits, shorter_step.logits)
    for layer in range(executor.num_layers):
        longer_k, longer_v = longer_baseline.gather_kv(longer_baseline.root_id, layer)
        forest_longer_k, forest_longer_v = forest.gather_kv(forest.root_id, layer)
        _assert_cross_kernel_matches(forest_longer_k, longer_k)
        _assert_cross_kernel_matches(forest_longer_v, longer_v)

        shorter_k, shorter_v = shorter_baseline.gather_kv(
            shorter_baseline.root_id, layer
        )
        forest_shorter_k, forest_shorter_v = forest.gather_kv(shorter_id, layer)
        _assert_cross_kernel_matches(forest_shorter_k, shorter_k)
        _assert_cross_kernel_matches(forest_shorter_v, shorter_v)


def test_forest_forward_captures_attention_binding_after_lock_acquisition(
    model_case: ModelCase,
    monkeypatch,
) -> None:
    executor = model_case.executor
    execution = executor.prefill(model_case.prompt[:5])
    child_id = executor.fork(execution, execution.root_id)
    branch_ids = (execution.root_id, child_id)
    token_ids = tuple(
        int(torch.argmax(execution.next_logits(branch_id)).item())
        for branch_id in branch_ids
    )
    model_module = import_module(executor.model.__class__.__module__)
    original_attention = model_module.eager_attention_forward

    def protected_attention(*args, **kwargs):
        return original_attention(*args, **kwargs)

    class RebindingLock:
        def __enter__(self):
            model_module.eager_attention_forward = protected_attention
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(executor_module, "_FOREST_FORWARD_LOCK", RebindingLock())
    try:
        executor.decode_batch(execution, branch_ids, token_ids)
        assert model_module.eager_attention_forward is protected_attention
    finally:
        model_module.eager_attention_forward = original_attention


def test_prefill_and_decode_share_the_forest_mutation_lock(
    model_case: ModelCase,
    monkeypatch,
) -> None:
    class CountingLock:
        def __init__(self) -> None:
            self.entries = 0

        def __enter__(self):
            self.entries += 1
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    lock = CountingLock()
    monkeypatch.setattr(executor_module, "_FOREST_FORWARD_LOCK", lock)

    execution = model_case.executor.prefill(model_case.prompt[:5])
    token_id = int(torch.argmax(execution.next_logits(execution.root_id)).item())
    model_case.executor.decode(execution, execution.root_id, token_id)

    assert lock.entries == 2


def test_greedy_tokens_equal_stock_huggingface_generate(
    model_case: ModelCase,
) -> None:
    executor = model_case.executor
    prompt = torch.tensor(
        [model_case.prompt[:6]], dtype=torch.long, device=executor.config.device
    )
    attention_mask = torch.ones_like(prompt)
    max_new_tokens = 4

    actual = executor.generate(prompt, max_new_tokens=max_new_tokens)
    pad_token_id = executor.model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = executor.model.config.eos_token_id or 0
    with torch.inference_mode():
        expected = executor.model.generate(
            prompt,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            use_cache=True,
        )

    assert torch.equal(actual.sequences.cpu(), expected.cpu())


def test_seeded_sampling_is_repeatable_and_matches_hf_distribution(
    model_case: ModelCase,
) -> None:
    executor = model_case.executor
    prompt = torch.tensor(
        [model_case.prompt[:6]], dtype=torch.long, device=executor.config.device
    )

    first = executor.generate(
        prompt,
        max_new_tokens=4,
        do_sample=True,
        seed=8675309,
        top_k=5,
    )
    second = executor.generate(
        prompt,
        max_new_tokens=4,
        do_sample=True,
        seed=8675309,
        top_k=5,
    )
    assert first.generated_ids == second.generated_ids

    sample_count = 256
    execution = executor.prefill(prompt)
    logits = execution.next_logits(execution.root_id)
    generator = torch.Generator(device=executor.config.device).manual_seed(20260718)
    autotree_samples = [
        executor.sample_next_token(logits, generator=generator, top_k=5)
        for _ in range(sample_count)
    ]

    batch_prompt = prompt.repeat(sample_count, 1)
    torch.manual_seed(20260718)
    pad_token_id = executor.model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = executor.model.config.eos_token_id or 0
    with torch.inference_mode():
        hf_sequences = executor.model.generate(
            batch_prompt,
            attention_mask=torch.ones_like(batch_prompt),
            do_sample=True,
            top_k=5,
            temperature=1.0,
            max_new_tokens=1,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    hf_samples = hf_sequences[:, -1].tolist()

    autotree_counts = Counter(autotree_samples)
    hf_counts = Counter(hf_samples)
    support = set(autotree_counts) | set(hf_counts)
    total_variation = 0.5 * sum(
        abs(autotree_counts[token] - hf_counts[token]) / sample_count
        for token in support
    )
    assert len(autotree_counts) <= 5
    assert len(hf_counts) <= 5
    assert total_variation < 0.15
