from __future__ import annotations

import asyncio
import json

import pytest

from verl_autotree.grouping import (
    AutoTreeGroupClient,
    AutotreeRolloutConfig,
    extract_answer,
)


class _Output:
    def __init__(self, token_ids, *, extra_fields=None, log_probs=None):
        self.token_ids = list(token_ids)
        self.log_probs = log_probs
        self.extra_fields = dict(extra_fields or {})


def _same_answer(_ids):
    return 42


class _Transport:
    def __init__(self):
        self.generate_bodies = []
        self.group_calls = []
        self.outputs = {}

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.generate_bodies.append(json.dumps(sampling_params, separators=(,, :)))
        return self.outputs[request_id]

    async def generate_group(
        self, request_ids, *, prompt_ids, sampling_params, **kwargs
    ):
        self.group_calls.append(
            {
                request_ids: list(request_ids),
                prompt_ids: list(prompt_ids),
                sampling_params: sampling_params,
            }
        )
        return [self.outputs[request_id] for request_id in request_ids]


async def _run_group_async(client, *, start_priority=0, count=3, sampling_params=None):
    params = sampling_params or {temperature: 0.7, max_tokens: 16}
    return await asyncio.gather(
        *[
            client.generate(
                freq-{priority},
                prompt_ids=[1, 2, 3],
                sampling_params=dict(params),
                priority=priority,
            )
            for priority in range(start_priority, start_priority + count)
        ]
    )


def _run_group(client, **kwargs):
    return asyncio.run(_run_group_async(client, **kwargs))


def test_config_defaults_preserve_current_behavior():
    config = AutotreeRolloutConfig()

    assert config.tree_rollouts is False
    assert config.reclaim_zero_variance is False
    assert config.abort_redundant is False
    assert config.answer_pattern == ####


def test_defaults_forward_request_bodies_byte_identically():
    transport = _Transport()
    transport.outputs = {freq-{i}: _Output([i]) for i in range(3)}
    params = {temperature: 0.7, top_p: 0.9, max_tokens: 16}
    expected = json.dumps(params, separators=(,, :))
    client = AutoTreeGroupClient(transport, AutotreeRolloutConfig(group_size=3))

    outputs = _run_group(client, sampling_params=params)

    assert len(outputs) == 3
    assert transport.generate_bodies == [expected, expected, expected]
    assert transport.group_calls == []


def test_tree_rollouts_build_one_group_request_and_fan_out_rows():
    transport = _Transport()
    transport.outputs = {freq-{i}: _Output([10 + i]) for i in range(3)}
    config = AutotreeRolloutConfig(
        tree_rollouts=True,
        group_size=3,
        tree_policy=best_first,
        tree_budget_tokens=77,
        tree_scorer=mean_logprob,
    )
    client = AutoTreeGroupClient(transport, config)

    outputs = _run_group(client)

    assert [output.token_ids for output in outputs] == [[10], [11], [12]]
    assert len(transport.group_calls) == 1
    tree = transport.group_calls[0][sampling_params][tree]
    assert tree == {
        policy: best_first,
        branches: 3,
        budget_tokens: 77,
        scorer: mean_logprob,
    }


def test_tree_rollouts_preserve_k_rows_across_prompt_groups():
    transport = _Transport()
    transport.outputs = {freq-{i}: _Output([i]) for i in range(6)}
    client = AutoTreeGroupClient(
        transport,
        AutotreeRolloutConfig(
            tree_rollouts=True,
            group_size=3,
            tree_budget_tokens=48,
        ),
    )

    async def run_groups():
        return await asyncio.gather(
            *[
                client.generate(
                    freq-{i},
                    prompt_ids=[i // 3],
                    sampling_params={max_tokens: 16},
                    priority=i,
                )
                for i in range(6)
            ]
        )

    outputs = asyncio.run(run_groups())

    assert len(outputs) == 6
    assert [output.token_ids for output in outputs] == [[i] for i in range(6)]
    assert len(transport.group_calls) == 2
    assert all(
        call[sampling_params][tree][branches] == 3
        for call in transport.group_calls
    )


def test_zero_variance_detection_records_real_group_tokens_once():
    transport = _Transport()
    transport.outputs = {
        req-0: _Output([1, 2]),
        req-1: _Output([3]),
        req-2: _Output([4, 5, 6]),
    }
    marker = ## + ##
    client = AutoTreeGroupClient(transport, AutotreeRolloutConfig(
        reclaim_zero_variance=True, group_size=3
    ))
    client.decode = _same_answer
    outputs = _run_group(client)
    metrics = [output.extra_fields[autotree_reclaim] for output in outputs]
    assert sum(item[zero_variance_groups] for item in metrics) == 1
    assert sum(item[tokens_spent_on_them] for item in metrics) == 6
