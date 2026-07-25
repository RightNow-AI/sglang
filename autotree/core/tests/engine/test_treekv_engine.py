from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import asdict, replace

import pytest
import torch

from autotree_core.engine import (
    BranchMerged,
    BranchPruned,
    BranchStarted,
    GenerationDone,
    GenerationRequest,
    KVCapacityExceededError,
    Message,
    TokenGenerated,
    TreeExecution,
    TreeKVEngine,
)


class ScriptedScheduler:
    """Drive one fork, one killed sibling, and one winning continuation."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()
        self._token_events = 0

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] != "token_sampled":
            return
        self._token_events += 1
        if self._token_events == 1:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": 0, "width": 2},
                    {"type": "continue", "branch": 1},
                    {"type": "continue", "branch": 2},
                ]
            )
        elif self._token_events == 3:
            self._commands.extend(
                [
                    {"type": "kill", "branch": 2, "reason": "beam_pruned"},
                    {"type": "finalize", "branch": 1},
                    {"type": "kill", "branch": 0, "reason": "tree_budget_exhausted"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class ExhaustionScheduler:
    def __init__(
        self,
        config: dict[str, object],
        observed_events: list[dict[str, object]],
    ) -> None:
        self.config = config
        self.observed_events = observed_events
        self._commands: deque[dict[str, object]] = deque()

    def feed_event(self, event: dict[str, object]) -> None:
        self.observed_events.append(event)
        if event["type"] == "token_sampled":
            self._commands.append({"type": "continue", "branch": event["branch"]})
        elif event["type"] == "branch_exhausted":
            self._commands.append({"type": "finalize", "branch": event["branch"]})

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class ContinueUntilCapacityScheduler:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] == "token_sampled":
            self._commands.append({"type": "continue", "branch": event["branch"]})

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class ConvergingScheduler:
    """Fork two greedy-identical children, then finalize the surviving branch."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()
        self._token_events = 0

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] != "token_sampled":
            return
        self._token_events += 1
        if self._token_events == 1:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": 0, "width": 2},
                    {"type": "continue", "branch": 1},
                    {"type": "continue", "branch": 2},
                ]
            )
        elif self._token_events == 3:
            self._commands.extend(
                [
                    {"type": "finalize", "branch": 1},
                    {"type": "finalize", "branch": 2},
                    {"type": "kill", "branch": 0, "reason": "fork_replaced"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class UnevenFinalizationScheduler:
    """Finalize one short branch and one longer branch."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()
        self._branch_two_tokens = 0

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] != "token_sampled":
            return
        branch_id = int(event["branch"])
        if branch_id == 0:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": 0, "width": 2},
                    {"type": "continue", "branch": 1},
                    {"type": "continue", "branch": 2},
                ]
            )
        elif branch_id == 2:
            self._branch_two_tokens += 1
        if branch_id == 2 and self._branch_two_tokens == 1:
            self._commands.extend(
                [
                    {"type": "finalize", "branch": 1},
                    {"type": "continue", "branch": 2},
                ]
            )
        elif branch_id == 2 and self._branch_two_tokens == 2:
            self._commands.append({"type": "continue", "branch": 2})
        elif branch_id == 2 and self._branch_two_tokens == 3:
            self._commands.extend(
                [
                    {"type": "finalize", "branch": 2},
                    {"type": "kill", "branch": 0, "reason": "fork_replaced"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class EosForkScheduler:
    """Fork non-terminal tokens, but finalize immediately when eos is explicit."""

    def __init__(
        self,
        config: dict[str, object],
        observed_events: list[dict[str, object]],
    ) -> None:
        self.config = config
        self.observed_events = observed_events
        self._commands: deque[dict[str, object]] = deque()

    def feed_event(self, event: dict[str, object]) -> None:
        self.observed_events.append(event)
        if event["type"] != "token_sampled":
            return
        if event.get("eos", False):
            self._commands.append({"type": "finalize", "branch": event["branch"]})
        else:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": event["branch"], "width": 2},
                    {"type": "finalize", "branch": 1},
                    {"type": "finalize", "branch": 2},
                    {"type": "kill", "branch": 0, "reason": "fork_replaced"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class FinalizeAllScheduler:
    """Fork the configured width, decode each child once, then finalize all."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()
        self._child_tokens = 0

    def feed_event(self, event: dict[str, object]) -> None:
        if event["type"] != "token_sampled":
            return
        branch_id = int(event["branch"])
        branches = int(self.config["branches"])
        if branch_id == 0:
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": 0, "width": branches},
                    *(
                        {"type": "continue", "branch": child}
                        for child in range(1, branches + 1)
                    ),
                ]
            )
            return
        self._child_tokens += 1
        if self._child_tokens == branches:
            self._commands.extend(
                [
                    *(
                        {"type": "finalize", "branch": child}
                        for child in range(1, branches + 1)
                    ),
                    {"type": "kill", "branch": 0, "reason": "tree_budget_exhausted"},
                ]
            )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands


class SharedBudgetScheduler:
    """Fork once, then spend the shared budget in complete live-branch rounds."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._commands: deque[dict[str, object]] = deque()
        self._states = {0: "active"}
        self._live: set[int] = set()
        self._round: set[int] = set()
        self._tokens = 0

    def feed_event(self, event: dict[str, object]) -> None:
        event_type = event["type"]
        branch_id = int(event["branch"])
        if event_type == "branch_exhausted":
            self._states[branch_id] = "finalized"
            self._live.discard(branch_id)
            self._round.discard(branch_id)
            self._commands = deque(
                command
                for command in self._commands
                if not (
                    int(command["branch"]) == branch_id
                    and command["type"] in {"continue", "finalize"}
                )
            )
            return
        if event_type != "token_sampled":
            return

        self._tokens += 1
        if branch_id == 0:
            branches = int(self.config["branches"])
            self._states[0] = "expanded"
            self._live = set(range(1, branches + 1))
            self._states.update({child: "active" for child in self._live})
            self._commands.extend(
                [
                    {"type": "fork_at", "branch": 0, "width": branches},
                    *(
                        {"type": "continue", "branch": child}
                        for child in sorted(self._live)
                    ),
                ]
            )
            return

        self._round.add(branch_id)
        if self._round != self._live:
            return
        self._round.clear()
        if self._tokens == int(self.config["budget_tokens"]):
            for child in sorted(self._live):
                self._states[child] = "finalized"
                self._commands.append({"type": "finalize", "branch": child})
            self._states[0] = "killed"
            self._commands.append(
                {"type": "kill", "branch": 0, "reason": "tree_budget_exhausted"}
            )
            return
        self._commands.extend(
            {"type": "continue", "branch": child}
            for child in sorted(self._live)
        )

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands

    def branch_state(self, branch: int) -> str | None:
        return self._states.get(branch)


class ConsensusPruningScheduler:
    """Model the Rust external-value handshake and speculative kill behavior."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._external = config["scorer"] == "external"
        self._commands: deque[dict[str, object]] = deque()
        self._states = {0: "active"}
        self._live: set[int] = set()
        self._pending: set[int] = set()
        self._round: set[int] = set()
        self._scores: dict[int, float] = {}
        self._tokens_per_branch: dict[int, int] = {}

    def feed_event(self, event: dict[str, object]) -> None:
        event_type = str(event["type"])
        branch_id = int(event["branch"])
        if event_type == "token_sampled":
            if self._external and not bool(event.get("eos", False)):
                self._pending.add(branch_id)
            if branch_id == 0:
                if not self._external:
                    self._fork_root()
                return
            self._tokens_per_branch[branch_id] += 1
            self._round.add(branch_id)
            if not self._external:
                self._finish_round()
            return
        if event_type == "value_scored":
            assert branch_id in self._pending
            self._pending.remove(branch_id)
            self._scores[branch_id] = float(event["score"])
            if branch_id == 0:
                self._fork_root()
            else:
                self._finish_round()
            return
        if event_type == "branch_exhausted":
            self._states[branch_id] = "finalized"
            self._live.discard(branch_id)
            self._pending.discard(branch_id)
            self._round.discard(branch_id)
            self._commands.append({"type": "finalize", "branch": branch_id})

    def poll_commands(self) -> list[dict[str, object]]:
        commands = list(self._commands)
        self._commands.clear()
        return commands

    def branch_state(self, branch: int) -> str | None:
        return self._states.get(branch)

    def _fork_root(self) -> None:
        branches = int(self.config["branches"])
        self._states[0] = "expanded"
        self._live = set(range(1, branches + 1))
        self._states.update({branch: "active" for branch in self._live})
        self._tokens_per_branch = {branch: 0 for branch in self._live}
        self._commands.extend(
            [
                {"type": "fork_at", "branch": 0, "width": branches},
                *(
                    {"type": "continue", "branch": branch}
                    for branch in sorted(self._live)
                ),
            ]
        )

    def _finish_round(self) -> None:
        if self._round != self._live or self._pending.intersection(self._live):
            return

        if self._external:
            best_score = max(self._scores[branch] for branch in self._live)
            victims = [
                branch
                for branch in sorted(self._live)
                if self._scores[branch] < best_score
            ]
            for branch in victims:
                self._live.remove(branch)
                self._states[branch] = "killed"
                self._commands.append(
                    {"type": "kill", "branch": branch, "reason": "speculative_kill"}
                )

        self._round.clear()
        if all(self._tokens_per_branch[branch] == 4 for branch in self._live):
            for branch in sorted(self._live):
                self._states[branch] = "finalized"
                self._commands.append({"type": "finalize", "branch": branch})
            self._states[0] = "killed"
            self._commands.append(
                {"type": "kill", "branch": 0, "reason": "tree_budget_exhausted"}
            )
            return

        self._commands.extend(
            {"type": "continue", "branch": branch}
            for branch in sorted(self._live)
        )


def request(*, budget_tokens: int = 3) -> GenerationRequest:
    return GenerationRequest(
        model="tiny-engine-model",
        messages=(Message(role="user", content="branch once"),),
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        stop=(),
        seed=41,
        user=None,
        tree=TreeExecution(
            policy="beam",
            branches=2,
            budget_tokens=budget_tokens,
            scorer=None,
        ),
    )


async def collect(engine: TreeKVEngine, generation_request: GenerationRequest):
    return [event async for event in engine.generate(generation_request)]


def emvpt_request(
    *,
    branches: int = 3,
    budget_tokens: int,
    check_interval: int = 4,
    warmup_tokens: int = 8,
    min_keep: int = 2,
    scorer: str | None = None,
) -> GenerationRequest:
    return replace(
        request(budget_tokens=budget_tokens),
        max_tokens=32,
        tree=TreeExecution(
            policy="emvpt",
            branches=branches,
            budget_tokens=budget_tokens,
            scorer=scorer,
            value_check_interval=check_interval,
            value_margin=0.35,
            value_min_keep=min_keep,
            value_warmup_tokens=warmup_tokens,
        ),
    )


def scripted_branch_samples(
    *,
    branch_logprobs: tuple[float, ...],
    rounds_before_prune: int,
    survivor_ids: tuple[int, ...] = (),
    survivor_rounds: int = 0,
) -> list[tuple[int, float]]:
    samples = [(10, -0.1)]
    samples.extend(
        (20 + branch_id, logprob)
        for _ in range(rounds_before_prune)
        for branch_id, logprob in enumerate(branch_logprobs, start=1)
    )
    samples.extend(
        (20 + branch_id, branch_logprobs[branch_id - 1])
        for _ in range(survivor_rounds)
        for branch_id in survivor_ids
    )
    return samples


@pytest.mark.parametrize(
    ("temperature", "top_p", "seed"),
    [
        pytest.param(0.5, 0.7, 11, id="nucleus-sampling"),
        pytest.param(2.0, 1.0, 29, id="non-unit-temperature"),
    ],
)
def test_sample_reports_unscaled_model_logprob(
    temperature: float,
    top_p: float,
    seed: int,
) -> None:
    logits = torch.tensor([3.0, 2.0, 1.0, -1.0])
    generation_request = replace(
        request(),
        temperature=temperature,
        top_p=top_p,
    )

    token_id, logprob = TreeKVEngine._sample(
        logits,
        generation_request,
        torch.Generator().manual_seed(seed),
    )

    expected = float(torch.log_softmax(logits.float(), dim=-1)[token_id].item())
    assert math.isfinite(logprob)
    assert logprob == pytest.approx(expected)


def test_null_seed_resolves_to_documented_zero_default() -> None:
    assert TreeKVEngine._resolve_seed(None) == 0
    assert TreeKVEngine._resolve_seed(17) == 17


def test_emvpt_uses_beam_scheduler_with_documented_defaults() -> None:
    tree = TreeExecution(
        policy="emvpt",
        branches=3,
        budget_tokens=64,
        scorer=None,
    )
    generation_request = replace(request(), tree=tree)

    assert tree.value_check_interval == 16
    assert tree.value_margin == pytest.approx(0.35)
    assert tree.value_min_keep == 2
    assert tree.value_warmup_tokens == 8
    assert TreeKVEngine._scheduler_config(generation_request)["policy"] == "beam"


def test_consensus_tree_params_use_documented_defaults() -> None:
    tree = TreeExecution(
        policy="beam",
        branches=3,
        budget_tokens=64,
        scorer="self_consistency",
    )

    assert tree.consensus_interval == 32
    assert tree.consensus_warmup == 64
    assert tree.min_survivors == 2


@pytest.mark.parametrize("scorer", [None, "logprob"])
def test_logprob_scheduler_config_is_unchanged(scorer: str | None) -> None:
    generation_request = replace(
        request(),
        tree=replace(request().tree, scorer=scorer),
    )

    assert TreeKVEngine._scheduler_config(generation_request) == {
        "policy": "beam",
        "branches": 2,
        "fork_width": 2,
        "fork_at_tokens": [1],
        "max_depth": 4,
        "budget_tokens": 3,
        "per_branch_token_budget": 4,
        "seed": 41,
        "scorer": "logprob",
    }


def test_fork_ids_events_and_kill_reclaim_real_tree_kv_pages(tiny_engine_case) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ScriptedScheduler,
    )

    events = asyncio.run(collect(engine, request()))

    starts = [event for event in events if isinstance(event, BranchStarted)]
    assert [(event.branch_id, event.parent_id) for event in starts] == [
        ("branch-0", None),
        ("branch-1", "branch-0"),
        ("branch-2", "branch-0"),
    ]
    assert any(
        isinstance(event, BranchPruned) and event.branch_id == "branch-2"
        for event in events
    )
    assert any(
        branch_id == 2 and after < before
        for branch_id, before, after in tiny_engine_case.executor.prune_accounting
    ), "a scheduler Kill must immediately release the killed leaf's KV reference"
    assert tiny_engine_case.executor.batch_decode_calls == [(1, 2)]
    done = next(event for event in events if isinstance(event, GenerationDone))
    token_events = [event for event in events if isinstance(event, TokenGenerated)]
    assert done.usage.completion_tokens == len(token_events) == 3
    assert all(event.token_id is not None for event in token_events)
    assert done.tree_summary is not None
    assert done.tree_summary.kv_reuse_ratio > 1.0
    assert set(done.tree_summary.final_scores) == {
        "branch-0",
        "branch-1",
        "branch-2",
    }


def test_convergent_children_batch_dedup_merge_and_measure_step_costs(
    tiny_engine_case,
) -> None:
    executor = type(tiny_engine_case.executor)(
        replace(tiny_engine_case.executor.config, page_size=2),
        model=tiny_engine_case.executor.model,
    )
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ConvergingScheduler,
        dedup_every_steps=2,
    )

    events = asyncio.run(collect(engine, request()))

    merges = [event for event in events if isinstance(event, BranchMerged)]
    assert [(event.branch_id, event.into_branch_id) for event in merges] == [
        ("branch-2", "branch-1")
    ]
    assert executor.batch_decode_calls == [(1, 2)]
    assert executor.dedup_calls == 1

    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.tree_summary is not None
    assert done.tree_summary.merged_count == 1
    assert done.tree_summary.pruned_count == 1
    assert done.tree_summary.kv_reuse_ratio == (
        done.counters.logical_tokens / done.counters.physical_tokens
    )
    assert done.counters.unique_tokens_per_step == (1, 1)
    assert done.counters.branch_tokens_per_step == (1, 2)
    assert asdict(done.counters)["unique_tokens_per_step"] == (1, 1)
    assert asdict(done.counters)["branch_tokens_per_step"] == (1, 2)


def test_same_seed_produces_identical_winning_completion(tiny_engine_case) -> None:
    def build() -> TreeKVEngine:
        return TreeKVEngine(
            model_id="tiny-engine-model",
            executor=tiny_engine_case.executor,
            tokenizer=tiny_engine_case.tokenizer,
            scheduler_factory=ScriptedScheduler,
        )

    first = asyncio.run(collect(build(), request()))
    second = asyncio.run(collect(build(), request()))

    first_done = next(event for event in first if isinstance(event, GenerationDone))
    second_done = next(event for event in second if isinstance(event, GenerationDone))
    assert first_done.text == second_done.text
    assert first_done.tree_summary == second_done.tree_summary


def test_winner_uses_scheduler_mean_score_across_fork(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=UnevenFinalizationScheduler,
    )
    samples = iter(
        [
            (10, -10.0),
            (11, -0.1),
            (12, -1.0),
            (13, -1.0),
            (14, -1.0),
        ]
    )
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )
    generation_request = replace(
        request(budget_tokens=5),
        tree=TreeExecution(
            policy="beam",
            branches=2,
            budget_tokens=5,
            scorer="logprob",
        ),
    )

    events = asyncio.run(collect(engine, generation_request))

    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.branch_id == "branch-2"
    assert done.tree_summary is not None
    assert done.tree_summary.winner_branch_id == "branch-2"
    assert done.tree_summary.final_scores["branch-2"] == pytest.approx(-3.25)
    assert done.tree_summary.final_scores["branch-1"] == pytest.approx(-5.05)


def test_logprob_event_output_is_unchanged(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ScriptedScheduler,
    )
    samples = iter([(10, -0.5), (11, -0.2), (12, -0.7)])
    decoded = {10: "root-", 11: "winner", 12: "loser"}
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )
    monkeypatch.setattr(
        tiny_engine_case.tokenizer,
        "decode",
        lambda token_ids, **_kwargs: decoded[token_ids[0]],
    )
    generation_request = replace(
        request(),
        tree=replace(request().tree, scorer="logprob"),
    )

    events = asyncio.run(collect(engine, generation_request))

    assert [event.type for event in events] == [
        "branch_started",
        "token",
        "branch_started",
        "branch_started",
        "token",
        "token",
        "branch_pruned",
        "branch_pruned",
        "done",
    ]
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.branch_id == "branch-1"
    assert done.text == "root-winner"
    assert done.usage.completion_tokens == 3
    assert done.tree_summary is not None
    assert done.tree_summary.tokens_spent_per_branch == {
        "branch-0": 1,
        "branch-1": 1,
        "branch-2": 1,
    }
    assert done.tree_summary.final_scores == pytest.approx(
        {"branch-0": -0.5, "branch-1": -0.35, "branch-2": -0.6}
    )


def test_real_scheduler_never_exceeds_requested_tree_budget(tiny_engine_case) -> None:
    pytest.importorskip("autotree_scheduler")
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
    )

    events = asyncio.run(collect(engine, request(budget_tokens=3)))
    done = next(event for event in events if isinstance(event, GenerationDone))

    assert done.usage.completion_tokens == 3
    assert sum(done.tree_summary.tokens_spent_per_branch.values()) == 3


@pytest.mark.parametrize("policy", ["beam", "best_first", "mcts"])
def test_real_scheduler_keeps_engine_lifecycle_aligned_during_dedup(
    tiny_engine_case,
    policy: str,
) -> None:
    pytest.importorskip("autotree_scheduler")
    executor = type(tiny_engine_case.executor)(
        replace(tiny_engine_case.executor.config, page_size=2),
        model=tiny_engine_case.executor.model,
    )
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=executor,
        tokenizer=tiny_engine_case.tokenizer,
    )

    generation_request = replace(
        request(budget_tokens=6),
        tree=TreeExecution(
            policy=policy,
            branches=2,
            budget_tokens=6,
            scorer=None,
        ),
    )
    events = asyncio.run(collect(engine, generation_request))

    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.usage.completion_tokens <= 6
    assert done.tree_summary is not None
    assert done.tree_summary.branch_count > 1
    assert done.tree_summary.kv_reuse_ratio > 1.0
    if policy == "beam":
        assert done.tree_summary.merged_count >= 1


def test_eos_feeds_branch_exhausted_and_finishes_with_stop(
    tiny_engine_case,
) -> None:
    observed_events: list[dict[str, object]] = []
    expected_id = int(
        tiny_engine_case.executor.prefill([5, 6, 7, 8]).next_logits(0).argmax().item()
    )
    tiny_engine_case.tokenizer.eos_token_id = expected_id
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=lambda config: ExhaustionScheduler(config, observed_events),
    )

    events = asyncio.run(collect(engine, replace(request(), tree=None)))

    assert [event["type"] for event in observed_events] == [
        "token_sampled",
        "branch_exhausted",
    ]
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.finish_reason == "stop"


def test_eos_token_is_not_forked_after_sampling(tiny_engine_case) -> None:
    observed_events: list[dict[str, object]] = []
    expected_id = int(
        tiny_engine_case.executor.prefill([5, 6, 7, 8]).next_logits(0).argmax().item()
    )
    tiny_engine_case.tokenizer.eos_token_id = expected_id
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=lambda config: EosForkScheduler(config, observed_events),
    )

    events = asyncio.run(collect(engine, request()))

    starts = [event for event in events if isinstance(event, BranchStarted)]
    assert [(event.branch_id, event.parent_id) for event in starts] == [
        ("branch-0", None)
    ]
    sampled = next(
        event for event in observed_events if event["type"] == "token_sampled"
    )
    assert sampled["eos"] is True
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.branch_id == "branch-0"
    assert done.finish_reason == "stop"


def test_self_consistency_uses_external_scheduler_during_execution(
    tiny_engine_case,
) -> None:
    observed_events: list[dict[str, object]] = []
    observed_configs: list[dict[str, object]] = []
    expected_id = int(
        tiny_engine_case.executor.prefill([5, 6, 7, 8]).next_logits(0).argmax().item()
    )
    tiny_engine_case.tokenizer.eos_token_id = expected_id

    def scheduler_factory(config: dict[str, object]) -> EosForkScheduler:
        observed_configs.append(config)
        return EosForkScheduler(config, observed_events)

    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=scheduler_factory,
    )
    generation_request = replace(
        request(),
        tree=TreeExecution(
            policy="beam",
            branches=2,
            budget_tokens=3,
            scorer="self_consistency",
        ),
    )

    events = asyncio.run(collect(engine, generation_request))

    assert observed_configs[0]["scorer"] == "external"
    assert observed_configs[0]["speculative_kill_margin"] == pytest.approx(0.0)
    assert [event["type"] for event in observed_events] == ["token_sampled"]
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.finish_reason == "stop"


def _run_execution_consensus_case(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scorer: str,
    outlier_answer: str = "9 ",
) -> tuple[list[object], GenerationDone]:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ConsensusPruningScheduler,
        dedup_every_steps=None,
    )
    samples_by_round = [
        [(10, -0.1)],
        [(21, -0.05), (22, -0.1), (23, -0.2)],
        [(31, -0.05), (32, -0.1), (33, -0.2)],
        [(41, -0.05), (42, -0.1), (43, -0.2)],
        [(51, -0.05), (52, -0.1), (53, -0.2)],
    ]
    if scorer == "self_consistency" and outlier_answer != "7 ":
        samples_by_round[3] = samples_by_round[3][:2]
        samples_by_round[4] = samples_by_round[4][:2]
    samples = iter(sample for round_samples in samples_by_round for sample in round_samples)
    decoded = {
        10: "Reasoning. ",
        21: "Answer: ",
        22: "Answer: ",
        23: "Answer: ",
        31: "7 ",
        32: "7 ",
        33: outlier_answer,
        41: "work ",
        42: "work ",
        43: "work ",
        51: "done",
        52: "done",
        53: "done",
    }
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )
    monkeypatch.setattr(
        tiny_engine_case.tokenizer,
        "decode",
        lambda token_ids, **_kwargs: decoded[token_ids[0]],
    )
    tiny_engine_case.tokenizer.eos_token_id = None
    generation_request = replace(
        request(budget_tokens=13),
        max_tokens=5,
        tree=TreeExecution(
            policy="beam",
            branches=3,
            budget_tokens=13,
            scorer=scorer,
            consensus_interval=1,
            consensus_warmup=3,
            min_survivors=2,
        ),
    )

    events = asyncio.run(collect(engine, generation_request))
    done = next(event for event in events if isinstance(event, GenerationDone))
    return events, done


def test_self_consistency_kills_diverging_branch_before_its_budget_is_spent(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, done = _run_execution_consensus_case(
        tiny_engine_case,
        monkeypatch,
        scorer="self_consistency",
    )

    assert any(
        isinstance(event, BranchPruned)
        and event.branch_id == "branch-3"
        and event.reason == "speculative_kill"
        for event in events
    )
    assert done.tree_summary is not None
    assert done.tree_summary.tokens_spent_per_branch == {
        "branch-0": 1,
        "branch-1": 4,
        "branch-2": 4,
        "branch-3": 2,
    }


def test_self_consistency_spends_fewer_tokens_than_logprob_at_equal_winner(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, logprob_done = _run_execution_consensus_case(
        tiny_engine_case,
        monkeypatch,
        scorer="logprob",
    )
    _, consensus_done = _run_execution_consensus_case(
        tiny_engine_case,
        monkeypatch,
        scorer="self_consistency",
    )

    assert consensus_done.branch_id == logprob_done.branch_id == "branch-1"
    assert consensus_done.text == logprob_done.text == "Reasoning. Answer: 7 work done"
    assert consensus_done.usage.completion_tokens == 11
    assert logprob_done.usage.completion_tokens == 13
    assert consensus_done.usage.completion_tokens < logprob_done.usage.completion_tokens


def test_self_consistency_all_agree_respects_min_survivors(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, done = _run_execution_consensus_case(
        tiny_engine_case,
        monkeypatch,
        scorer="self_consistency",
        outlier_answer="7 ",
    )

    assert not any(
        isinstance(event, BranchPruned) and event.reason == "speculative_kill"
        for event in events
    )
    assert done.tree_summary is not None
    assert done.tree_summary.tokens_spent_per_branch == {
        "branch-0": 1,
        "branch-1": 4,
        "branch-2": 4,
        "branch-3": 4,
    }


def _run_self_consistency_case(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
    *,
    answers: list[str],
    logprobs: list[float],
) -> GenerationDone:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=FinalizeAllScheduler,
    )
    token_ids = iter(range(20, 21 + len(answers)))
    samples = iter(zip(token_ids, logprobs, strict=True))
    decoded = {20: "Reasoning. "}
    decoded.update(
        {token_id: text for token_id, text in zip(range(21, 21 + len(answers)), answers)}
    )
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )
    monkeypatch.setattr(
        tiny_engine_case.tokenizer,
        "decode",
        lambda token_ids, **_kwargs: decoded[token_ids[0]],
    )
    generation_request = replace(
        request(budget_tokens=1 + len(answers)),
        tree=TreeExecution(
            policy="beam",
            branches=len(answers),
            budget_tokens=1 + len(answers),
            scorer="self_consistency",
        ),
    )

    events = asyncio.run(collect(engine, generation_request))
    return next(event for event in events if isinstance(event, GenerationDone))


def test_self_consistency_majority_beats_logprob_favorite(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _run_self_consistency_case(
        tiny_engine_case,
        monkeypatch,
        answers=["Answer: 7.", "Answer: 7.", "Answer: 9."],
        logprobs=[-0.1, -2.0, -1.0, -0.01],
    )

    assert done.branch_id == "branch-2"
    assert done.text.endswith("Answer: 7.")
    assert done.tree_summary is not None
    assert done.tree_summary.winner_branch_id == "branch-2"
    assert done.tree_summary.scorer == "self_consistency"


def test_self_consistency_group_tie_breaks_by_summed_logprob(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _run_self_consistency_case(
        tiny_engine_case,
        monkeypatch,
        answers=["Answer: 7.", "Answer: 7.", "Answer: 9.", "Answer: 9."],
        logprobs=[-0.1, -1.0, -2.0, -0.2, -0.3],
    )

    assert done.branch_id == "branch-3"
    assert done.text.endswith("Answer: 9.")


def test_self_consistency_all_singletons_fall_back_to_logprob(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _run_self_consistency_case(
        tiny_engine_case,
        monkeypatch,
        answers=["Answer: 7.", "Answer: 8.", "Answer: 9."],
        logprobs=[-0.1, -1.0, -0.2, -0.5],
    )

    assert done.branch_id == "branch-2"


def test_self_consistency_selects_winner_at_exact_tree_budget(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _run_self_consistency_case(
        tiny_engine_case,
        monkeypatch,
        answers=["Answer: 5.", "Answer: 5.", "Answer: 6."],
        logprobs=[-0.1, -0.8, -0.7, -0.01],
    )

    assert done.usage.completion_tokens == 4
    assert done.branch_id == "branch-2"


def test_emvpt_prunes_weak_branch_at_first_eligible_check(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=SharedBudgetScheduler,
    )
    samples = iter(
        scripted_branch_samples(
            branch_logprobs=(-0.1, -0.2, -1.0),
            rounds_before_prune=8,
            survivor_ids=(1, 2),
            survivor_rounds=2,
        )
    )
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )

    events = asyncio.run(collect(engine, emvpt_request(budget_tokens=29)))

    value_prunes = [
        event
        for event in events
        if isinstance(event, BranchPruned) and event.reason == "emvpt_below_margin"
    ]
    assert [event.branch_id for event in value_prunes] == ["branch-3"]
    assert sum(
        isinstance(event, TokenGenerated) and event.branch_id == "branch-3"
        for event in events
    ) == 8
    assert any(
        branch_id == 3 and after < before
        for branch_id, before, after in tiny_engine_case.executor.prune_accounting
    )
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.tree_summary is not None
    assert done.tree_summary.value_estimates["branch-3"] == pytest.approx(-1.0)
    assert done.tree_summary.pruned_at_tokens["branch-3"] == 8


def test_emvpt_respects_min_keep(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=SharedBudgetScheduler,
    )
    samples = iter(
        scripted_branch_samples(
            branch_logprobs=(-0.1, -1.0, -2.0),
            rounds_before_prune=8,
            survivor_ids=(1, 2),
            survivor_rounds=2,
        )
    )
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )

    events = asyncio.run(collect(engine, emvpt_request(budget_tokens=29)))

    value_prunes = [
        event.branch_id
        for event in events
        if isinstance(event, BranchPruned) and event.reason == "emvpt_below_margin"
    ]
    assert value_prunes == ["branch-3"]
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.tree_summary is not None
    assert done.tree_summary.pruned_at_tokens == {
        "branch-0": None,
        "branch-1": None,
        "branch-2": None,
        "branch-3": 8,
    }


def test_emvpt_respects_warmup(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=SharedBudgetScheduler,
    )
    samples = iter(
        scripted_branch_samples(
            branch_logprobs=(-0.1, -0.2, -2.0),
            rounds_before_prune=7,
        )
    )
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )

    events = asyncio.run(
        collect(
            engine,
            emvpt_request(
                budget_tokens=22,
                check_interval=1,
                warmup_tokens=8,
            ),
        )
    )

    assert not any(
        isinstance(event, BranchPruned) and event.reason == "emvpt_below_margin"
        for event in events
    )
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.tree_summary is not None
    assert all(value is None for value in done.tree_summary.pruned_at_tokens.values())


def test_emvpt_reuses_pruned_branch_budget_for_survivors(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=SharedBudgetScheduler,
    )
    samples = iter(
        scripted_branch_samples(
            branch_logprobs=(-0.1, -0.2, -1.0),
            rounds_before_prune=8,
            survivor_ids=(1, 2),
            survivor_rounds=2,
        )
    )
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )

    events = asyncio.run(collect(engine, emvpt_request(budget_tokens=29)))

    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.usage.completion_tokens == 29
    assert done.tree_summary is not None
    assert done.tree_summary.tokens_spent_per_branch == {
        "branch-0": 1,
        "branch-1": 10,
        "branch-2": 10,
        "branch-3": 8,
    }


def test_default_policy_does_not_enable_value_pruning(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=SharedBudgetScheduler,
    )
    samples = iter(
        scripted_branch_samples(
            branch_logprobs=(-0.1, -0.2, -2.0),
            rounds_before_prune=2,
        )
    )
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )
    generation_request = replace(
        request(budget_tokens=7),
        max_tokens=32,
        tree=TreeExecution(
            policy="beam",
            branches=3,
            budget_tokens=7,
            scorer=None,
        ),
    )

    events = asyncio.run(collect(engine, generation_request))

    assert not any(
        isinstance(event, BranchPruned) and event.reason == "emvpt_below_margin"
        for event in events
    )
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.tree_summary is not None
    assert done.tree_summary.value_estimates is None
    assert done.tree_summary.pruned_at_tokens is None
    assert "value_estimates" not in done.tree_summary.to_dict()
    assert "pruned_at_tokens" not in done.tree_summary.to_dict()


def test_emvpt_self_consistency_votes_only_among_survivors(
    tiny_engine_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=SharedBudgetScheduler,
    )
    samples = iter(
        [
            (10, -0.1),
            (21, -0.1),
            (22, -0.2),
            (23, -0.05),
            (24, -2.0),
            (31, -0.1),
            (32, -0.2),
            (33, -0.05),
            (34, -2.0),
            (41, -0.1),
            (42, -0.2),
            (43, -0.05),
            (51, -0.1),
            (52, -0.2),
            (53, -0.05),
        ]
    )
    decoded = {
        10: "Reasoning. ",
        21: "work ",
        22: "work ",
        23: "work ",
        24: "work ",
        31: "work ",
        32: "work ",
        33: "work ",
        34: "work ",
        41: "Answer: 7. ",
        42: "Answer: 7. ",
        43: "Answer: 9. ",
        51: "done",
        52: "done",
        53: "done",
    }
    monkeypatch.setattr(
        engine,
        "_sample",
        lambda _logits, _request, _generator: next(samples),
    )
    monkeypatch.setattr(
        tiny_engine_case.tokenizer,
        "decode",
        lambda token_ids, **_kwargs: decoded[token_ids[0]],
    )

    events = asyncio.run(
        collect(
            engine,
            emvpt_request(
                branches=4,
                budget_tokens=15,
                check_interval=2,
                warmup_tokens=2,
                scorer="self_consistency",
            ),
        )
    )

    assert any(
        isinstance(event, BranchPruned)
        and event.branch_id == "branch-4"
        and event.reason == "emvpt_below_margin"
        for event in events
    )
    done = next(event for event in events if isinstance(event, GenerationDone))
    assert done.branch_id == "branch-1"
    assert done.text.endswith("Answer: 7. done")
    assert done.tree_summary is not None
    assert done.tree_summary.scorer == "self_consistency"


def test_engine_rejects_unknown_scorer(tiny_engine_case) -> None:
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=tiny_engine_case.executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ScriptedScheduler,
    )
    generation_request = replace(
        request(),
        tree=replace(request().tree, scorer="external"),
    )

    with pytest.raises(ValueError, match="self_consistency"):
        asyncio.run(collect(engine, generation_request))


def test_mid_decode_capacity_exhaustion_is_promoted_to_engine_error(
    tiny_engine_case,
) -> None:
    executor = type(tiny_engine_case.executor)(
        replace(tiny_engine_case.executor.config, capacity_pages=2),
        model=tiny_engine_case.executor.model,
    )
    engine = TreeKVEngine(
        model_id="tiny-engine-model",
        executor=executor,
        tokenizer=tiny_engine_case.tokenizer,
        scheduler_factory=ContinueUntilCapacityScheduler,
    )
    generation_request = replace(request(), max_tokens=8, tree=None)

    with pytest.raises(KVCapacityExceededError) as raised:
        asyncio.run(collect(engine, generation_request))

    assert raised.value.phase == "decode"
    assert raised.value.required_pages == 1
    assert raised.value.available_pages == 0
    assert raised.value.capacity_pages == 2


def test_engine_device_and_dtype_reach_executor_config(monkeypatch) -> None:
    import autotree_core.engine.treekv as treekv_module

    captured: dict[str, object] = {}

    class CapturingExecutor:
        def __init__(self, config):
            captured["config"] = config
            self.config = config

    monkeypatch.setattr(treekv_module, "ModelExecutor", CapturingExecutor)
    monkeypatch.setattr(
        treekv_module.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, model_id: type("Cfg", (), {"n_positions": 64})()),
    )
    monkeypatch.setattr(
        treekv_module.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, model_id: object()),
    )

    TreeKVEngine(
        model_id="captured-model",
        scheduler_factory=ScriptedScheduler,
        device="cpu",
        dtype="bfloat16",
    )

    config = captured["config"]
    assert config.device == torch.device("cpu")
    assert config.dtype is torch.bfloat16


def test_engine_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        TreeKVEngine(model_id="any-model", dtype="int8")
