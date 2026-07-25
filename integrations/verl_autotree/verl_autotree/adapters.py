"""Lazy verl adapters for the AutoTree-enabled SGLang engine."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping

__all__ = ["AutoTreeServerAdapter", "AutoTreeHttpServer", "AutoTreeReplica"]

_TREE_KEYS = ("policy", "branches", "budget_tokens", "scorer")
_TREE_POLICIES = {"beam", "best_first", "mcts"}


@dataclass(frozen=True)
class _TreeBlock:
    policy: str
    branches: int
    budget_tokens: int
    scorer: str | None


def _extract_tree_block(sampling_params: Mapping[str, Any]) -> _TreeBlock | None:
    """Extract and validate the optional AutoTree sampling block."""
    raw = sampling_params.get("tree")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("sampling_params.tree must be a mapping")

    missing = [key for key in _TREE_KEYS[:3] if key not in raw]
    if missing:
        raise ValueError(f"sampling_params.tree is missing required fields: {', '.join(missing)}")
    unknown = sorted(set(raw) - set(_TREE_KEYS))
    if unknown:
        raise ValueError(f"sampling_params.tree has unknown fields: {', '.join(unknown)}")

    policy = raw["policy"]
    branches = raw["branches"]
    budget_tokens = raw["budget_tokens"]
    scorer = raw.get("scorer")
    if policy not in _TREE_POLICIES:
        raise ValueError(f"sampling_params.tree.policy must be one of {sorted(_TREE_POLICIES)}")
    if isinstance(branches, bool) or not isinstance(branches, int) or branches < 1:
        raise ValueError("sampling_params.tree.branches must be a positive integer")
    if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) or budget_tokens < 1:
        raise ValueError("sampling_params.tree.budget_tokens must be a positive integer")
    if scorer is not None and not isinstance(scorer, str):
        raise ValueError("sampling_params.tree.scorer must be a string or null")
    return _TreeBlock(policy, branches, budget_tokens, scorer)


def _build_server_adapter() -> type:
    server_adapter = import_module(
        "verl.workers.rollout.sglang_rollout.sglang_rollout"
    ).ServerAdapter

    class AutoTreeServerAdapter(server_adapter):
        """SGLang's unchanged CUDA-IPC weight-sync adapter."""

    AutoTreeServerAdapter.__module__ = __name__
    AutoTreeServerAdapter.__qualname__ = "AutoTreeServerAdapter"
    return AutoTreeServerAdapter


def _build_http_server() -> type:
    server_module = import_module(
        "verl.workers.rollout.sglang_rollout.async_sglang_server"
    )
    sglang_http_server = server_module.SGLangHttpServer
    token_output = import_module("verl.workers.rollout.replica").TokenOutput

    class AutoTreeHttpServer(sglang_http_server):
        """SGLang HTTP server with AutoTree request dispatch."""

        async def generate(
            self,
            prompt_ids: Any,
            sampling_params: dict[str, Any],
            request_id: str,
            image_data: list[Any] | None = None,
            video_data: list[Any] | None = None,
            bootstrap_host: str | None = None,
            bootstrap_port: int | None = None,
            bootstrap_room: int | None = None,
        ) -> Any:
            tree = _extract_tree_block(sampling_params)
            if tree is None:
                return await super().generate(
                    prompt_ids,
                    sampling_params,
                    request_id,
                    image_data=image_data,
                    video_data=video_data,
                    bootstrap_host=bootstrap_host,
                    bootstrap_port=bootstrap_port,
                    bootstrap_room=bootstrap_room,
                )

            try:
                generate_req_input = import_module(
                    "sglang.srt.managers.io_struct"
                ).GenerateReqInput
                tree_module = import_module("sglang.srt.tree")
                tree_generate_req_input = tree_module.TreeGenerateReqInput
                tree_params = tree_module.TreeParams
                tree_result = tree_module.TreeResult
            except (AttributeError, ImportError) as error:
                raise NotImplementedError(
                    "AutoTree generation requires an SGLang engine build with "
                    "sglang.srt.tree (the /v1/tree/completions engine feature)"
                ) from error

            params = dict(sampling_params)
            params.pop("tree")
            return_logprob = bool(params.pop("logprobs", False))
            if "max_tokens" in params and "max_new_tokens" not in params:
                params["max_new_tokens"] = params.pop("max_tokens")

            base_request = generate_req_input(
                rid=request_id,
                input_ids=prompt_ids,
                sampling_params=params,
                return_logprob=return_logprob,
                image_data=image_data,
            )
            if bootstrap_room is not None:
                base_request.bootstrap_host = bootstrap_host
                base_request.bootstrap_port = bootstrap_port
                base_request.bootstrap_room = bootstrap_room
            if getattr(self.model_config, "lora_rank", 0) > 0:
                utils = import_module("verl.workers.rollout.sglang_rollout.utils")
                base_request.lora_path = utils.SGLANG_LORA_NAME

            tree_request = tree_generate_req_input(
                base=base_request,
                tree=tree_params(
                    policy=tree.policy,
                    branches=tree.branches,
                    budget_tokens=tree.budget_tokens,
                    scorer=tree.scorer,
                ),
            )
            tree_request.tree.validate()
            result = await self.tokenizer_manager.generate_request(tree_request, None).__anext__()
            token_ids, log_probs, finish_reason, summary = _coerce_plain_result(
                result, tree_result
            )
            return token_output(
                token_ids=token_ids,
                log_probs=log_probs,
                stop_reason="length" if finish_reason == "length" else "completed",
                extra_fields={"tree_summary": summary},
            )

    AutoTreeHttpServer.__module__ = __name__
    AutoTreeHttpServer.__qualname__ = "AutoTreeHttpServer"
    return AutoTreeHttpServer



def _coerce_plain_result(result, tree_result_cls=None):
    """Normalize whatever the engine yielded into (token_ids, logprobs, reason, summary).

    The tokenizer manager yields PLAIN DICTS for tree requests, the same shape
    the serving layer consumes at serving_tree.py:309 via result.get("meta_info").
    The previous code asserted isinstance(result, TreeResult) and raised on the
    first real call, so this path had never run against a live engine.

    It also read result.winner_log_probs, which does not exist on TreeResult
    (its fields are winner_text, winner_token_ids, prompt_tokens,
    completion_tokens, summary, finish_reason, counters, branch_events). getattr
    with a default meant log_probs was silently always None, so a trainer got no
    logprobs at all and could not compute a policy gradient. Real per-token
    logprobs live in meta_info["output_token_logprobs"], as sglang triples of
    (logprob, token_id, text).

    A TreeResult object is still accepted so a future engine that yields one
    keeps working.
    """
    if isinstance(result, dict):
        meta = result.get("meta_info") or {}
        raw = meta.get("output_token_logprobs")
        log_probs = None
        token_ids = None
        if raw:
            log_probs = [float(entry[0]) for entry in raw]
            token_ids = [int(entry[1]) for entry in raw]
        if token_ids is None:
            token_ids = list(result.get("output_ids") or meta.get("output_ids") or [])
        finish = meta.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type")
        summary = result.get("tree") or {}
        return token_ids, log_probs, finish, summary

    if tree_result_cls is not None and not isinstance(result, tree_result_cls):
        raise RuntimeError(
            "AutoTree engine returned an unsupported envelope: "
            f"{type(result).__name__}"
        )
    summary = result.summary.to_dict() if hasattr(result.summary, "to_dict") else result.summary
    return list(result.winner_token_ids), None, result.finish_reason, summary

def _build_replica() -> type:
    server_module = import_module(
        "verl.workers.rollout.sglang_rollout.async_sglang_server"
    )
    sglang_replica = server_module.SGLangReplica
    ray = import_module("ray")
    http_server = __getattr__("AutoTreeHttpServer")
    remote_http_server = ray.remote(http_server)

    class AutoTreeReplica(sglang_replica):
        """SGLang replica whose actors use :class:`AutoTreeHttpServer`."""

        server_class = remote_http_server

    AutoTreeReplica.__module__ = __name__
    AutoTreeReplica.__qualname__ = "AutoTreeReplica"
    return AutoTreeReplica


_BUILDERS = {
    "AutoTreeServerAdapter": _build_server_adapter,
    "AutoTreeHttpServer": _build_http_server,
    "AutoTreeReplica": _build_replica,
}


def __getattr__(name: str) -> Any:
    builder = _BUILDERS.get(name)
    if builder is None:
        raise AttributeError(name)
    value = builder()
    globals()[name] = value
    return value
