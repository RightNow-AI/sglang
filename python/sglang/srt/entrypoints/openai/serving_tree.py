"""Serving implementation for the AutoTree-compatible tree completions API."""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.protocol_tree import (
    TreeCompletionChoice,
    TreeCompletionRequest,
    TreeCompletionResponse,
    TreeBranchMergedEvent,
    TreeBranchPrunedEvent,
    TreeBranchStartedEvent,
    TreeCountersResponse,
    TreeDoneEvent,
    TreeResponseMessage,
    TreeSummaryResponse,
    TreeTokenEvent,
    TreeUsage,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.tree import (
    TreeBranchEvent,
    TreeCounters,
    TreeGenerateReqInput,
    TreeParams,
    TreeResult,
    TreeSummary,
)
from sglang.srt.tree.memo import MemoStore, canonical_key
from sglang.srt.tree.params import (
    DEFAULT_CONSENSUS_INTERVAL,
    DEFAULT_CONSENSUS_WARMUP,
    DEFAULT_MIN_SURVIVORS,
)
from sglang.srt.tree.selection import env_int

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


class OpenAIServingTree(OpenAIServingBase):
    """Handler for ``POST /v1/tree/completions``."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        chat_serving: OpenAIServingChat,
    ) -> None:
        super().__init__(tokenizer_manager)
        self.chat_serving = chat_serving
        self._memo_store = (
            self._create_memo_store() if self._memo_feature_enabled() else None
        )

    def _request_id_prefix(self) -> str:
        return "tree-"

    def _validate_request(self, request: TreeCompletionRequest) -> Optional[str]:
        if not request.messages:
            return "Messages cannot be empty."
        return None

    @staticmethod
    def _memo_feature_enabled() -> bool:
        return os.environ.get("AUTOTREE_MEMO") == "1"

    @staticmethod
    def _create_memo_store() -> MemoStore:
        path = os.environ.get("AUTOTREE_MEMO_PATH") or os.path.join(
            tempfile.gettempdir(), f"sglang-autotree-memo-{os.getpid()}.jsonl"
        )
        max_entries = env_int("AUTOTREE_MEMO_MAX_ENTRIES", 10000)
        return MemoStore(
            path,
            max_entries=max_entries,
            namespace="tree-serving",
        )

    def _active_memo_store(self) -> Optional[MemoStore]:
        if not self._memo_feature_enabled():
            return None
        return getattr(self, "_memo_store", None)

    @staticmethod
    def _rendered_prompt_for_memo(adapted_request: TreeGenerateReqInput) -> Any:
        base_request = adapted_request.base
        text = getattr(base_request, "text", None)
        if text is not None:
            return text
        input_ids = getattr(base_request, "input_ids", None)
        if input_ids is not None:
            return input_ids
        raise ValueError("Tree memo requires a rendered text or token-id prompt.")

    def _memo_key(
        self,
        adapted_request: TreeGenerateReqInput,
        request: TreeCompletionRequest,
    ) -> str:
        sampling_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.resolved_max_tokens,
        }
        if request.seed is not None:
            sampling_params.update(
                {
                    "seed": request.resolved_seed,
                    "seeded_determinism_requested": True,
                }
            )
        return canonical_key(
            self._rendered_prompt_for_memo(adapted_request),
            request.model,
            sampling_params,
            request.context_version,
        )

    def memo_stats(self) -> dict[str, Any]:
        store = getattr(self, "_memo_store", None)
        if store is None:
            return {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "tokens_saved": 0,
                "agreement_rate": None,
            }
        stats = store.stats()
        return {
            "hits": stats["hits"],
            "misses": stats["misses"],
            "hit_rate": stats["hit_rate"],
            "tokens_saved": stats["tokens_saved_estimate"],
            "agreement_rate": stats["agreement_rate"],
        }

    def clear_memo(self) -> dict[str, Any]:
        store = getattr(self, "_memo_store", None)
        if store is not None:
            store.clear()
        return self.memo_stats()

    def _convert_to_internal_request(
        self,
        request: TreeCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[TreeGenerateReqInput, TreeCompletionRequest]:
        """Render the chat prompt and wrap it with fixed tree parameters."""
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[message.model_dump() for message in request.messages],
            max_tokens=request.resolved_max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            seed=request.resolved_seed,
            stream=request.stream,
            stream_options=(
                request.stream_options.model_dump()
                if request.stream_options is not None
                else None
            ),
            n=1,
            user=request.user,
        )
        base_request, _ = self.chat_serving._convert_to_internal_request(
            chat_request, raw_request
        )
        base_request.return_logprob = request.tree.branches > 1 or request.stream
        # The scheduler-side tree runtime consumes only numeric logprob values.
        # Avoid detokenizing every returned logprob token in TokenizerManager.
        base_request.return_text_in_logprobs = False

        verifier = getattr(request.tree, "verifier", None)
        if hasattr(verifier, "model_dump"):
            verifier = verifier.model_dump()

        tree_request = TreeGenerateReqInput(
            base=base_request,
            tree=TreeParams(
                policy=request.tree.policy,
                branches=request.tree.branches,
                budget_tokens=request.tree.budget_tokens,
                scorer=request.tree.scorer,
                fork_at_text=request.tree.fork_at_text,
                fork_at_entropy=request.tree.fork_at_entropy,
                adaptive_width=request.tree.adaptive_width,
                # Read defensively: a payload parsed by an older schema will
                # not carry the consensus knobs, and a missing knob must fall
                # back to its documented default rather than raise.
                consensus_warmup=getattr(
                    request.tree, "consensus_warmup", DEFAULT_CONSENSUS_WARMUP
                ),
                consensus_interval=getattr(
                    request.tree, "consensus_interval", DEFAULT_CONSENSUS_INTERVAL
                ),
                min_survivors=getattr(
                    request.tree, "min_survivors", DEFAULT_MIN_SURVIVORS
                ),
                verifier=verifier,
            ),
        )
        tree_request.tree.validate()
        return tree_request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: TreeGenerateReqInput,
        request: TreeCompletionRequest,
        raw_request: Request,
    ) -> Union[TreeCompletionResponse, ORJSONResponse]:
        memo_store = self._active_memo_store()
        if adapted_request.tree.verifier is not None:
            memo_store = None
        memo_key = None
        if memo_store is not None:
            memo_key = self._memo_key(adapted_request, request)
            memo_entry = memo_store.get(
                memo_key,
                model=request.model,
                context_version=request.context_version,
            )
            if memo_entry is not None:
                return self._build_memo_response(request, memo_key, memo_entry)

        try:
            generator = self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            )
            result = await generator.__anext__()
        except ValueError as error:
            return self.create_error_response(str(error))

        if not isinstance(result, TreeResult):
            result = self._coerce_plain_result(result, adapted_request)
        if result is None:
            return self.create_error_response(
                "Tree scheduler returned an invalid result envelope.",
                err_type="InternalServerError",
                status_code=500,
            )
        response = self._build_completion_response(request, result)
        if memo_store is not None and memo_key is not None:
            memo_store.put(
                memo_key,
                answer_text=result.winner_text,
                extracted_answer=self._extract_answer(result.winner_text),
                model=request.model,
                context_version=request.context_version,
                n_tokens_saved=self._memo_tokens_saved(result),
            )
        return response

    def _build_memo_response(
        self,
        request: TreeCompletionRequest,
        memo_key: str,
        memo_entry: dict[str, Any],
    ) -> TreeCompletionResponse:
        result = TreeResult(
            winner_text=memo_entry["answer_text"],
            winner_token_ids=[],
            prompt_tokens=0,
            completion_tokens=0,
            summary=TreeSummary(
                policy=request.tree.policy,
                branch_count=request.tree.branches,
                pruned_count=0,
                merged_count=0,
                winner_branch_id="memo",
                tokens_spent_per_branch={},
                final_scores={},
                scorer=request.tree.scorer,
                kv_reuse_ratio=None,
                branch_answers={},
                served_from_memo=True,
                memo_key=memo_key,
            ),
            finish_reason="stop",
        )
        return self._build_completion_response(request, result)

    @staticmethod
    def _memo_tokens_saved(result: TreeResult) -> int:
        prompt_tokens = max(0, int(result.prompt_tokens))
        branch_tokens = sum(
            max(0, int(tokens))
            for tokens in result.summary.tokens_spent_per_branch.values()
        )
        completion_tokens = branch_tokens or max(0, int(result.completion_tokens))
        return prompt_tokens + completion_tokens

    def _coerce_plain_result(self, result, adapted_request) -> Optional[TreeResult]:
        """Phase-1 fallback: the runtime engine executes the tree in the
        scheduler (fork, prefix-KV sharing, budget finalize) and returns the
        winning branch as an ordinary completion. Until the per-branch trace is
        surfaced through the wire, build a minimal TreeResult that reports only
        what is known: the winner's text and the requested branch count. Every
        per-branch statistic is left null rather than fabricated.
        """
        if not isinstance(result, dict):
            return None
        text = result.get("text")
        if text is None:
            return None
        meta = result.get("meta_info") or {}
        prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
        completion_tokens = int(meta.get("completion_tokens", 0) or 0)
        finish = meta.get("finish_reason")
        finish_reason = (
            finish.get("type", "stop") if isinstance(finish, dict) else "stop"
        )
        params = getattr(adapted_request, "tree", None)
        branch_count = int(getattr(params, "branches", 1) or 1)
        scorer = getattr(params, "scorer", None)
        policy = getattr(params, "policy", "beam")

        # The scheduler runtime publishes its live trace through the
        # customized_info channel; the last snapshot is authoritative.
        snapshots = meta.get("autotree")
        snap = None
        if isinstance(snapshots, list):
            # token-aligned channel: entries are None padding except where the
            # runtime placed a snapshot; the last real one is authoritative
            real = [s for s in snapshots if isinstance(s, dict)]
            snap = real[-1] if real else None
        if isinstance(snap, dict):
            branches = snap.get("branches") or {}
            winner_id = str(snap.get("winner_branch_id") or "0")
            winner_ids = list(meta.get("output_ids", []) or [])
            used_scorer = scorer or "mean_logprob"
            verifier_used = bool(
                snap.get("verifier_used", False)
                or getattr(params, "verifier", None) is not None
            )

            # Self-consistency winner selection: when the final snapshot carries
            # every branch's token ids, detokenize each, extract the final
            # answer, and take the majority. Ties and no-answer cases fall back
            # to the value proxy's leading branch.
            branch_texts = self._detokenize_branches(branches)
            branch_answers = (
                {bid: self._extract_answer(t) for bid, t in branch_texts.items()}
                if branch_texts
                else {}
            )
            if branch_texts:
                selected = winner_id if verifier_used else self._self_consistency_vote(
                    branches, branch_texts
                )
                if selected is not None and selected in branch_texts:
                    winner_id = selected
                    text = branch_texts[selected]
                    winner_ids = list(
                        (branches.get(selected) or {}).get("output_ids") or []
                    )
                    completion_tokens = len(winner_ids) or completion_tokens
                    if not verifier_used:
                        used_scorer = "self_consistency"

            summary = TreeSummary(
                policy=snap.get("policy") or policy,
                branch_count=int(snap.get("branch_count") or branch_count),
                pruned_count=int(snap.get("pruned_count") or 0),
                merged_count=0,
                winner_branch_id=winner_id,
                tokens_spent_per_branch={
                    bid: int(b.get("tokens", 0)) for bid, b in branches.items()
                },
                final_scores={
                    bid: float(b.get("mean_logprob", 0.0))
                    for bid, b in branches.items()
                    if b.get("tokens")
                },
                scorer=used_scorer,
                kv_reuse_ratio=None,
                branch_answers=branch_answers,
                verifier_used=verifier_used,
                verifier_approved_count=int(
                    snap.get("verifier_approved_count", 0) or 0
                ),
                verifier_fell_back=bool(
                    snap.get("verifier_fell_back", False)
                ),
            )
            return TreeResult(
                winner_text=text,
                winner_token_ids=winner_ids,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                summary=summary,
                finish_reason=finish_reason,
            )

        summary = TreeSummary(
            policy=policy,
            branch_count=branch_count,
            pruned_count=0,
            merged_count=0,
            winner_branch_id="0",
            tokens_spent_per_branch={},
            final_scores={},
            scorer=scorer,
            kv_reuse_ratio=None,
            verifier_used=getattr(params, "verifier", None) is not None,
            verifier_approved_count=0,
            verifier_fell_back=getattr(params, "verifier", None) is not None,
        )
        return TreeResult(
            winner_text=text,
            winner_token_ids=list(meta.get("output_ids", []) or []),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            summary=summary,
            finish_reason=finish_reason,
        )

    def _detokenize_branches(self, branches: dict) -> Optional[dict]:
        """Decode each branch's output ids to text, trimmed at the first EOS
        (the parent runs with ignore_eos while it waits for its siblings, so
        anything past its natural EOS is scaffolding, not answer)."""
        tokenizer = getattr(self.tokenizer_manager, "tokenizer", None)
        if tokenizer is None:
            return None
        eos_id = getattr(tokenizer, "eos_token_id", None)
        texts = {}
        for bid, info in branches.items():
            ids = list((info or {}).get("output_ids") or [])
            if not ids:
                continue
            if eos_id is not None and eos_id in ids:
                ids = ids[: ids.index(eos_id)]
            texts[bid] = tokenizer.decode(ids, skip_special_tokens=True)
        return texts or None

    @staticmethod
    def _extract_answer(text: str) -> Optional[str]:
        """Final-answer extraction, shared with the scheduler runtime so the
        wire-visible branch_answers and the engine-side votes key identically
        (boxed, then '#### N', then the Answer: line, then last number)."""
        from sglang.srt.tree.answers import extract_answer_text

        return extract_answer_text(text)

    def _self_consistency_vote(
        self, branches: dict, branch_texts: dict
    ) -> Optional[str]:
        """Majority vote over extracted answers; ties break toward the higher
        mean-logprob branch. Returns None when no branch yields an answer."""
        answers = {
            bid: self._extract_answer(t) for bid, t in branch_texts.items()
        }
        counts: dict = {}
        for bid, ans in answers.items():
            if ans is not None:
                counts.setdefault(ans, []).append(bid)
        if not counts:
            return None

        def _lp(bid):
            return float((branches.get(bid) or {}).get("mean_logprob", -1e9))

        from sglang.srt.tree import selection

        if selection.vote_mode() == "weighted":
            # Confidence-weighted class selection (opt-in), mirroring the engine
            # runtime so wire branch_answers and the engine winner agree. The
            # plurality path below is the default and stays byte-identical.
            win_key = selection.weighted_winning_key(
                (ans, _lp(bid)) for bid, ans in answers.items() if ans is not None
            )
            if win_key is not None and win_key in counts:
                return max(counts[win_key], key=_lp)
        best_answer = max(
            counts.items(),
            key=lambda kv: (len(kv[1]), max(_lp(b) for b in kv[1])),
        )
        return max(best_answer[1], key=_lp)

    async def _handle_streaming_request(
        self,
        adapted_request: TreeGenerateReqInput,
        request: TreeCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ORJSONResponse]:
        generator = self._generate_tree_stream(adapted_request, raw_request)
        try:
            first_chunk = await generator.__anext__()
        except ValueError as error:
            return self.create_error_response(str(error))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request.base),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _generate_tree_stream(
        self,
        adapted_request: TreeGenerateReqInput,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        token_indices = defaultdict(int)
        saw_result = False
        runtime_event_count = 0
        source = self.tokenizer_manager.generate_request(
            adapted_request, raw_request
        )
        try:
            async for item in source:
                if isinstance(item, TreeBranchEvent):
                    payload = self._branch_event_payload(item, token_indices)
                    if payload is not None:
                        yield self._sse(payload.type, payload.model_dump_json())
                    continue

                if isinstance(item, TreeResult):
                    for event in item.branch_events:
                        payload = self._branch_event_payload(event, token_indices)
                        if payload is not None:
                            yield self._sse(
                                payload.type, payload.model_dump_json()
                            )
                    result = item
                elif isinstance(item, dict):
                    snapshot = self._latest_tree_snapshot(item)
                    if snapshot is not None:
                        events = snapshot.get("events") or []
                        if runtime_event_count > len(events):
                            raise ValueError(
                                "Tree scheduler stream event history regressed."
                            )
                        for raw_event in events[runtime_event_count:]:
                            event = self._coerce_branch_event(raw_event)
                            payload = self._branch_event_payload(
                                event, token_indices
                            )
                            if payload is not None:
                                yield self._sse(
                                    payload.type, payload.model_dump_json()
                                )
                        runtime_event_count = len(events)

                    finish_reason = (item.get("meta_info") or {}).get(
                        "finish_reason"
                    )
                    if finish_reason is None:
                        continue
                    result = self._coerce_plain_result(item, adapted_request)
                    if result is None:
                        raise ValueError(
                            "Tree scheduler returned an invalid stream envelope."
                        )
                    tree_completion_tokens = sum(
                        result.summary.tokens_spent_per_branch.values()
                    )
                    if tree_completion_tokens:
                        result.completion_tokens = tree_completion_tokens
                else:
                    raise ValueError(
                        "Tree scheduler returned an invalid stream envelope."
                    )

                if saw_result:
                    raise ValueError(
                        "Tree scheduler returned more than one final result."
                    )
                saw_result = True
                done = self._done_event(result)
                yield self._sse(done.type, done.model_dump_json())

            if not saw_result:
                raise ValueError("Tree scheduler stream ended without a final result.")
            yield "data: [DONE]\n\n"
        finally:
            close = getattr(source, "aclose", None)
            if callable(close):
                await close()
            if not saw_result:
                abort = getattr(self.tokenizer_manager, "abort_request", None)
                rid = getattr(getattr(adapted_request, "base", None), "rid", None)
                if callable(abort) and rid:
                    abort(rid)

    @staticmethod
    def _latest_tree_snapshot(item: dict) -> Optional[dict]:
        snapshots = (item.get("meta_info") or {}).get("autotree")
        if not isinstance(snapshots, list):
            return None
        real = [snapshot for snapshot in snapshots if isinstance(snapshot, dict)]
        return real[-1] if real else None

    @staticmethod
    def _coerce_branch_event(raw_event: Any) -> TreeBranchEvent:
        if isinstance(raw_event, TreeBranchEvent):
            return raw_event
        if not isinstance(raw_event, dict):
            raise ValueError("Tree scheduler returned an invalid branch event.")
        fields = {
            key: raw_event.get(key)
            for key in (
                "event",
                "branch_id",
                "parent_id",
                "token_id",
                "text",
                "score",
                "reason",
            )
        }
        return TreeBranchEvent(**fields)

    def _branch_event_payload(self, event: TreeBranchEvent, token_indices):
        if event.event == "forked":
            return TreeBranchStartedEvent(
                branch_id=event.branch_id, parent_id=event.parent_id
            )
        if event.event == "token":
            text = event.text
            if text is None and event.token_id is not None:
                tokenizer = getattr(self.tokenizer_manager, "tokenizer", None)
                if tokenizer is not None:
                    try:
                        text = tokenizer.decode(
                            [event.token_id], skip_special_tokens=False
                        )
                    except Exception:
                        text = None
            if text is None or event.score is None:
                raise ValueError("Tree token event is missing text or logprob.")
            token_index = token_indices[event.branch_id]
            token_indices[event.branch_id] += 1
            return TreeTokenEvent(
                branch_id=event.branch_id,
                token_index=token_index,
                token=text,
                token_id=event.token_id,
                logprob=event.score,
            )
        if event.event == "pruned":
            if not event.reason:
                raise ValueError("Tree pruned event is missing a reason.")
            return TreeBranchPrunedEvent(
                branch_id=event.branch_id, reason=event.reason
            )
        if event.event == "merged":
            if not event.parent_id:
                raise ValueError("Tree merged event is missing its target branch.")
            return TreeBranchMergedEvent(
                branch_id=event.branch_id, into_branch_id=event.parent_id
            )
        if event.event == "finalized":
            return None
        raise ValueError(f"Unknown tree branch event: {event.event}")

    def _done_event(self, result: TreeResult) -> TreeDoneEvent:
        summary = self._summary(result.summary)
        counters = result.counters or TreeCounters()
        return TreeDoneEvent(
            branch_id=summary.winner_branch_id,
            text=result.winner_text,
            finish_reason=result.finish_reason,
            usage=self._usage(result),
            counters=TreeCountersResponse(
                logical_tokens=counters.logical_tokens,
                physical_tokens=counters.physical_tokens,
                useful_tokens=counters.useful_tokens,
                elapsed_seconds=counters.elapsed_seconds,
                ttft_seconds=counters.ttft_seconds,
                unique_tokens_per_step=counters.unique_tokens_per_step,
                branch_tokens_per_step=counters.branch_tokens_per_step,
            ),
            tree=summary,
        )

    @staticmethod
    def _sse(event_type: str, payload: str) -> str:
        return f"event: {event_type}\ndata: {payload}\n\n"

    def _build_completion_response(
        self,
        request: TreeCompletionRequest,
        result: TreeResult,
        *,
        created: Optional[int] = None,
        response_id: Optional[str] = None,
    ) -> TreeCompletionResponse:
        usage = self._usage(result)
        response_kwargs = {}
        if created is not None:
            response_kwargs["created"] = created
        if response_id is not None:
            response_kwargs["id"] = response_id
        return TreeCompletionResponse(
            **response_kwargs,
            model=request.model,
            choices=[
                TreeCompletionChoice(
                    message=TreeResponseMessage(content=result.winner_text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=usage,
            tree=self._summary(result.summary),
        )

    @staticmethod
    def _usage(result: TreeResult) -> TreeUsage:
        return TreeUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        )

    @staticmethod
    def _summary(summary: TreeSummary) -> TreeSummaryResponse:
        if summary.winner_branch_id is None:
            raise ValueError("Tree result is missing winner_branch_id")
        # kv_reuse_ratio is null in the phase-1 fallback (per-branch trace not
        # yet surfaced through the wire); it is populated once the runtime emits
        # the full tree envelope. A null here is honest, not an error.
        return TreeSummaryResponse(
            policy=summary.policy,
            branch_count=summary.branch_count,
            pruned_count=summary.pruned_count,
            merged_count=summary.merged_count,
            winner_branch_id=summary.winner_branch_id,
            tokens_spent_per_branch=summary.tokens_spent_per_branch,
            final_scores=summary.final_scores,
            scorer=summary.scorer,
            kv_reuse_ratio=summary.kv_reuse_ratio,
            branch_answers=getattr(summary, "branch_answers", {}) or {},
            served_from_memo=getattr(summary, "served_from_memo", False),
            memo_key=getattr(summary, "memo_key", None),
            verifier_used=getattr(summary, "verifier_used", False),
            verifier_approved_count=getattr(
                summary, "verifier_approved_count", 0
            ),
            verifier_fell_back=getattr(summary, "verifier_fell_back", False),
        )
