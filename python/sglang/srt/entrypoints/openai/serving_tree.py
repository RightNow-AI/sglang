"""Serving implementation for the AutoTree-compatible tree completions API."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, AsyncGenerator, Optional, Union

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

    def _request_id_prefix(self) -> str:
        return "tree-"

    def _validate_request(self, request: TreeCompletionRequest) -> Optional[str]:
        if not request.messages:
            return "Messages cannot be empty."
        return None

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
        base_request.return_logprob = True
        base_request.return_text_in_logprobs = True

        tree_request = TreeGenerateReqInput(
            base=base_request,
            tree=TreeParams(
                policy=request.tree.policy,
                branches=request.tree.branches,
                budget_tokens=request.tree.budget_tokens,
                scorer=request.tree.scorer,
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
        try:
            generator = self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            )
            result = await generator.__anext__()
        except ValueError as error:
            return self.create_error_response(str(error))

        if not isinstance(result, TreeResult):
            return self.create_error_response(
                "Tree scheduler returned an invalid result envelope.",
                err_type="InternalServerError",
                status_code=500,
            )
        return self._build_completion_response(request, result)

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
        async for item in self.tokenizer_manager.generate_request(
            adapted_request, raw_request
        ):
            if isinstance(item, TreeBranchEvent):
                payload = self._branch_event_payload(item, token_indices)
                if payload is not None:
                    yield self._sse(payload.type, payload.model_dump_json())
                continue
            if not isinstance(item, TreeResult):
                raise ValueError("Tree scheduler returned an invalid stream envelope.")
            if saw_result:
                raise ValueError("Tree scheduler returned more than one final result.")
            saw_result = True
            done = self._done_event(item)
            yield self._sse(done.type, done.model_dump_json())

        if not saw_result:
            raise ValueError("Tree scheduler stream ended without a final result.")
        yield "data: [DONE]\n\n"

    def _branch_event_payload(self, event: TreeBranchEvent, token_indices):
        if event.event == "forked":
            return TreeBranchStartedEvent(
                branch_id=event.branch_id, parent_id=event.parent_id
            )
        if event.event == "token":
            if event.text is None or event.score is None:
                raise ValueError("Tree token event is missing text or logprob.")
            token_index = token_indices[event.branch_id]
            token_indices[event.branch_id] += 1
            return TreeTokenEvent(
                branch_id=event.branch_id,
                token_index=token_index,
                token=event.text,
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
        if summary.kv_reuse_ratio is None:
            raise ValueError("Tree result is missing kv_reuse_ratio")
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
        )
