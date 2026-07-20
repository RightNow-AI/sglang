"""Serving implementation for the AutoTree-compatible tree completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.protocol_tree import (
    TreeCompletionChoice,
    TreeCompletionRequest,
    TreeCompletionResponse,
    TreeResponseMessage,
    TreeSummaryResponse,
    TreeUsage,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.tree import TreeGenerateReqInput, TreeParams, TreeResult, TreeSummary

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
