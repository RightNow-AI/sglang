"""Serving implementation for the AutoTree-compatible tree completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.protocol_tree import TreeCompletionRequest
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.tree import TreeGenerateReqInput, TreeParams

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
