from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from autotree_core.modeling import ModelExecutor, ModelExecutorConfig


class NumericTokenizer:
    eos_token_id = None

    def encode(self, _text: str, *, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is True
        return [5, 6, 7, 8]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return f"<{token_ids[0]}>"


class RecordingExecutor(ModelExecutor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prune_accounting: list[tuple[int, int, int]] = []
        self.batch_decode_calls: list[tuple[int, ...]] = []
        self.dedup_calls = 0

    def decode_batch(self, execution, branch_ids, token_ids):
        self.batch_decode_calls.append(tuple(branch_ids))
        return super().decode_batch(execution, branch_ids, token_ids)

    def deduplicate(self, execution) -> int:
        self.dedup_calls += 1
        return super().deduplicate(execution)

    def prune(self, execution, branch_id: int) -> None:
        before = execution.pool.used_pages
        super().prune(execution, branch_id)
        self.prune_accounting.append((branch_id, before, execution.pool.used_pages))


@dataclass
class TinyEngineCase:
    executor: RecordingExecutor
    tokenizer: NumericTokenizer


@pytest.fixture
def tiny_engine_case() -> TinyEngineCase:
    torch.manual_seed(1729)
    model = GPT2LMHeadModel(
        GPT2Config(
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
    ).eval()
    executor = RecordingExecutor(
        ModelExecutorConfig(
            model_id="tiny-engine-model",
            page_size=4,
            capacity_pages=32,
            local_files_only=True,
        ),
        model=model,
    )
    return TinyEngineCase(executor=executor, tokenizer=NumericTokenizer())
