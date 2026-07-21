import runpy
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_apply_splice_registers_tree_after_chat_and_is_idempotent(tmp_path):
    site = tmp_path / "sglang"
    write(
        site / "srt" / "managers" / "scheduler.py",
        """from sglang.srt.managers.io_struct import (
)

class Scheduler:
    def init_request_dispatcher(self):
        routes = [
                (TokenizedGenerateReqInput, self.handle_generate_request),
        ]

    def handle_generate_request(
        self, recv
    ):
        pass
""",
    )
    write(
        site
        / "srt"
        / "managers"
        / "scheduler_components"
        / "batch_result_processor.py",
        """class Processor:
    def prefill(self, req, next_token_id):
                    self._maybe_update_reasoning_tokens(req, next_token_id)

    def decode(self, req, next_token_id):
            req.output_ids.extend(next_token_id)
""",
    )
    write(
        site / "srt" / "managers" / "tokenizer_manager.py",
        """class TokenizerManager:
    async def generate_request(self, obj):
        # Normalize the request
        obj.normalize_batch_and_arguments()
        if obj.is_single:
                    tokenized_obj = await self._tokenize_one_request(obj)
""",
    )
    http_path = site / "srt" / "entrypoints" / "http_server.py"
    write(
        http_path,
        """from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion

async def lifespan(fast_api_app):
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion()
    fast_api_app.state.openai_serving_chat = (
        _global_state.tokenizer_manager.serving_chat_class(
            _global_state.tokenizer_manager, _global_state.template_manager
        )
    )

@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def completions():
    pass
""",
    )
    script = (
        Path(__file__).resolve().parents[4] / "tools" / "splice" / "apply_splice.py"
    )

    original_argv = sys.argv
    try:
        sys.argv = [str(script), str(site)]
        runpy.run_path(str(script), run_name="__main__")
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv

    patched = http_path.read_text(encoding="utf-8")
    chat_index = patched.index("fast_api_app.state.openai_serving_chat =")
    tree_index = patched.index("fast_api_app.state.openai_serving_tree =")

    assert chat_index < tree_index
    assert "fast_api_app.state.openai_serving_chat\n    )" in patched
    assert patched.count("fast_api_app.state.openai_serving_tree =") == 1
