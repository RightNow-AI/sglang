"""Apply the phase-1 tree runtime splice to an installed sglang 0.5.15 tree.

Anchored string edits only; every miss is a loud failure. Idempotent.
Usage: python3 apply_splice.py <site-packages/sglang dir>
"""

import sys
from pathlib import Path

SITE = Path(sys.argv[1])
MARK = "# [autotree-splice]"


def patch(path: Path, anchor: str, insert: str, before: bool = False) -> None:
    text = path.read_text()
    if insert.strip().splitlines()[0] in text:
        print(f"  = {path.name}: already patched at this site")
        return
    if anchor not in text:
        raise SystemExit(f"ANCHOR MISS in {path}: {anchor[:80]!r}")
    new = (insert + anchor) if before else (anchor + insert)
    path.write_text(text.replace(anchor, new, 1))
    print(f"  + {path.name}: patched")


srt = SITE / "srt"

# 1. scheduler.py: dispatcher entry + lazy handler
sched = srt / "managers" / "scheduler.py"
patch(
    sched,
    "                (TokenizedGenerateReqInput, self.handle_generate_request),",
    f"\n                (TreeSpliceInput, self._autotree_dispatch),  {MARK}",
)
patch(
    sched,
    "    def handle_generate_request(",
    f"""    def _autotree_dispatch(self, recv):  {MARK}
        from sglang.srt.tree.tree_runtime import install

        if not hasattr(self, "tree_runtime"):
            install(self)
        return self.tree_runtime.handle_tree_request(recv)

""",
    before=True,
)
# import for the dispatcher entry type
patch(
    sched,
    "from sglang.srt.managers.io_struct import (",
    f"from sglang.srt.tree.tree_runtime import (  {MARK}\n"
    "    TokenizedTreeGenerateReqInput as TreeSpliceInput,\n"
    ")\n",
    before=True,
)

# 2. batch_result_processor.py: prefill-done + decode token hooks
brp = srt / "managers" / "scheduler_components" / "batch_result_processor.py"
patch(
    brp,
    "                    self._maybe_update_reasoning_tokens(req, next_token_id)",
    f"""
                    tr = getattr(self, "tree_runtime", None)  {MARK}
                    if tr is not None:
                        tr.on_prefill_done(req)""",
    before=False,
)
patch(
    brp,
    "            req.output_ids.extend(next_token_id)",
    f"""
            tr = getattr(self, "tree_runtime", None)  {MARK}
            if tr is not None:
                tr.on_token(req, next_token_id, None)""",
)

# 3. tokenizer_manager.py: envelope unwrap + tokenized wrap
tm = srt / "managers" / "tokenizer_manager.py"
patch(
    tm,
    "    async def generate_request(",
    f"""    def _autotree_unwrap(self, obj):  {MARK}
        tree = getattr(obj, "tree", None)
        base = getattr(obj, "base", None)
        if tree is not None and base is not None:
            params = tree if isinstance(tree, dict) else {{
                "policy": getattr(tree, "policy", "beam"),
                "branches": getattr(tree, "branches", 1),
                "budget_tokens": getattr(tree, "budget_tokens", 0),
                "scorer": getattr(tree, "scorer", None),
            }}
            setattr(base, "_autotree_params", params)
            return base
        return obj

""",
    before=True,
)
patch(
    tm,
    "        # Normalize the request\n        obj.normalize_batch_and_arguments()",
    f"        obj = self._autotree_unwrap(obj)  {MARK}\n",
    before=True,
)
patch(
    tm,
    "                    tokenized_obj = await self._tokenize_one_request(obj)",
    f"""
                    _tp = getattr(obj, "_autotree_params", None)  {MARK}
                    if _tp is not None:
                        from sglang.srt.tree.tree_runtime import (
                            TokenizedTreeGenerateReqInput,
                        )
                        tokenized_obj = TokenizedTreeGenerateReqInput(
                            base=tokenized_obj, tree=_tp
                        )""",
)

# 4. http_server.py: serving object + route
http = srt / "entrypoints" / "http_server.py"
patch(
    http,
    "from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion",
    f"from sglang.srt.entrypoints.openai.serving_tree import OpenAIServingTree  {MARK}\n",
    before=True,
)
patch(
    http,
    "    fast_api_app.state.openai_serving_completion = ",
    f"""    fast_api_app.state.openai_serving_tree = OpenAIServingTree(  {MARK}
        _global_state.tokenizer_manager, fast_api_app.state.openai_serving_chat
    )
""",
    before=True,
)
patch(
    http,
    '@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])',
    f"""@app.post("/v1/tree/completions", dependencies=[Depends(validate_json_request)])  {MARK}
async def openai_v1_tree_completions(request: TreeSpliceRequest, raw_request: Request):
    return await raw_request.app.state.openai_serving_tree.handle_request(
        request, raw_request
    )


""",
    before=True,
)
patch(
    http,
    "from sglang.srt.entrypoints.openai.serving_tree import OpenAIServingTree",
    f"\nfrom sglang.srt.entrypoints.openai.protocol_tree import (  {MARK}\n"
    "    TreeCompletionRequest as TreeSpliceRequest,\n"
    ")",
)

print("splice applied")
