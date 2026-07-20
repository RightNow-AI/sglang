import sys
from pathlib import Path
SITE = Path(sys.argv[1])
brp = SITE / "srt" / "managers" / "scheduler_components" / "batch_result_processor.py"
s = brp.read_text()
old1 = """                    tr = getattr(self, "tree_runtime", None)  # [autotree-splice]
                    if tr is not None:
                        tr.on_prefill_done(req)"""
new1 = """                    from sglang.srt.tree.tree_runtime import get_active as _tr_get  # [autotree-splice]
                    tr = _tr_get()
                    if tr is not None:
                        tr.on_prefill_done(req)"""
if old1 in s:
    s = s.replace(old1, new1)
else:
    assert "[autotree-splice]" in s, "prefill hook missing entirely"
anchor = "            req.output_ids.extend(next_token_id)"
decode_hook = anchor + """
            from sglang.srt.tree.tree_runtime import get_active as _tr_get2  # [autotree-splice-decode]
            _tr = _tr_get2()
            if _tr is not None:
                _tr.on_token(req, next_token_id, next_token_logprobs[i] if next_token_logprobs is not None else None)"""
if "[autotree-splice-decode]" not in s:
    assert anchor in s, "decode anchor not found"
    s = s.replace(anchor, decode_hook, 1)
elif "_tr.on_token(req, next_token_id, None)" in s:
    # upgrade an older splice that passed no logprob
    s = s.replace(
        "                _tr.on_token(req, next_token_id, None)",
        "                _tr.on_token(req, next_token_id, next_token_logprobs[i] if next_token_logprobs is not None else None)",
        1,
    )
brp.write_text(s)
print("both hooks in place")
