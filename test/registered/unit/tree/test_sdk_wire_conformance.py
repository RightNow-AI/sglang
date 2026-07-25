"""The SDK must accept what the engine actually sends, and vice versa.

The SDK pins its models with extra="forbid", which is the right choice: it
catches silent wire drift. But it only works if the models are kept in sync
with the engine, and they were not. Before this test:

- the engine ALWAYS sends branch_answers, served_from_memo and memo_key in
  TreeSummary, none of which the SDK declared, so extra="forbid" rejected
  EVERY production response
- the engine accepts fork_at_text, fork_at_entropy, adaptive_width,
  consensus_warmup, consensus_interval and min_survivors, none of which the
  SDK declared, so a valid request was rejected client side before it was sent

Both directions were broken and nothing caught it, because the SDK was only
ever tested against hand written fixtures that matched the SDK rather than
against the engine's real dataclasses.

This test compares the two by parsing the source, so it needs neither the SDK
installed nor the engine importable, and it fails the moment they drift again.
"""

import re
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="per-commit-cpu")

REPO = Path(__file__).resolve().parents[4]
SDK_MODELS = REPO / "autotree" / "sdk" / "autotree_sdk" / "models.py"
ENGINE_PARAMS = REPO / "python" / "sglang" / "srt" / "tree" / "params.py"


def _fields(text, header, stop):
    match = re.search(rf"{header}(.*?)(?={stop}|\Z)", text, re.S)
    if not match:
        return []
    return re.findall(r"^\s{4}(\w+)\s*:", match.group(1), re.M)


def sdk_fields(cls):
    return _fields(SDK_MODELS.read_text(encoding="utf-8", errors="replace"),
                   rf"class {cls}\(BaseModel\):", r"\nclass ")


def engine_fields(cls):
    return _fields(ENGINE_PARAMS.read_text(encoding="utf-8", errors="replace"),
                   rf"class {cls}:", r"\n@dataclasses|\nclass ")


def test_sdk_accepts_every_field_the_engine_sends():
    """extra=forbid means a missing field is a hard client-side failure."""
    engine = engine_fields("TreeSummary")
    sdk = sdk_fields("TreeSummary")
    assert engine, "could not parse the engine TreeSummary"
    missing = [f for f in engine if f not in sdk]
    assert not missing, (
        f"SDK TreeSummary rejects fields the engine sends: {missing}. "
        "With extra=forbid this fails every real response."
    )


def test_sdk_accepts_every_tree_parameter_the_engine_supports():
    engine = engine_fields("TreeParams")
    sdk = sdk_fields("TreeParameters")
    assert engine, "could not parse the engine TreeParams"
    missing = [f for f in engine if f not in sdk]
    assert not missing, (
        f"SDK TreeParameters rejects engine parameters: {missing}. "
        "Users cannot send them even though the engine accepts them."
    )


def test_sdk_does_not_invent_fields_the_engine_never_sends():
    """Drift in the other direction: a required SDK field the engine omits."""
    engine = set(engine_fields("TreeSummary"))
    invented = [f for f in sdk_fields("TreeSummary") if f not in engine]
    assert not invented, (
        f"SDK TreeSummary declares fields the engine never sends: {invented}"
    )


def test_the_conformance_check_can_actually_see_both_files():
    """Guard the guard: a bad path would make every check above vacuously pass."""
    assert SDK_MODELS.exists(), f"SDK models not found at {SDK_MODELS}"
    assert ENGINE_PARAMS.exists(), f"engine params not found at {ENGINE_PARAMS}"
    assert len(engine_fields("TreeSummary")) >= 9
    assert len(sdk_fields("TreeSummary")) >= 9
