"""Apply the two AutoTree registry entries to a verl checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

BASE_ENTRY = '    ("autotree", "async"): "verl_autotree.adapters.AutoTreeServerAdapter",'
REPLICA_ENTRY = (
    'RolloutReplicaRegistry.register("autotree", lambda: '
    '__import__("verl_autotree.adapters", fromlist=["AutoTreeReplica"]).AutoTreeReplica)'
)

_BASE_ANCHOR = '    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",'
_REPLICA_ANCHOR = 'RolloutReplicaRegistry.register("sglang", _load_sglang)'

SPECIAL_CASE_GUIDANCE = (
    "Manual follow-up required for verl @ 6a6242f (not edited by this script):",
    '  verl/checkpoint_engine/base.py:306 - mirror the name == "sglang" condition for "autotree".',
    '  verl/workers/rollout/replica.py:394 - mirror the rollout == "sglang" special case for "autotree".',
    '  verl/workers/rollout/llm_server.py:520 - mirror the rollout == "sglang" special case for "autotree".',
    "  verl/workers/rollout/replica.py:327 - make the autotree loader reuse the _load_sglang vLLM-mock setup.",
)


def _insert_after(path: Path, anchor: str, entry: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if entry in text:
        return False
    if anchor not in text:
        raise RuntimeError(f"Could not find expected registry anchor in {path}: {anchor}")
    newline = "\r\n" if "\r\n" in text else "\n"
    updated = text.replace(anchor, f"{anchor}{newline}{entry}", 1)
    path.write_text(updated, encoding="utf-8", newline="")
    return True


def patch_verl(checkout: str | Path) -> tuple[bool, bool]:
    """Patch the weight-sync and replica registries, idempotently."""
    root = Path(checkout).expanduser().resolve()
    base_path = root / "verl" / "workers" / "rollout" / "base.py"
    replica_path = root / "verl" / "workers" / "rollout" / "replica.py"
    missing = [path for path in (base_path, replica_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Not a compatible verl checkout; missing: " + ", ".join(str(path) for path in missing)
        )
    return (
        _insert_after(base_path, _BASE_ANCHOR, BASE_ENTRY),
        _insert_after(replica_path, _REPLICA_ANCHOR, REPLICA_ENTRY),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", help="Path to the verl checkout")
    args = parser.parse_args(argv)
    base_changed, replica_changed = patch_verl(args.checkout)
    print(f"base.py registry: {'patched' if base_changed else 'already present'}")
    print(f"replica.py registry: {'patched' if replica_changed else 'already present'}")
    print("\n".join(SPECIAL_CASE_GUIDANCE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
