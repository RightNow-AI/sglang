from __future__ import annotations

from pathlib import Path

from verl_autotree.patch_verl import BASE_ENTRY, REPLICA_ENTRY, main, patch_verl


BASE_FIXTURE = '''_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}
'''

REPLICA_FIXTURE = '''# Register built-in types
RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)
RolloutReplicaRegistry.register("trtllm", _load_trtllm)
'''


def _checkout(tmp_path: Path) -> Path:
    rollout = tmp_path / "verl" / "workers" / "rollout"
    rollout.mkdir(parents=True)
    (rollout / "base.py").write_text(BASE_FIXTURE, encoding="utf-8")
    (rollout / "replica.py").write_text(REPLICA_FIXTURE, encoding="utf-8")
    return tmp_path


def test_patch_inserts_exact_registry_lines_and_is_idempotent(tmp_path):
    checkout = _checkout(tmp_path)

    assert patch_verl(checkout) == (True, True)
    assert patch_verl(checkout) == (False, False)

    base_text = (checkout / "verl/workers/rollout/base.py").read_text(encoding="utf-8")
    replica_text = (checkout / "verl/workers/rollout/replica.py").read_text(encoding="utf-8")
    assert base_text.count(BASE_ENTRY) == 1
    assert replica_text.count(REPLICA_ENTRY) == 1


def test_cli_prints_manual_special_case_guidance(tmp_path, capsys):
    checkout = _checkout(tmp_path)

    assert main([str(checkout)]) == 0

    output = capsys.readouterr().out
    assert "verl/checkpoint_engine/base.py:306" in output
    assert "verl/workers/rollout/replica.py:394" in output
    assert "verl/workers/rollout/llm_server.py:520" in output
    assert "verl/workers/rollout/replica.py:327" in output
    assert "not edited by this script" in output
