"""CPU dry-run coverage for the decode benchmark CLI."""

from __future__ import annotations

from importlib import import_module


def test_benchmark_cpu_dry_run_executes_end_to_end(capsys) -> None:
    benchmark = import_module("autotree_core.kernels.bench_decode")

    exit_code = benchmark.main(["--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "reference" in output.lower()
    assert "tokens/sec" in output.lower()
    assert "branches=4" in output
