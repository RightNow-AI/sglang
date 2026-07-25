from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
DISABLE_MARKERS = (
    "disable-cuda-graph",
    "disable_cuda_graph",
    "disable-decode-cuda-graph",
    "cuda-graph-backend-decode disabled",
)


def test_autotree_run_paths_do_not_disable_cuda_graphs():
    paths = [ROOT / "REPRODUCE.md", ROOT / "README.md"]
    paths.extend((ROOT / "docs" / "autotree").rglob("*"))
    paths.extend((ROOT / "bench").rglob("*"))
    offenders = []
    for path in sorted(path for path in paths if path.is_file()):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in DISABLE_MARKERS:
            if marker in text:
                offenders.append(f"{path.relative_to(ROOT)}: {marker}")
    assert offenders == []


def test_shared_read_fallback_is_off_by_default():
    source = (
        ROOT / "python" / "sglang" / "srt" / "model_executor" / "forward_batch_info.py"
    ).read_text(encoding="utf-8")
    assert 'environ.get("AUTOTREE_SHARED_READ", "0")' in source
    assert 'environ.get("AUTOTREE_SHARED_READ", "1")' not in source
