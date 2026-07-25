import json

from thoughtbench.aggregate import aggregate_results, write_leaderboard


def _artifact(label: str, accuracy: float) -> dict:
    return {
        "meta": {
            "engine_label": label,
            "model": "model",
            "base_url_redacted": "http://endpoint.test",
            "arm": "single",
            "params": {},
            "seeds": [1],
            "git_sha": None,
            "started_at": "2026-01-01T00:00:00+00:00",
        },
        "tasks": [
            {
                "id": "a",
                "seed": 1,
                "correct": accuracy == 1,
                "answer": "1",
                "gold": "1",
                "tokens": 2,
                "wall_s": 0.1,
            }
        ],
        "summary": {
            "accuracy": accuracy,
            "ci_low": 0,
            "ci_high": 1,
            "mean_tokens": 2,
            "median_tokens": 2,
            "tokens_per_correct": 2 if accuracy else None,
            "n_tasks": 1,
        },
    }


def test_aggregator_merges_current_results_and_ignores_preserved_legacy_shapes(tmp_path) -> None:
    (tmp_path / "b.json").write_text(json.dumps(_artifact("b", 0)), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(_artifact("a", 1)), encoding="utf-8")
    (tmp_path / "legacy.json").write_text(json.dumps({"schema_version": "legacy"}), encoding="utf-8")

    leaderboard = aggregate_results(tmp_path)

    assert [entry.meta.engine_label for entry in leaderboard.entries] == ["a", "b"]
    written, path = write_leaderboard(tmp_path)
    assert path == tmp_path / "leaderboard.json"
    assert len(written.entries) == 2
    assert json.loads(path.read_text(encoding="utf-8"))["entries"][0]["meta"]["engine_label"] == "a"
