import json

import pytest

from thoughtbench.tasks import TaskFileError, load_tasks


def test_load_tasks_returns_exact_file_hash_and_typed_tasks(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "prompt": "question",
                "answer": "answer",
                "grader": "exact-match",
                "tags": ["fixture"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tasks, digest = load_tasks(path)

    assert tasks[0].id == "one"
    assert len(digest) == 64


def test_load_tasks_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    row = {
        "id": "duplicate",
        "prompt": "question",
        "answer": "answer",
        "grader": "exact-match",
        "tags": [],
    }
    path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")

    with pytest.raises(TaskFileError, match="duplicate task id"):
        load_tasks(path)


def test_load_tasks_rejects_unknown_fields(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "prompt": "question",
                "answer": "answer",
                "grader": "numeric",
                "tags": [],
                "dataset": "not-allowed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskFileError, match="extra_forbidden"):
        load_tasks(path)
