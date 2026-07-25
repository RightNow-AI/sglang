"""JSONL task loading with strict duplicate and shape checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from .models import Task


class TaskFileError(ValueError):
    """Raised when a task JSONL file is malformed."""


def load_tasks(path: Path) -> tuple[list[Task], str]:
    """Load tasks and return their exact-file SHA-256 provenance hash."""

    raw = path.read_bytes()
    tasks: list[Task] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            task = Task.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise TaskFileError(f"{path}:{line_number}: {exc}") from exc
        if task.id in seen:
            raise TaskFileError(f"{path}:{line_number}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise TaskFileError(f"{path}: task file is empty")
    return tasks, hashlib.sha256(raw).hexdigest()
