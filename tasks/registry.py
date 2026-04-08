# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Task registry — single source of truth.

Both the /tasks endpoint and baseline_agent.py import from here.
Adding a new task = add one line to TASK_REGISTRY.
"""

from tasks.base_task import BaseTask
from tasks.task_01_easy import Task01SingleZone
from tasks.task_02_medium import Task02MultiZone
from tasks.task_03_hard import Task03CascadingHazards

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    Task01SingleZone.task_id: Task01SingleZone,
    Task02MultiZone.task_id: Task02MultiZone,
    Task03CascadingHazards.task_id: Task03CascadingHazards,
}


def get_task(task_id: str) -> BaseTask:
    cls = TASK_REGISTRY.get(task_id)
    if cls is None:
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY)}")
    return cls()


def list_tasks() -> list[dict]:
    return [cls().to_dict() for cls in TASK_REGISTRY.values()]
