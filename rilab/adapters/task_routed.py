"""Task-specific adapter for mixed Turn/Response deployments.

The 2x2 component experiment must use each task's native feature layout:
a v1.4.0 Turn model expects the v3 adapter, while a v1.3.0 Response model
expects the v2 adapter.  One agent therefore needs two adapters, selected by
the current decision task.
"""

from __future__ import annotations

from rilab.adapters import create_adapter
from rilab.decision import AdapterError


class TaskRoutedAdapter:
    profile = "task_routed"

    def __init__(self, turn_profile: str, response_profile: str):
        if turn_profile == response_profile:
            raise AdapterError("task-routed adapter requires distinct task profiles")
        self.profiles = {
            "turn": turn_profile,
            "response": response_profile,
        }
        self.adapters = {
            task: create_adapter(profile)
            for task, profile in self.profiles.items()
        }

    def profile_for_task(self, task: str) -> str:
        try:
            return self.profiles[task]
        except KeyError as exc:
            raise AdapterError(f"unsupported task for adapter: {task}") from exc

    def project_for_task(self, task: str, state: dict) -> dict:
        try:
            adapter = self.adapters[task]
        except KeyError as exc:
            raise AdapterError(f"unsupported task for adapter: {task}") from exc
        return adapter.project(state)

    def project(self, state: dict) -> dict:
        raise AdapterError("task_routed adapter requires an explicit task")
