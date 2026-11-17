"""Backend that routes Turn and Response to different model services.

Both sub-backends are ordinary HttpPredictBackend instances.  Each one talks
to its native service and therefore validates that service's own feature and
calibration contract.  The wrapper only exposes the task-specific half.
"""

from __future__ import annotations

from copy import deepcopy

from rilab.adapters.task_routed import TaskRoutedAdapter
from rilab.backends.http_backend import HttpPredictBackend
from rilab.decision import AdapterError


class TaskRoutedBackend:
    profile = "task_routed"

    def __init__(self, backend_id: str, config: dict):
        self.backend_id = backend_id
        self.config = deepcopy(config)
        self.turn_backend = None
        self.response_backend = None
        try:
            turn_config = deepcopy(config["turn_backend"])
            response_config = deepcopy(config["response_backend"])
            turn_config.setdefault("timeout_seconds", 30.0)
            response_config.setdefault("timeout_seconds", 30.0)
            self.turn_backend = HttpPredictBackend(
                f"{backend_id}:turn", turn_config
            )
            self.response_backend = HttpPredictBackend(
                f"{backend_id}:response", response_config
            )
            self.metadata = {
                "turn": self.turn_backend.metadata["turn"],
                "response": self.response_backend.metadata["response"],
                "turn_service": {
                    "backend_id": self.turn_backend.backend_id,
                    "profile": self.turn_backend.profile,
                    "metadata": self.turn_backend.metadata,
                    "calibration": self.turn_backend.calibration,
                },
                "response_service": {
                    "backend_id": self.response_backend.backend_id,
                    "profile": self.response_backend.profile,
                    "metadata": self.response_backend.metadata,
                    "calibration": self.response_backend.calibration,
                },
            }
            self.calibration = {
                "turn": self.turn_backend.calibration,
                "response": self.response_backend.calibration,
            }
        except BaseException:
            self.close()
            raise

    @property
    def requests(self) -> int:
        return (
            (self.turn_backend.requests if self.turn_backend else 0)
            + (self.response_backend.requests if self.response_backend else 0)
        )

    def profile_for_task(self, task: str) -> str:
        if task == "turn" and self.turn_backend is not None:
            return self.turn_backend.profile
        if task == "response" and self.response_backend is not None:
            return self.response_backend.profile
        raise AdapterError(f"unsupported task for backend: {task}")

    def create_adapter(self):
        return TaskRoutedAdapter(
            self.profile_for_task("turn"),
            self.profile_for_task("response"),
        )

    def validate_adapter(self, adapter) -> None:
        if not isinstance(adapter, TaskRoutedAdapter):
            raise AdapterError("task-routed backend requires TaskRoutedAdapter")
        expected = {
            "turn": self.profile_for_task("turn"),
            "response": self.profile_for_task("response"),
        }
        if adapter.profiles != expected:
            raise AdapterError(
                f"task-routed adapter profiles differ from backend: "
                f"{adapter.profiles} != {expected}"
            )

    def predict(self, task: str, state: dict, candidates: list[dict]) -> dict:
        if task == "turn" and self.turn_backend is not None:
            return self.turn_backend.predict(task, state, candidates)
        if task == "response" and self.response_backend is not None:
            return self.response_backend.predict(task, state, candidates)
        raise AdapterError(f"unsupported task for backend: {task}")

    def check_unchanged(self) -> None:
        if self.turn_backend is not None:
            self.turn_backend.check_unchanged()
        if self.response_backend is not None:
            self.response_backend.check_unchanged()

    def close(self) -> None:
        if self.turn_backend is not None:
            self.turn_backend.close()
        if self.response_backend is not None:
            self.response_backend.close()
