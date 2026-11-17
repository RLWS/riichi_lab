"""Backend factories for single-service and task-routed model deployments."""

from rilab.backends.http_backend import BackendRegistry, HttpPredictBackend
from rilab.backends.task_routed import TaskRoutedBackend


def create_backend(backend_id: str, config: dict):
    if config.get("backend_type") == "task_routed":
        return TaskRoutedBackend(backend_id, config)
    return HttpPredictBackend(backend_id, config)


__all__ = [
    "BackendRegistry",
    "HttpPredictBackend",
    "TaskRoutedBackend",
    "create_backend",
]
