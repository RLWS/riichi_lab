"""Backend factories for single-service and task-routed model deployments."""

from rilab.backends.http_backend import BackendRegistry, HttpPredictBackend
from rilab.backends.task_routed import TaskRoutedBackend
from rilab.backends.veto_turn import VetoTurnBackend


def create_backend(backend_id: str, config: dict):
    backend_type = config.get("backend_type")
    if backend_type == "task_routed":
        return TaskRoutedBackend(backend_id, config)
    if backend_type == "veto_turn":
        return VetoTurnBackend(backend_id, config)
    return HttpPredictBackend(backend_id, config)


__all__ = [
    "BackendRegistry",
    "HttpPredictBackend",
    "TaskRoutedBackend",
    "VetoTurnBackend",
    "create_backend",
]
