from typing import Protocol


class PredictBackend(Protocol):
    backend_id: str
    profile: str

    def predict(self, task: str, state: dict, candidates: list[dict]) -> dict: ...
