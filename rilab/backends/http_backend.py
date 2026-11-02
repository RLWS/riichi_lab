"""Independent connection pools and metadata contracts for each model instance."""

from __future__ import annotations

from copy import deepcopy
import os

import httpx

from rilab.config import PROFILES
from rilab.decision import AdapterError


class HttpPredictBackend:
    def __init__(self, backend_id: str, config: dict, *, client=None):
        self.backend_id = backend_id
        self.config = deepcopy(config)
        self.profile = config["adapter_profile"]
        self.expected = PROFILES[self.profile]
        headers = {}
        if config.get("api_key_env"):
            headers["X-API-Key"] = os.environ[config["api_key_env"]]
        self.client = client or httpx.Client(
            base_url=config["api_base"].rstrip("/"),
            timeout=config["timeout_seconds"],
            headers=headers,
        )
        try:
            self.metadata = self._read_models()
        except BaseException:
            self.client.close()
            raise
        self.requests = 0

    def _read_models(self) -> dict:
        response = self.client.get("/v1/models")
        response.raise_for_status()
        try:
            models = response.json()["models"]
            for index, task in enumerate(("turn", "response")):
                info = models[task]
                if (
                    info["task"] != task
                    or info["state_dim"] != self.expected["state_dims"][index]
                    or info["action_dim"] != self.expected["action_dims"][index]
                    or info.get("feature_schema") != self.expected["feature_schema"]
                    or type(info["candidate_limit"]) is not int
                    or info["candidate_limit"] <= 0
                ):
                    raise AdapterError(
                        f"incompatible model contract: {self.backend_id}/{task}"
                    )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(f"invalid models response: {self.backend_id}") from exc
        return models

    def predict(self, task: str, state: dict, candidates: list[dict]) -> dict:
        if task not in ("turn", "response") or not candidates:
            raise AdapterError("invalid prediction task/candidates")
        if len(candidates) > self.metadata[task]["candidate_limit"]:
            raise AdapterError(f"{self.backend_id}/{task}: candidate limit exceeded")
        request = {
            "state": state,
            "legal_actions": candidates,
            "top_k": min(5, len(candidates)),
        }
        if task == "response":
            request["pass_bias"] = self.config.get("response", {}).get("pass_bias")
        response = self.client.post(f"/v1/predict/{task}", json=request)
        response.raise_for_status()
        result = response.json()
        if result.get("task") != task or result.get("model") != self.metadata[task]:
            raise AdapterError("prediction task/model changed during run")
        if result.get("selected_action") not in candidates:
            raise AdapterError("prediction selected an unsubmitted candidate")
        self.requests += 1
        return result

    def check_unchanged(self) -> None:
        if self._read_models() != self.metadata:
            raise AdapterError("model metadata changed during run")

    def close(self):
        self.client.close()


class BackendRegistry:
    def __init__(self, configs: dict, *, factory=HttpPredictBackend):
        self.backends = {}
        try:
            for name, config in configs.items():
                self.backends[name] = factory(name, config)
        except BaseException:
            self.close()
            raise

    def __getitem__(self, backend_id):
        return self.backends[backend_id]

    def close(self):
        for backend in self.backends.values():
            backend.close()
