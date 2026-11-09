"""Independent connection pools and metadata contracts for each model instance."""

from __future__ import annotations

from copy import deepcopy
import math
import os

import httpx

from rilab.config import (
    PROFILES,
    calibration_contract,
    valid_action_biases,
)
from rilab.decision import AdapterError


class HttpPredictBackend:
    def __init__(self, backend_id: str, config: dict, *, client=None):
        self.backend_id = backend_id
        self.config = deepcopy(config)
        self.profile = config["adapter_profile"]
        self.expected = PROFILES[self.profile]
        self.calibration_contract = calibration_contract(config)
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
            self.calibration = self._expected_calibration()
        except BaseException:
            self.client.close()
            raise
        self.requests = 0

    def _read_models(self) -> dict:
        response = self.client.get("/v1/models")
        response.raise_for_status()
        return self._validate_models(response.json())

    def _validate_models(self, payload) -> dict:
        try:
            models = payload["models"]
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
            response = models["response"]
            if self.calibration_contract == "legacy_pass_bias":
                if "action_biases" in response:
                    raise AdapterError(
                        f"legacy service exposed action_biases: {self.backend_id}"
                    )
                default = response.get("default_pass_bias")
                if default is not None and (
                    type(default) not in (int, float)
                    or not math.isfinite(default)
                    or default < 0
                ):
                    raise AdapterError(
                        f"invalid default_pass_bias: {self.backend_id}"
                    )
                configured = self.config.get("response", {}).get("pass_bias")
                if configured is None and default is None:
                    raise AdapterError(
                        f"legacy service has no effective pass_bias: {self.backend_id}"
                    )
            else:
                if "default_pass_bias" in response:
                    raise AdapterError(
                        f"action_biases service exposed default_pass_bias: "
                        f"{self.backend_id}"
                    )
                self._check_action_calibration(response, "models")
            penalty = response.get("no_yaku_call_penalty", 0.0)
            if (
                type(penalty) not in (int, float)
                or not math.isfinite(penalty)
                or penalty < 0
            ):
                raise AdapterError(
                    f"invalid service no_yaku_call_penalty: {self.backend_id}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AdapterError):
                raise
            raise AdapterError(f"invalid models response: {self.backend_id}") from exc
        return models

    def _check_action_calibration(self, info: dict, source: str) -> None:
        configured = self.config.get("response", {})
        biases = info.get("action_biases")
        if not valid_action_biases(biases):
            raise AdapterError(
                f"invalid {source} action_biases: {self.backend_id}"
            )
        if biases != configured.get("action_biases"):
            raise AdapterError(
                f"{source} action_biases differ from config: {self.backend_id}"
            )
        if info.get("no_yaku_call_penalty") != configured.get(
            "no_yaku_call_penalty"
        ):
            raise AdapterError(
                f"{source} no_yaku_call_penalty differs from config: "
                f"{self.backend_id}"
            )

    def _expected_calibration(self) -> dict:
        response = self.metadata["response"]
        if self.calibration_contract == "legacy_pass_bias":
            bias = self.config.get("response", {}).get("pass_bias")
            if bias is None:
                bias = response["default_pass_bias"]
            return {
                "pass_bias": bias,
                "no_yaku_call_penalty": response.get("no_yaku_call_penalty", 0.0),
            }
        return {
            "calibration_contract": "action_biases_v1",
            "action_biases": dict(response["action_biases"]),
            "no_yaku_call_penalty": response["no_yaku_call_penalty"],
        }

    def predict(self, task: str, state: dict, candidates: list[dict]) -> dict:
        request = self._prediction_request(task, state, candidates)
        response = self.client.post(f"/v1/predict/{task}", json=request)
        response.raise_for_status()
        return self._validate_prediction(task, candidates, response.json())

    def _prediction_request(self, task, state, candidates):
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
            if self.calibration_contract == "legacy_pass_bias":
                # Legacy requests must stay byte-for-byte on the old contract.
                request["pass_bias"] = self.config.get("response", {}).get(
                    "pass_bias"
                )
            else:
                # Never send pass_bias (not even null) to a v1.4.0 service.
                request["action_biases"] = dict(
                    self.config["response"]["action_biases"]
                )
        return request

    def _validate_prediction(self, task, candidates, result):
        if result.get("task") != task or result.get("model") != self.metadata[task]:
            raise AdapterError("prediction task/model changed during run")
        if result.get("selected_action") not in candidates:
            raise AdapterError("prediction selected an unsubmitted candidate")
        if task == "response":
            self._validate_response_calibration(result)
            raw = result.get("raw_selected_action")
            if raw not in candidates:
                raise AdapterError("prediction raw action is not a submitted candidate")
            if result.get("selection_mode") != "calibrated":
                raise AdapterError("response selection mode changed")
        elif any(
            key in result for key in ("pass_bias", "action_biases")
        ):
            raise AdapterError("turn response unexpectedly carried calibration")
        self.requests += 1
        return result

    def _validate_response_calibration(self, result: dict) -> None:
        if self.calibration_contract == "legacy_pass_bias":
            if "action_biases" in result:
                raise AdapterError(
                    "legacy response unexpectedly carried action_biases"
                )
            if result.get("pass_bias") != self.calibration["pass_bias"]:
                raise AdapterError("response pass_bias differs from pinned value")
        else:
            if "pass_bias" in result:
                raise AdapterError(
                    "action_biases response unexpectedly carried pass_bias"
                )
            if result.get("action_biases") != self.calibration["action_biases"]:
                raise AdapterError(
                    "response action_biases differ from pinned request"
                )
        if result.get("no_yaku_call_penalty") != self.calibration[
            "no_yaku_call_penalty"
        ]:
            raise AdapterError(
                "response no_yaku_call_penalty differs from pinned value"
            )

    def check_unchanged(self) -> None:
        if self._read_models() != self.metadata:
            raise AdapterError("model metadata changed during run")

    def close(self):
        self.client.close()


class AsyncHttpPredictBackend(HttpPredictBackend):
    """Online pool: metadata warm-up and cancellable HTTP, no worker threads."""

    def __init__(self, backend_id, config, *, client=None):
        self.backend_id = backend_id
        self.config = deepcopy(config)
        self.profile = config["adapter_profile"]
        self.expected = PROFILES[self.profile]
        self.calibration_contract = calibration_contract(config)
        headers = {}
        if config.get("api_key_env"):
            headers["X-API-Key"] = os.environ[config["api_key_env"]]
        self.client = client or httpx.AsyncClient(
            base_url=config["api_base"].rstrip("/"),
            timeout=config["timeout_seconds"],
            headers=headers,
        )
        self.requests = 0
        self.metadata = None

    async def warmup(self):
        self.metadata = await self._read_models()
        self.calibration = self._expected_calibration()

    async def _read_models(self):
        response = await self.client.get("/v1/models")
        response.raise_for_status()
        return self._validate_models(response.json())

    async def predict(self, task, state, candidates):
        if self.metadata is None:
            raise AdapterError("online backend must be warmed up before connecting")
        request = self._prediction_request(task, state, candidates)
        response = await self.client.post(f"/v1/predict/{task}", json=request)
        response.raise_for_status()
        return self._validate_prediction(task, candidates, response.json())

    async def check_unchanged(self):
        if await self._read_models() != self.metadata:
            raise AdapterError("model metadata changed during run")

    async def close(self):
        await self.client.aclose()


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
