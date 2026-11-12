from copy import deepcopy
from pathlib import Path

import pytest
import httpx

from rilab.backends.http_backend import HttpPredictBackend
from rilab.online.ranked import load_ranked_config
from scripts.run_v140_online import (
    loopback_config,
    validation_config,
    validation_passed,
)


def test_v140_ranked_budget_and_validation_keep_frozen_model():
    config = load_ranked_config(
        Path(__file__).parents[1] / "configs/arena_ranked_v140.yaml"
    )
    before = deepcopy(config)
    validation = validation_config(config)
    assert config == before
    assert config["ranked"]["max_games"] == 10
    assert config["backend"]["api_key_env"] == "TRAIN_MJ_API_KEY_V140"
    assert validation["online"]["match_rounds"] == 4
    assert validation["backend"] == config["backend"]
    assert validation["backend"]["adapter_profile"] == "full_state_v3"
    assert "pass_bias" not in validation["backend"]["response"]
    assert "ranked" not in validation


def test_local_launcher_does_not_require_manual_deployment_key(monkeypatch):
    monkeypatch.delenv("TRAIN_MJ_API_KEY_V140", raising=False)
    config = load_ranked_config(
        Path(__file__).parents[1] / "configs/arena_ranked_v140.yaml"
    )
    before = deepcopy(config)
    local = loopback_config(config)
    assert config == before
    assert "api_key_env" not in local["backend"]
    assert "api_key_env" not in validation_config(local)["backend"]
    assert local["backend"]["response"] == config["backend"]["response"]


@pytest.mark.parametrize("local", [False, True])
def test_v140_backend_sends_key_only_for_authenticated_deployment(monkeypatch, local):
    config = load_ranked_config(
        Path(__file__).parents[1] / "configs/arena_ranked_v140.yaml"
    )
    if local:
        config = loopback_config(config)
        monkeypatch.delenv("TRAIN_MJ_API_KEY_V140", raising=False)
    else:
        monkeypatch.setenv("TRAIN_MJ_API_KEY_V140", "test-backend-key")
    models = {
        task: {
            "task": task,
            "feature_schema": "full_state_v3",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "candidate_limit": 64,
        }
        for task, state_dim, action_dim in (
            ("turn", 11969, 249),
            ("response", 12012, 251),
        )
    }
    models["response"].update(config["backend"]["response"])
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers.get("X-API-Key") == (
            None if local else "test-backend-key"
        )
        return httpx.Response(200, json={"models": models})

    client_factory = httpx.Client
    monkeypatch.setattr(
        "rilab.backends.http_backend.httpx.Client",
        lambda **kwargs: client_factory(
            transport=httpx.MockTransport(handle), **kwargs
        ),
    )
    backend = HttpPredictBackend("v140", config["backend"])
    try:
        assert len(requests) == 1
    finally:
        backend.close()


@pytest.mark.parametrize("fault", [None, "protocol", "fallback", "defaulted"])
def test_activation_gate_rejects_unclean_validation(fault):
    review = {
        "protocol_passed": True,
        "fallback_request_ids": [],
        "defaulted_request_ids": [],
    }
    if fault == "protocol":
        review["protocol_passed"] = False
    elif fault:
        review[f"{fault}_request_ids"] = [1]
    assert validation_passed(review) == (fault is None)
