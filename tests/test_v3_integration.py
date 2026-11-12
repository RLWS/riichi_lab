"""Keep v3 wire states distinct from the model feature/calibration contract."""

from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest
import yaml

from rilab.adapters import create_adapter
from rilab.backends.http_backend import HttpPredictBackend
from rilab.config import load_arena_config
from rilab.decision import AdapterError

ROOT = Path(__file__).parents[1]


def test_v3_projection_keeps_public_wire_schema_and_copies_state():
    state = json.loads(
        (ROOT / "tests/fixtures/http_contract/golden_turn_request.json").read_text()
    )["state"]
    before = deepcopy(state)
    projected = create_adapter("full_state_v3").project(state)
    assert projected.pop("own_discards") == state["all_discards"][state["actor"]]
    assert projected == create_adapter("full_state_v2").project(state)
    projected["hand"].clear()
    assert state == before


def test_v3_own_discards_overrides_stale_alias_without_mutation():
    state = {
        "actor": 2,
        "all_discards": [[], [], ["0m", "1z"], []],
        "own_discards": ["9p"],
        "public_events": [],
        "discard_history": [],
    }
    projected = create_adapter("full_state_v3").project(state)
    assert projected["own_discards"] == ["0m", "1z"]
    projected["own_discards"].clear()
    assert state["all_discards"][2] == ["0m", "1z"]
    assert state["own_discards"] == ["9p"]


@pytest.mark.parametrize("wrong_contract", [False, True])
def test_v3_http_checks_dimensions_and_omits_legacy_bias(wrong_contract):
    config = load_arena_config(ROOT / "configs/arena_v140_smoke.yaml")["backends"][
        "v1_4_0"
    ]
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
    models["response"].update(config["response"])
    if wrong_contract:
        models["response"]["action_dim"] = 91
    requests = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"models": models})
        payload = json.loads(request.content)
        requests.append(payload)
        action = payload["legal_actions"][0]
        return httpx.Response(
            200,
            json={
                "task": "response",
                "model": models["response"],
                "selected_action": action,
                "raw_selected_action": action,
                "selection_mode": "calibrated",
                **config["response"],
            },
        )

    client = httpx.Client(
        base_url="http://model", transport=httpx.MockTransport(handle)
    )
    if wrong_contract:
        with pytest.raises(AdapterError, match="incompatible model contract"):
            HttpPredictBackend("v3", config, client=client)
        return
    backend = HttpPredictBackend("v3", config, client=client)
    try:
        backend.predict("response", {}, [{"kind": "pass"}])
        assert "pass_bias" not in requests[0]
        assert requests[0]["action_biases"] == config["response"]["action_biases"]
    finally:
        backend.close()


def test_v3_config_rejects_legacy_bias_even_null(tmp_path):
    config = load_arena_config(ROOT / "configs/arena_v140_smoke.yaml")
    config["backends"]["v1_4_0"]["response"]["pass_bias"] = None
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="must not carry pass_bias"):
        load_arena_config(path)
