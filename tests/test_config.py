from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from rilab.config import load_arena_config

CONFIG = Path(__file__).parents[1] / "configs/arena_selfplay.yaml"


def test_versioned_backend_profiles_and_per_backend_calibration(tmp_path):
    config = load_arena_config(CONFIG)
    assert config["backends"]["v1_2_0"]["adapter_profile"] == "legacy_v1_2"
    assert config["backends"]["v1_3_0"]["adapter_profile"] == "full_state_v2"
    config["backends"]["v1_2_0"]["response"]["pass_bias"] = 0
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = load_arena_config(path)
    assert result["backends"]["v1_2_0"]["response"]["pass_bias"] == 0
    assert result["backends"]["v1_3_0"]["response"]["pass_bias"] is None


@pytest.mark.parametrize(
    "fault",
    [
        "reference",
        "seat_count",
        "profile",
        "revision",
        "hash",
        "endpoint",
        "duplicate_id",
        "rotation",
        "bias",
    ],
)
def test_invalid_backend_contract_rejected(fault, tmp_path):
    config = deepcopy(load_arena_config(CONFIG))
    backend = config["backends"]["v1_3_0"]
    if fault == "reference":
        config["matchups"][0]["seats"][0] = "unknown"
    elif fault == "seat_count":
        config["matchups"][0]["seats"].pop()
    elif fault == "profile":
        backend["adapter_profile"] = "legacy_v1_2"
    elif fault == "revision":
        backend["revision"]["tag"] = "master"
    elif fault == "hash":
        backend["model_identity"]["turn_sha256"] = ""
    elif fault == "endpoint":
        backend["api_base"] = config["backends"]["v1_2_0"]["api_base"]
    elif fault == "duplicate_id":
        config["matchups"][1]["id"] = config["matchups"][0]["id"]
    elif fault == "rotation":
        config["matchups"][0]["rotation"] = "unique6"
    elif fault == "bias":
        backend["response"]["pass_bias"] = -1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_arena_config(path)
