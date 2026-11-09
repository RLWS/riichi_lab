from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest
import yaml

from rilab.backends.http_backend import HttpPredictBackend
from rilab.config import PROFILES, load_arena_config, validate_expected_calibration
from rilab.decision import AdapterError
from rilab.screening import (
    BASELINE_ID,
    ENGINEERING_SEEDS,
    SCREENING_SEEDS,
    bootstrap_screening,
    build_native_config,
    build_native_protocol,
    empty_counts,
    load_handoff,
    load_service_configs,
    recommendation,
    validate_handoff,
    validate_screening_plan,
)

ROOT = Path(__file__).parents[1]
HANDOFF = ROOT.parent / "train_mj/configs/experiments/v1_4_0_step1"
PLAN = HANDOFF / "screening_plan.yaml"
if not PLAN.is_file():
    pytest.skip(
        "v1.4.0 handoff config is not present next to this checkout",
        allow_module_level=True,
    )


def new_backend_config():
    return {
        "mode": "http",
        "api_base": "http://new",
        "adapter_profile": "full_state_v2_action_biases",
        "calibration_contract": "action_biases_v1",
        "revision": {"commit": PROFILES["full_state_v2_action_biases"]["commit"]},
        "timeout_seconds": 30.0,
        "response": {
            "action_biases": {"ron": 0.0, "kan": 0.4, "pon": 1.1, "chi": 1.3},
            "no_yaku_call_penalty": 0.3,
        },
        "model_identity": {
            "turn_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        },
    }


def new_models_metadata():
    return {
        "models": {
            "turn": {
                "feature_schema": "full_state_v2",
                "task": "turn",
                "state_dim": 11969,
                "action_dim": 89,
                "candidate_limit": 64,
            },
            "response": {
                "feature_schema": "full_state_v2",
                "task": "response",
                "state_dim": 12012,
                "action_dim": 91,
                "candidate_limit": 16,
                "action_biases": {"ron": 0.0, "kan": 0.4, "pon": 1.1, "chi": 1.3},
                "no_yaku_call_penalty": 0.3,
            },
        }
    }


def test_handoff_spec_is_frozen_and_native_configs_are_generated():
    plan = load_handoff(PLAN)
    validate_handoff(plan, handoff_dir=HANDOFF)
    services = load_service_configs(plan, HANDOFF)
    assert services["baseline"]["sha256"]
    assert set(services["candidates"]) == {
        "A_more_calls",
        "B_balanced",
        "C_fewer_calls",
    }
    for stage, seeds, budget in (
        ("smoke", ENGINEERING_SEEDS, 16),
        ("screen", SCREENING_SEEDS, 1024),
    ):
        for candidate in services["candidates"]:
            config = build_native_config(
                plan, stage=stage, candidate_id=candidate
            )
            protocol = build_native_protocol(
                plan, stage=stage, candidate_id=candidate
            )
            schedule = validate_screening_plan(protocol, config)
            assert len(schedule) == budget
            assert config["engine"]["seeds"] == seeds
            assert config["backends"][candidate]["response"]["action_biases"] == (
                next(c for c in plan["candidates"] if c["id"] == candidate)[
                    "action_biases"
                ]
            )
            assert "pass_bias" not in config["backends"][candidate]["response"]
            assert config["backends"][BASELINE_ID]["response"]["pass_bias"] == 1.5


def test_changed_handoff_parameters_seeds_or_budget_are_rejected():
    plan = load_handoff(PLAN)
    for fault in ("bias", "seed", "budget", "penalty", "tag"):
        changed = deepcopy(plan)
        if fault == "bias":
            changed["candidates"][0]["action_biases"]["chi"] += 0.1
        elif fault == "seed":
            changed["seeds"]["screening"].pop()
        elif fault == "budget":
            changed["design"]["screening_half_games_per_candidate"] -= 1
        elif fault == "penalty":
            changed["candidates"][1]["no_yaku_call_penalty"] = 0.0
        else:
            changed["source"]["candidate"]["release_tag_exists"] = True
        with pytest.raises(ValueError):
            validate_handoff(changed, handoff_dir=HANDOFF)


@pytest.mark.parametrize(
    "fault",
    ["pass_bias_null", "pass_bias_scalar", "missing_key", "unknown", "negative", "nan"],
)
def test_new_profile_rejects_legacy_or_incomplete_calibration(fault, tmp_path):
    config = {
        "config_version": 1,
        "engine": {"game_mode": "4p-red-half", "rule": "default_tenhou", "seeds": [1]},
        "backends": {
            "candidate": new_backend_config(),
        },
        "matchups": [
            {
                "id": "control",
                "seats": ["candidate"] * 4,
                "focal_seat": 0,
                "rotation": "cyclic4",
                "paired_group": "upgrade",
            }
        ],
        "evaluation": {
            "primary_paired_group": "upgrade",
            "cluster": "seed_with_all_rotations",
            "confidence_level": 0.95,
        },
    }
    response = config["backends"]["candidate"]["response"]
    if fault == "pass_bias_null":
        response["pass_bias"] = None
    elif fault == "pass_bias_scalar":
        response["pass_bias"] = 1.5
    elif fault == "missing_key":
        response["action_biases"].pop("chi")
    elif fault == "unknown":
        response["action_biases"]["foo"] = 1.0
    elif fault == "negative":
        response["action_biases"]["ron"] = -0.1
    else:
        response["action_biases"]["ron"] = float("nan")
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_arena_config(path)


def test_commit_pinned_profile_must_not_claim_a_tag(tmp_path):
    config = {
        "config_version": 1,
        "engine": {"game_mode": "4p-red-half", "rule": "default_tenhou", "seeds": [1]},
        "backends": {"candidate": new_backend_config()},
        "matchups": [
            {
                "id": "control",
                "seats": ["candidate"] * 4,
                "focal_seat": 0,
                "rotation": "cyclic4",
                "paired_group": "upgrade",
            }
        ],
        "evaluation": {
            "primary_paired_group": "upgrade",
            "cluster": "seed_with_all_rotations",
            "confidence_level": 0.95,
        },
    }
    config["backends"]["candidate"]["revision"]["tag"] = "v1.4.0"
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_arena_config(path)


def test_http_backend_builds_exactly_one_contract_request_and_checks_echo():
    config = new_backend_config()
    metadata = new_models_metadata()
    requests = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=metadata)
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "task": "response",
                "selection_mode": "calibrated",
                "action_biases": payload["action_biases"],
                "no_yaku_call_penalty": 0.3,
                "selected_action": payload["legal_actions"][0],
                "raw_selected_action": payload["legal_actions"][0],
                "predictions": [],
                "model": metadata["models"]["response"],
            },
        )

    client = httpx.Client(base_url="http://new", transport=httpx.MockTransport(handle))
    backend = HttpPredictBackend("candidate", config, client=client)
    backend.predict("response", {}, [{"kind": "pass"}])
    assert requests[0]["action_biases"] == config["response"]["action_biases"]
    assert "pass_bias" not in requests[0]
    assert backend.calibration["action_biases"] == config["response"]["action_biases"]
    backend.close()


def test_http_backend_rejects_cross_contract_echo_or_metadata():
    config = new_backend_config()
    metadata = new_models_metadata()
    metadata["models"]["response"]["default_pass_bias"] = 1.5
    client = httpx.Client(
        base_url="http://new",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=metadata)),
    )
    with pytest.raises(AdapterError, match="default_pass_bias"):
        HttpPredictBackend("candidate", config, client=client)
    assert client.is_closed

    metadata = new_models_metadata()
    config = new_backend_config()

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=metadata)
        return httpx.Response(
            200,
            json={
                "task": "response",
                "selection_mode": "calibrated",
                "pass_bias": 1.5,
                "action_biases": config["response"]["action_biases"],
                "no_yaku_call_penalty": 0.3,
                "selected_action": {"kind": "pass"},
                "raw_selected_action": {"kind": "pass"},
                "predictions": [],
                "model": metadata["models"]["response"],
            },
        )

    client = httpx.Client(base_url="http://new", transport=httpx.MockTransport(handle))
    backend = HttpPredictBackend("candidate", config, client=client)
    with pytest.raises(AdapterError, match="pass_bias"):
        backend.predict("response", {}, [{"kind": "pass"}])
    backend.close()


def test_expected_calibration_helper_rejects_mixed_contracts():
    backend = new_backend_config()
    with pytest.raises(ValueError):
        validate_expected_calibration(
            backend,
            {
                "pass_bias": 1.5,
                "no_yaku_call_penalty": 0.3,
            },
        )
    validate_expected_calibration(
        backend,
        {
            "calibration_contract": "action_biases_v1",
            "action_biases": backend["response"]["action_biases"],
            "no_yaku_call_penalty": 0.3,
        },
    )


def test_bootstrap_uses_the_same_seed_indices_and_recommendation_rule():
    clusters = {}
    for candidate_id, rank_delta in (
        ("A_more_calls", 0.0),
        ("B_balanced", 0.1),
        ("C_fewer_calls", -0.1),
    ):
        rows = []
        for seed in range(8):
            candidate = empty_counts()
            control = empty_counts()
            candidate.update(games=4, rank_sum=int(10 + rank_delta * 4), hands=20)
            control.update(games=4, rank_sum=10, hands=20)
            rows.append({"candidate": candidate, "control": control})
        clusters[candidate_id] = rows
    result = bootstrap_screening(clusters, iterations=200, rng_seed=20261109)
    for candidate_id in clusters:
        assert result[candidate_id]["differences"]["mean_rank"]["ci95"]
    cross = result["cross_candidate"]["A_more_calls_minus_C_fewer_calls"][
        "mean_rank"
    ]
    assert cross["value"] > 0
    groups = {}
    for candidate_id in clusters:
        groups[candidate_id] = {
            "differences": result[candidate_id]["differences"],
            "focused": {
                "candidate": {
                    "mean_final_points": {"value": 25000.0},
                }
            },
        }
    rec = recommendation(groups, engineering_ok=True, artifact_ok=True)
    assert rec["selected"] == "C_fewer_calls"
    assert rec["status"] == "pending_independent_confirmation"
    assert recommendation(
        groups, engineering_ok=False, artifact_ok=True
    )["status"] == "no_recommendation"


def test_service_configs_match_the_plan():
    plan = load_handoff(PLAN)
    services = load_service_configs(plan, HANDOFF)
    assert services["baseline"]["values"]["models"]["response"]["pass_bias"][
        "default"
    ] == 1.5
    for candidate_id, item in services["candidates"].items():
        expected = next(c for c in plan["candidates"] if c["id"] == candidate_id)
        assert item["values"]["models"]["response"]["action_biases"] == expected[
            "action_biases"
        ]
        assert item["values"]["models"]["response"]["no_yaku_call_penalty"] == 0.3
