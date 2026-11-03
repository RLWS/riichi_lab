from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from rilab.config import load_arena_config
from rilab.strength import register_protocol, validate_plan

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/arena_strength.yaml"
PLAN = ROOT / "configs/strength_protocol.yaml"


@pytest.fixture
def planned():
    return yaml.safe_load(PLAN.read_text(encoding="utf-8")), load_arena_config(CONFIG)


def test_fixed_independent_seed_budget(planned):
    plan, config = planned
    schedule = validate_plan(plan, config)
    assert len(schedule) == 1792
    assert len(set(g["seed"] for g in schedule)) == 128
    assert not set(range(1, 9)) & set(config["engine"]["seeds"])


@pytest.mark.parametrize(
    "fault", ["draft", "seed", "budget", "overlap", "gain", "guard", "stop"]
)
def test_reject_changed_or_incomplete_plan(planned, fault):
    plan, config = planned
    if fault == "draft":
        plan["status"] = "draft"
    elif fault == "seed":
        config["engine"]["seeds"].pop()
    elif fault == "budget":
        plan["design"]["half_game_budget"] -= 1
    elif fault == "overlap":
        plan["design"]["excluded_engineering_or_calibration_seeds"].append(10001)
    elif fault == "gain":
        plan["decision"]["minimum_meaningful_rank_gain"] = 0
    elif fault == "guard":
        plan["decision"]["last_place_rate_guardrail"] = float("nan")
    else:
        plan["stopping"]["no_peeking_to_change_budget_or_models"] = False
    with pytest.raises(ValueError):
        validate_plan(plan, config)


def deployed(config):
    runtime = deepcopy(config)
    metadata = {}
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    for i, (name, b) in enumerate(runtime["backends"].items()):
        b["api_base"] = f"http://127.0.0.1:{12000 + i}"
        b["deployment"] = {
            "source_dirty": False,
            "verified_weight_sha256": b["model_identity"],
        }
        c = plan["expected_calibration"][name]
        metadata[name] = {
            "response": {
                "default_pass_bias": c["pass_bias"],
                "no_yaku_call_penalty": c["no_yaku_call_penalty"],
            }
        }
    return runtime, metadata


def test_registration_seals_deployment_before_batch(planned, tmp_path):
    _, config = planned
    runtime, metadata = deployed(config)
    record = register_protocol(PLAN, CONFIG, runtime, metadata, tmp_path)
    path = tmp_path / "protocol.registered.json"
    assert json.loads(path.read_text()) == record
    ref = runtime["evaluation"]["strength_registration"]
    assert ref["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["status"] == "registered_before_launch"
    assert not (tmp_path / "batch").exists()
    assert record["planned_config"] == config


@pytest.mark.parametrize("fault", ["calibration", "weight", "dirty", "late", "matchup"])
def test_refuse_bad_deployment_or_late_registration(planned, tmp_path, fault):
    _, config = planned
    runtime, metadata = deployed(config)
    if fault == "calibration":
        metadata["v1_2_0"]["response"]["default_pass_bias"] = 0
    elif fault == "weight":
        runtime["backends"]["v1_2_0"]["deployment"]["verified_weight_sha256"] = {}
    elif fault == "dirty":
        runtime["backends"]["v1_2_0"]["deployment"]["source_dirty"] = True
    elif fault == "late":
        (tmp_path / "batch").mkdir()
    else:
        runtime["matchups"][2]["seats"] = list(
            reversed(runtime["matchups"][2]["seats"])
        )
    with pytest.raises(ValueError):
        register_protocol(PLAN, CONFIG, runtime, metadata, tmp_path)
    assert not (tmp_path / "protocol.registered.json").exists()


def test_launcher_dry_run_never_calls_services_or_creates_outputs(
    tmp_path, monkeypatch
):
    import scripts.run_strength as launcher

    monkeypatch.setattr(
        launcher, "run_integration", lambda _: pytest.fail("must not launch")
    )
    out = tmp_path / "dry"
    launcher.main(["--output-dir", str(out), "--dry-run"])
    assert not out.exists()


def test_strength_launcher_always_selects_one_batch(tmp_path, monkeypatch):
    import scripts.run_strength as launcher

    calls = []
    monkeypatch.setattr(launcher, "run_integration", calls.append)
    launcher.main(["--output-dir", str(tmp_path / "not_run")])
    assert len(calls) == 1
    args = calls[0]
    assert args[args.index("--batch-runs") + 1] == "1"
    assert "--strength-protocol" in args
    assert args[args.index("--batch-timeout") + 1] == "86400"
    assert not (tmp_path / "not_run").exists()
