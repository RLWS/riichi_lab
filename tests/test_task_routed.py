from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from rilab.adapters.task_routed import TaskRoutedAdapter
from rilab.backends.task_routed import TaskRoutedBackend
from rilab.w2 import analyze_batch, build_schedule


class FakeBackend:
    def __init__(self, backend_id, config):
        self.backend_id = backend_id
        self.profile = config["adapter_profile"]
        self.config = config
        self.metadata = {
            "turn": {"task": "turn", "feature_schema": self.profile},
            "response": {"task": "response", "feature_schema": self.profile},
        }
        self.calibration = {"fake": True}
        self.requests = 0
        self.closed = False

    def predict(self, task, state, candidates):
        self.requests += 1
        return {
            "task": task,
            "model": self.metadata[task],
            "selected_action": candidates[0],
            "raw_selected_action": candidates[0],
            "selection_mode": "calibrated",
        }

    def check_unchanged(self):
        return None

    def close(self):
        self.closed = True


def test_task_routed_adapter_profiles():
    adapter = TaskRoutedAdapter("full_state_v2", "full_state_v3")
    assert adapter.profile_for_task("turn") == "full_state_v2"
    assert adapter.profile_for_task("response") == "full_state_v3"


def test_task_routed_backend_routes_and_validates():
    config = {
        "turn_backend": {
            "adapter_profile": "full_state_v3",
            "api_base": "http://127.0.0.1:1",
            "timeout_seconds": 1.0,
            "response": {},
        },
        "response_backend": {
            "adapter_profile": "full_state_v2",
            "api_base": "http://127.0.0.1:2",
            "timeout_seconds": 1.0,
            "response": {},
        },
    }
    with patch("rilab.backends.task_routed.HttpPredictBackend", FakeBackend):
        backend = TaskRoutedBackend("B", config)
        assert backend.profile_for_task("turn") == "full_state_v3"
        assert backend.profile_for_task("response") == "full_state_v2"
        adapter = backend.create_adapter()
        backend.validate_adapter(adapter)
        turn = backend.predict("turn", {}, [{"kind": "pass"}])
        response = backend.predict("response", {}, [{"kind": "pass"}])
        assert turn["model"]["feature_schema"] == "full_state_v3"
        assert response["model"]["feature_schema"] == "full_state_v2"
        assert backend.requests == 2
        backend.check_unchanged()
        backend.close()
        assert backend.turn_backend.closed
        assert backend.response_backend.closed


def test_w2_schedule_uses_a_as_control():
    plan = {
        "smoke_seeds": [40001],
        "formal_seeds": list(range(42001, 42129)),
    }
    smoke = build_schedule(plan, plan["smoke_seeds"])
    assert len(smoke) == 16
    assert smoke[0]["seats"] == ["v130", "v130", "v130", "v130"]
    assert smoke[4]["seats"][0] == "B"
    assert smoke[8]["seats"][0] == "C"
    assert smoke[12]["seats"][0] == "D"
    formal = build_schedule(plan, plan["formal_seeds"])
    assert len(formal) == 2048


def _synthetic_batch(tmp_path: Path, plan: dict) -> Path:
    batch = tmp_path / "batch"
    batch.mkdir()
    games = []
    ranks = {
        "A": [2.5, 2.5, 2.5, 2.5],
        "B": [2.6, 2.6, 2.6, 2.6],
        "C": [2.7, 2.7, 2.7, 2.7],
        "D": [3.0, 3.0, 3.0, 3.0],
    }
    game_id = 0
    for seed in plan["formal_seeds"]:
        for arm in ("A", "B", "C", "D"):
            for rotation in range(4):
                games.append(
                    {
                        "game_id": f"g{game_id:05d}",
                        "seed": seed,
                        "matchup_id": arm,
                        "rotation": rotation,
                        "focal_seat": rotation,
                        "seats": ["v130"] * 4,
                        "ranks": ranks[arm],
                        "scores": [25000] * 4,
                        "fallbacks": 0,
                    }
                )
                game_id += 1
    (batch / "summary.json").write_text(
        json.dumps({"games": games, "stage": "W2"}), encoding="utf-8"
    )
    return batch


def test_w2_analysis_reports_d_minus_a_and_exploratory_components(tmp_path):
    plan = {
        "formal_seeds": list(range(42001, 42129)),
        "evaluation": {
            "bootstrap_resamples": 100,
            "bootstrap_seed": 20261117,
        },
    }
    batch = _synthetic_batch(tmp_path, plan)
    result = analyze_batch(batch, plan)
    assert result["completed_games"] == 2048
    assert result["primary"]["name"] == "D-A"
    assert result["primary"]["rank_difference"]["mean"] > 0
    for name in ("B-A", "C-A", "D-B", "D-C"):
        assert result["components"][name]["exploratory"] is True
        assert result["components"][name]["unadjusted"] is True
    assert result["derived"]["turn_effect"]["exploratory"] is True
