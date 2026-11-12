import json
from pathlib import Path

import pytest
import yaml

from rilab.config import load_arena_config
from rilab.scheduling import expand_schedule
from scripts.verify_v140_engineering import verify

CONFIG = Path(__file__).parents[1] / "configs/arena_v140_engineering.yaml"


def test_engineering_schedule_has_complete_rotations_per_seed():
    config = load_arena_config(CONFIG)
    schedule = expand_schedule(config)
    assert len(schedule) == 112
    for seed in config["engine"]["seeds"]:
        rows = [row for row in schedule if row["seed"] == seed]
        assert len(rows) == 14
        for matchup, count in (
            ("candidate_vs_baseline", 4),
            ("baseline_control", 4),
            ("two_vs_two", 6),
        ):
            assert {r["rotation"] for r in rows if r["matchup_id"] == matchup} == set(
                range(count)
            )


def test_verifier_rejects_smoke_budget(tmp_path):
    config = load_arena_config(CONFIG)
    config["engine"]["seeds"] = [40001]
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="eight-seed"):
        verify(tmp_path, path)


@pytest.mark.parametrize(
    "fault", ["valid", "replay_identical", "runs", "games_per_run"]
)
def test_verifier_rejects_incomplete_or_different_runs(tmp_path, fault):
    launch = {"valid": True, "replay_identical": True, "runs": 2, "games_per_run": 112}
    launch[fault] = False if fault in ("valid", "replay_identical") else 1
    (tmp_path / "verification.json").write_text(json.dumps(launch), encoding="utf-8")
    with pytest.raises(ValueError, match="two complete identical runs"):
        verify(tmp_path, CONFIG)
