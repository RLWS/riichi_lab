import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from riichienv import Observation

from rilab.adapters import create_adapter
from rilab.cli import cmd_parity
from rilab.decision import DecisionContext
from rilab.mapping.action_mapping import bind_candidates
from rilab.parity import convert_event
from rilab.tracker.state_tracker import StateTracker
from scripts.capture_directed import capture_case

ROOT = Path(__file__).parents[1] / "tests/fixtures"


@pytest.mark.parametrize("profile", ["legacy_v1_2", "full_state_v2"])
def test_frozen_xml_states_replay(profile):
    adapter = create_adapter(profile)
    decisions = 0
    for path in sorted((ROOT / "parity" / profile).glob("*.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        assert data["differences"] == []
        trackers = [StateTracker(i) for i in range(4)]
        pre_reach = {}
        for entry in data["record"]:
            if "match_rounds" in entry:
                for tracker in trackers:
                    tracker.match_rounds = entry["match_rounds"]
            elif "event" in entry:
                tracker = trackers[entry["seat"]]
                event = entry["event"]
                if event["type"] == "start_kyoku":
                    assert "?" not in event["tehais"][entry["seat"]]
                    assert all(
                        hand == ["?"] * 13
                        for p, hand in enumerate(event["tehais"])
                        if p != entry["seat"]
                    )
                if event["type"] == "reach" and event["actor"] == entry["seat"]:
                    pre_reach[entry["seat"]] = tracker.snapshot()
                tracker.apply(event)
            elif "confirmed_missed_ron" in entry:
                trackers[entry["confirmed_missed_ron"]].confirm_missed_ron()
            else:
                decision = entry["decision"]
                tracker = trackers[decision["actor"]]
                if (
                    decision["task"] == "turn"
                    and decision["label"]["kind"] == "riichi_discard"
                ):
                    state = pre_reach[decision["actor"]]
                else:
                    chankan = (
                        decision["task"] == "response"
                        and tracker.last_event["type"] == "kakan"
                    )
                    target = (
                        tracker.kan_target
                        if chankan
                        else {
                            "source": tracker.last_discard_actor,
                            "tile": tracker.last_discard_tile,
                        }
                    )
                    context = DecisionContext(
                        decision["task"],
                        tracker.hand_id,
                        tracker.player_id,
                        target["source"],
                        target["tile"],
                        chankan,
                    )
                    state = tracker.snapshot(context)
                assert adapter.project(state) == decision["state"], (path, decisions)
                decisions += 1
    assert decisions == 14038


def test_converter_masks_private_tiles_and_preserves_exact_red_kan():
    event = SimpleNamespace(event_type="draw", actor=2, payload={"tile": "0m"})
    assert convert_event(event, 1) == {"type": "tsumo", "actor": 2, "pai": "?"}
    assert convert_event(event, 2)["pai"] == "5mr"
    event = SimpleNamespace(
        event_type="meld",
        actor=0,
        payload={
            "kind": "kakan",
            "called_tile": "5m",
            "source": 2,
            "consumed_tiles": ["5m", "5m", "0m"],
        },
    )
    assert convert_event(event, 1) == {
        "type": "kakan",
        "actor": 0,
        "pai": "5mr",
        "consumed": ["5m"] * 3,
    }


def test_red_chankan_fixture_reproduces_and_keeps_actual_legal_tile():
    saved = json.loads((ROOT / "engine/directed/red_chankan.json").read_text())
    assert capture_case("red_chankan") == saved
    obs = Observation.deserialize_from_base64(saved["responses"][0]["observation"])
    tracker = StateTracker(1)
    tracker.consume(obs.new_events())
    context = tracker.context(obs.legal_actions())
    state = tracker.snapshot(context)
    assert state["response_target"] == {"source": 0, "tile": "0m"}
    assert state["all_melds"][0][0]["called_tile"] == "5m"
    binding = next(
        b
        for b in bind_candidates(obs, context, state)
        if b.candidate["kind"] == "chankan"
    )
    assert binding.candidate["tile"] == "0m"


def test_known_defect_does_not_turn_parity_gate_green(capsys):
    with patch("rilab.parity_runner.run_parity", return_value={"valid": False}):
        assert cmd_parity(SimpleNamespace(config="unused", output_dir="unused")) == 1
    assert '"valid": false' in capsys.readouterr().out


def test_runner_keeps_known_failure_with_passing_http_contract(tmp_path, monkeypatch):
    import zipfile
    import yaml
    from rilab.config import PROFILES
    from rilab.parity_runner import run_parity

    root = tmp_path / "train"
    (root / "tests/fixtures/tenhou").mkdir(parents=True)
    config = tmp_path / "parity.yaml"
    config.write_text(
        yaml.safe_dump(
            {"parity": {"train_mj_root": str(root), "profiles": ["full_state_v2"]}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rilab.parity_runner.subprocess.check_output",
        lambda *a, **k: PROFILES["full_state_v2"]["commit"] + "\n",
    )

    def run(command, **kwargs):
        if command[0] == "git":
            with zipfile.ZipFile(command[command.index("--output") + 1], "w"):
                pass
            return SimpleNamespace(returncode=0)
        if "check_xml_parity.py" in command[1]:
            out = Path(command[command.index("--output-dir") + 1])
            out.mkdir(parents=True)
            (out / "summary.json").write_text(
                json.dumps(
                    {
                        "valid": False,
                        "xml_valid": True,
                        "boundary": {"defect": "DEF-001", "valid": False},
                    }
                )
            )
            return SimpleNamespace(returncode=1)
        out = Path(command[command.index("--output") + 1])
        out.write_text(json.dumps({"valid": True, "checks": []}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("rilab.parity_runner.subprocess.run", run)
    report = run_parity(config, tmp_path / "report")
    assert report["valid"] is False
    assert report["profiles"][0]["http_contract"]["valid"] is True
    assert report["profiles"][0]["boundary"]["defect"] == "DEF-001"
