"""A0 夹具加载与覆盖验收（开发计划 A0 退出检查 1、方案 §2 夹具要求）。

固化采集时已验证的事实，防止夹具被误删/误改：
- 引擎夹具可加载、序列化往返全通过；
- 必需边界类别覆盖（抢杠和双响由定向引擎设置补齐）；
- 立直四事件序列（tsumo→reach→dahai→reach_accepted）与杠后岭上摸牌真实存在。
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from riichienv import Observation

from scripts.capture_fixtures import action_to_json, observation_state, capture_game
from scripts.capture_directed import capture_case

FIXTURES = Path(__file__).parent / "fixtures"
ENGINE = FIXTURES / "engine"
HTTP = FIXTURES / "http_contract"


def load_index() -> dict:
    return json.loads((ENGINE / "index.json").read_text(encoding="utf-8"))


def load_game(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def test_engine_fixtures_load_and_roundtrip() -> None:
    index = load_index()
    games = [g for g in index["games"] if "error" not in g]
    files = sorted((ENGINE / "games").glob("*.json.gz"))
    assert len(files) == len(games) == 60
    for game in games:
        data = load_game(ENGINE / "games" / game["file"])
        assert data["decisions"], game["file"]
        assert data["mjai_log"], game["file"]
        assert game["serialize_roundtrip_ok"] is True
        snapshots = 0
        for decision in data["decisions"]:
            actions = decision["legal_actions"]
            assert decision["chosen"] in actions
            for action in actions:
                assert isinstance(action["type"], str) and action["type"]
                assert "engine" in action
                if action["type"] == "dahai":
                    assert action["pai"] and action["engine"]["tile"] is not None
                if action["type"] in ("chi", "pon", "daiminkan", "ankan", "kakan"):
                    assert action["consumed"] and action["engine"]["consume_tiles"]
            if "observation" in decision:
                snapshots += 1
                obs = Observation.deserialize_from_base64(decision["observation"])
                assert obs.player_id == decision["player_id"]
                assert list(obs.new_events()) == decision["new_events"]
                assert obs.drawn_tile == decision["drawn_tile"]
                assert observation_state(obs) == decision["observation_state"]
                assert [action_to_json(a) for a in obs.legal_actions()] == actions
        assert snapshots >= 4, game["file"]
    assert index["engine"]["package"] == "riichienv"
    assert index["engine"]["rule"] == "default_tenhou"


def test_required_categories_covered() -> None:
    index = load_index()
    assert index["coverage"]["uncovered"] == []
    covered = index["coverage"]["covered"]
    for category in index["coverage"]["required"]:
        assert covered[category] or category in index["coverage"]["directed"], category


def test_seed_reproduces_decisions_and_results() -> None:
    saved = load_game(ENGINE / "games/game_0001.json.gz")
    actual = capture_game(1)
    for key in ("decisions", "mjai_log", "scores", "ranks"):
        assert actual[key] == saved[key], key


@pytest.mark.parametrize("kind", ["chankan", "multi_ron"])
def test_directed_windows_are_reproducible(kind: str) -> None:
    path = ENGINE / load_index()["coverage"]["directed"][kind]
    saved = json.loads(path.read_text(encoding="utf-8"))
    actual = capture_case(kind)
    assert actual == saved
    assert [r["player_id"] for r in saved["responses"]] == (
        [1] if kind == "chankan" else [1, 3]
    )
    wins = [e for e in saved["mjai_log"] if e["type"] == "hora"]
    assert len(wins) == len(saved["responses"])


def test_riichi_four_event_sequence_present() -> None:
    hits = 0
    for path in sorted((ENGINE / "games").glob("*.json.gz")):
        log = [json.loads(e) for e in load_game(path)["mjai_log"]]
        for i in range(len(log) - 3):
            w = log[i : i + 4]
            if (
                w[0].get("type") == "tsumo"
                and w[1].get("type") == "reach"
                and w[2].get("type") == "dahai"
                and w[3].get("type") == "reach_accepted"
                and w[0].get("actor")
                == w[1].get("actor")
                == w[2].get("actor")
                == w[3].get("actor")
            ):
                hits += 1
                break
    assert hits >= 30, f"riichi sequence unexpectedly rare: {hits}"


def test_rinshan_draw_after_kan_present() -> None:
    """杠后岭上摸牌：ankan/daiminkan 后同 actor 摸牌；中间可隔一条 kan-dora 事件。

    契约顺序为"杠事件先于新宝牌事件"（full-state-api.md），故 kan->dora->tsumo 同样算岭上；
    加杠未被抢和时同样有岭上摸牌；三类杠分别检查，防止遗漏。
    """
    hits = 0
    kan_types = set()
    for path in sorted((ENGINE / "games").glob("*.json.gz")):
        log = [json.loads(e) for e in load_game(path)["mjai_log"]]
        for i in range(len(log) - 1):
            if log[i].get("type") not in ("ankan", "daiminkan", "kakan"):
                continue
            follow = log[i + 1 :]
            if follow and follow[0].get("type") == "dora":
                follow = follow[1:]
            if (
                follow
                and follow[0].get("type") == "tsumo"
                and follow[0].get("actor") == log[i].get("actor")
            ):
                hits += 1
                kan_types.add(log[i]["type"])
    assert hits > 0
    assert kan_types == {"ankan", "daiminkan", "kakan"}


def test_http_contract_fixtures_valid() -> None:
    turn_req = json.loads(
        (HTTP / "golden_turn_request.json").read_text(encoding="utf-8")
    )
    resp_req = json.loads(
        (HTTP / "golden_response_request.json").read_text(encoding="utf-8")
    )
    assert turn_req["state"]["feature_schema"] == "full_state_v2"
    assert resp_req["state"]["response_target"] == {"source": 0, "tile": "3m"}
    for action in turn_req["legal_actions"]:
        if action["kind"] in ("discard", "riichi_discard"):
            assert turn_req["state"]["hand"].get(action["tile"], 0) > 0

    models = json.loads((HTTP / "baseline_models.json").read_text(encoding="utf-8"))
    for task in ("turn", "response"):
        info = models["models"]["models"][task]
        assert info["feature_schema"] == "full_state_v2"
        assert info["candidate_limit"] > 0

    for name, req in (
        ("baseline_turn_response.json", turn_req),
        ("baseline_response_response.json", resp_req),
    ):
        baseline = json.loads((HTTP / name).read_text(encoding="utf-8"))
        legal = {json.dumps(a, sort_keys=True) for a in req["legal_actions"]}
        selected = baseline["selected_action"]
        assert json.dumps(selected, sort_keys=True) in legal


def test_http_requests_come_from_engine_observations() -> None:
    from scripts.capture_http_requests import initial_request

    source = json.loads((HTTP / "engine_source.json").read_text(encoding="utf-8"))
    for task in ("turn", "response"):
        obs = Observation.deserialize_from_base64(source[f"{task}_observation"])
        request = json.loads(
            (HTTP / f"golden_{task}_request.json").read_text(encoding="utf-8")
        )
        assert initial_request(obs, response=task == "response") == request
