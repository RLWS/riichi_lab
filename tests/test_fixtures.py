"""A0 夹具加载与覆盖验收（开发计划 A0 退出检查 1、方案 §2 夹具要求）。

固化采集时已验证的事实，防止夹具被误删/误改：
- 引擎夹具可加载、序列化往返全通过；
- 必需边界类别覆盖（chankan 例外：A0 未命中，登记为 A1/A2 定向构造待办）；
- 立直四事件序列（tsumo→reach→dahai→reach_accepted）与杠后岭上摸牌真实存在。
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
ENGINE = FIXTURES / "engine"
HTTP = FIXTURES / "http_contract"

EXCEPTED_UNCOVERED = ["chankan"]  # 登记待办：不用随机运行代替罕见边界覆盖（方案 §2）


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
        assert all(d["serialize_roundtrip_legal_count_ok"] for d in data["decisions"])
    assert index["engine"]["package"] == "riichienv"
    assert index["engine"]["rule"] == "default_tenhou"


def test_required_categories_covered() -> None:
    index = load_index()
    assert index["coverage"]["uncovered"] == EXCEPTED_UNCOVERED
    covered = index["coverage"]["covered"]
    for category in index["coverage"]["required"]:
        if category in EXCEPTED_UNCOVERED:
            assert not covered[category]
        else:
            assert covered[category], f"category lost coverage: {category}"


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
                and w[0].get("actor") == w[1].get("actor") == w[2].get("actor") == w[3].get("actor")
            ):
                hits += 1
                break
    assert hits >= 30, f"riichi sequence unexpectedly rare: {hits}"


def test_rinshan_draw_after_kan_present() -> None:
    """杠后岭上摸牌：ankan/daiminkan 后同 actor 摸牌；中间可隔一条 kan-dora 事件。

    契约顺序为"杠事件先于新宝牌事件"（full-state-api.md），故 kan->dora->tsumo 同样算岭上；
    kakan 按规则无岭上摸牌，不作要求。60 局中 2 局（seeds 8/9）无杠，故阈值取 55。
    """
    hits = 0
    for path in sorted((ENGINE / "games").glob("*.json.gz")):
        log = [json.loads(e) for e in load_game(path)["mjai_log"]]
        for i in range(len(log) - 1):
            if log[i].get("type") not in ("ankan", "daiminkan"):
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
                break
    assert hits >= 55, f"rinshan draw after kan unexpectedly rare: {hits}"


def test_http_contract_fixtures_valid() -> None:
    turn_req = json.loads((HTTP / "golden_turn_request.json").read_text(encoding="utf-8"))
    resp_req = json.loads((HTTP / "golden_response_request.json").read_text(encoding="utf-8"))
    assert turn_req["state"]["feature_schema"] == "full_state_v2"
    assert resp_req["state"]["response_target"] == {"source": 0, "tile": "3m"}

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
