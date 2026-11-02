"""Reproducible engine setups for chankan and double ron (not complete games).

Run: python -m scripts.capture_directed
"""

from __future__ import annotations

import json
from pathlib import Path

from riichienv import ActionType, GameRule, RiichiEnv

from scripts.capture_fixtures import action_to_json, observation_state


def setup_events(kind: str) -> list[dict]:
    hands = [
        ["4m", "4m", "1m", "2m", "3m", "7p", "8p", "9p", "7s", "8s", "9s", "P", "P"],
        ["1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "S", "S", "5m", "6m"],
        ["?"] * 13,
        ["?"] * 13,
    ]
    if kind == "multi_ron":
        hands[0] = [
            "4m",
            "1m",
            "2m",
            "3m",
            "7p",
            "8p",
            "9p",
            "7s",
            "8s",
            "9s",
            "F",
            "F",
            "C",
        ]
        hands[3] = [
            "4p",
            "5p",
            "6p",
            "4s",
            "5s",
            "6s",
            "P",
            "P",
            "P",
            "S",
            "S",
            "5m",
            "6m",
        ]
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000] * 4,
            "dora_marker": "9s",
            "tehais": hands,
        }
    ]
    if kind == "chankan":
        events += [
            {"type": "dahai", "actor": 2, "pai": "4m", "tsumogiri": False},
            {
                "type": "pon",
                "actor": 0,
                "target": 2,
                "pai": "4m",
                "consumed": ["4m", "4m"],
            },
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
            {"type": "tsumo", "actor": 1, "pai": "9p"},
            {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": True},
            {"type": "tsumo", "actor": 0, "pai": "4m"},
        ]
    elif kind == "multi_ron":
        events.append({"type": "tsumo", "actor": 0, "pai": "9m"})
    else:
        raise ValueError(kind)
    return events


def capture_case(kind: str) -> dict:
    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=1)
    events = setup_events(kind)
    for event in events:
        env.apply_event(event)
    action_type = ActionType.Kakan if kind == "chankan" else ActionType.Discard
    before = env.get_observation(0)
    action = next(
        a
        for a in before.legal_actions()
        if a.action_type == action_type and json.loads(a.to_mjai()).get("pai") == "4m"
    )
    observations = env.step({0: action})
    seats = [1] if kind == "chankan" else [1, 3]
    responses = []
    wins = {}
    for seat in seats:
        obs = observations[seat]
        wins[seat] = next(
            a for a in obs.legal_actions() if a.action_type == ActionType.Ron
        )
        responses.append(
            {
                "player_id": seat,
                "observation": obs.serialize_to_base64(),
                "state": observation_state(obs),
                "legal_actions": [action_to_json(a) for a in obs.legal_actions()],
                "chosen": action_to_json(wins[seat]),
            }
        )
    env.step(wins)
    return {
        "kind": kind,
        "source": "directed_engine_setup",
        "setup_events": events,
        "trigger": action_to_json(action),
        "responses": responses,
        "mjai_log": env.mjai_log,
    }


def main() -> None:
    out = Path("tests/fixtures/engine/directed")
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("chankan", "multi_ron"):
        case = capture_case(kind)
        (out / f"{kind}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(kind, "response seats", [r["player_id"] for r in case["responses"]])
    index_path = out.parent / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["coverage"]["directed"] = {
        kind: f"directed/{kind}.json" for kind in ("chankan", "multi_ron")
    }
    index["coverage"]["uncovered"] = [
        kind
        for kind in index["coverage"]["required"]
        if not index["coverage"]["covered"][kind]
        and kind not in index["coverage"]["directed"]
    ]
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
