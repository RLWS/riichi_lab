from __future__ import annotations

from rilab.w3 import (
    decision_metrics,
    dora_tile,
    hand_metrics,
    state_features,
)


def test_dora_tile_cycles():
    assert dora_tile("9m") == "1m"
    assert dora_tile("4z") == "1z"
    assert dora_tile("7z") == "5z"


def test_state_features_dora_and_opponent_riichi():
    state = {
        "actor": 0,
        "hand": {"5z": 2, "1m": 1},
        "own_melds": [],
        "dora_indicators": ["7z"],
        "scores": [25000, 25000, 25000, 25000],
        "riichi_declared_by_player": [False, True, False, False],
        "dealer": 1,
        "turn_number": 4,
        "round_number": 0,
        "remaining_tiles": 60,
    }
    features = state_features(state)
    assert features["dora_count"] == 2
    assert features["opp_riichi_count"] == 1
    assert features["is_dealer"] is False


def test_hand_metrics_open_riichi_win_and_deal():
    events = [
        {"type": "start_kyoku"},
        {"type": "reach_accepted", "actor": 0},
        {"type": "chi", "actor": 0},
        {"type": "hora", "actor": 0, "target": 0, "deltas": [1000, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "start_kyoku"},
        {"type": "reach_accepted", "actor": 1},
        {"type": "chi", "actor": 0},
        {"type": "hora", "actor": 1, "target": 0, "deltas": [-1000, 1000, 0, 0]},
        {"type": "end_kyoku"},
    ]
    hands = hand_metrics(events, 0)
    assert len(hands) == 2
    assert hands[0]["open"] and hands[0]["riichi"] and hands[0]["win"]
    assert hands[0]["income"] == 1000
    assert hands[1]["deal"] and hands[1]["loss"] == 1000
    assert hands[1]["calls_after_opp_riichi"] == 1
    assert hands[0]["calls_after_opp_riichi"] == 0


def test_decision_metrics_riichi_and_ron_divergence():
    state = {
        "actor": 0,
        "hand": {"1m": 1},
        "own_melds": [],
        "dora_indicators": [],
        "scores": [25000] * 4,
        "riichi_declared_by_player": [False, True, False, False],
        "dealer": 0,
        "turn_number": 3,
        "round_number": 0,
        "remaining_tiles": 60,
    }
    records = [
        {
            "player_id": 0,
            "task": "turn",
            "state": state,
            "candidates": [
                {"kind": "discard", "tile": "1m"},
                {"kind": "riichi_discard", "tile": "1m"},
            ],
            "selected_action": {"kind": "discard", "tile": "1m"},
        },
        {
            "player_id": 0,
            "task": "response",
            "state": state,
            "candidates": [
                {"kind": "pass"},
                {"kind": "ron", "tile": "1m"},
            ],
            "selected_action": {"kind": "pass"},
        },
    ]
    metrics = decision_metrics(records, 0)
    assert metrics["turn"]["riichi_legal"] == 1
    assert metrics["turn"]["riichi_pass"] == 1
    assert metrics["response"]["ron_legal"] == 1
    assert metrics["response"]["missed_ron"] == 1
