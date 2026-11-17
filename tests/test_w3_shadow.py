from __future__ import annotations

from rilab.w3_shadow import (
    action_class,
    expected_source_backend,
    normalize_state,
)


def _state():
    return {
        "actor": 1,
        "feature_schema": "full_state_v2",
        "all_discards": [["1m"], ["2m", "3m"], [], []],
        "own_discards": ["9m"],
    }


def test_normalize_state_keeps_public_v2_contract():
    v2 = normalize_state(_state(), "full_state_v2")
    v3 = normalize_state(_state(), "full_state_v3")
    assert v2["feature_schema"] == "full_state_v2"
    assert "own_discards" not in v2
    assert v3["feature_schema"] == "full_state_v2"
    assert v3["own_discards"] == ["2m", "3m"]


def test_expected_source_backend():
    assert expected_source_backend("A", "turn") == "v130"
    assert expected_source_backend("B", "turn") == "v140"
    assert expected_source_backend("C", "response") == "v140"
    assert expected_source_backend("D", "response") == "v140"
    assert expected_source_backend("B", "response") == "v130"


def test_action_class():
    assert action_class("response", {"kind": "chi"}) == "chi"
    assert action_class("turn", {"kind": "riichi_discard"}) == "riichi_discard"
