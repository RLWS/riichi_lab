from __future__ import annotations

from rilab.w3d import (
    TILE_TYPE_IDS,
    _max_remaining_candidate,
    _meld_objects,
    _paired_stats,
    _visible_counts,
    aggregate,
    evaluate_hand,
    evaluate_post_discard,
)

EMPTY_STATE = {
    "hand": {},
    "own_melds": [],
    "all_discards": [],
    "all_melds": [],
    "dora_indicators": [],
    "seat_wind": 0,
    "round_wind": 0,
    "honba": 0,
    "riichi_sticks": 0,
}


def _state(hand: dict, **overrides) -> dict:
    return {**EMPTY_STATE, "hand": hand, **overrides}


def test_tile_type_ids_cover_34_types() -> None:
    assert len(TILE_TYPE_IDS) == 34
    assert len(set(TILE_TYPE_IDS)) == 34
    assert TILE_TYPE_IDS[4] == 17  # 5m uses the non-red copy


def test_evaluate_hand_closed_tenpai_wait_and_remaining() -> None:
    hand = {
        "1m": 1,
        "2m": 1,
        "3m": 1,
        "4m": 1,
        "5m": 1,
        "6m": 1,
        "7m": 1,
        "8m": 1,
        "9m": 1,
        "1p": 1,
        "2p": 1,
        "3p": 1,
        "1s": 1,
    }
    result = evaluate_hand(_state(hand), hand, riichi=True)
    assert result["ok"] is True
    assert result["tenpai"] is True
    assert result["shanten"] == 0
    assert result["waits"] == ["1s"]
    assert result["wait_count"] == 1
    assert result["remaining"] == 3
    assert result["max_han"] is not None
    assert result["max_han"] >= 1


def test_evaluate_post_discard_uses_pre_discard_visibility() -> None:
    hand = {
        "1m": 1,
        "2m": 1,
        "3m": 1,
        "4m": 1,
        "5m": 1,
        "6m": 1,
        "7m": 1,
        "8m": 1,
        "9m": 1,
        "1p": 1,
        "2p": 1,
        "3p": 1,
        "1s": 2,
    }
    result = evaluate_post_discard(_state(hand), "1s", riichi=True)
    assert result["ok"] is True
    assert result["tenpai"] is True
    assert result["waits"] == ["1s"]
    assert result["remaining"] == 2


def test_evaluate_post_discard_rejects_missing_tile() -> None:
    hand = {"1m": 1, "2m": 1, "3m": 1}
    result = evaluate_post_discard(_state(hand), "9s", riichi=False)
    assert result["ok"] is False
    assert "discard not in hand" in result["error"]


def test_ankan_does_not_double_count_called_tile() -> None:
    ankan = {
        "kind": "ankan",
        "consumed_tiles": ["0p", "5p", "5p", "5p"],
        "called_tile": "0p",
        "source": 0,
    }
    melds = _meld_objects([ankan])
    assert len(melds) == 1
    assert len(melds[0].tiles) == 4
    visible = _visible_counts(_state({}, all_melds=[[ankan]]))
    assert sum(visible.values()) == 4
    assert visible["5p"] == 4


def test_kakan_keeps_the_added_tile() -> None:
    kakan = {
        "kind": "kakan",
        "consumed_tiles": ["1z", "1z", "1z"],
        "called_tile": "1z",
        "source": 2,
    }
    melds = _meld_objects([kakan])
    assert len(melds) == 1
    assert len(melds[0].tiles) == 4
    visible = _visible_counts(_state({}, all_melds=[[kakan]]))
    assert visible["1z"] == 4


def test_evaluate_hand_counts_visible_red_five() -> None:
    hand = {
        "1m": 1,
        "2m": 1,
        "3m": 1,
        "4m": 1,
        "5m": 2,
        "6m": 1,
        "7m": 1,
        "8m": 1,
        "9m": 1,
        "1p": 1,
        "2p": 1,
        "3p": 1,
    }
    state = _state(hand, all_discards=[["0m"]])
    result = evaluate_hand(state, hand, riichi=True)
    assert result["waits"] == ["5m"]
    assert result["remaining"] == 1


def test_visible_counts_normalizes_red_fives() -> None:
    hand_state = _state(
        {"0m": 1, "5m": 2},
        all_discards=[["5m"]],
    )
    assert _visible_counts(hand_state)["5m"] == 4
    discard_state = _state(
        {"5m": 3},
        all_discards=[["0m"]],
    )
    assert _visible_counts(discard_state)["5m"] == 4


def test_max_remaining_candidate_prefers_more_remaining() -> None:
    candidates = [
        {"tile": "1m", "remaining": 2, "max_han": 5, "wait_count": 1},
        {"tile": "2m", "remaining": 6, "max_han": 2, "wait_count": 2},
    ]
    assert _max_remaining_candidate(candidates)["tile"] == "2m"


def test_paired_stats_reports_direction() -> None:
    rows = [
        {
            "left": {"ok": True, "tenpai": True, "remaining": 5, "max_han": 3},
            "right": {"ok": True, "tenpai": True, "remaining": 4, "max_han": 3},
        },
        {
            "left": {"ok": True, "tenpai": True, "remaining": 3, "max_han": 2},
            "right": {"ok": True, "tenpai": True, "remaining": 4, "max_han": 1},
        },
    ]
    result = _paired_stats(rows, "left", "right")
    assert result["n"] == 2
    assert result["mean_delta_remaining"] == 0
    assert result["remaining_better_equal_worse"] == [1, 0, 1]
    assert result["mean_delta_han"] == 0.5
    assert result["han_better_equal_worse"] == [1, 1, 0]


def test_aggregate_builds_source_and_pair_keys() -> None:
    row = {
        "arm": "A",
        "task": "turn",
        "strata": ["riichi_selected"],
        "source_class": "riichi_discard",
        "other_class": "riichi_discard",
        "game_id": "g1",
        "hand_id": 1,
        "hand": {"win": True, "deal": False, "income": 1000, "loss": 0},
        "source_eval": {
            "ok": True,
            "tenpai": True,
            "remaining": 4,
            "max_han": 2,
            "wait_count": 1,
        },
        "riichi_candidate_eval": {
            "ok": True,
            "tenpai": True,
            "remaining": 4,
            "max_han": 2,
            "wait_count": 1,
        },
        "riichi_candidates": [{"tile": "1m"}],
    }
    result = aggregate([row])
    item = result["by_source"]["A/turn/riichi_selected"]
    assert item["n"] == 1
    assert item["source_shape"]["tenpai"] == 1
    assert item["source_vs_candidate"]["n"] == 1
    pair = result["by_source_other"]["A/turn/riichi_selected/riichi_discard/riichi_discard"]
    assert pair["n"] == 1
