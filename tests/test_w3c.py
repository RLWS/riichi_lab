from __future__ import annotations

from rilab.w3c import _stats, aggregate


def _row(source_class="discard", other_class="riichi_discard"):
    return {
        "arm": "A",
        "task": "turn",
        "stratum": "riichi_legal_pass",
        "strata": ["riichi_legal_pass"],
        "source_class": source_class,
        "other_class": other_class,
        "game_id": "g1",
        "hand_id": 1,
        "outcome_missing": False,
        "hand": {
            "win": True,
            "deal": False,
            "income": 1000,
            "loss": 0,
            "riichi": False,
            "open": False,
        },
    }


def test_stats_counts_hands_and_income():
    rows = [_row(), _row()]
    stats = _stats(rows)
    assert stats["n"] == 2
    assert stats["unique_hands"] == 1
    assert stats["wins"] == 2
    assert stats["income_per_hand"] == 1000
    assert stats["win_rate"] == 1.0


def test_aggregate_by_source_and_other():
    result = aggregate([_row()])
    assert result["matched"] == 1
    assert result["by_source"]["A/turn/riichi_legal_pass/discard"]["n"] == 1
    assert (
        result["by_source_other"]["A/turn/riichi_legal_pass/discard/riichi_discard"][
            "n"
        ]
        == 1
    )
