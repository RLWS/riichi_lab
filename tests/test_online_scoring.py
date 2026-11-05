import json
from pathlib import Path

import pytest

from rilab.ranked_evaluation import (
    arm_summary,
    compare_displays,
    prepare_rows,
    rounded_counts,
    standardized_online,
)
from rilab.ranked_snapshot import extract_session, snapshot
from rilab.scoring import rank_score, rank_scorecard

ROOT = Path(__file__).parents[1]


def test_linear_score_is_monotonic_and_does_not_invent_strength():
    assert rank_score(1) == 100 and rank_score(4) == 0 and rank_score(2.5) == 50
    with pytest.raises(ValueError):
        rank_score(0)
    with pytest.raises(ValueError):
        rank_score(float("nan"))


def test_observed_gain_direction_and_practical_gate_are_separate():
    metric = {
        "candidate": {"value": 2.43},
        "control": {"value": 2.5},
        "ci95": [-0.17, 0.03],
    }
    last = {"ci95": [-0.05, 0.017]}
    result = rank_scorecard(metric, last, minimum_gain=0.05, guardrail=0.02)
    assert result["observed_direction"] == "improvement"
    assert result["observed_rank_gain"] == pytest.approx(0.07)
    assert result["direction_evidence"] == "uncertain"
    assert result["practical_gain"]["supported"] is False
    assert result["last_place_guardrail"]["passed"] is True
    metric["ci95"] = [-0.15, -0.01]
    result = rank_scorecard(metric, last, minimum_gain=0.05, guardrail=0.02)
    assert result["direction_evidence"] == "improvement_supported"
    assert result["practical_gain"]["supported"] is False


def test_screenshot_integer_counts_and_conditional_intervals():
    assert rounded_counts(294, [23.8, 22.1, 27.2, 26.9]) == [70, 65, 80, 79]
    assert rounded_counts(334, [28.4, 26.3, 22.8, 22.5]) == [95, 88, 76, 75]
    doc = json.loads(
        (
            ROOT / "tests/fixtures/evaluation/online_screenshots_20261105.json"
        ).read_text()
    )
    result = compare_displays(doc)
    assert result["observed_rank_gain"] == pytest.approx(0.17921300256629635)
    assert result["rank_score_gain"] == pytest.approx(5.973766752209879)
    assert result["hand_metric_intervals"] is None
    assert "unverified" in result["inference_status"]
    doc["versions"]["v1.2.0"]["displayed_mean_rank"] = 2.0
    with pytest.raises(ValueError, match="conflicts"):
        compare_displays(doc)


@pytest.mark.parametrize(
    "n,percentages", [(10000, [25, 25, 25, 25]), (294, [25, 25, 25, 25])]
)
def test_ambiguous_or_inconsistent_rounding_is_rejected(n, percentages):
    with pytest.raises(ValueError):
        rounded_counts(n, percentages)


def row(version, rank, hour, seat=0):
    return {
        "terminal": True,
        "started": True,
        "version": version,
        "rank_low": rank,
        "rank_high": rank,
        "rank_mid": rank,
        "seat": seat,
        "hour": hour,
        "points": 25000,
        "batch_status": "failed",
        "protocol_passed": False,
        "hands": 2,
        "wins": 1,
        "deal_ins": 0,
        "riichi": 0,
        "calls": 1,
    }


def test_hour_and_seat_standardization_removes_sample_mix_advantage():
    rows = (
        [row("a", 1, 1)] * 8
        + [row("b", 1, 1)] * 2
        + [row("a", 4, 2)] * 2
        + [row("b", 4, 2)] * 8
    )
    # Pooled ranks suggest a large advantage, but both versions tie in each hour.
    assert arm_summary([r for r in rows if r["version"] == "a"])[
        "mean_rank_bounds"
    ] == [1.6, 1.6]
    result = standardized_online(rows, "a", "b", iterations=100)
    assert result["rank_difference_bounds"] == [0, 0]
    assert result["conservative_hour_cluster_ci95"] == [0, 0]
    assert result["time_clusters"] == 2
    # Disjoint seats have no comparable support, even during the same hour.
    assert (
        standardized_online([row("a", 1, 1, 0), row("b", 4, 1, 1)], "a", "b")[
            "available"
        ]
        is False
    )


def test_failed_batches_completed_games_and_unknown_outcomes_are_visible():
    rows = [row("a", 1, 1), row("a", 4, 2)]
    rows.append({**row("a", 2, 3), "terminal": False})
    result = arm_summary(rows)
    assert result["terminal_games"] == 2
    assert result["nonpassing_terminal_games_retained"] == 2
    assert result["nonterminal_started"] == 1
    assert result["all_started_rank_bounds_with_missing"] == [2, 3]


def test_ties_remain_bounds_without_platform_tie_rule():
    game = {
        "source_id": "x/1",
        "match_id": None,
        "seat": 1,
        "scores": [30000, 25000, 25000, 20000],
        "ranks": None,
        "trace_mtime_utc": 3600,
        "hands": 1,
        "wins": 0,
        "deal_ins": 0,
        "riichi": 0,
        "calls": 0,
    }
    source = {
        "batches": [
            {
                "batch_id": "b",
                "backend": {"revision": {"tag": "a"}},
                "status": "failed",
                "games": [game],
            }
        ]
    }
    rows, _ = prepare_rows(source)
    assert rows[0]["rank_low"] == 2 and rows[0]["rank_high"] == 3
    game["ranks"] = [1, 4, 2, 3]
    with pytest.raises(ValueError, match="conflict"):
        prepare_rows(source)


def test_readonly_extractor_exports_no_auth_or_hidden_hands_and_deduplicates_ron(
    tmp_path,
):
    path = tmp_path / "session.jsonl"
    messages = [
        {"type": "start_game", "id": 0},
        {"type": "start_kyoku", "tehais": [["secret-hand"]]},
        {"type": "hora", "actor": 1, "target": 0},
        {"type": "hora", "actor": 2, "target": 0},
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [10000, 30000, 30000, 30000]},
    ]
    records = [{"kind": "sent", "message": {"token": "secret-auth"}}] + [
        {"kind": "received", "message": m} for m in messages
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    before = path.read_bytes()
    result = extract_session(path)
    assert result["deal_ins"] == 1 and result["hands"] == 1
    assert "secret-auth" not in json.dumps(result) and "secret-hand" not in json.dumps(
        result
    )
    assert path.read_bytes() == before


def test_snapshot_keeps_missing_trace_and_unfinished_batch(tmp_path):
    root = tmp_path / "project/outputs"
    batch = root / "b"
    (batch / "game_0001").mkdir(parents=True)
    manifest = {
        "mode": "ranked",
        "config": {"backend": {"revision": {"tag": "v1.3.0"}}},
    }
    (batch / "manifest.json").write_text(json.dumps(manifest))
    result = snapshot(root)
    assert len(result["batches"]) == 1
    assert result["read_errors"][0]["reason"] == "missing_trace"
    assert result["read_only"] is True
