from __future__ import annotations

import json
from pathlib import Path

import pytest

from rilab.w3e import (
    ARM_IDS,
    build_schedule,
    capture_filter,
    count_events,
    fisher_exact_two_sided,
    load_plan,
    newcombe_difference,
    primary_decision,
    validate_plan,
    wilson_interval,
    analyze_events,
    attach_outcomes,
    group_rows,
)

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs/w3e/w3e_turn_riichi_plan.yaml"


def _plan():
    plan = load_plan(PLAN_PATH)
    validate_plan(plan)
    return plan


def _eval(**overrides):
    base = {
        "ok": True,
        "error": None,
        "tenpai": True,
        "shanten": 0,
        "waits": ["1m"],
        "wait_count": 1,
        "remaining": 4,
        "max_han": 2,
        "max_fu": 30,
        "winning_wait_count": 1,
        "good_wait": False,
        "bad_wait": False,
    }
    base.update(overrides)
    return base


def _event(
    *,
    arm: str,
    game_id: str = "g1",
    hand_id: int = 1,
    seed: int = 43001,
    win: bool = False,
    deal: bool = False,
    opp_riichi_late: bool = False,
    both_riichi: bool = False,
    same_tile: bool = True,
) -> dict:
    return {
        "game_id": game_id,
        "seed": seed,
        "arm": arm,
        "rotation": 0,
        "focal": 0,
        "decision_id": hand_id,
        "hand_id": hand_id,
        "task": "turn",
        "source_class": "riichi_discard",
        "other_class": "discard",
        "both_riichi": both_riichi,
        "same_tile": same_tile,
        "opp_riichi_count": 1 if opp_riichi_late else 0,
        "turn_number": 15 if opp_riichi_late else 5,
        "late": opp_riichi_late,
        "opp_riichi_late": opp_riichi_late,
        "source_eval": _eval(),
        "other_eval": None,
        "error": None,
        "hand": {
            "win": win,
            "deal": deal,
            "income": 8000 if win else 0,
            "loss": 3000 if deal else 0,
            "riichi": True,
            "open": False,
        },
        "outcome_missing": False,
    }


# ---------------------------------------------------------------------------
# plan and schedule
# ---------------------------------------------------------------------------


def test_plan_loads_and_validates_preregistered_values():
    plan = _plan()
    assert plan["purpose"] == "w3e_turn_riichi"
    assert plan["stage1_seeds"] == list(range(43001, 43065))
    assert plan["stage2_seeds"] == list(range(43065, 43129))
    assert plan["smoke_seeds"] == [43501, 43502]
    assert set(plan["arms"]) == {"A", "B"}
    assert plan["analysis"]["minimum_unique_hands_per_group"] == 24
    assert plan["analysis"]["extension"]["extend_if_below"] == 24


def test_plan_rejects_forbidden_w2_seed():
    plan = _plan()
    plan["stage1_seeds"] = [42001] + plan["stage1_seeds"][1:]
    with pytest.raises(ValueError, match="forbidden seeds"):
        validate_plan(plan)


def test_plan_rejects_overlapping_stage_seeds():
    plan = _plan()
    plan["stage2_seeds"] = [plan["stage1_seeds"][0]] + plan["stage2_seeds"][1:]
    with pytest.raises(ValueError, match="disjoint"):
        validate_plan(plan)


def test_plan_rejects_wrong_arm():
    plan = _plan()
    plan["arms"]["B"] = {"turn": "v140", "response": "v140"}
    with pytest.raises(ValueError, match="arm definition"):
        validate_plan(plan)


def test_schedule_is_two_arms_four_rotations_per_seed():
    plan = _plan()
    schedule = build_schedule(plan, plan["stage1_seeds"][:2])
    assert len(schedule) == 2 * len(ARM_IDS) * 4
    first = schedule[:8]
    assert [item["matchup_id"] for item in first] == ["A"] * 4 + ["B"] * 4
    assert [item["focal_seat"] for item in first] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert all(item["seats"] == ["v130"] * 4 for item in first[:4])
    assert all(item["seats"].count("B") == 1 for item in first[4:])
    assert all(item["seats"][item["focal_seat"]] == "B" for item in first[4:])


def test_capture_filter_selects_only_focal_turn_riichi():
    spec = {"focal_seat": 2}
    riichi = {"kind": "riichi_discard", "tile": "1m"}
    assert capture_filter(
        spec,
        {"player_id": 2, "task": "turn", "selected_action": riichi},
    )
    assert not capture_filter(
        spec,
        {"player_id": 1, "task": "turn", "selected_action": riichi},
    )
    assert not capture_filter(
        spec,
        {"player_id": 2, "task": "response", "selected_action": riichi},
    )
    assert not capture_filter(
        spec,
        {
            "player_id": 2,
            "task": "turn",
            "selected_action": {"kind": "discard", "tile": "1m"},
        },
    )


# ---------------------------------------------------------------------------
# statistical helpers
# ---------------------------------------------------------------------------


def test_wilson_interval_contains_point_and_handles_zero():
    interval = wilson_interval(3, 8)
    assert interval["low"] < 3 / 8 < interval["high"]
    empty = wilson_interval(0, 10)
    assert empty["low"] == 0.0
    assert 0.0 < empty["high"] < 0.4
    assert wilson_interval(0, 0) == {"low": None, "high": None}


def test_newcombe_difference_is_signed_and_symmetric():
    forward = newcombe_difference(20, 50, 5, 50)
    backward = newcombe_difference(5, 50, 20, 50)
    assert forward["difference"] == pytest.approx(0.30)
    assert backward["difference"] == pytest.approx(-0.30)
    assert forward["ci95"][0] == pytest.approx(-backward["ci95"][1])
    assert forward["ci95"][1] == pytest.approx(-backward["ci95"][0])


def test_fisher_exact_strong_table_is_small():
    assert fisher_exact_two_sided(10, 0, 0, 10) < 0.001
    assert 0.0 <= fisher_exact_two_sided(3, 7, 1, 9) <= 1.0


def test_primary_decision_rules():
    plan = _plan()
    base = {
        "left": {"n": 30},
        "right": {"n": 30},
        "difference": 0.20,
        "newcombe_ci95": [0.05, 0.35],
    }
    assert primary_decision(base, plan)[0] == "replicated"
    underpowered = {**base, "left": {"n": 10}, "right": {"n": 30}}
    assert primary_decision(underpowered, plan)[0] == "underpowered"
    rejected = {
        **base,
        "difference": 0.02,
        "newcombe_ci95": [-0.05, 0.08],
    }
    assert primary_decision(rejected, plan)[0] == "rejected"
    inconclusive = {
        **base,
        "difference": 0.12,
        "newcombe_ci95": [-0.01, 0.25],
    }
    assert primary_decision(inconclusive, plan)[0] == "inconclusive"


# ---------------------------------------------------------------------------
# count-only and analysis
# ---------------------------------------------------------------------------


def test_count_phase_has_no_outcome_fields_and_uses_b_extra_count():
    plan = _plan()
    rows = [
        _event(arm="B", hand_id=1),
        _event(arm="B", hand_id=2, opp_riichi_late=True),
        _event(arm="A", hand_id=3),
    ]
    summary = {
        "completed_games": 3,
        "expected_games": 3,
        "games": [{"fallbacks": 0}, {"fallbacks": 0}, {"fallbacks": 0}],
    }
    counts = count_events(
        rows,
        {
            "captured_with_state": 3,
            "common_support_limit": {"turn": 64, "response": 64},
        },
        stage="stage1",
        plan=plan,
        summary=summary,
        audit_games=3,
        plan_sha256="0" * 64,
    )
    assert counts["divergence"]["b_extra_unique_hands"] == 2
    assert counts["divergence"]["a_mirror_unique_hands"] == 1
    assert counts["outcomes_inspected"] is False
    assert counts["extension_rule"]["extension_required"] is True
    serialized = json.dumps(counts)
    assert "win_rate" not in serialized
    assert "deal_rate" not in serialized
    assert "income" not in serialized


def test_group_deduplicates_by_hand():
    rows = [_event(arm="B", hand_id=1), _event(arm="B", hand_id=1)]
    groups = group_rows(rows)
    assert len(groups["B_extra"]) == 1


def test_analyze_replicates_synthetic_association():
    plan = _plan()
    plan["analysis"] = {**plan["analysis"], "bootstrap_resamples": 200}
    plan["_plan_sha256"] = "0" * 64
    events = []
    for index in range(30):
        events.append(
            _event(
                arm="B",
                hand_id=index + 1,
                seed=43001 + index,
                deal=index < 12,
                win=index < 6,
                opp_riichi_late=index < 15,
            )
        )
    for index in range(30):
        events.append(
            _event(
                arm="A",
                hand_id=100 + index,
                seed=43001 + index,
                deal=index < 3,
                win=index < 9,
                opp_riichi_late=index < 3,
            )
        )
    result = analyze_events(events, plan, scope="stage1", stages=["stage1"])
    assert result["groups"]["B_extra"]["n"] == 30
    assert result["groups"]["A_mirror"]["n"] == 30
    assert result["primary"]["decision"] == "replicated"
    report = __import__("rilab.w3e", fromlist=["render_report"]).render_report(result)
    assert "B_extra" in report
    assert "Primary" in report


def test_attach_outcomes_links_hand_id(tmp_path):
    batch = tmp_path / "batch"
    (batch / "games").mkdir(parents=True)
    events = [
        {"type": "start_kyoku"},
        {"type": "hora", "actor": 0, "target": 0, "deltas": [8000, -3000, -2500, -2500]},
        {"type": "end_kyoku"},
    ]
    (batch / "games" / "g1.mjai.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    rows = attach_outcomes(
        [{"game_id": "g1", "focal": 0, "hand_id": 1, "error": None}], batch
    )
    assert rows[0]["outcome_missing"] is False
    assert rows[0]["hand"]["win"] is True
    assert rows[0]["hand"]["income"] == 8000
