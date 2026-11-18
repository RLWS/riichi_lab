from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from rilab.w5 import (
    ARM_IDS,
    analyze_stages,
    build_schedule,
    capture_filter,
    count_document,
    load_plan,
    load_stage_pairs,
    mcnemar_exact,
    paired_binary,
    primary_decision,
    split_first_veto_pairs,
    validate_plan,
)

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs/w5/w5_riichi_veto_plan.yaml"


def _plan():
    plan = load_plan(PLAN_PATH)
    validate_plan(plan)
    return plan


def _game(game_id, arm, seed=44001, rotation=0, ranks=(1, 2, 3, 4)):
    return {
        "game_id": game_id,
        "seed": seed,
        "rotation": rotation,
        "matchup_id": arm,
        "focal_seat": rotation,
        "seats": [arm, "v130", "v130", "v130"],
        "ranks": list(ranks),
        "scores": [30000, 25000, 20000, 15000],
        "fallbacks": 0,
        "valid": True,
    }


def _record(
    decision_id,
    selected,
    *,
    hand_id=1,
    player_id=0,
    veto=False,
    new_events=None,
):
    prediction = {"veto": {"applied": True}} if veto else {}
    return {
        "decision_id": decision_id,
        "player_id": player_id,
        "task": "turn",
        "hand_id": hand_id,
        "selected_action": selected,
        "engine_action": selected,
        "new_events": new_events or [f"event-{decision_id}"],
        "prediction": prediction,
    }


def _write_decisions(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_game_log(path: Path, *, win=False, deal=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    deltas = [0, 0, 0, 0]
    target = 0
    actor = 0
    if win:
        deltas = [8000, -3000, -2500, -2500]
    elif deal:
        actor = 1
        deltas = [-3000, 3000, 0, 0]
    events = [
        {"type": "start_kyoku"},
        {"type": "hora", "actor": actor, "target": target, "deltas": deltas},
        {"type": "end_kyoku"},
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )


def _batch(tmp_path, *, prefix_divergence=False, treatment_deal=True):
    batch = tmp_path / "batch"
    summary = {
        "completed_games": 2,
        "expected_games": 2,
        "games": [_game("g00000", "C"), _game("g00001", "T")],
    }
    (batch).mkdir(parents=True)
    (batch / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    riichi = {"kind": "riichi_discard", "tile": "3p"}
    discard = {"kind": "discard", "tile": "1m"}
    control_first = _record(0, {"kind": "discard", "tile": "9m"})
    if prefix_divergence:
        control_first["engine_action"] = {"kind": "discard", "tile": "8m"}
    _write_decisions(
        batch / "decisions" / "g00000.jsonl.gz",
        [control_first, _record(1, riichi)],
    )
    _write_decisions(
        batch / "decisions" / "g00001.jsonl.gz",
        [
            _record(0, {"kind": "discard", "tile": "9m"}),
            _record(1, discard, veto=True),
        ],
    )
    _write_game_log(batch / "games" / "g00000.mjai.jsonl", win=not treatment_deal)
    _write_game_log(batch / "games" / "g00001.mjai.jsonl", deal=treatment_deal)
    return batch


def _pair(seed, *, treatment_deal, control_deal, treatment_win=False, control_win=False):
    return {
        "seed": seed,
        "treatment": {
            "deal": treatment_deal,
            "win": treatment_win,
            "income": 8000 if treatment_win else 0,
            "loss": 3000 if treatment_deal else 0,
        },
        "control": {
            "deal": control_deal,
            "win": control_win,
            "income": 8000 if control_win else 0,
            "loss": 3000 if control_deal else 0,
        },
    }


# ---------------------------------------------------------------------------
# plan and schedule
# ---------------------------------------------------------------------------


def test_plan_loads_and_validates_preregistered_values():
    plan = _plan()
    assert plan["purpose"] == "w5_riichi_veto"
    assert plan["stage1_seeds"] == list(range(44001, 44065))
    assert plan["stage2_seeds"] == list(range(44065, 44129))
    assert plan["smoke_seeds"] == [44501, 44502]
    assert set(plan["arms"]) == {"C", "T"}
    assert plan["veto"]["condition"] == {
        "opp_riichi_count_min": 1,
        "turn_number_min_exclusive": 12,
    }
    assert plan["analysis"]["minimum_pairs"] == 24
    assert plan["analysis"]["extension"]["extend_if_below"] == 24


def test_plan_rejects_forbidden_seed():
    plan = _plan()
    plan["stage1_seeds"] = [43001] + plan["stage1_seeds"][1:]
    with pytest.raises(ValueError, match="forbidden seeds"):
        validate_plan(plan)


def test_plan_rejects_overlapping_stage_seeds():
    plan = _plan()
    plan["stage2_seeds"] = [plan["stage1_seeds"][0]] + plan["stage2_seeds"][1:]
    with pytest.raises(ValueError, match="disjoint"):
        validate_plan(plan)


def test_plan_rejects_wrong_arm():
    plan = _plan()
    plan["arms"]["T"] = {"turn": "v140", "response": "v130", "veto": None}
    with pytest.raises(ValueError, match="arm definition"):
        validate_plan(plan)


def test_plan_rejects_changed_veto_rule():
    plan = _plan()
    plan["veto"]["condition"]["turn_number_min_exclusive"] = 10
    with pytest.raises(ValueError, match="turn_number_min_exclusive"):
        validate_plan(plan)


def test_schedule_is_two_arms_four_rotations_per_seed():
    plan = _plan()
    schedule = build_schedule(plan, plan["stage1_seeds"][:2])
    assert len(schedule) == 2 * len(ARM_IDS) * 4
    first = schedule[:8]
    assert [item["matchup_id"] for item in first] == ["C"] * 4 + ["T"] * 4
    assert [item["focal_seat"] for item in first] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert all(item["seats"].count("C") == 1 for item in first[:4])
    assert all(item["seats"].count("T") == 1 for item in first[4:])
    assert all(item["seats"][item["focal_seat"]] == "T" for item in first[4:])


def test_capture_filter_selects_only_applied_veto():
    spec = {"focal_seat": 2}
    applied = {
        "player_id": 2,
        "task": "turn",
        "prediction": {"veto": {"applied": True}},
    }
    assert capture_filter(spec, applied)
    assert not capture_filter(
        spec, {**applied, "prediction": {"veto": {"applied": False}}}
    )
    assert not capture_filter(spec, {**applied, "player_id": 1})
    assert not capture_filter(spec, {**applied, "task": "response"})


# ---------------------------------------------------------------------------
# count-only phase
# ---------------------------------------------------------------------------


def test_count_document_has_no_outcomes_and_uses_first_veto_pairs():
    plan = _plan()
    summary = {
        "completed_games": 4,
        "expected_games": 4,
        "games": [
            _game("g00000", "C"),
            _game("g00001", "T"),
            _game("g00002", "C", rotation=1),
            _game("g00003", "T", rotation=1),
        ],
    }
    rows = [
        {
            "game_id": "g00001",
            "seed": 44001,
            "rotation": 0,
            "focal": 0,
            "decision_id": 5,
            "hand_id": 1,
            "veto_count": 1,
            "veto_decision_ids": [5],
        }
    ]
    counts = count_document(
        rows,
        {
            "treatment_games": 2,
            "veto_events": 1,
            "games_with_veto": 1,
            "first_vetoes": 1,
        },
        stage="stage1",
        plan=plan,
        summary=summary,
        audit_games=4,
        plan_sha256="0" * 64,
    )
    assert counts["first_veto_pairs"] == 1
    assert counts["extension_rule"]["extension_required"] is True
    assert counts["outcomes_inspected"] is False
    serialized = json.dumps(counts)
    for token in ("win_rate", "deal_rate", "income", "loss"):
        assert token not in serialized


def test_count_marks_missing_control_game_unpaired():
    summary = {"games": [_game("g00001", "T")]}
    rows = [
        {
            "game_id": "g00001",
            "seed": 44001,
            "rotation": 0,
            "focal": 0,
            "decision_id": 5,
            "hand_id": 1,
            "veto_count": 1,
            "veto_decision_ids": [5],
        }
    ]
    pairs, unpaired = split_first_veto_pairs(rows, summary)
    assert pairs == []
    assert unpaired[0]["reason"] == "control_game_missing"


# ---------------------------------------------------------------------------
# pairing / analysis
# ---------------------------------------------------------------------------


def test_load_stage_pairs_links_first_veto_hand(tmp_path):
    batch = _batch(tmp_path, treatment_deal=True)
    pairs, unpaired, meta = load_stage_pairs(batch)
    assert len(pairs) == 1
    assert unpaired == []
    pair = pairs[0]
    assert pair["seed"] == 44001
    assert pair["rotation"] == 0
    assert pair["hand_id"] == 1
    assert pair["treatment"]["deal"] is True
    assert pair["control"]["deal"] is False
    assert meta["veto_events"] == 1


def test_load_stage_pairs_marks_prior_divergence_unpaired(tmp_path):
    batch = _batch(tmp_path, prefix_divergence=True)
    pairs, unpaired, _ = load_stage_pairs(batch)
    assert pairs == []
    assert unpaired[0]["reason"].startswith("prior_divergence:")


def test_mcnemar_exact_matches_binomial_tail():
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert mcnemar_exact(0, 0) is None


def test_primary_decision_rules():
    plan = _plan()
    base = {
        "n_pairs": 30,
        "difference": -0.30,
        "seed_cluster_bootstrap": {"ci95": [-0.50, -0.10]},
    }
    assert primary_decision(base, plan)[0] == "supported"
    assert primary_decision({**base, "n_pairs": 23}, plan)[0] == "underpowered"
    assert (
        primary_decision({**base, "difference": -0.05}, plan)[0]
        == "not_supported"
    )
    assert (
        primary_decision(
            {**base, "seed_cluster_bootstrap": {"ci95": [-0.50, 0.01]}}, plan
        )[0]
        == "not_supported"
    )


def test_paired_binary_reports_paired_risk_difference_and_bootstrap():
    plan = _plan()
    plan["analysis"] = {**plan["analysis"], "bootstrap_resamples": 200}
    pairs = []
    for index in range(30):
        treatment_deal = index < 2
        control_deal = index < 12
        pairs.append(
            _pair(
                44001 + index,
                treatment_deal=treatment_deal,
                control_deal=control_deal,
            )
        )
    result = paired_binary(pairs, "deal", plan)
    assert result["n_pairs"] == 30
    assert result["treatment"]["count"] == 2
    assert result["control"]["count"] == 12
    assert result["difference"] == pytest.approx(-10 / 30)
    assert result["discordant"]["control_only"] == 10
    assert result["discordant"]["treatment_only"] == 0
    assert result["seed_cluster_bootstrap"]["ci95"][1] < 0


def test_analyze_stages_pairs_synthetic_first_veto_hand(tmp_path):
    plan = _plan()
    plan["analysis"] = {**plan["analysis"], "bootstrap_resamples": 200}
    batch = _batch(tmp_path, treatment_deal=True)
    pairs, unpaired, meta = load_stage_pairs(batch)
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    result = analyze_stages(
        [
            {
                "stage": "smoke",
                "pairs": pairs,
                "unpaired": unpaired,
                "meta": meta,
                "summary": summary,
            }
        ],
        plan,
        scope="smoke",
    )
    assert result["first_veto_pairs"] == 1
    assert result["primary"]["treatment"]["count"] == 1
    assert result["primary"]["control"]["count"] == 0
    assert result["primary"]["decision"] == "underpowered"
    assert result["veto_events"]["veto_events"] == 1
    assert "W5 Riichi veto report" in __import__(
        "rilab.w5", fromlist=["render_report"]
    ).render_report(result)
