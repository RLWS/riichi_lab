from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from rilab.artifacts import encode
from rilab.config import load_arena_config
from rilab.evaluation import (
    METRICS,
    classify_rank,
    evaluate_run,
    metric_counts,
    paired_bootstrap,
    percentile,
    seat_counters,
)
from rilab.scheduling import expand_schedule

ROOT = Path(__file__).parents[1]


def counts(rank=2, hands=10, wins=1):
    value = dict.fromkeys({f for pair in METRICS.values() for f in pair}, 0)
    value.update(games=4, rank_sum=4 * rank, hands=hands, wins=wins)
    return value


def test_paired_seed_resampling_preserves_covariance():
    # Both arms vary strongly by seed; paired differences are exactly zero.
    clusters = [
        {"candidate": counts(rank=r), "control": counts(rank=r)} for r in [1, 4, 2]
    ]
    result = paired_bootstrap(clusters, iterations=1000)
    assert all(m["difference"] == 0 and m["ci95"] == [0, 0] for m in result.values())


def test_seed_cluster_bootstrap_known_discrete_distribution_and_repeatability():
    clusters = [
        {"candidate": counts(rank=r), "control": counts(rank=2)} for r in [1, 3]
    ]
    result = paired_bootstrap(clusters, iterations=1000, rng_seed=17)
    assert result["mean_rank"]["difference"] == 0
    assert result["mean_rank"]["ci95"] == [-1, 1]
    assert paired_bootstrap(clusters, iterations=1000, rng_seed=17) == result


def test_rates_use_pooled_actor_hand_denominators_not_average_seed_rates():
    clusters = [
        {"candidate": counts(hands=1, wins=1)},
        {"candidate": counts(hands=9, wins=0)},
    ]
    assert metric_counts(clusters, "candidate", "win_rate") == {
        "numerator": 1,
        "denominator": 10,
        "value": 0.1,
    }
    assert percentile([1, 3, 5], 0.25) == 2


def test_multiple_ron_and_calls_count_player_hands_once():
    events = [
        {"type": "start_kyoku"},
        {"type": "chi", "actor": 1},
        {"type": "pon", "actor": 1},
        {"type": "ankan", "actor": 2},
        {"type": "kakan", "actor": 1},
        {"type": "reach", "actor": 3},
        {"type": "hora", "actor": 1, "target": 0, "tsumo": False},
        {"type": "hora", "actor": 2, "target": 0, "tsumo": False},
        {"type": "end_kyoku"},
        {"type": "start_kyoku"},
        {"type": "reach", "actor": 3},
        {"type": "reach_accepted", "actor": 3},
        {"type": "hora", "actor": 3, "target": 3, "tsumo": True},
        {"type": "end_kyoku"},
    ]
    values = seat_counters(events)
    assert [v["hands"] for v in values] == [2] * 4
    assert [v["wins"] for v in values] == [0, 1, 1, 1]
    assert [v["deal_ins"] for v in values] == [1, 0, 0, 0]
    assert [v["called_hands"] for v in values] == [0, 1, 0, 0]
    assert values[3]["riichi_declared"] == 2 and values[3]["riichi_accepted"] == 1


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"type": "end_kyoku"}],
        [{"type": "start_kyoku"}],
        [{"type": "start_kyoku"}, {"type": "start_kyoku"}],
    ],
)
def test_reject_incomplete_or_invalid_hands(events):
    with pytest.raises(ValueError):
        seat_counters(events)


@pytest.mark.parametrize(
    "interval,eligible,expected",
    [
        ([-0.3, -0.1], True, "improvement"),
        ([0.1, 0.3], True, "regression"),
        ([-0.3, 0.1], True, "insufficient_evidence"),
        ([-0.02, -0.01], True, "insufficient_evidence"),
        ([-0.3, -0.1], False, "insufficient_evidence"),
    ],
)
def test_meaningful_gain_and_exploratory_gate(interval, eligible, expected):
    assert classify_rank(interval, 0.05, eligible=eligible) == expected


def synthetic_run(tmp_path, monkeypatch):
    """Known experiment: candidate focal ranks 1, control focal ranks 2."""
    import rilab.evaluation as evaluation

    config = load_arena_config(ROOT / "configs/arena_selfplay.yaml")
    config["engine"]["seeds"] = [101, 102]
    schedule = expand_schedule(config)
    run = tmp_path / "source"
    (run / "games").mkdir(parents=True)
    games = []
    for spec in schedule:
        rank = 1 if spec["matchup_id"] == "candidate_vs_baseline" else 2
        focal = spec["focal_seat"]
        ranks = [1, 2, 3, 4]
        if focal is not None:
            other = ranks.index(rank)
            ranks[focal], ranks[other] = ranks[other], ranks[focal]
        games.append(
            {
                **spec,
                "valid": True,
                "fallbacks": 0,
                "ranks": ranks,
                "scores": [25000] * 4,
            }
        )
        events = [
            {"type": "start_game"},
            {"type": "start_kyoku"},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        (run / "games" / f"{spec['game_id']}.mjai.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events), encoding="utf-8"
        )
    manifest = {"config": config, "schedule": schedule, "models": {"synthetic": True}}
    summary = {
        "valid": True,
        "games": games,
        "completed_games": len(games),
        "expected_games": len(games),
        "fallbacks": 0,
        "replay_digest": hashlib.sha256(encode(games).encode()).hexdigest(),
    }
    for name, value in [("manifest", manifest), ("summary", summary)]:
        (run / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    # A3's audit is tested with real gzip/action logs in test_batch; isolate A4 here.
    monkeypatch.setattr(evaluation, "audit_batch", lambda _: deepcopy(games))
    return run, summary, manifest


def test_evaluator_focal_only_and_two_vs_two_separation(tmp_path, monkeypatch):
    run, _, _ = synthetic_run(tmp_path, monkeypatch)
    report = evaluate_run(run, tmp_path / "report", iterations=100)
    assert report["seed_clusters"] == 2
    assert report["source_games"] == 28 and report["primary_focal_games"] == 16
    assert report["primary"]["mean_rank"]["difference"] == -1
    assert report["primary"]["mean_rank"]["candidate"]["denominator"] == 8
    assert (
        report["supplementary"]["two_vs_two"]["metrics"]["mean_rank"]["candidate"][
            "denominator"
        ]
        == 24
    )
    assert report["strength_verdict"] == "insufficient_evidence"
    assert (tmp_path / "report/report.md").is_file()


@pytest.mark.parametrize("fault", ["invalid", "missing", "schedule", "digest"])
def test_evaluator_rejects_bad_source_without_writing_report(
    tmp_path, monkeypatch, fault
):
    run, summary, manifest = synthetic_run(tmp_path, monkeypatch)
    if fault == "invalid":
        summary["valid"] = False
    elif fault == "missing":
        summary["games"].pop()
    elif fault == "schedule":
        manifest["schedule"].pop()
    else:
        summary["replay_digest"] = "invalid"
    for name, value in [("manifest", manifest), ("summary", summary)]:
        (run / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_run(run, tmp_path / "report", iterations=100)
    assert not (tmp_path / "report").exists()
