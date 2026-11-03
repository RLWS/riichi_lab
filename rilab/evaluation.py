"""Seed-cluster paired statistics; existing A3 data is exploratory only."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random

from rilab.artifacts import audit_batch, encode, write_json
from rilab.scheduling import expand_schedule

METRICS = {
    "mean_rank": ("rank_sum", "games"),
    "last_place_rate": ("last_places", "games"),
    "mean_final_points": ("point_sum", "games"),
    "win_rate": ("wins", "hands"),
    "deal_in_rate": ("deal_ins", "hands"),
    "riichi_rate": ("riichi_accepted", "hands"),
    "riichi_declaration_rate": ("riichi_declared", "hands"),
    "call_rate": ("called_hands", "hands"),
}


def seat_counters(events: list[dict]) -> list[dict]:
    """Count actor-hand outcomes, deduplicating the payer of a multiple ron."""
    counters = [
        dict.fromkeys(sorted({field for pair in METRICS.values() for field in pair}), 0)
        for _ in range(4)
    ]
    flags = None
    started = ended = 0
    for event in events:
        kind = event["type"]
        if kind == "start_kyoku":
            if flags is not None:
                raise ValueError("hand started before previous end_kyoku")
            started += 1
            flags = {
                key: set()
                for key in (
                    "wins",
                    "deal_ins",
                    "riichi_accepted",
                    "riichi_declared",
                    "called_hands",
                )
            }
        elif kind == "end_kyoku":
            if flags is None:
                raise ValueError("end_kyoku without hand")
            for p, counts in enumerate(counters):
                counts["hands"] += 1
                for key, players in flags.items():
                    counts[key] += p in players
            flags = None
            ended += 1
        elif kind in (
            "hora",
            "reach",
            "reach_accepted",
            "chi",
            "pon",
            "daiminkan",
            "ankan",
            "kakan",
        ):
            if flags is None:
                raise ValueError("hand event outside a hand")
            p = event["actor"]
            if type(p) is not int or not 0 <= p < 4:
                raise ValueError("invalid event actor")
            if kind == "hora":
                flags["wins"].add(p)
                if not event.get("tsumo", event["target"] == p):
                    target = event["target"]
                    if type(target) is not int or not 0 <= target < 4 or target == p:
                        raise ValueError("invalid ron target")
                    flags["deal_ins"].add(target)
            elif kind == "reach":
                flags["riichi_declared"].add(p)
            elif kind == "reach_accepted":
                flags["riichi_accepted"].add(p)
            elif kind in ("chi", "pon", "daiminkan"):
                flags["called_hands"].add(p)
    if flags is not None or started == 0 or started != ended:
        raise ValueError("incomplete hand events")
    return counters


def add_counts(target, source):
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def metric_counts(clusters, label, metric):
    numerator, denominator = METRICS[metric]
    num = sum(c[label].get(numerator, 0) for c in clusters)
    den = sum(c[label].get(denominator, 0) for c in clusters)
    if den <= 0:
        raise ValueError("empty metric denominator")
    return {"numerator": num, "denominator": den, "value": num / den}


def percentile(values, probability):
    """Linear interpolation at (n-1)*p, with a sorted input."""
    index = (len(values) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def paired_bootstrap(
    clusters, *, labels=("candidate", "control"), iterations=10000, rng_seed=20261103
):
    """Resample complete seed clusters; reuse draws across metrics and arms."""
    if len(clusters) < 2 or type(iterations) is not int or iterations < 100:
        raise ValueError("bootstrap requires at least two seed clusters and 100 draws")
    rng = random.Random(rng_seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(iterations):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        for metric in METRICS:
            a, b = (metric_counts(sampled, label, metric)["value"] for label in labels)
            draws[metric].append(a - b)
    result = {}
    for metric in METRICS:
        a, b = (metric_counts(clusters, label, metric) for label in labels)
        ordered = sorted(draws[metric])
        result[metric] = {
            labels[0]: a,
            labels[1]: b,
            "difference": a["value"] - b["value"],
            "ci95": [percentile(ordered, 0.025), percentile(ordered, 0.975)],
            "direction": "higher_is_better"
            if metric in ("mean_final_points", "win_rate")
            else "lower_is_better"
            if metric in ("mean_rank", "last_place_rate", "deal_in_rate")
            else "descriptive_only",
        }
    return result


def classify_rank(ci95, minimum_gain, *, eligible):
    """Lower candidate-minus-control rank is better; exploratory data cannot pass."""
    if (
        not math.isfinite(minimum_gain)
        or minimum_gain < 0
        or len(ci95) != 2
        or not all(math.isfinite(x) for x in ci95)
        or ci95[0] > ci95[1]
    ):
        raise ValueError("invalid rank threshold or interval")
    if not eligible:
        return "insufficient_evidence"
    if minimum_gain <= 0:
        raise ValueError("formal classification requires a positive meaningful gain")
    if ci95[1] < 0 and ci95[1] <= -minimum_gain:
        return "improvement"
    if ci95[0] > 0 and ci95[0] >= minimum_gain:
        return "regression"
    return "insufficient_evidence"


def evaluate_run(run_dir, output_dir, *, iterations=10000, rng_seed=20261103):
    """Validate a complete A3 run and produce a non-promotional exploratory report."""
    if type(iterations) is not int or iterations < 100 or type(rng_seed) is not int:
        raise ValueError("invalid bootstrap settings")
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    expected = expand_schedule(config)
    if manifest["schedule"] != expected:
        raise ValueError("manifest schedule differs from configured complete schedule")
    games = audit_batch(run)
    if (
        summary.get("valid") is not True
        or summary.get("games") != games
        or summary.get("completed_games") != len(expected)
        or summary.get("expected_games") != len(expected)
        or summary.get("fallbacks") != 0
    ):
        raise ValueError("invalid or incomplete source run")
    if (
        summary.get("replay_digest")
        != hashlib.sha256(encode(games).encode()).hexdigest()
    ):
        raise ValueError("source summary digest mismatch")
    group = config["evaluation"]["primary_paired_group"]
    members = [m for m in config["matchups"] if m.get("paired_group") == group]
    controls = [m for m in members if len(set(m["seats"])) == 1]
    if len(members) != 2 or len(controls) != 1:
        raise ValueError("primary group requires one homogeneous baseline control")
    control = controls[0]
    candidate = next(m for m in members if m is not control)
    baseline_id = control["seats"][control["focal_seat"]]
    candidate_id = candidate["seats"][candidate["focal_seat"]]
    if candidate_id == baseline_id:
        raise ValueError("candidate and baseline identities must differ")
    main = {
        seed: {"candidate": {}, "control": {}} for seed in config["engine"]["seeds"]
    }
    supplements = defaultdict(dict)
    observations = []
    for game in games:
        events = [
            json.loads(line)
            for line in (run / "games" / f"{game['game_id']}.mjai.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if (
            not events
            or events[0]["type"] != "start_game"
            or events[-1]["type"] != "end_game"
        ):
            raise ValueError("incomplete game lifecycle")
        counters = seat_counters(events)
        if (
            len(game["ranks"]) != 4
            or any(type(r) is not int for r in game["ranks"])
            or sorted(game["ranks"]) != [1, 2, 3, 4]
            or len(game["scores"]) != 4
            or any(type(s) is not int for s in game["scores"])
        ):
            raise ValueError("invalid authoritative final ranks/scores")
        for p, counts in enumerate(counters):
            counts.update(
                games=1,
                rank_sum=game["ranks"][p],
                last_places=int(game["ranks"][p] == 4),
                point_sum=game["scores"][p],
            )
        if game["paired_group"] == group:
            label = "candidate" if game["matchup_id"] == candidate["id"] else "control"
            counts = counters[game["focal_seat"]]
            add_counts(main[game["seed"]][label], counts)
            observations.append(
                {
                    "game_id": game["game_id"],
                    "seed": game["seed"],
                    "rotation": game["rotation"],
                    "label": label,
                    "focal_seat": game["focal_seat"],
                    "counts": counts,
                }
            )
        elif game["focal_seat"] is None:
            spec = next(m for m in config["matchups"] if m["id"] == game["matchup_id"])
            if spec["rotation"] != "unique6" or set(game["seats"]) != {
                candidate_id,
                baseline_id,
            }:
                raise ValueError("unsupported supplementary experiment identities")
            by_seed = supplements[game["matchup_id"]].setdefault(
                game["seed"], {"candidate": {}, "control": {}}
            )
            for p, counts in enumerate(counters):
                label = "candidate" if game["seats"][p] == candidate_id else "control"
                add_counts(by_seed[label], counts)
        else:
            raise ValueError(
                "unreported extra matchup; choose an explicit experiment config"
            )
    clusters = list(main.values())
    primary = paired_bootstrap(clusters, iterations=iterations, rng_seed=rng_seed)
    secondary = {
        name: {
            "design": "two_vs_two_same_table_seed_cluster; no team outcomes",
            "seed_clusters": len(values),
            "metrics": paired_bootstrap(
                list(values.values()), iterations=iterations, rng_seed=rng_seed
            ),
        }
        for name, values in supplements.items()
    }
    report = {
        "stage": "A4",
        "valid": True,
        "analysis_kind": "exploratory",
        "strength_verdict": classify_rank(
            primary["mean_rank"]["ci95"], 0, eligible=False
        ),
        "ineligibility_reasons": [
            "A3 engineering seeds were already observed; not a preregistered independent strength experiment"
        ],
        "seeds": list(main),
        "seed_clusters": len(clusters),
        "source_games": len(games),
        "primary_focal_games": len(observations),
        "comparison": {
            "candidate": candidate_id,
            "control": baseline_id,
            "paired_group": group,
            "estimand": "complete deployed policies, including different calibration settings",
        },
        "bootstrap": {
            "method": "paired seed-cluster percentile; ratio-of-sums recomputed per draw",
            "iterations": iterations,
            "rng_seed": rng_seed,
            "confidence_level": 0.95,
            "interval_scope": "marginal descriptive intervals, not simultaneous promotion tests",
            "independent_units": "seeds; all four rotations in both arms resampled together",
        },
        "counting_conventions": {
            "hands": "actor-hand exposures; all start_kyoku including abortive draws",
            "wins": "each winner once per hand; multiple ron credits each winner",
            "deal_ins": "payer once per hand regardless of the number of ron winners; tsumo is not a deal-in",
            "riichi_rate": "accepted riichi hands / actor-hand exposures; declaration rate reported separately",
            "call_rate": "hands with chi/pon/daiminkan / actor-hand exposures; repeated calls count once, ankan alone does not count",
            "points_and_ranks": "integer final points and authoritative engine ranks; no re-ranking of ties",
            "rates": "ratio of pooled numerators and denominators, recomputed jointly per seed-cluster bootstrap draw",
            "replays": "analyze separately for reproducibility; never pool repeated runs as independent samples",
        },
        "implementation": {
            "python": platform.python_version(),
            "evaluation_py_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "primary": primary,
        "supplementary": secondary,
        "per_seed_primary_counts": [
            {"seed": seed, **counts} for seed, counts in main.items()
        ],
        "focal_observations": observations,
        "backends": config["backends"],
        "model_metadata": manifest["models"],
        "known_exceptions": manifest.get("known_exceptions", []),
        "source": {
            "path": str(run.resolve()),
            "replay_digest": summary["replay_digest"],
            "manifest_sha256": hashlib.sha256(
                (run / "manifest.json").read_bytes()
            ).hexdigest(),
            "summary_sha256": hashlib.sha256(
                (run / "summary.json").read_bytes()
            ).hexdigest(),
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "report.json", report)
    rank = primary["mean_rank"]
    text = (
        "# A4 探索性统计报告\n\n结论：证据不足。现有 A3 数据仅用于验收统计工具，不作正式强度结论。\n\n"
        f"种子簇：{len(clusters)}；主实验焦点半庄：{len(observations)}；补充 2v2 单独报告。\n\n"
        f"平均顺位差（候选−基线，低为好）：{rank['difference']:.6f}，95% 区间 {rank['ci95']}。\n\n"
        "两版本校准不同，比较的是完整部署策略。DEF-001 保留为已知例外。完整分子、分母与辅助指标见 report.json。\n"
    )
    (out / "report.md").write_text(text, encoding="utf-8")
    return report
