"""Observational online scorecards; keep opponent uncertainty and censoring visible."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
import json
import hashlib
from pathlib import Path
import random
from statistics import NormalDist

from rilab.artifacts import write_json
from rilab.evaluation import percentile
from rilab.scoring import rank_score, rank_scorecard


def rounded_counts(games, percentages, precision=1):
    """Reconstruct only uniquely identified counts; never use fractional pseudo-games."""
    if type(games) is not int or games <= 0 or len(percentages) != 4:
        raise ValueError("invalid four-player rank distribution")
    half = Decimal(10) ** -precision / 2
    counts = []
    for value in percentages:
        p = Decimal(str(value))
        lo = max(
            0, int(((p - half) * games / 100).to_integral_value(rounding=ROUND_CEILING))
        )
        hi = min(
            games,
            int(((p + half) * games / 100).to_integral_value(rounding=ROUND_CEILING))
            - 1,
        )
        if lo != hi:
            raise ValueError(
                "rounded percentages do not uniquely identify integer counts"
            )
        counts.append(lo)
    if sum(counts) != games:
        raise ValueError("rounded percentages are inconsistent with game count")
    return counts


def histogram_stats(counts):
    if (
        len(counts) != 4
        or any(type(n) is not int or n < 0 for n in counts)
        or sum(counts) < 2
    ):
        raise ValueError("need at least two games with four integer rank counts")
    n = sum(counts)
    mean = sum((i + 1) * c for i, c in enumerate(counts)) / n
    variance = sum(c * (i + 1 - mean) ** 2 for i, c in enumerate(counts)) / (n - 1)
    return {
        "games": n,
        "rank_counts": counts,
        "mean_rank": mean,
        "rank_score_0_100": rank_score(mean),
        "top_two_rate": sum(counts[:2]) / n,
        "first_rate": counts[0] / n,
        "last_rate": counts[3] / n,
        "sample_rank_variance": variance,
    }


def compare_displays(document):
    versions = document["versions"]
    result = {}
    for version, source in versions.items():
        counts = source.get("rank_counts")
        reconstructed = counts is None
        if reconstructed:
            counts = rounded_counts(source["games"], source["rank_percentages"])
        if sum(counts) != source["games"]:
            raise ValueError("rank counts differ from declared games")
        if "displayed_mean_rank" in source:
            exact = (
                Decimal(sum((i + 1) * n for i, n in enumerate(counts)))
                / source["games"]
            )
            if exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) != Decimal(
                str(source["displayed_mean_rank"])
            ):
                raise ValueError("displayed mean rank conflicts with rank distribution")
        result[version] = {
            **histogram_stats(counts),
            "count_provenance": "unique reconstruction from rounded screenshot"
            if reconstructed
            else "supplied counts",
            "displayed_metrics": source,
        }
    candidate, baseline = (result[document[k]] for k in ("candidate", "baseline"))
    diff = candidate["mean_rank"] - baseline["mean_rank"]
    se = (
        candidate["sample_rank_variance"] / candidate["games"]
        + baseline["sample_rank_variance"] / baseline["games"]
    ) ** 0.5
    z = NormalDist().inv_cdf(0.975)
    return {
        "versions": result,
        "observed_rank_gain": -diff,
        "rank_score_gain": -diff * 100 / 3,
        "top_two_rate_gain": candidate["top_two_rate"] - baseline["top_two_rate"],
        "first_rate_gain": candidate["first_rate"] - baseline["first_rate"],
        "last_rate_change": candidate["last_rate"] - baseline["last_rate"],
        "unadjusted_iid_normal_rank_difference_ci95": [diff - z * se, diff + z * se],
        "inference_status": "conditional diagnostic only; independence, equal opponent pool and time coverage unverified; no preregistered causal promotion claim",
        "hand_metric_intervals": None,
        "missing_information": [
            "per-hand denominators for displayed rates",
            "opponent identities/ratings",
            "match times and shared games",
        ],
    }


def prepare_rows(snapshot):
    rows = []
    seen = set()
    duplicates = []
    for batch in snapshot["batches"]:
        version = batch["backend"]["revision"]["tag"]
        for original in batch["games"]:
            identifier = original.get("match_id")
            # No fabricated game IDs or de-duplication based solely on coincident scores.
            key = (
                (version, identifier, original["seat"])
                if identifier is not None
                else (version, original["source_id"])
            )
            if key in seen:
                duplicates.append(original["source_id"])
                continue
            seen.add(key)
            row = {
                **original,
                "version": version,
                "batch_id": batch["batch_id"],
                "batch_status": batch["status"],
            }
            seat, scores = row.get("seat"), row.get("scores")
            row["started"] = type(seat) is int and 0 <= seat < 4
            row["terminal"] = (
                row["started"]
                and isinstance(scores, list)
                and len(scores) == 4
                and all(type(x) is int for x in scores)
            )
            if row["terminal"]:
                own = scores[seat]
                ranks = row.get("ranks")
                score_low, score_high = (
                    1 + sum(s > own for s in scores),
                    sum(s >= own for s in scores),
                )
                if ranks is not None:
                    if (
                        not isinstance(ranks, list)
                        or len(ranks) != 4
                        or any(type(r) is not int for r in ranks)
                        or sorted(ranks) != [1, 2, 3, 4]
                        or not score_low <= ranks[seat] <= score_high
                    ):
                        raise ValueError(
                            "authoritative ranks conflict with terminal scores"
                        )
                    low = high = ranks[seat]
                    provenance = "authoritative_ranks"
                else:
                    low, high = score_low, score_high
                    provenance = (
                        "unique_score_order"
                        if low == high
                        else "unresolved_platform_tie_rule"
                    )
                row.update(
                    rank_low=low,
                    rank_high=high,
                    rank_mid=(low + high) / 2,
                    rank_provenance=provenance,
                    points=own,
                    hour=int(row["trace_mtime_utc"] // 3600),
                )
            rows.append(row)
    return rows, duplicates


def arm_summary(rows):
    terminal = [r for r in rows if r["terminal"]]
    started = [r for r in rows if r["started"]]
    if not terminal:
        raise ValueError("no terminal outcomes for version")
    low, high = (
        sum(r[k] for r in terminal) / len(terminal) for k in ("rank_low", "rank_high")
    )
    missing = len(started) - len(terminal)
    n = len(started)
    hands = sum(r["hands"] for r in terminal)
    return {
        "sessions": len(rows),
        "started_sessions": n,
        "terminal_games": len(terminal),
        "nonterminal_started": missing,
        "sessions_without_observed_start": len(rows) - n,
        "mean_rank_bounds": [low, high],
        "mean_rank_midpoint_descriptive": (low + high) / 2,
        "rank_score_bounds": [rank_score(high), rank_score(low)],
        "unresolved_ties": sum(r["rank_low"] != r["rank_high"] for r in terminal),
        "all_started_rank_bounds_with_missing": [
            (sum(r["rank_low"] for r in terminal) + missing) / n,
            (sum(r["rank_high"] for r in terminal) + 4 * missing) / n,
        ],
        "mean_final_points": sum(r["points"] for r in terminal) / len(terminal),
        "nonpassing_terminal_games_retained": sum(
            r.get("protocol_passed") is not True for r in terminal
        ),
        "batch_status_counts": dict(Counter(r["batch_status"] for r in rows)),
        "exact_rank_counts": dict(
            Counter(r["rank_low"] for r in terminal if r["rank_low"] == r["rank_high"])
        ),
        "rates": {
            key: {
                "numerator": sum(r[key] for r in terminal),
                "denominator": hands,
                "value": sum(r[key] for r in terminal) / hands if hands else None,
            }
            for key in ("wins", "deal_ins", "riichi", "calls")
        },
        "rate_scope": "observed completed hands in terminal traces only; screenshot denominators may differ",
    }


def standardized_online(
    rows, candidate, baseline, *, iterations=10000, rng_seed=20261105
):
    if type(iterations) is not int or iterations < 100 or type(rng_seed) is not int:
        raise ValueError("invalid hour-cluster bootstrap settings")
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["terminal"]:
            cells[row["hour"], row["seat"]][row["version"]].append(row)
    hours = defaultdict(
        lambda: {
            "weight": 0,
            "low_sum": 0,
            "high_sum": 0,
            "candidate_games": 0,
            "baseline_games": 0,
        }
    )
    cell_rows = []
    for (hour, seat), arms in sorted(cells.items()):
        a, b = arms.get(candidate, []), arms.get(baseline, [])
        if not a or not b:
            continue
        weight = min(len(a), len(b))
        low = sum(r["rank_low"] for r in a) / len(a) - sum(
            r["rank_high"] for r in b
        ) / len(b)
        high = sum(r["rank_high"] for r in a) / len(a) - sum(
            r["rank_low"] for r in b
        ) / len(b)
        block = hours[hour]
        block["weight"] += weight
        block["low_sum"] += weight * low
        block["high_sum"] += weight * high
        block["candidate_games"] += len(a)
        block["baseline_games"] += len(b)
        cell_rows.append(
            {
                "hour_utc": datetime.fromtimestamp(
                    hour * 3600, timezone.utc
                ).isoformat(),
                "seat": seat,
                "candidate_games": len(a),
                "baseline_games": len(b),
                "overlap_weight": weight,
            }
        )
    blocks = list(hours.values())
    if not blocks:
        return {"available": False, "reason": "no common hour/seat support"}

    def bounds(sample):
        weight = sum(x["weight"] for x in sample)
        return [
            sum(x["low_sum"] for x in sample) / weight,
            sum(x["high_sum"] for x in sample) / weight,
        ]

    interval = None
    if len(blocks) >= 2:
        rng = random.Random(rng_seed)
        lo, hi = [], []
        for _ in range(iterations):
            value = bounds([blocks[rng.randrange(len(blocks))] for _ in blocks])
            lo.append(value[0])
            hi.append(value[1])
        interval = [percentile(sorted(lo), 0.025), percentile(sorted(hi), 0.975)]
    return {
        "available": True,
        "time_clusters": len(blocks),
        "common_cells": cell_rows,
        "included_candidate_games": sum(x["candidate_games"] for x in blocks),
        "included_baseline_games": sum(x["baseline_games"] for x in blocks),
        "rank_difference_bounds": bounds(blocks),
        "conservative_hour_cluster_ci95": interval,
        "method": "same UTC trace-completion hour and initial seat; min(arm counts) overlap weights; resample whole hours; widen for unresolved ties",
        "limitations": "observational sensitivity analysis, not randomized; opponent difficulty and account rating unavailable; hour is trace persistence time, not authoritative match start",
    }


def evaluate_online(
    snapshot_path, display_path, output_dir, *, prior_path=None, identity_path=None
):
    source = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    display = json.loads(Path(display_path).read_text(encoding="utf-8"))
    candidate, baseline = display["candidate"], display["baseline"]
    rows, duplicates = prepare_rows(source)
    arms = {
        version: arm_summary([r for r in rows if r["version"] == version])
        for version in (candidate, baseline)
    }
    a, b = arms[candidate], arms[baseline]
    gains = [
        b["mean_rank_bounds"][0] - a["mean_rank_bounds"][1],
        b["mean_rank_bounds"][1] - a["mean_rank_bounds"][0],
    ]
    report = {
        "valid": True,
        "analysis_kind": "online_observational",
        "candidate": candidate,
        "baseline": baseline,
        "observed_rank_gain_bounds": gains,
        "observed_rank_score_gain_bounds": [gain * 100 / 3 for gain in gains],
        "observed_leader": candidate
        if gains[0] > 0
        else baseline
        if gains[1] < 0
        else "unresolved",
        "causal_or_formal_promotion_verdict": "not_established_by_observational_data",
        "arms": arms,
        "screenshot_analysis": compare_displays(display),
        "time_seat_sensitivity": standardized_online(rows, candidate, baseline),
        "records_read": len(rows),
        "duplicate_records": duplicates,
        "read_errors": source["read_errors"],
        "screenshot_counts_not_assumed_same_cohort": {
            v: {
                "cloud_terminal": arms[v]["terminal_games"],
                "screenshot_games": display["versions"][v]["games"],
            }
            for v in arms
        },
        "deployment": [
            {
                "batch_id": batch["batch_id"],
                "backend": batch["backend"],
                "models": batch["models"],
                "current_weight_checks": batch["current_weight_checks"],
            }
            for batch in source["batches"]
        ],
        "source": {
            "captured_at": source["captured_at"],
            "read_only": source["read_only"],
            "extractor_sha256": source.get("extractor_sha256"),
            "snapshot_sha256": hashlib.sha256(
                Path(snapshot_path).read_bytes()
            ).hexdigest(),
            "display_transcription_sha256": hashlib.sha256(
                Path(display_path).read_bytes()
            ).hexdigest(),
        },
        "implementation_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "ranked_evaluation.py",
                "ranked_snapshot.py",
                "scoring.py",
                "evaluation.py",
            )
        },
        "limits": [
            "Do not pool local selfplay and online ranked means as the same estimand",
            "Do not exclude completed games from failed batches based on batch status; operational faults and unknown outcomes are reported",
            "Displayed all-time counts and cloud traces do not fully align; do not add them together",
            "Platform ratings are contextual observations, not interchangeable with linear rank scores",
            "Opponent identities/ratings and authoritative tie rules absent from stored MJAI protocol",
        ],
    }
    if prior_path:
        prior = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        prior = prior.get("formal_report", prior)
        report["prior_selfplay"] = {
            "primary": prior["primary"],
            "strength_verdict": prior["strength_verdict"],
            "scope": prior["comparison"],
            "historical_gate_preserved": True,
        }
        thresholds = prior.get("decision", {}).get("registered_thresholds", {})
        report["prior_selfplay"]["version_scorecard"] = rank_scorecard(
            prior["primary"]["mean_rank"],
            prior["primary"]["last_place_rate"],
            minimum_gain=thresholds.get("minimum_meaningful_rank_gain"),
            guardrail=thresholds.get("last_place_rate_guardrail"),
            scope="fixed baseline selfplay only",
        )
        if identity_path:
            identity = json.loads(Path(identity_path).read_text(encoding="utf-8"))
            cloud = identity["cloud_checkpoint"]
            checks = []
            for batch in source["batches"]:
                if batch["backend"]["revision"]["tag"] != candidate:
                    continue
                old = next(
                    b
                    for b in prior["backends"].values()
                    if b["revision"]["tag"] == candidate
                )
                for task in ("turn", "response"):
                    local = next(
                        (
                            c
                            for c in identity["local_checkpoints"]
                            if c["file_sha256"]
                            == old["model_identity"][task + "_sha256"]
                        ),
                        None,
                    )
                    if (
                        local is None
                        or batch["current_weight_checks"][task]["current_file_sha256"]
                        != cloud["file_sha256"]
                    ):
                        raise ValueError(
                            "tensor identity proof does not match observed file artifacts"
                        )
                    checks.append(
                        {
                            "batch_id": batch["batch_id"],
                            "task": task,
                            "inference_parameters_equal": local["tensor_fingerprints"][
                                task
                            ]
                            == cloud["tensor_fingerprints"][task],
                            "code_revision_equal": batch["backend"]["revision"][
                                "commit"
                            ]
                            == old["revision"]["commit"],
                            "file_artifact_changed": local["file_sha256"]
                            != cloud["file_sha256"],
                        }
                    )
            report["parameter_identity_check"] = {
                "checks": checks,
                "proof": identity,
                "scope": "current checkpoint tensors versus pinned A4 files; not a historical in-memory model attestation",
            }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "report.json", report)
    (out / "report.md").write_text(
        "# 线上版本比较\n\n线上观测领先："
        + report["observed_leader"]
        + "。\n\n"
        + f"云端终局顺位收益范围（处理未确认同分规则）：{gains}。截图观测收益：{report['screenshot_analysis']['observed_rank_gain']:.6f}。\n\n"
        + "分开报告观测收益、时间/席位敏感性和正式部署资格。对手强度未调整，旧 A4 门槛结论保持原样。详见 report.json。\n",
        encoding="utf-8",
    )
    return report
