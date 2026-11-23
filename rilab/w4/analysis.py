"""Account/date/seat/score-vector linkage and descriptive W4 comparison."""

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, quantiles

from .public import CONFIG, read, save
from .traces import PROFILES

BOTS = (493, 521)


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def link(games, sessions, tolerance):
    index = defaultdict(list)
    for g in games:
        index[g["bot"], g["seat"], tuple(g["scores"])].append(g)
    links = {}
    claims = defaultdict(list)
    for s in sessions:
        key = s["trace_sha256"]
        if key in links:
            links[key]["aliases"].append(s["source_id"])
            continue
        candidates = [
            g
            for g in index[s["bot"], s["seat"], tuple(s["scores"] or [])]
            if abs(utc(g["played_at"]) - s["trace_mtime_utc"]) <= tolerance
        ]
        state = "incomplete" if not s["scores"] else "unmatched"
        row = dict(
            s, aliases=[], candidates=[g["game_id"] for g in candidates], status=state
        )
        if len(candidates) > 1:
            row["status"] = "ambiguous"
        for g in candidates:
            claims[s["bot"], g["game_id"]].append(row)
        links[key] = row
    for (_, game_id), rows in claims.items():
        for r in rows:
            if len(rows) == len(r["candidates"]) == 1:
                r.update(game_id=game_id, status="unique")
            else:
                r["status"] = "ambiguous"
    return list(links.values())


def outcome(rows):
    return [
        mean(g["rank"] for g in rows),
        mean(int(g["rank"] == 4) for g in rows),
        mean(g["score"] for g in rows),
    ]


def contrast(rows, keys):
    cells = defaultdict(lambda: defaultdict(list))
    for g in rows:
        cells[tuple(g[k] for k in keys)][g["bot"]].append(g)
    sums = [0.0, 0.0, 0.0]
    n = Counter()
    dates = set()
    count = weight = 0
    for group in cells.values():
        if not all(group[b] for b in BOTS):
            continue
        w = min(len(group[b]) for b in BOTS)
        for i, (new, old) in enumerate(zip(outcome(group[521]), outcome(group[493]))):
            sums[i] += w * (new - old)
        weight += w
        count += 1
        for b in BOTS:
            n[b] += len(group[b])
            dates.update(g["date"] for g in group[b])
    return dict(
        cells=count,
        represented=n,
        weight=weight,
        days=sorted(dates),
        delta=[x / weight for x in sums] if weight else None,
    )


def comparison(recent):
    report = {}
    for g in recent:
        g["date"] = g["played_at"][:10]
        g["hour"] = g["played_at"][:13]
        g["self_band"] = g["rating_before"] // 100
        for key in ("rating_before", "mu_before"):
            values = [p[key] for p in g["opponents"]]
            g["opp_" + key] = mean(values) if None not in values else None
        g["opp_band"] = (
            int(mean(p["rating_before"] for p in g["opponents"]) // 100)
            if all(p["rating_before"] is not None for p in g["opponents"])
            else None
        )
    for keys in [
        [],
        ["date"],
        ["date", "seat"],
        ["date", "seat", "opp_band"],
        ["date", "seat", "opp_band", "self_band"],
        ["hour", "seat", "opp_band", "self_band"],
    ]:
        data = [g for g in recent if all(g[k] is not None for k in keys)]
        report["+".join(keys) or "crude"] = contrast(data, keys)
    dates = sorted(set(g["date"] for g in recent))
    # Avoid treating opposite edges of a wide band as actual covariate overlap.
    measures = ("rating_before", "mu_before", "opp_rating_before", "opp_mu_before")

    def shared_range(rows):
        arms = [
            [
                g
                for g in rows
                if g["bot"] == b and all(g[k] is not None for k in measures)
            ]
            for b in BOTS
        ]
        if not all(arms):
            return {}, []
        bounds = {
            k: [
                max(min(g[k] for g in a) for a in arms),
                min(max(g[k] for g in a) for a in arms),
            ]
            for k in measures
        }
        return bounds, [
            g
            for a in arms
            for g in a
            if all(lo <= g[k] <= hi for k, (lo, hi) in bounds.items())
        ]

    bounds, overlap = shared_range(recent)
    report["across_date_range"] = dict(
        bounds=bounds, **contrast(overlap, ["seat", "self_band", "opp_band"])
    )
    report["within_date_range"] = {}
    for date in dates:
        bounds, overlap = shared_range([g for g in recent if g["date"] == date])
        report["within_date_range"][date] = dict(
            bounds=bounds, **contrast(overlap, ["seat", "self_band", "opp_band"])
        )
    report["leave_day_out"] = {
        d: contrast(
            [g for g in recent if g["date"] != d],
            ["date", "seat", "opp_band", "self_band"],
        )
        for d in dates
    }

    # Resample complete calendar days jointly across accounts, including shared games.
    def totals(rows):
        return [
            len(rows),
            sum(g["rank"] for g in rows),
            sum(g["rank"] == 4 for g in rows),
            sum(g["score"] for g in rows),
        ]

    rng = random.Random(20261118)
    groups = {
        d: [
            totals([g for g in recent if g["date"] == d and g["bot"] == b])
            for b in BOTS
        ]
        for d in dates
    }
    draws = []
    for _ in range(10000):
        sample = rng.choices(dates, k=len(dates))
        old, new = [
            [sum(groups[d][arm][i] for d in sample) for i in range(4)] for arm in (0, 1)
        ]
        if old[0] and new[0]:
            draws.append([new[i] / new[0] - old[i] / old[0] for i in (1, 2, 3)])
    intervals = [
        quantiles(values, n=1000, method="inclusive") for values in zip(*draws)
    ]
    report["day_cluster_bootstrap"] = dict(
        valid=len(draws), days=dates, ci=[[q[24], q[974]] for q in intervals]
    )
    return report


def profile(rows):
    rows = sorted(rows, key=lambda g: g["played_at"])
    return dict(
        n=len(rows),
        start=rows[0]["played_at"],
        end=rows[-1]["played_at"],
        outcome=outcome(rows),
        ranks=Counter(g["rank"] for g in rows),
        seats=Counter(g["seat"] for g in rows),
        days=Counter(g["played_at"][:10] for g in rows),
        rating_bins=Counter(g["rating_before"] // 100 * 100 for g in rows),
        mean_rating=mean(g["rating_before"] for g in rows),
        mean_mu=mean(g["mu_before"] for g in rows),
        rating=[
            min(g["rating_before"] for g in rows),
            median(g["rating_before"] for g in rows),
            max(g["rating_before"] for g in rows),
        ],
        mu=[
            min(g["mu_before"] for g in rows),
            median(g["mu_before"] for g in rows),
            max(g["mu_before"] for g in rows),
        ],
        flagged=sum(g["is_disconnected"] or g["is_penalized"] for g in rows),
    )


def behavior(rows, root, shadow=None):
    stats = Counter()
    cal = Counter()
    schemas = Counter()
    for g, s, trace in rows:
        outcomes = trace["hand_results"]
        chosen = {}
        stats["games"] += 1
        for hand in outcomes:
            prefix = "open_" if hand["open"] else "closed_"
            stats[prefix + "hands"] += 1
            for k in ("win", "deal", "loss", "income"):
                stats[prefix + k] += hand[k]
        with gzip.open(
            root / "decisions" / (s["trace_sha256"] + ".jsonl.gz"), "rt"
        ) as stream:
            for r in map(json.loads, stream):
                if shadow is not None:
                    shadow.write(
                        json.dumps(
                            dict(
                                r,
                                game_id=g["game_id"],
                                bot=g["bot"],
                                trace_sha256=s["trace_sha256"],
                                model_identity=s["model_identity"],
                            ),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                pred = r.get("prediction", {})
                ex = r.get("execution", {})
                actual = ex.get("actual_action") or {}
                kinds = {a["kind"] for a in r["candidates"]}
                act = actual.get("type") if ex.get("status") == "accepted" else None
                sel = pred.get("selected_action", {}).get("kind")
                schemas[pred.get("model", {}).get("feature_schema", "missing")] += 1
                stats["requests"] += 1
                stats["missing_prediction"] += not bool(pred)
                stats["forced_without_prediction"] += (
                    bool(r.get("forced_riichi_discard")) and not pred
                )
                stats["missing_accepted"] += act is None
                stats["fallback"] += bool(r.get("reply", {}).get("fallback"))
                if r["task"] == "response":
                    if pred:
                        cal[
                            json.dumps(
                                [
                                    pred.get(k)
                                    for k in (
                                        "action_biases",
                                        "pass_bias",
                                        "no_yaku_call_penalty",
                                    )
                                ],
                                sort_keys=True,
                            )
                        ] += 1
                    for action in ("chi", "pon", "daiminkan"):
                        if action in kinds:
                            stats[action + "_opp"] += 1
                            stats[action + "_model"] += sel == action
                            stats[action + "_accepted"] += act == action
                if (
                    r["task"] == "turn"
                    and "riichi_discard" in kinds
                    and r["opponent_accepted"]
                    and r["state"]["turn_number"] > 12
                ):
                    stats["late_opp"] += 1
                    stats["late_model"] += sel == "riichi_discard"
                    stats["late_reach"] += act == "reach"
                    stats["late_unknown"] += act is None
                    # Fixed first eligible decision per hand for conditional outcomes.
                    chosen.setdefault(r["hand_index"], act)
        for i, act in chosen.items():
            if i >= len(outcomes):
                stats["late_incomplete"] += 1
                continue
            prefix = (
                "late_"
                + ("reach" if act == "reach" else "other" if act else "unknown")
                + "_"
            )
            stats[prefix + "hands"] += 1
            for k in ("deal", "win", "income", "loss"):
                stats[prefix + k] += outcomes[i][k]
    return dict(counts=stats, calibration=cal, schema=schemas)


def main(public_only=False):
    cfg = read(CONFIG)
    root = Path(cfg["output"])
    recent = read(root / "recent.json")
    games = read(root / "history.json")
    summary = {"comparison": comparison(recent)}
    sets = []
    for bot in BOTS:
        arm = [g for g in recent if g["bot"] == bot]
        history = sorted(
            [g for g in games if g["bot"] == bot], key=lambda g: g["played_at"]
        )
        opponents = [p for g in arm for p in g["opponents"]]
        opponent_means = {}
        for key in ("rating_before", "mu_before"):
            values = [p[key] for p in opponents if p[key] is not None]
            opponent_means[key] = mean(values) if values else None
        ids = Counter(p["bot_id"] for p in opponents)
        sets.append(set(ids))
        ordered_recent = sorted(arm, key=lambda g: g["played_at"])
        window = cfg["window"]
        summary[str(bot)] = dict(
            recent=profile(arm),
            history=profile(history),
            rating_discontinuities=sum(
                a["rating_before"] + a["rating_delta"] != b["rating_before"]
                for a, b in zip(history, history[1:])
            ),
            ties=sum(len(set(g["scores"])) != 4 for g in arm),
            anomalous_score_sum=sum(sum(g["scores"]) != 100000 for g in arm),
            account_names=sorted({g["bot_name"] for g in arm}),
            preceding=profile(history[-2 * window : -window]),
            quarters=[
                profile(ordered_recent[i : i + 50]) for i in range(0, len(arm), 50)
            ],
            opponent_ids=ids,
            opponent_means=opponent_means,
            missing_opponent=sum(
                any(p[k] is None for k in ("rating_before", "mu_before"))
                for p in opponents
            ),
            flagged_opponent_seats=sum(
                p["is_disconnected"] or p["is_penalized"] for p in opponents
            ),
        )
    summary["shared_opponents"] = sorted(sets[0] & sets[1])
    for bot in BOTS:
        counts = summary[str(bot)]["opponent_ids"]
        summary[str(bot)]["shared_opponent_seats"] = sum(
            counts[k] for k in summary["shared_opponents"]
        )
    summary["shared_recent_games"] = len(recent) - len({g["game_id"] for g in recent})
    save(root / "public_analysis.json", summary)
    if public_only:
        return
    games = read(root / "tables.json")
    sessions = []
    for batch in read(cfg["snapshot"])["batches"]:
        backend = batch["backend"]
        if backend["adapter_profile"] in PROFILES:
            sessions.extend(
                dict(
                    g,
                    bot=PROFILES[backend["adapter_profile"]],
                    model_identity=backend["model_identity"],
                )
                for g in batch["games"]
            )
    local = read(root / "local_sessions.json")
    traces = {s["trace_sha256"]: s for s in local}
    if (root / "cloud_traces.json").exists():
        traces.update(read(root / "cloud_traces.json"))
    sessions.extend({k: v for k, v in s.items() if k != "hand_results"} for s in local)
    links = link(games, sessions, cfg["link_tolerance_seconds"])
    by_game = {(s["bot"], s["game_id"]): s for s in links if s["status"] == "unique"}
    for g in games:
        if (s := by_game.get((g["bot"], g["game_id"]))) is not None:
            s.update(
                {
                    k: g[k]
                    for k in (
                        "played_at",
                        "rank",
                        "score",
                        "rating_before",
                        "mu_before",
                        "rating_delta",
                        "opponents",
                        "recent",
                        "bot_name",
                    )
                }
            )
            s["decision_shard"] = (
                "decisions/" + s["trace_sha256"] + ".jsonl.gz"
                if s["trace_sha256"] in traces
                else None
            )
    save(root / "linked_games.json", links)
    for bot in BOTS:
        result = summary[str(bot)]
        result["links"] = Counter(s["status"] for s in links if s["bot"] == bot)
        matched = [
            (g, by_game[(bot, g["game_id"])])
            for g in games
            if g["bot"] == bot and (bot, g["game_id"]) in by_game
        ]
        for label, pairs in [
            ("recent", [(g, s) for g, s in matched if g["recent"]]),
            ("older", [(g, s) for g, s in matched if not g["recent"]]),
        ]:
            eligible = [
                (g, s, traces[s["trace_sha256"]])
                for g, s in pairs
                if s["trace_sha256"] in traces
                and s["model_identity"] == cfg["expected_model_identity"][str(bot)]
            ]
            result[label + "_linked"] = len(pairs)
            result[label + "_identity"] = Counter(
                json.dumps(s["model_identity"], sort_keys=True) for g, s in pairs
            )
            if label == "recent":
                batches = defaultdict(list)
                for g, s in pairs:
                    batches[s["source_id"].rsplit("/", 1)[0]].append(g)
                result["batch_sizes"] = {
                    batch: len(rows) for batch, rows in batches.items()
                }
                result["leave_batch_out"] = {
                    batch: outcome(
                        [
                            g
                            for other, rows in batches.items()
                            if batch != other
                            for g in rows
                        ]
                    )
                    for batch in batches
                    if len(batches) > 1
                }
                path = root / f"shadow_recent_{bot}.jsonl.gz"
                tmp = path.with_suffix(".tmp")
                with gzip.open(tmp, "wt", encoding="utf8", compresslevel=2) as stream:
                    result[label + "_behavior"] = behavior(eligible, root, stream)
                tmp.replace(path)
            else:
                result[label + "_behavior"] = behavior(eligible, root)
            if label == "recent":
                keep = [g for g, s, t in eligible if not s.get("fallbacks", 0)]
                result["without_snapshot_fallback"] = dict(
                    n=len(keep), outcome=outcome(keep) if keep else None
                )
    save(root / "analysis.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-only", action="store_true")
    main(parser.parse_args().public_only)
