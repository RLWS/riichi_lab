"""W3b same-state shadow replay.

For sampled W2 decision states, send the same state/candidate set to the native
v1.3.0 and v1.4.0 services and compare their selected actions.  The state is
normalized to each service's public feature contract: v2 drops own_discards;
v3 adds the adapter's own_discards alias.  No model or schedule is changed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random

from rilab.w3 import read_json, read_jsonl_gz, state_features

CALL_KINDS = ("chi", "pon", "daiminkan", "ron")


def _sample_evenly(items: list[dict], count: int) -> list[dict]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def sample_records(
    w2_dir: str | Path,
    *,
    games_per_arm: int = 32,
    max_turn_per_arm: int = 200,
    max_response_per_arm: int = 200,
) -> list[dict]:
    batch = Path(w2_dir)
    summary = read_json(batch / "summary.json")
    by_arm = defaultdict(list)
    for game in summary.get("games", []):
        by_arm[game.get("matchup_id")].append(game)
    records = []
    for arm in ("A", "B", "C", "D"):
        games = by_arm.get(arm, [])
        if not games:
            continue
        step = max(1, len(games) // games_per_arm)
        selected = games[::step][:games_per_arm]
        turn = []
        response = []
        for game in selected:
            focal = game.get("focal_seat")
            if focal is None:
                continue
            path = batch / "decisions" / f"{game['game_id']}.jsonl.gz"
            if not path.is_file():
                continue
            for record in read_jsonl_gz(path):
                if record.get("player_id") != focal:
                    continue
                task = record.get("task")
                item = {
                    "game_id": game["game_id"],
                    "arm": arm,
                    "seed": game.get("seed"),
                    "rotation": game.get("rotation"),
                    "focal": focal,
                    "decision_id": record.get("decision_id"),
                    "task": task,
                    "state": record.get("state"),
                    "candidates": record.get("candidates") or [],
                    "selected_action": record.get("selected_action"),
                    "prediction": record.get("prediction"),
                }
                if not item["candidates"]:
                    continue
                if task == "turn":
                    turn.append(item)
                elif task == "response":
                    response.append(item)
        records.extend(_sample_evenly(turn, max_turn_per_arm))
        records.extend(_sample_evenly(response, max_response_per_arm))
    return records


def _record_strata(record: dict, features: dict) -> list[str]:
    task = record.get("task")
    selected = record.get("selected_action") or {}
    selected_kind = selected.get("kind")
    kinds = {candidate.get("kind") for candidate in record.get("candidates") or []}
    opp = features.get("opp_riichi_count", 0)
    turn = features.get("turn_number")
    late = isinstance(turn, (int, float)) and turn > 12
    open_hand = features.get("own_meld_count", 0) > 0
    dora = features.get("dora_count", 0)
    strata = []
    if task == "turn":
        if selected_kind == "riichi_discard":
            strata.append("riichi_selected")
        if "riichi_discard" in kinds and selected_kind != "riichi_discard":
            strata.append("riichi_legal_pass")
        if opp > 0:
            strata.append("opp_riichi")
        if opp > 0 and late:
            strata.append("opp_riichi_late")
        if late:
            strata.append("late")
        if open_hand:
            strata.append("open_hand")
    elif task == "response":
        if kinds & set(CALL_KINDS):
            strata.append("call_legal")
        if selected_kind in CALL_KINDS:
            strata.append("call_selected")
        if "ron" in kinds:
            strata.append("ron_legal")
        if opp > 0:
            strata.append("opp_riichi")
        if opp > 0 and late:
            strata.append("opp_riichi_late")
        if late:
            strata.append("late")
        if open_hand:
            strata.append("open_hand")
        if dora >= 2:
            strata.append("high_dora")
    return strata


def sample_stratified_records(
    w2_dir: str | Path,
    *,
    games_per_arm: int = 512,
    max_per_stratum: int = 100,
    seed: int = 20261117,
) -> list[dict]:
    batch = Path(w2_dir)
    summary = read_json(batch / "summary.json")
    by_arm = defaultdict(list)
    for game in summary.get("games", []):
        by_arm[game.get("matchup_id")].append(game)
    rng = random.Random(seed)
    reservoirs = defaultdict(list)
    seen = Counter()

    def offer(key: str, record: dict) -> None:
        seen[key] += 1
        bucket = reservoirs[key]
        if len(bucket) < max_per_stratum:
            bucket.append(record)
            return
        index = rng.randrange(seen[key])
        if index < max_per_stratum:
            bucket[index] = record

    for arm in ("A", "B", "C", "D"):
        games = by_arm.get(arm, [])
        if not games:
            continue
        step = max(1, len(games) // games_per_arm)
        selected = games[::step][:games_per_arm]
        for game in selected:
            focal = game.get("focal_seat")
            if focal is None:
                continue
            path = batch / "decisions" / f"{game['game_id']}.jsonl.gz"
            if not path.is_file():
                continue
            for record in read_jsonl_gz(path):
                if record.get("player_id") != focal:
                    continue
                task = record.get("task")
                candidates = record.get("candidates") or []
                if not candidates:
                    continue
                features = state_features(record.get("state") or {})
                strata = _record_strata(record, features)
                if not strata:
                    continue
                item = {
                    "game_id": game["game_id"],
                    "arm": arm,
                    "seed": game.get("seed"),
                    "rotation": game.get("rotation"),
                    "focal": focal,
                    "decision_id": record.get("decision_id"),
                    "task": task,
                    "state": record.get("state"),
                    "candidates": candidates,
                    "selected_action": record.get("selected_action"),
                    "prediction": record.get("prediction"),
                    "features": features,
                    "strata": strata,
                }
                for stratum in strata:
                    offer(f"{arm}/{task}/{stratum}", item)

    unique = {}
    for key, records in reservoirs.items():
        for record in records:
            identity = (
                record["arm"],
                record["task"],
                record["game_id"],
                record["decision_id"],
            )
            if identity not in unique:
                unique[identity] = {
                    **record,
                    "strata": set(record["strata"]),
                }
            else:
                unique[identity]["strata"].update(record["strata"])
    output = []
    for record in unique.values():
        record["strata"] = sorted(record["strata"])
        output.append(record)
    output.sort(
        key=lambda item: (
            item["arm"],
            item["task"],
            item["seed"] or 0,
            item["rotation"] or 0,
            item["decision_id"] or 0,
        )
    )
    return output
    normalized = deepcopy(state)
    # full_state_v2 is the public HTTP state contract for both services.
    # The v3 checkpoint uses the same state plus the own_discards alias;
    # the v3 candidate encoding happens inside the service.
    normalized["feature_schema"] = "full_state_v2"
    if profile == "full_state_v3":
        actor = normalized.get("actor")
        all_discards = normalized.get("all_discards") or []
        if actor is not None and actor < len(all_discards):
            normalized["own_discards"] = list(all_discards[actor])
        else:
            normalized.setdefault("own_discards", [])
    else:
        normalized.pop("own_discards", None)
    return normalized


def normalize_state(state: dict, profile: str) -> dict:
    normalized = deepcopy(state)
    # full_state_v2 is the public HTTP state contract for both services.
    # The v3 checkpoint uses the same state plus the own_discards alias;
    # the v3 candidate encoding happens inside the service.
    normalized["feature_schema"] = "full_state_v2"
    if profile == "full_state_v3":
        actor = normalized.get("actor")
        all_discards = normalized.get("all_discards") or []
        if actor is not None and actor < len(all_discards):
            normalized["own_discards"] = list(all_discards[actor])
        else:
            normalized.setdefault("own_discards", [])
    else:
        normalized.pop("own_discards", None)
    return normalized


def action_key(action) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def action_class(task: str, action) -> str:
    if not isinstance(action, dict):
        return "invalid"
    return str(action.get("kind"))


def expected_source_backend(arm: str, task: str) -> str:
    if task == "turn":
        return "v140" if arm in ("B", "D") else "v130"
    return "v140" if arm in ("C", "D") else "v130"


def replay_records(records: list[dict], v130, v140) -> list[dict]:
    rows = []
    for record in records:
        task = record["task"]
        base = {
            "game_id": record["game_id"],
            "arm": record["arm"],
            "seed": record["seed"],
            "rotation": record["rotation"],
            "focal": record["focal"],
            "decision_id": record["decision_id"],
            "task": task,
            "source_selected": record["selected_action"],
            "features": state_features(record["state"]),
            "strata": record.get("strata", []),
            "candidates": len(record["candidates"]),
        }
        try:
            v130_result = v130.predict(
                task,
                normalize_state(record["state"], "full_state_v2"),
                record["candidates"],
            )
            v140_result = v140.predict(
                task,
                normalize_state(record["state"], "full_state_v3"),
                record["candidates"],
            )
            v130_action = v130_result.get("selected_action")
            v140_action = v140_result.get("selected_action")
            source_key = action_key(record["selected_action"])
            expected = expected_source_backend(record["arm"], task)
            rows.append(
                {
                    **base,
                    "v130_selected": v130_action,
                    "v140_selected": v140_action,
                    "v130_raw": v130_result.get("raw_selected_action"),
                    "v140_raw": v140_result.get("raw_selected_action"),
                    "v130_class": action_class(task, v130_action),
                    "v140_class": action_class(task, v140_action),
                    "agree": action_key(v130_action) == action_key(v140_action),
                    "expected_source": expected,
                    "source_match_expected": (
                        action_key(v130_action) == source_key
                        if expected == "v130"
                        else action_key(v140_action) == source_key
                    ),
                    "source_match_v130": action_key(v130_action) == source_key,
                    "source_match_v140": action_key(v140_action) == source_key,
                    "error": None,
                }
            )
        except BaseException as exc:
            rows.append({**base, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def _is_call(response_class: str) -> bool:
    return response_class in ("chi", "pon", "daiminkan", "ron")


def _is_riichi(turn_class: str) -> bool:
    return turn_class == "riichi_discard"


def _turn_bucket(turn_number) -> str:
    if not isinstance(turn_number, (int, float)):
        return "unknown"
    if turn_number <= 6:
        return "early"
    if turn_number <= 12:
        return "mid"
    return "late"


def _counter_row(rows: list[dict]) -> dict:
    counter = Counter()
    class_pairs = Counter()
    for row in rows:
        if row.get("error"):
            counter["errors"] += 1
            continue
        counter["n"] += 1
        counter["agree"] += bool(row["agree"])
        counter["source_match"] += bool(row["source_match_expected"])
        counter["v130_class_" + row["v130_class"]] += 1
        counter["v140_class_" + row["v140_class"]] += 1
        class_pairs[f"{row['v130_class']}->{row['v140_class']}"] += 1
        if row["task"] == "response":
            v130_call = _is_call(row["v130_class"])
            v140_call = _is_call(row["v140_class"])
            counter["v130_call"] += v130_call
            counter["v140_call"] += v140_call
            counter["both_call"] += v130_call and v140_call
            counter["both_pass"] += not v130_call and not v140_call
            counter["v130_call_only"] += v130_call and not v140_call
            counter["v140_call_only"] += v140_call and not v130_call
        else:
            v130_riichi = _is_riichi(row["v130_class"])
            v140_riichi = _is_riichi(row["v140_class"])
            counter["v130_riichi"] += v130_riichi
            counter["v140_riichi"] += v140_riichi
            counter["both_riichi"] += v130_riichi and v140_riichi
            counter["v130_riichi_only"] += v130_riichi and not v140_riichi
            counter["v140_riichi_only"] += v140_riichi and not v130_riichi
    counter["class_pairs"] = dict(class_pairs)
    counter["agree_rate"] = counter["agree"] / counter["n"] if counter["n"] else None
    counter["source_match_rate"] = (
        counter["source_match"] / counter["n"] if counter["n"] else None
    )
    for prefix in ("v130_call", "v140_call", "both_call", "both_pass",
                   "v130_call_only", "v140_call_only", "v130_riichi",
                   "v140_riichi", "both_riichi", "v130_riichi_only",
                   "v140_riichi_only"):
        counter[prefix + "_rate"] = (
            counter[prefix] / counter["n"] if counter["n"] else None
        )
    return dict(counter)


def aggregate(rows: list[dict]) -> dict:
    by_arm_task = defaultdict(list)
    by_stratum = defaultdict(list)
    by_named = defaultdict(list)
    for row in rows:
        by_arm_task[(row["arm"], row["task"])].append(row)
        for stratum in row.get("strata", []):
            by_named[(row["arm"], row["task"], stratum)].append(row)
        features = row.get("features") or {}
        opp = "opp_riichi" if features.get("opp_riichi_count", 0) else "no_opp_riichi"
        bucket = _turn_bucket(features.get("turn_number"))
        by_stratum[(row["arm"], row["task"], opp, bucket)].append(row)
    return {
        "rows": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "by_arm_task": {
            f"{arm}/{task}": _counter_row(group)
            for (arm, task), group in sorted(by_arm_task.items())
        },
        "by_named_stratum": {
            f"{arm}/{task}/{stratum}": _counter_row(group)
            for (arm, task, stratum), group in sorted(by_named.items())
        },
        "by_stratum": {
            f"{arm}/{task}/{opp}/{bucket}": _counter_row(group)
            for (arm, task, opp, bucket), group in sorted(by_stratum.items())
        },
        "limitations": [
            "Same-state one-step comparison, not downstream trajectory.",
            "Sampled W2 decisions; local CPU services.",
            "No shanten/wait-value calculation.",
        ],
    }


def render_report(result: dict) -> str:
    lines = [
        "# W3b same-state shadow replay",
        "",
        f"- sampled decisions: `{result['rows']}`",
        f"- errors: `{result['errors']}`",
        "",
        "## Agreement / source match",
        "",
        "| Arm/Task | n | v130=v140 | expected source match |",
        "|---|---:|---:|---:|",
    ]
    for key, item in result["by_arm_task"].items():
        lines.append(
            f"| {key} | {item.get('n', 0)} | "
            f"{_pct(item.get('agree_rate'))} | "
            f"{_pct(item.get('source_match_rate'))} |"
        )
    lines += [
        "",
        "## Response call/pass transitions",
        "",
        "| Arm | n | v130 call | v140 call | both call | both pass | v130 call only | v140 call only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_arm_task"].items():
        if not key.endswith("/response"):
            continue
        lines.append(
            f"| {key.split('/')[0]} | {item.get('n', 0)} | "
            f"{_pct(item.get('v130_call_rate'))} | "
            f"{_pct(item.get('v140_call_rate'))} | "
            f"{_pct(item.get('both_call_rate'))} | "
            f"{_pct(item.get('both_pass_rate'))} | "
            f"{_pct(item.get('v130_call_only_rate'))} | "
            f"{_pct(item.get('v140_call_only_rate'))} |"
        )
    lines += [
        "",
        "## Turn riichi/other transitions",
        "",
        "| Arm | n | v130 riichi | v140 riichi | both riichi | v130 only | v140 only |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_arm_task"].items():
        if not key.endswith("/turn"):
            continue
        lines.append(
            f"| {key.split('/')[0]} | {item.get('n', 0)} | "
            f"{_pct(item.get('v130_riichi_rate'))} | "
            f"{_pct(item.get('v140_riichi_rate'))} | "
            f"{_pct(item.get('both_riichi_rate'))} | "
            f"{_pct(item.get('v130_riichi_only_rate'))} | "
            f"{_pct(item.get('v140_riichi_only_rate'))} |"
        )
    lines += [
        "",
        "## Named strata",
        "",
        "| Stratum | n | agree | source match | v130 | v140 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in result.get("by_named_stratum", {}).items():
        arm, task, stratum = key.split("/", 2)
        if task == "turn" and stratum not in (
            "riichi_selected",
            "riichi_legal_pass",
            "opp_riichi_late",
        ):
            continue
        if task == "response" and stratum not in (
            "call_legal",
            "opp_riichi",
            "opp_riichi_late",
            "ron_legal",
            "high_dora",
        ):
            continue
        if task == "turn":
            v130_value = item.get("v130_riichi_rate")
            v140_value = item.get("v140_riichi_rate")
        else:
            v130_value = item.get("v130_call_rate")
            v140_value = item.get("v140_call_rate")
        lines.append(
            f"| {key} | {item.get('n', 0)} | "
            f"{_pct(item.get('agree_rate'))} | "
            f"{_pct(item.get('source_match_rate'))} | "
            f"{_pct(v130_value)} | {_pct(v140_value)} |"
        )
    lines += [
        "",
        "## Key strata",
        "",
        "Feature-based C response and D turn strata:",
        "",
        "```json",
        json.dumps(
            {
                key: value
                for key, value in result["by_stratum"].items()
                if (
                    key.startswith("C/response/")
                    or key.startswith("D/turn/")
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
    ]
    return "\n".join(lines)


def _pct(value):
    return f"{value:.2%}" if isinstance(value, (int, float)) else "n/a"
