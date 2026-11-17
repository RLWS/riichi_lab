"""W3c outcome linkage for W3b2 stratified shadow states.

Reads the W3b2 results and joins each sampled decision to the source hand's
actual outcome.  This is descriptive: the executed source action is observed,
the other version's action is only a shadow counterfactual and is not executed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from rilab.w3 import hand_metrics, read_json, read_jsonl, read_jsonl_gz
from rilab.w3_shadow import action_class


def _stats(rows: list[dict]) -> dict:
    unique = {(row["game_id"], row["hand_id"]) for row in rows}
    wins = sum(1 for row in rows if row["hand"]["win"])
    deals = sum(1 for row in rows if row["hand"]["deal"])
    income = sum(row["hand"]["income"] for row in rows)
    loss = sum(row["hand"]["loss"] for row in rows)
    return {
        "n": len(rows),
        "unique_hands": len(unique),
        "wins": wins,
        "deals": deals,
        "win_rate": wins / len(rows) if rows else None,
        "deal_rate": deals / len(rows) if rows else None,
        "income_per_hand": income / len(rows) if rows else None,
        "loss_per_hand": loss / len(rows) if rows else None,
        "mean_win_income": income / wins if wins else None,
        "mean_deal_loss": loss / deals if deals else None,
        "hand_riichi_rate": (
            sum(1 for row in rows if row["hand"]["riichi"]) / len(rows)
            if rows
            else None
        ),
        "hand_open_rate": (
            sum(1 for row in rows if row["hand"]["open"]) / len(rows)
            if rows
            else None
        ),
    }


def load_outcome_rows(w3b2_dir: str | Path, w2_dir: str | Path) -> list[dict]:
    results = [
        json.loads(line)
        for line in (Path(w3b2_dir) / "results.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    by_game = defaultdict(list)
    for row in results:
        by_game[row["game_id"]].append(row)
    output = []
    for game_id, game_rows in by_game.items():
        focal = game_rows[0]["focal"]
        game_dir = Path(w2_dir) / "games" / f"{game_id}.mjai.jsonl"
        decision_path = Path(w2_dir) / "decisions" / f"{game_id}.jsonl.gz"
        if not game_dir.is_file() or not decision_path.is_file():
            for row in game_rows:
                output.append({**row, "hand_id": None, "outcome_missing": True})
            continue
        events = list(read_jsonl(game_dir))
        hands = hand_metrics(events, focal)
        hand_by_decision = {}
        for record in read_jsonl_gz(decision_path):
            if record.get("player_id") != focal:
                continue
            hand_by_decision[record.get("decision_id")] = record.get("hand_id")
        for row in game_rows:
            hand_id = hand_by_decision.get(row["decision_id"])
            if not isinstance(hand_id, int) or not 1 <= hand_id <= len(hands):
                output.append(
                    {
                        **row,
                        "hand_id": hand_id,
                        "outcome_missing": True,
                        "source_class": action_class(row["task"], row["source_selected"]),
                        "other_class": (
                            row["v140_class"]
                            if row["expected_source"] == "v130"
                            else row["v130_class"]
                        ),
                    }
                )
                continue
            hand = hands[hand_id - 1]
            output.append(
                {
                    **row,
                    "hand_id": hand_id,
                    "outcome_missing": False,
                    "source_class": action_class(row["task"], row["source_selected"]),
                    "other_class": (
                        row["v140_class"]
                        if row["expected_source"] == "v130"
                        else row["v130_class"]
                    ),
                    "hand": hand,
                }
            )
    return output


def aggregate(rows: list[dict]) -> dict:
    by_source = defaultdict(list)
    by_source_other = defaultdict(list)
    for row in rows:
        if row.get("outcome_missing"):
            continue
        for stratum in row.get("strata", []):
            key = (row["arm"], row["task"], stratum, row["source_class"])
            by_source[key].append(row)
            by_source_other[key + (row["other_class"],)].append(row)
    return {
        "rows": len(rows),
        "matched": sum(1 for row in rows if not row.get("outcome_missing")),
        "missing": sum(1 for row in rows if row.get("outcome_missing")),
        "by_source": {
            "/".join(key): _stats(group)
            for key, group in sorted(by_source.items())
        },
        "by_source_other": {
            "/".join(key): _stats(group)
            for key, group in sorted(by_source_other.items())
        },
        "limitations": [
            "Outcome linkage is descriptive: the source action was executed; the other version's action is a shadow counterfactual and was not executed.",
            "Multiple sampled decisions can come from the same hand; unique_hands is reported separately.",
            "No shanten/wait-value calculation; local fixed-opponent games.",
        ],
    }


def _fmt(value, pct=False):
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.2%}" if pct else f"{value:.2f}"


def render_report(result: dict) -> str:
    lines = [
        "# W3c outcome linkage",
        "",
        f"- rows: `{result['rows']}`; matched: `{result['matched']}`; missing: `{result['missing']}`",
        "",
        "## Turn: selected riichi",
        "",
        "| Arm | n | unique hands | Win | Deal | Income/hand | Loss/hand |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source"].items():
        arm, task, stratum, source_class = key.split("/", 3)
        if task != "turn" or stratum != "riichi_selected":
            continue
        lines.append(
            f"| {arm} | {item['n']} | {item['unique_hands']} | "
            f"{_fmt(item['win_rate'], True)} | {_fmt(item['deal_rate'], True)} | "
            f"{_fmt(item['income_per_hand'])} | {_fmt(item['loss_per_hand'])} |"
        )
    lines += [
        "",
        "## Turn: riichi legal but source passed",
        "",
        "`other_class` is what the other Turn version would do on the same state.",
        "",
        "| Arm | source | other | n | unique | Win | Deal | Hand riichi | Income/hand | Loss/hand |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source_other"].items():
        arm, task, stratum, source_class, other_class = key.split("/", 4)
        if task != "turn" or stratum != "riichi_legal_pass":
            continue
        lines.append(
            f"| {arm} | {source_class} | {other_class} | {item['n']} | "
            f"{item['unique_hands']} | {_fmt(item['win_rate'], True)} | "
            f"{_fmt(item['deal_rate'], True)} | {_fmt(item['hand_riichi_rate'], True)} | "
            f"{_fmt(item['income_per_hand'])} | {_fmt(item['loss_per_hand'])} |"
        )
    lines += [
        "",
        "## Turn: opponent riichi + late",
        "",
        "| Arm | source | n | Win | Deal | Income/hand | Loss/hand |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source"].items():
        arm, task, stratum, source_class = key.split("/", 3)
        if task != "turn" or stratum != "opp_riichi_late":
            continue
        lines.append(
            f"| {arm} | {source_class} | {item['n']} | "
            f"{_fmt(item['win_rate'], True)} | {_fmt(item['deal_rate'], True)} | "
            f"{_fmt(item['income_per_hand'])} | {_fmt(item['loss_per_hand'])} |"
        )
    lines += [
        "",
        "## Response: opponent riichi",
        "",
        "| Arm | source | other | n | unique | Win | Deal | Income/hand | Loss/hand |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source_other"].items():
        arm, task, stratum, source_class, other_class = key.split("/", 4)
        if task != "response" or stratum != "opp_riichi":
            continue
        lines.append(
            f"| {arm} | {source_class} | {other_class} | {item['n']} | "
            f"{item['unique_hands']} | {_fmt(item['win_rate'], True)} | "
            f"{_fmt(item['deal_rate'], True)} | {_fmt(item['income_per_hand'])} | "
            f"{_fmt(item['loss_per_hand'])} |"
        )
    lines += [
        "",
        "## Response: high dora",
        "",
        "| Arm | source | other | n | unique | Win | Deal | Income/hand | Loss/hand |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source_other"].items():
        arm, task, stratum, source_class, other_class = key.split("/", 4)
        if task != "response" or stratum != "high_dora":
            continue
        lines.append(
            f"| {arm} | {source_class} | {other_class} | {item['n']} | "
            f"{item['unique_hands']} | {_fmt(item['win_rate'], True)} | "
            f"{_fmt(item['deal_rate'], True)} | {_fmt(item['income_per_hand'])} | "
            f"{_fmt(item['loss_per_hand'])} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
    ]
    return "\n".join(lines)
