"""W3 diagnostics for local decision logs.

This module is intentionally read-only: it consumes W2/A3 event and decision
artifacts, computes per-arm/per-stratum descriptive metrics, and never calls a
model or changes a schedule.  The online 200-game sample is not included here
because the current cloud snapshot does not export model inputs; that part
needs a separate sanitized trace export.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_gz(path: str | Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dora_tile(indicator: str) -> str:
    if not indicator or len(indicator) < 2:
        return indicator
    suit = indicator[-1]
    rank = int(indicator[0])
    if suit in ("m", "p", "s"):
        return f"{rank % 9 + 1}{suit}"
    if suit == "z":
        if 1 <= rank <= 4:
            return f"{rank % 4 + 1}z"
        if 5 <= rank <= 7:
            return f"{5 + (rank - 4) % 3}z"
    return indicator


def state_features(state: dict) -> dict:
    actor = state.get("actor")
    hand = state.get("hand", {})
    own_melds = state.get("own_melds", [])
    indicators = state.get("dora_indicators", [])
    dora_tiles = [dora_tile(item) for item in indicators]
    tiles = Counter(hand)
    for meld in own_melds:
        for tile in meld.get("consumed_tiles", []):
            tiles[tile] += 1
        if meld.get("called_tile"):
            tiles[meld["called_tile"]] += 1
    dora_count = sum(tiles[tile] for tile in dora_tiles)
    scores = state.get("scores") or [0, 0, 0, 0]
    riichi_by = state.get("riichi_declared_by_player") or [False] * 4
    own_riichi = bool(riichi_by[actor]) if actor is not None else False
    opp_riichi = sum(bool(item) for item in riichi_by) - (1 if own_riichi else 0)
    return {
        "hand_size": sum(hand.values()),
        "dora_count": dora_count,
        "own_riichi": own_riichi,
        "opp_riichi_count": max(0, opp_riichi),
        "turn_number": state.get("turn_number"),
        "round_number": state.get("round_number"),
        "remaining_tiles": state.get("remaining_tiles"),
        "score": scores[actor] if actor is not None and actor < len(scores) else None,
        "is_dealer": actor == state.get("dealer") if actor is not None else False,
        "own_meld_count": len(own_melds),
        "hand": dict(hand),
        "dora_indicators": list(indicators),
        "dora_tiles": dora_tiles,
    }


def hand_metrics(events: list[dict], focal: int) -> list[dict]:
    hands = []
    current = None

    def finalize():
        nonlocal current
        if current is not None:
            hands.append(current)
            current = None

    for event in events:
        kind = event.get("type")
        if kind == "start_kyoku":
            finalize()
            current = {
                "open": False,
                "riichi": False,
                "win": False,
                "deal": False,
                "income": 0,
                "loss": 0,
                "calls_after_opp_riichi": 0,
                "opp_riichi": False,
                "turn_number": 0,
            }
        elif current is None:
            continue
        elif kind == "tsumo":
            current["turn_number"] = max(current["turn_number"], 0)
        elif kind == "dahai":
            current["turn_number"] += 1
        elif kind == "reach_accepted":
            if event.get("actor") == focal:
                current["riichi"] = True
            else:
                current["opp_riichi"] = True
        elif kind in ("chi", "pon", "daiminkan"):
            if event.get("actor") == focal:
                current["open"] = True
                if current["opp_riichi"]:
                    current["calls_after_opp_riichi"] += 1
        elif kind == "hora":
            deltas = event.get("deltas") or [0, 0, 0, 0]
            if event.get("actor") == focal:
                current["win"] = True
                current["income"] += deltas[focal]
            elif event.get("target") == focal:
                current["deal"] = True
                current["loss"] += -deltas[focal]
        elif kind in ("end_kyoku", "ryukyoku"):
            finalize()
    finalize()
    return hands


def aggregate_hands(hands: list[dict]) -> dict:
    total = len(hands)
    wins = sum(1 for hand in hands if hand["win"])
    deals = sum(1 for hand in hands if hand["deal"])
    open_hands = [hand for hand in hands if hand["open"]]
    riichi_hands = [hand for hand in hands if hand["riichi"]]
    closed_nonriichi = [
        hand for hand in hands if not hand["open"] and not hand["riichi"]
    ]
    return {
        "hands": total,
        "wins": wins,
        "deals": deals,
        "win_rate": wins / total if total else None,
        "deal_rate": deals / total if total else None,
        "open_hands": len(open_hands),
        "open_wins": sum(1 for hand in open_hands if hand["win"]),
        "open_deals": sum(1 for hand in open_hands if hand["deal"]),
        "open_win_rate": (
            sum(1 for hand in open_hands if hand["win"]) / len(open_hands)
            if open_hands
            else None
        ),
        "open_deal_rate": (
            sum(1 for hand in open_hands if hand["deal"]) / len(open_hands)
            if open_hands
            else None
        ),
        "riichi_hands": len(riichi_hands),
        "riichi_wins": sum(1 for hand in riichi_hands if hand["win"]),
        "riichi_deals": sum(1 for hand in riichi_hands if hand["deal"]),
        "riichi_win_rate": (
            sum(1 for hand in riichi_hands if hand["win"]) / len(riichi_hands)
            if riichi_hands
            else None
        ),
        "riichi_deal_rate": (
            sum(1 for hand in riichi_hands if hand["deal"]) / len(riichi_hands)
            if riichi_hands
            else None
        ),
        "closed_nonriichi_hands": len(closed_nonriichi),
        "closed_nonriichi_wins": sum(1 for hand in closed_nonriichi if hand["win"]),
        "closed_nonriichi_deals": sum(1 for hand in closed_nonriichi if hand["deal"]),
        "closed_nonriichi_win_rate": (
            sum(1 for hand in closed_nonriichi if hand["win"])
            / len(closed_nonriichi)
            if closed_nonriichi
            else None
        ),
        "closed_nonriichi_deal_rate": (
            sum(1 for hand in closed_nonriichi if hand["deal"])
            / len(closed_nonriichi)
            if closed_nonriichi
            else None
        ),
        "income": sum(hand["income"] for hand in hands),
        "loss": sum(hand["loss"] for hand in hands),
        "mean_win_income": (
            sum(hand["income"] for hand in hands) / wins if wins else None
        ),
        "mean_deal_loss": (
            sum(hand["loss"] for hand in hands) / deals if deals else None
        ),
        "calls_after_opp_riichi": sum(
            hand["calls_after_opp_riichi"] for hand in hands
        ),
    }


TURN_KINDS = ("discard", "riichi_discard", "tsumo", "ankan", "kakan", "kyuushu_kyuuhai")
CALL_KINDS = ("chi", "pon", "daiminkan", "ron")


def decision_metrics(records: list[dict], focal: int) -> dict:
    turn = Counter()
    response = Counter()
    turn_numbers = []
    turn_opp_riichi = []
    turn_dora = []
    response_numbers = []
    response_opp_riichi = []
    response_dora = []
    for record in records:
        if record.get("player_id") != focal:
            continue
        task = record.get("task")
        selected = record.get("selected_action") or {}
        kind = selected.get("kind")
        candidates = record.get("candidates") or []
        kinds = [candidate.get("kind") for candidate in candidates]
        features = state_features(record.get("state") or {})
        if task == "turn":
            turn["decisions"] += 1
            turn["selected_" + str(kind)] += 1
            for candidate_kind in set(kinds):
                if candidate_kind is not None:
                    turn["legal_" + str(candidate_kind)] += 1
            if "riichi_discard" in kinds:
                turn["riichi_legal"] += 1
                if kind != "riichi_discard":
                    turn["riichi_pass"] += 1
            if "tsumo" in kinds:
                turn["tsumo_legal"] += 1
                if kind != "tsumo":
                    turn["missed_tsumo"] += 1
            turn_numbers.append(features["turn_number"] or 0)
            turn_opp_riichi.append(features["opp_riichi_count"])
            turn_dora.append(features["dora_count"])
        elif task == "response":
            response["decisions"] += 1
            response["selected_" + str(kind)] += 1
            call_legal = any(candidate_kind in CALL_KINDS for candidate_kind in kinds)
            if call_legal:
                response["call_legal"] += 1
                if kind not in CALL_KINDS:
                    response["pass_when_call_legal"] += 1
            if kind in CALL_KINDS:
                response["call_selected"] += 1
                if features["opp_riichi_count"] > 0:
                    response["calls_after_opp_riichi"] += 1
            if "ron" in kinds:
                response["ron_legal"] += 1
                if kind != "ron":
                    response["missed_ron"] += 1
            for candidate_kind in ("chi", "pon", "daiminkan", "ron"):
                if candidate_kind in kinds:
                    response["legal_" + candidate_kind] += 1
            response_numbers.append(features["turn_number"] or 0)
            response_opp_riichi.append(features["opp_riichi_count"])
            response_dora.append(features["dora_count"])
    turn["mean_turn_number"] = _mean(turn_numbers)
    turn["mean_opp_riichi"] = _mean(turn_opp_riichi)
    turn["mean_dora"] = _mean(turn_dora)
    response["mean_turn_number"] = _mean(response_numbers)
    response["mean_opp_riichi"] = _mean(response_opp_riichi)
    response["mean_dora"] = _mean(response_dora)
    response["call_rate"] = (
        response["call_selected"] / response["decisions"]
        if response["decisions"]
        else None
    )
    response["pass_when_call_legal_rate"] = (
        response["pass_when_call_legal"] / response["call_legal"]
        if response["call_legal"]
        else None
    )
    response["ron_miss_rate"] = (
        response["missed_ron"] / response["ron_legal"] if response["ron_legal"] else None
    )
    turn["riichi_rate"] = (
        turn["selected_riichi_discard"] / turn["decisions"]
        if turn["decisions"]
        else None
    )
    turn["riichi_pass_rate"] = (
        turn["riichi_pass"] / turn["riichi_legal"] if turn["riichi_legal"] else None
    )
    turn["tsumo_miss_rate"] = (
        turn["missed_tsumo"] / turn["tsumo_legal"] if turn["tsumo_legal"] else None
    )
    return {"turn": dict(turn), "response": dict(response)}


def _mean(values):
    return sum(values) / len(values) if values else None


def _merge_counts(target: Counter, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, int):
            target[key] += value


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _add_hand_target(target: dict, source: dict) -> None:
    for key, value in source.items():
        if key in (
            "win_rate",
            "deal_rate",
            "open_win_rate",
            "open_deal_rate",
            "riichi_win_rate",
            "riichi_deal_rate",
            "closed_nonriichi_win_rate",
            "closed_nonriichi_deal_rate",
            "mean_win_income",
            "mean_deal_loss",
        ):
            continue
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        else:
            target.setdefault(key, value)


def _merge_hand_aggregates(items: list[dict]) -> dict:
    merged = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
    hands = merged.get("hands", 0)
    wins = merged.get("wins", 0)
    deals = merged.get("deals", 0)
    merged["win_rate"] = wins / hands if hands else None
    merged["deal_rate"] = deals / hands if hands else None
    for prefix in ("open", "riichi", "closed_nonriichi"):
        n = merged.get(f"{prefix}_hands", 0)
        merged[f"{prefix}_win_rate"] = (
            merged.get(f"{prefix}_wins", 0) / n if n else None
        )
        merged[f"{prefix}_deal_rate"] = (
            merged.get(f"{prefix}_deals", 0) / n if n else None
        )
    merged["mean_win_income"] = merged.get("income", 0) / wins if wins else None
    merged["mean_deal_loss"] = merged.get("loss", 0) / deals if deals else None
    return merged


def summarize_batch(batch_dir: str | Path, *, decision_stride: int = 1) -> dict:
    batch = Path(batch_dir)
    summary = read_json(batch / "summary.json")
    games = summary.get("games", [])
    arms = defaultdict(list)
    decisions = defaultdict(list)
    arm_counts = defaultdict(int)
    for index, game in enumerate(games):
        game_id = game["game_id"]
        focal = game.get("focal_seat")
        if focal is None:
            continue
        events = list(read_jsonl(batch / "games" / f"{game_id}.mjai.jsonl"))
        hands = hand_metrics(events, focal)
        arm = game.get("matchup_id") or "unknown"
        sample_decisions = arm_counts[arm] % decision_stride == 0
        arm_counts[arm] += 1
        arms[arm].append(
            {
                "game_id": game_id,
                "seed": game.get("seed"),
                "rotation": game.get("rotation"),
                "rank": game["ranks"][focal],
                "points": game["scores"][focal],
                "hands": hands,
                "hand_metrics": aggregate_hands(hands),
            }
        )
        if sample_decisions:
            path = batch / "decisions" / f"{game_id}.jsonl.gz"
            if path.is_file():
                decisions[arm].append(
                    {
                        "game_id": game_id,
                        "seed": game.get("seed"),
                        "rotation": game.get("rotation"),
                        "focal": focal,
                        "metrics": decision_metrics(
                            list(read_jsonl_gz(path)), focal
                        ),
                    }
                )
    result = {}
    for arm, rows in arms.items():
        hand_metrics_all = [row["hand_metrics"] for row in rows]
        merged_hands = _merge_hand_aggregates(hand_metrics_all)
        decision_rows = decisions.get(arm, [])
        merged_turn = Counter()
        merged_response = Counter()
        for row in decision_rows:
            _merge_counts(merged_turn, row["metrics"]["turn"])
            _merge_counts(merged_response, row["metrics"]["response"])
        decision_summary = {
            "turn": dict(merged_turn),
            "response": dict(merged_response),
        }
        if merged_turn.get("decisions"):
            decision_summary["turn"]["riichi_rate"] = (
                merged_turn.get("selected_riichi_discard", 0)
                / merged_turn["decisions"]
            )
            decision_summary["turn"]["riichi_pass_rate"] = _rate(
                merged_turn.get("riichi_pass", 0),
                merged_turn.get("riichi_legal", 0),
            )
            decision_summary["turn"]["tsumo_miss_rate"] = _rate(
                merged_turn.get("missed_tsumo", 0),
                merged_turn.get("tsumo_legal", 0),
            )
        if merged_response.get("decisions"):
            decision_summary["response"]["call_rate"] = _rate(
                merged_response.get("call_selected", 0),
                merged_response["decisions"],
            )
            decision_summary["response"]["pass_when_call_legal_rate"] = _rate(
                merged_response.get("pass_when_call_legal", 0),
                merged_response.get("call_legal", 0),
            )
            decision_summary["response"]["ron_miss_rate"] = _rate(
                merged_response.get("missed_ron", 0),
                merged_response.get("ron_legal", 0),
            )
        result[arm] = {
            "games": len(rows),
            "mean_rank": sum(row["rank"] for row in rows) / len(rows),
            "fourth_rate": sum(1 for row in rows if row["rank"] == 4) / len(rows),
            "mean_points": sum(row["points"] for row in rows) / len(rows),
            "hands": merged_hands,
            "decisions": decision_summary,
            "decision_games": len(decision_rows),
            "per_seed": _per_seed(rows),
        }
    return result


def _per_seed(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)
    output = {}
    for seed, items in grouped.items():
        output[str(seed)] = {
            "games": len(items),
            "mean_rank": sum(item["rank"] for item in items) / len(items),
            "fourth_rate": sum(1 for item in items if item["rank"] == 4)
            / len(items),
            "mean_points": sum(item["points"] for item in items) / len(items),
        }
    return output


def summarize_a3(a3_dir: str | Path, *, decision_stride: int = 1) -> dict:
    batch = Path(a3_dir)
    summary = read_json(batch / "summary.json")
    groups = defaultdict(list)
    for index, game in enumerate(summary.get("games", [])):
        if game.get("matchup_id") != "candidate_vs_baseline":
            continue
        focal = game.get("focal_seat")
        if focal is None:
            continue
        game_id = game["game_id"]
        hands = hand_metrics(
            list(read_jsonl(batch / "games" / f"{game_id}.mjai.jsonl")), focal
        )
        row = {
            "game_id": game_id,
            "seed": game.get("seed"),
            "rotation": game.get("rotation"),
            "rank": game["ranks"][focal],
            "points": game["scores"][focal],
            "hand_metrics": aggregate_hands(hands),
        }
        groups["A3_new_new"].append(row)
    result = {}
    for name, rows in groups.items():
        merged = _merge_hand_aggregates([row["hand_metrics"] for row in rows])
        result[name] = {
            "games": len(rows),
            "mean_rank": sum(row["rank"] for row in rows) / len(rows),
            "fourth_rate": sum(1 for row in rows if row["rank"] == 4) / len(rows),
            "mean_points": sum(row["points"] for row in rows) / len(rows),
            "hands": merged,
            "per_seed": _per_seed(rows),
        }
    return result


def seed_sensitivity(w2_formal: dict, w2_smoke: dict, a3: dict) -> dict:
    def contrast(rows, left, right):
        values = []
        for seed, item in rows.get(left, {}).get("per_seed", {}).items():
            other = rows.get(right, {}).get("per_seed", {}).get(seed)
            if other:
                values.append(item["mean_rank"] - other["mean_rank"])
        return {
            "n": len(values),
            "mean": _mean(values),
            "positive": sum(1 for value in values if value > 0),
            "negative": sum(1 for value in values if value < 0),
            "zero": sum(1 for value in values if value == 0),
        }

    return {
        "w2_formal": {
            "D-A": contrast(w2_formal, "D", "A"),
            "C-A": contrast(w2_formal, "C", "A"),
            "B-A": contrast(w2_formal, "B", "A"),
            "D-C": contrast(w2_formal, "D", "C"),
        },
        "w2_smoke": {
            "D-A": contrast(w2_smoke, "D", "A"),
            "C-A": contrast(w2_smoke, "C", "A"),
        },
        "a3_new_new": a3.get("A3_new_new", {}).get("per_seed", {}),
    }


def render_report(result: dict) -> str:
    lines = [
        "# W3 local diagnostics",
        "",
        "Read-only diagnostics from W2 and A3 event/decision logs. "
        "Online 200 is not included because the cloud snapshot does not export model inputs.",
        "",
        "## W2 arms",
        "",
        "| Arm | Games | Mean rank | 4th | Win | Deal | Open | Riichi | Riichi deal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("A", "B", "C", "D"):
        item = result["w2_formal"].get(arm)
        if not item:
            continue
        hands = item["hands"]
        lines.append(
            f"| {arm} | {item['games']} | {item['mean_rank']:.4f} | "
            f"{item['fourth_rate']:.2%} | {hands['win_rate']:.2%} | "
            f"{hands['deal_rate']:.2%} | "
            f"{_rate(hands['open_hands'], hands['hands']):.2%} | "
            f"{_rate(hands['riichi_hands'], hands['hands']):.2%} | "
            f"{hands['riichi_deal_rate']:.2%} |"
        )
    lines += [
        "",
        "## W2 decision diagnostics (focal seat, sampled games)",
        "",
        "| Arm | Turn dec | Riichi sel | Riichi pass/legal | Tsumo miss/legal | Resp dec | Call rate | Pass when call legal | Ron miss/legal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("A", "B", "C", "D"):
        item = result["w2_formal"].get(arm)
        if not item:
            continue
        turn = item["decisions"]["turn"]
        response = item["decisions"]["response"]
        lines.append(
            f"| {arm} | {turn.get('decisions', 0)} | "
            f"{turn.get('selected_riichi_discard', 0)} | "
            f"{turn.get('riichi_pass', 0)}/{turn.get('riichi_legal', 0)} | "
            f"{turn.get('missed_tsumo', 0)}/{turn.get('tsumo_legal', 0)} | "
            f"{response.get('decisions', 0)} | "
            f"{_nullable_pct(response.get('call_rate'))} | "
            f"{_nullable_pct(response.get('pass_when_call_legal_rate'))} | "
            f"{response.get('missed_ron', 0)}/{response.get('ron_legal', 0)} |"
        )
    lines += [
        "",
        "## A3 vs W2 D",
        "",
        "| Group | Games | Mean rank | 4th | Win | Deal | Riichi | Riichi deal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A3_new_new",):
        item = result["a3"].get(name)
        if not item:
            continue
        hands = item["hands"]
        lines.append(
            f"| {name} | {item['games']} | {item['mean_rank']:.4f} | "
            f"{item['fourth_rate']:.2%} | {hands['win_rate']:.2%} | "
            f"{hands['deal_rate']:.2%} | "
            f"{_rate(hands['riichi_hands'], hands['hands']):.2%} | "
            f"{hands['riichi_deal_rate']:.2%} |"
        )
    d = result["w2_formal"].get("D")
    if d:
        hands = d["hands"]
        lines.append(
            f"| W2_D | {d['games']} | {d['mean_rank']:.4f} | "
            f"{d['fourth_rate']:.2%} | {hands['win_rate']:.2%} | "
            f"{hands['deal_rate']:.2%} | "
            f"{_rate(hands['riichi_hands'], hands['hands']):.2%} | "
            f"{hands['riichi_deal_rate']:.2%} |"
        )
    lines += [
        "",
        "## Seed sensitivity",
        "",
        "```json",
        json.dumps(result["seed_sensitivity"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Limitations",
        "",
        "- Descriptive, not causal; no shanten/wait-value calculator was added.",
        "- Decision diagnostics are sampled; event metrics cover all completed games.",
        "- Online 200 requires a sanitized export of cloud request states.",
        "",
    ]
    return "\n".join(lines)


def _nullable_pct(value):
    return f"{value:.2%}" if isinstance(value, (int, float)) else "n/a"
