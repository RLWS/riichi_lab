"""W3d shanten/wait/value enrichment for W3b2/W3c states.

Reads W3c outcome rows, re-loads the original W2 decision state/candidates,
and evaluates the post-discard (turn) or current (response) hand with
riichienv's native HandEvaluator.  The source and shadow actions remain
descriptive; this module only adds hand-shape context.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from riichienv import calculate_shanten
from riichienv import _riichienv as rust_core
from riichienv.convert import mpsz_to_tid, tid_to_mpsz
from riichienv.hand import Conditions, HandEvaluator, Meld

from rilab.w3 import read_jsonl_gz

MELD_TYPES = {
    "chi": rust_core.MeldType.Chi,
    "pon": rust_core.MeldType.Pon,
    "daiminkan": rust_core.MeldType.Daiminkan,
    "ankan": rust_core.MeldType.Ankan,
    "kakan": rust_core.MeldType.Kakan,
}
OPEN_MELDS = {"chi", "pon", "daiminkan"}


def _tile_type_ids() -> list[int]:
    """One representative 136-id per tile type (5 uses the non-red copy)."""
    ids = []
    for base in (0, 36, 72):
        for number in range(1, 10):
            ids.append(base + 4 * (number - 1) + (1 if number == 5 else 0))
    ids.extend(108 + 4 * (number - 1) for number in range(1, 8))
    return ids


TILE_TYPE_IDS = _tile_type_ids()



def _tile_ids(tile: str, count: int) -> list[int]:
    if count <= 0:
        return []
    base = mpsz_to_tid(tile)
    if tile in ("5m", "5p", "5s"):
        sequence = [base, base + 1, base + 2, base - 1]
    else:
        sequence = [base + index for index in range(4)]
    return sequence[:count]


def _hand_tiles(hand: dict) -> list[int]:
    tiles = []
    for tile, count in hand.items():
        tiles.extend(_tile_ids(tile, int(count)))
    return sorted(tiles)


def _meld_objects(own_melds: list[dict]) -> list[Meld]:
    melds = []
    for meld in own_melds:
        kind = meld.get("kind")
        if kind not in MELD_TYPES:
            continue
        consumed = meld.get("consumed_tiles", [])
        counts = Counter(consumed)
        called = meld.get("called_tile")
        if called and (kind != "ankan" or called not in consumed):
            counts[called] += 1
        tiles = []
        for tile, count in counts.items():
            tiles.extend(_tile_ids(tile, int(count)))
        melds.append(
            Meld(
                MELD_TYPES[kind],
                sorted(tiles),
                kind in OPEN_MELDS,
                from_who=meld.get("source", 0),
                called_tile=mpsz_to_tid(called) if called else None,
            )
        )
    return melds


def _visible_counts(state: dict) -> Counter:
    visible = Counter()
    for discards in state.get("all_discards", []):
        visible.update(discards)
    for melds in state.get("all_melds", []):
        for meld in melds:
            consumed = meld.get("consumed_tiles", [])
            visible.update(consumed)
            called = meld.get("called_tile")
            if called and (
                meld.get("kind") != "ankan" or called not in consumed
            ):
                visible[called] += 1
    visible.update(state.get("hand", {}))
    visible.update(state.get("dora_indicators", []))
    return visible


def _conditions(state: dict, riichi: bool) -> Conditions:
    return Conditions(
        tsumo=False,
        riichi=riichi,
        player_wind=int(state.get("seat_wind", 0)),
        round_wind=int(state.get("round_wind", 0)),
        honba=int(state.get("honba", 0)),
        riichi_sticks=int(state.get("riichi_sticks", 0)),
    )


def _empty_eval(error: str) -> dict:
    return {
        "ok": False,
        "error": error,
        "tenpai": False,
        "shanten": None,
        "waits": [],
        "wait_count": 0,
        "remaining": None,
        "max_han": None,
        "max_fu": None,
        "winning_wait_count": 0,
        "good_wait": False,
        "bad_wait": False,
    }


def evaluate_hand(state: dict, hand: dict, *, riichi: bool) -> dict:
    tiles = _hand_tiles(hand)
    melds = _meld_objects(state.get("own_melds", []))
    dora_ids = [mpsz_to_tid(tile) for tile in state.get("dora_indicators", [])]
    conditions = _conditions(state, riichi)
    try:
        evaluator = HandEvaluator(tiles, melds)
        shanten = calculate_shanten(tiles) if not melds else None
        wait_details = []
        for tile_id in TILE_TYPE_IDS:
            result = evaluator.calc(tile_id, dora_ids, conditions)
            if not getattr(result, "has_win_shape", False):
                continue
            han = getattr(result, "han", 0) or 0
            fu = getattr(result, "fu", 0) or 0
            wait_details.append(
                {
                    "tile": tid_to_mpsz(tile_id),
                    "is_win": bool(getattr(result, "is_win", False)),
                    "han": han,
                    "fu": fu,
                }
            )
    except Exception as exc:
        return _empty_eval(f"{type(exc).__name__}: {exc}")
    visible = _visible_counts(state)
    for item in wait_details:
        item["remaining"] = max(0, 4 - visible[item["tile"]])
    wait_tiles = [item["tile"] for item in wait_details]
    remaining = sum(item["remaining"] for item in wait_details)
    winning = [item for item in wait_details if item["is_win"]]
    max_han = max((item["han"] for item in winning), default=None)
    max_fu = max(
        (item["fu"] for item in winning if item["han"] == max_han), default=None
    )
    tenpai = bool(wait_details)
    return {
        "ok": True,
        "error": None,
        "tenpai": tenpai,
        "shanten": shanten,
        "waits": wait_tiles,
        "wait_count": len(wait_tiles),
        "remaining": remaining,
        "max_han": max_han,
        "max_fu": max_fu,
        "winning_wait_count": len(winning),
        "good_wait": tenpai and remaining >= 6,
        "bad_wait": tenpai and remaining <= 3,
    }


def _post_discard_hand(state: dict, discard: str) -> dict | None:
    hand = Counter(state.get("hand", {}))
    if hand.get(discard, 0) <= 0:
        return None
    hand[discard] -= 1
    if hand[discard] == 0:
        del hand[discard]
    return dict(hand)


def evaluate_post_discard(state: dict, discard: str, *, riichi: bool) -> dict:
    hand = _post_discard_hand(state, discard)
    if hand is None:
        return _empty_eval(f"discard not in hand: {discard}")
    return evaluate_hand(state, hand, riichi=riichi)


def _riichi_candidates(record: dict) -> list[dict]:
    return [
        candidate
        for candidate in record.get("candidates", [])
        if candidate.get("kind") == "riichi_discard"
    ]


def _other_selected(row: dict) -> dict | None:
    key = "v140_selected" if row.get("expected_source") == "v130" else "v130_selected"
    return row.get(key)


def _max_remaining_candidate(evals: list[dict]) -> dict | None:
    if not evals:
        return None
    return max(
        evals,
        key=lambda item: (
            item.get("remaining") if item.get("remaining") is not None else -1,
            item.get("max_han") if item.get("max_han") is not None else -1,
            item.get("wait_count") or 0,
        ),
    )


def load_rows(w3c_dir: str | Path, w2_dir: str | Path) -> list[dict]:
    outcomes = [
        json.loads(line)
        for line in (Path(w3c_dir) / "outcomes.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    by_game = defaultdict(list)
    for row in outcomes:
        by_game[row["game_id"]].append(row)
    output = []
    for game_id, game_rows in by_game.items():
        focal = game_rows[0]["focal"]
        decision_path = Path(w2_dir) / "decisions" / f"{game_id}.jsonl.gz"
        if not decision_path.is_file():
            continue
        records = {}
        for record in read_jsonl_gz(decision_path):
            if record.get("player_id") == focal:
                records[record.get("decision_id")] = record
        for row in game_rows:
            record = records.get(row["decision_id"])
            if record is None:
                output.append({**row, "shape_missing": True})
                continue
            state = record.get("state") or {}
            source = row.get("source_selected") or {}
            source_kind = source.get("kind")
            source_eval = None
            source_riichi_eval = None
            if row["task"] == "turn" and source_kind in (
                "discard",
                "riichi_discard",
            ):
                source_eval = evaluate_post_discard(
                    state,
                    source.get("tile"),
                    riichi=source_kind == "riichi_discard",
                )
                if "riichi_legal_pass" in row.get("strata", []):
                    source_riichi_eval = evaluate_post_discard(
                        state, source.get("tile"), riichi=True
                    )
            current_eval = None
            if row["task"] == "response":
                current_eval = evaluate_hand(
                    state,
                    state.get("hand", {}),
                    riichi=bool(state.get("riichi_declared")),
                )
            candidate_evals = []
            if row["task"] == "turn":
                for candidate in _riichi_candidates(record):
                    candidate_evals.append(
                        {
                            "tile": candidate.get("tile"),
                            **evaluate_post_discard(
                                state, candidate.get("tile"), riichi=True
                            ),
                        }
                    )
            candidate_eval = _max_remaining_candidate(candidate_evals)
            riichi_other_eval = None
            other_action = _other_selected(row)
            if (
                row["task"] == "turn"
                and other_action
                and other_action.get("kind") == "riichi_discard"
            ):
                riichi_other_eval = evaluate_post_discard(
                    state, other_action.get("tile"), riichi=True
                )
            output.append(
                {
                    **row,
                    "shape_missing": False,
                    "source_eval": source_eval,
                    "source_riichi_eval": source_riichi_eval,
                    "current_eval": current_eval,
                    "riichi_candidate_eval": candidate_eval,
                    "riichi_candidates": candidate_evals,
                    "riichi_other_eval": riichi_other_eval,
                }
            )
    return output


def _shape_stats(rows: list[dict], key: str) -> dict:
    evals = [row[key] for row in rows if row.get(key) and row[key].get("ok")]
    tenpai = [item for item in evals if item.get("tenpai")]
    waits = [item.get("wait_count", 0) for item in tenpai]
    remaining = [item["remaining"] for item in tenpai if item.get("remaining") is not None]
    hans = [item["max_han"] for item in tenpai if item.get("max_han") is not None]
    winning_waits = [item.get("winning_wait_count", 0) for item in tenpai]
    return {
        "n": len(rows),
        "shaped": len(evals),
        "tenpai": len(tenpai),
        "tenpai_rate": len(tenpai) / len(rows) if rows else None,
        "mean_wait_count": sum(waits) / len(waits) if waits else None,
        "mean_remaining": sum(remaining) / len(remaining) if remaining else None,
        "mean_max_han": sum(hans) / len(hans) if hans else None,
        "han_known": len(hans),
        "winning_wait_rate": (
            sum(winning_waits) / len(winning_waits) if winning_waits else None
        ),
        "good_wait_rate": (
            sum(1 for item in tenpai if item.get("good_wait")) / len(tenpai)
            if tenpai
            else None
        ),
        "bad_wait_rate": (
            sum(1 for item in tenpai if item.get("bad_wait")) / len(tenpai)
            if tenpai
            else None
        ),
    }


def _paired_stats(rows: list[dict], left_key: str, right_key: str) -> dict:
    pairs = []
    for row in rows:
        left = row.get(left_key)
        right = row.get(right_key)
        if not left or not right or not left.get("ok") or not right.get("ok"):
            continue
        if not left.get("tenpai") or not right.get("tenpai"):
            continue
        pairs.append((left, right))
    d_rem = [
        left["remaining"] - right["remaining"]
        for left, right in pairs
        if left.get("remaining") is not None and right.get("remaining") is not None
    ]
    d_han = [
        left["max_han"] - right["max_han"]
        for left, right in pairs
        if left.get("max_han") is not None and right.get("max_han") is not None
    ]
    return {
        "n": len(pairs),
        "mean_delta_remaining": sum(d_rem) / len(d_rem) if d_rem else None,
        "remaining_better_equal_worse": [
            sum(1 for value in d_rem if value > 0),
            sum(1 for value in d_rem if value == 0),
            sum(1 for value in d_rem if value < 0),
        ],
        "mean_delta_han": sum(d_han) / len(d_han) if d_han else None,
        "han_better_equal_worse": [
            sum(1 for value in d_han if value > 0),
            sum(1 for value in d_han if value == 0),
            sum(1 for value in d_han if value < 0),
        ],
    }


def _outcome_stats(rows: list[dict]) -> dict:
    wins = sum(1 for row in rows if row.get("hand", {}).get("win"))
    deals = sum(1 for row in rows if row.get("hand", {}).get("deal"))
    income = sum(row.get("hand", {}).get("income", 0) for row in rows)
    loss = sum(row.get("hand", {}).get("loss", 0) for row in rows)
    return {
        "n": len(rows),
        "unique_hands": len({(row["game_id"], row["hand_id"]) for row in rows}),
        "win_rate": wins / len(rows) if rows else None,
        "deal_rate": deals / len(rows) if rows else None,
        "income_per_hand": income / len(rows) if rows else None,
        "loss_per_hand": loss / len(rows) if rows else None,
    }


def aggregate(rows: list[dict]) -> dict:
    by_source = defaultdict(list)
    by_source_other = defaultdict(list)
    for row in rows:
        if row.get("shape_missing"):
            continue
        for stratum in row.get("strata", []):
            key = (row["arm"], row["task"], stratum)
            by_source[key].append(row)
            by_source_other[key + (row["source_class"], row["other_class"])].append(
                row
            )
    result = {
        "rows": len(rows),
        "shaped": sum(1 for row in rows if not row.get("shape_missing")),
        "by_source": {},
        "by_source_other": {},
        "limitations": [
            "Shanten/waits are computed from the recorded state; meld hands use wait enumeration only, not a closed-hand shanten.",
            "Wait tiles are enumerated structurally with HandEvaluator.has_win_shape; remaining counts subtract visible tiles in hand, discards, melds, and dora indicators.",
            "max_han is only defined when the wait can actually win (yaku exists); open yaku-less waits therefore contribute no han value.",
            "The riichi candidate evaluation is post-discard with riichi; `riichi_candidate_shape` is the legal riichi discard with the most remaining waits, then max han (a max-rem heuristic, not a full EV ranking), so it is not necessarily the model's best action.",
            "`riichi_other_shape` is the other version's actually selected riichi discard when that version chose riichi; it is exact for that shadow action.",
            "Response current-hand shape is pre-call, not post-call.",
            "Descriptive; small cells remain small.",
        ],
    }
    for key, group in sorted(by_source.items()):
        name = "/".join(key)
        result["by_source"][name] = {
            **_outcome_stats(group),
            "source_shape": _shape_stats(group, "source_eval"),
            "source_riichi_shape": _shape_stats(group, "source_riichi_eval"),
            "current_shape": _shape_stats(group, "current_eval"),
            "riichi_candidate_shape": _shape_stats(group, "riichi_candidate_eval"),
            "riichi_other_shape": _shape_stats(group, "riichi_other_eval"),
            "source_vs_candidate": _paired_stats(
                group, "source_eval", "riichi_candidate_eval"
            ),
            "source_riichi_vs_candidate": _paired_stats(
                group, "source_riichi_eval", "riichi_candidate_eval"
            ),
            "source_riichi_vs_other": _paired_stats(
                group, "source_riichi_eval", "riichi_other_eval"
            ),
            "mean_riichi_candidates": (
                sum(len(row.get("riichi_candidates") or []) for row in group)
                / len(group)
                if group
                else None
            ),
        }
    for key, group in sorted(by_source_other.items()):
        name = "/".join(key)
        result["by_source_other"][name] = {
            **_outcome_stats(group),
            "source_shape": _shape_stats(group, "source_eval"),
            "source_riichi_shape": _shape_stats(group, "source_riichi_eval"),
            "current_shape": _shape_stats(group, "current_eval"),
            "riichi_candidate_shape": _shape_stats(group, "riichi_candidate_eval"),
            "riichi_other_shape": _shape_stats(group, "riichi_other_eval"),
            "source_vs_candidate": _paired_stats(
                group, "source_eval", "riichi_candidate_eval"
            ),
            "source_riichi_vs_candidate": _paired_stats(
                group, "source_riichi_eval", "riichi_candidate_eval"
            ),
            "source_riichi_vs_other": _paired_stats(
                group, "source_riichi_eval", "riichi_other_eval"
            ),
            "mean_riichi_candidates": (
                sum(len(row.get("riichi_candidates") or []) for row in group)
                / len(group)
                if group
                else None
            ),
        }
    return result


def _pct(value):
    return f"{value:.2%}" if isinstance(value, (int, float)) else "n/a"


def _num(value, digits=2):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def render_report(result: dict) -> str:
    lines = [
        "# W3d shanten/wait/value enrichment",
        "",
        f"- rows: `{result['rows']}`; shaped: `{result['shaped']}`",
        "",
        "## Turn riichi selected",
        "",
        "| Arm | n | Tenpai | Wait tiles | Remaining waits | Max han | Win | Deal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source"].items():
        arm, task, stratum = key.split("/", 2)
        if task != "turn" or stratum != "riichi_selected":
            continue
        shape = item["source_shape"]
        lines.append(
            f"| {arm} | {item['n']} | {_pct(shape['tenpai_rate'])} | "
            f"{_num(shape['mean_wait_count'])} | {_num(shape['mean_remaining'])} | "
            f"{_num(shape['mean_max_han'])} | {_pct(item['win_rate'])} | "
            f"{_pct(item['deal_rate'])} |"
        )
    lines += [
        "",
        "### Riichi discard choice vs max-rem alternative",
        "",
        "`source_vs_candidate` compares the selected riichi discard with the max-remaining riichi discard on the same state.",
        "",
        "| Arm | paired n | Δrem | source better/equal/worse | Δhan | source better/equal/worse |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source"].items():
        arm, task, stratum = key.split("/", 2)
        if task != "turn" or stratum != "riichi_selected":
            continue
        paired = item["source_vs_candidate"]
        rem = paired["remaining_better_equal_worse"]
        han = paired["han_better_equal_worse"]
        lines.append(
            f"| {arm} | {paired['n']} | {_num(paired['mean_delta_remaining'], 3)} | "
            f"{rem[0]}/{rem[1]}/{rem[2]} | {_num(paired['mean_delta_han'], 3)} | "
            f"{han[0]}/{han[1]}/{han[2]} |"
        )
    lines += [
        "",
        "## Turn riichi legal pass: source shape and outcome",
        "",
        "Only `source_class=discard`; source shape is the chosen discard, evaluated with riichi for the han column.",
        "",
        "| Arm | Other | n | Source tenpai | Source waits | Source rem | Source riichi han | Win | Deal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source_other"].items():
        arm, task, stratum, source_class, other_class = key.split("/", 4)
        if (
            task != "turn"
            or stratum != "riichi_legal_pass"
            or source_class != "discard"
        ):
            continue
        source = item["source_shape"]
        source_riichi = item["source_riichi_shape"]
        lines.append(
            f"| {arm} | {other_class} | {item['n']} | "
            f"{_pct(source['tenpai_rate'])} | {_num(source['mean_wait_count'])} | "
            f"{_num(source['mean_remaining'])} | {_num(source_riichi['mean_max_han'])} | "
            f"{_pct(item['win_rate'])} | {_pct(item['deal_rate'])} |"
        )
    lines += [
        "",
        "### Paired source vs max-rem riichi candidate",
        "",
        "Only states where both the source discard and the max-rem riichi candidate are tenpai.",
        "",
        "| Arm | Other | paired n | Δrem | source better/equal/worse | Δhan | source better/equal/worse |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source_other"].items():
        arm, task, stratum, source_class, other_class = key.split("/", 4)
        if (
            task != "turn"
            or stratum != "riichi_legal_pass"
            or source_class != "discard"
        ):
            continue
        paired = item["source_riichi_vs_candidate"]
        rem = paired["remaining_better_equal_worse"]
        han = paired["han_better_equal_worse"]
        lines.append(
            f"| {arm} | {other_class} | {paired['n']} | "
            f"{_num(paired['mean_delta_remaining'], 2)} | "
            f"{rem[0]}/{rem[1]}/{rem[2]} | {_num(paired['mean_delta_han'], 2)} | "
            f"{han[0]}/{han[1]}/{han[2]} |"
        )
    lines += [
        "",
        "### Paired source vs the other version's exact riichi discard",
        "",
        "Only states where the other Turn version actually chose riichi; both sides are evaluated as riichi discards.",
        "",
        "| Arm | paired n | Δrem | source better/equal/worse | Δhan | source better/equal/worse |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source_other"].items():
        arm, task, stratum, source_class, other_class = key.split("/", 4)
        if (
            task != "turn"
            or stratum != "riichi_legal_pass"
            or source_class != "discard"
            or other_class != "riichi_discard"
        ):
            continue
        paired = item["source_riichi_vs_other"]
        rem = paired["remaining_better_equal_worse"]
        han = paired["han_better_equal_worse"]
        lines.append(
            f"| {arm} | {paired['n']} | {_num(paired['mean_delta_remaining'], 2)} | "
            f"{rem[0]}/{rem[1]}/{rem[2]} | {_num(paired['mean_delta_han'], 2)} | "
            f"{han[0]}/{han[1]}/{han[2]} |"
        )
    lines += [
        "",
        "## Turn opponent riichi + late",
        "",
        "| Arm | n | Tenpai | Remaining waits | Max han | Win | Deal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source"].items():
        arm, task, stratum = key.split("/", 2)
        if task != "turn" or stratum != "opp_riichi_late":
            continue
        shape = item["source_shape"]
        lines.append(
            f"| {arm} | {item['n']} | {_pct(shape['tenpai_rate'])} | "
            f"{_num(shape['mean_remaining'])} | {_num(shape['mean_max_han'])} | "
            f"{_pct(item['win_rate'])} | {_pct(item['deal_rate'])} |"
        )
    lines += [
        "",
        "## Response: high dora",
        "",
        "| Arm | n | Tenpai | Wait tiles | Remaining waits | Max han | Win | Deal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source"].items():
        arm, task, stratum = key.split("/", 2)
        if task != "response" or stratum != "high_dora":
            continue
        shape = item["current_shape"]
        lines.append(
            f"| {arm} | {item['n']} | {_pct(shape['tenpai_rate'])} | "
            f"{_num(shape['mean_wait_count'])} | {_num(shape['mean_remaining'])} | "
            f"{_num(shape['mean_max_han'])} | {_pct(item['win_rate'])} | "
            f"{_pct(item['deal_rate'])} |"
        )
    lines += [
        "",
        "## Response: opponent riichi",
        "",
        "| Arm | n | Tenpai | Wait tiles | Remaining waits | Max han | Win | Deal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in result["by_source"].items():
        arm, task, stratum = key.split("/", 2)
        if task != "response" or stratum != "opp_riichi":
            continue
        shape = item["current_shape"]
        lines.append(
            f"| {arm} | {item['n']} | {_pct(shape['tenpai_rate'])} | "
            f"{_num(shape['mean_wait_count'])} | {_num(shape['mean_remaining'])} | "
            f"{_num(shape['mean_max_han'])} | {_pct(item['win_rate'])} | "
            f"{_pct(item['deal_rate'])} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
    ]
    return "\n".join(lines)
