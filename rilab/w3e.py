"""W3e: Turn-only replication of the W3d "risky extra riichi" hypothesis.

Design (pre-registered in docs/w3e-turn-riichi-preregistration-2026-11-18.md):

* Arm A = old v1.3.0 Turn + old v1.3.0 Response.
* Arm B = new v1.4.0 Turn + old v1.3.0 Response.
* Response is fixed to v1.3.0, so only Turn changes.
* Full state is captured only for focal-seat executed ``riichi_discard``
  decisions.  The other Turn service is replayed on the same state to classify
  declaration-threshold divergence.
* ``B_extra`` = B declared riichi where old Turn would discard.
* ``A_mirror`` = A declared riichi where new Turn would discard.
* Primary endpoint is hand-level deal rate, B_extra versus A_mirror.

The module is split into deterministic pieces (schedule/registration/counts)
and an outcome-analysis piece that is run only after the pre-registered
count-only extension decision.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random

import yaml

from rilab.w2 import sha256_file, sha256_json, service_configs
from rilab.w3 import hand_metrics, read_json, read_jsonl, read_jsonl_gz, state_features
from rilab.w3_shadow import action_class, normalize_state
from rilab.w3d import _paired_stats, _shape_stats, evaluate_post_discard

PURPOSE = "w3e_turn_riichi"
ARM_IDS = ("A", "B")
EXPECTED_ARMS = {
    "A": {"turn": "v130", "response": "v130"},
    "B": {"turn": "v140", "response": "v130"},
}
STAGE_SEED_COUNTS = {"smoke": 2, "stage1": 64, "stage2": 64}
FORBIDDEN_SEED_RANGES = ((40001, 40008), (41001, 41128), (42001, 42128))
Z95 = 1.959963984540054
RIICHI_KIND = "riichi_discard"
DISCARD_KIND = "discard"


# ---------------------------------------------------------------------------
# plan / schedule / registration
# ---------------------------------------------------------------------------


def _expand_range(value, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element inclusive range")
    start, end = value
    if type(start) is not int or type(end) is not int or start > end:
        raise ValueError(f"{name} must be an increasing integer range")
    return list(range(start, end + 1))


def load_plan(path: str | Path) -> dict:
    plan_path = Path(path)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("W3e plan must be a mapping")
    engine = plan.get("engine")
    if not isinstance(engine, dict):
        raise ValueError("W3e plan requires engine")
    for stage in STAGE_SEED_COUNTS:
        range_key = f"{stage}_seed_range"
        list_key = f"{stage}_seeds"
        if range_key in engine:
            plan[list_key] = _expand_range(engine[range_key], range_key)
        elif list_key in engine:
            value = engine[list_key]
            if not isinstance(value, list):
                raise ValueError(f"{list_key} must be a list")
            plan[list_key] = list(value)
        else:
            raise ValueError(f"W3e plan requires {range_key} or {list_key}")
    return plan


def _validate_seed_list(name: str, seeds, count: int) -> list[int]:
    if (
        not isinstance(seeds, list)
        or len(seeds) != count
        or len(set(seeds)) != count
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise ValueError(f"{name} must contain {count} unique uint32 seeds")
    return list(seeds)


def validate_plan(plan: dict) -> None:
    if plan.get("plan_version") != 1:
        raise ValueError("W3e plan_version must be 1")
    if plan.get("purpose") != PURPOSE:
        raise ValueError("W3e purpose must be w3e_turn_riichi")
    engine = plan.get("engine")
    if not isinstance(engine, dict):
        raise ValueError("W3e plan requires engine")
    if engine.get("game_mode") != "4p-red-half" or engine.get("rule") != "default_tenhou":
        raise ValueError("W3e only supports 4p-red-half/default_tenhou")

    # This validates the two native service blocks (paths, profiles, hashes,
    # calibration fields) with the same rules as W2.
    service_configs(plan)

    arms = plan.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_IDS):
        raise ValueError("W3e arms must be exactly A/B")
    for arm, expected in EXPECTED_ARMS.items():
        if arms.get(arm) != expected:
            raise ValueError(f"W3e arm definition differs: {arm}")

    seed_sets = {}
    for stage, count in STAGE_SEED_COUNTS.items():
        seed_sets[stage] = _validate_seed_list(
            f"{stage}_seeds", plan.get(f"{stage}_seeds"), count
        )
    names = list(STAGE_SEED_COUNTS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if set(seed_sets[left]) & set(seed_sets[right]):
                raise ValueError(f"{left}_seeds and {right}_seeds must be disjoint")
    for start, end in FORBIDDEN_SEED_RANGES:
        forbidden = set(range(start, end + 1))
        for stage, seeds in seed_sets.items():
            overlap = sorted(set(seeds) & forbidden)
            if overlap:
                raise ValueError(
                    f"{stage}_seeds reuse forbidden seeds {start}-{end}: {overlap[:3]}"
                )

    capture = plan.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("W3e plan requires capture")
    if capture.get("tasks") != ["turn"]:
        raise ValueError("W3e capture.tasks must be [turn]")
    if capture.get("players") != "focal":
        raise ValueError("W3e capture.players must be focal")
    if capture.get("selected_kinds") != [RIICHI_KIND]:
        raise ValueError("W3e capture.selected_kinds must be [riichi_discard]")

    analysis = plan.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("W3e plan requires analysis")
    if analysis.get("primary_endpoint") != "b_extra_vs_a_mirror_deal_rate":
        raise ValueError("W3e primary_endpoint is fixed")
    if analysis.get("primary_unit") != "unique_hand":
        raise ValueError("W3e primary_unit is fixed")
    if analysis.get("direction") != "b_extra_higher_deal_rate":
        raise ValueError("W3e direction is fixed")
    if analysis.get("confidence_level") != 0.95:
        raise ValueError("W3e confidence_level must be 0.95")
    if analysis.get("wilson_z") != Z95:
        raise ValueError("W3e wilson_z must be the two-sided 95% normal quantile")
    if analysis.get("bootstrap_resamples") != 10000:
        raise ValueError("W3e bootstrap_resamples must be 10000")
    if analysis.get("bootstrap_seed") != 20261118:
        raise ValueError("W3e bootstrap_seed must be 20261118")
    if analysis.get("minimum_unique_hands_per_group") != 24:
        raise ValueError("W3e minimum_unique_hands_per_group must be 24")
    if analysis.get("replication_min_risk_difference") != 0.10:
        raise ValueError("W3e replication_min_risk_difference must be 0.10")
    if analysis.get("rejection_max_upper_bound") != 0.10:
        raise ValueError("W3e rejection_max_upper_bound must be 0.10")
    extension = analysis.get("extension")
    if not isinstance(extension, dict):
        raise ValueError("W3e plan requires analysis.extension")
    if extension.get("extend_if_below") != 24:
        raise ValueError("W3e extension.extend_if_below must be 24")
    if extension.get("decision_basis") != "count_only_no_outcomes":
        raise ValueError("W3e extension decision must use counts only")
    forbidden_ranges = analysis.get("forbidden_seed_ranges")
    if forbidden_ranges != [list(pair) for pair in FORBIDDEN_SEED_RANGES]:
        raise ValueError("W3e forbidden_seed_ranges differ from the pre-registration")


def build_schedule(plan: dict, seeds: list[int]) -> list[dict]:
    validate_plan({**plan, "stage1_seeds": list(plan["stage1_seeds"]),
                   "stage2_seeds": list(plan["stage2_seeds"]),
                   "smoke_seeds": list(plan["smoke_seeds"])})
    schedule = []
    for seed in seeds:
        for arm in ARM_IDS:
            base = ["v130"] * 4 if arm == "A" else ["B", "v130", "v130", "v130"]
            for rotation in range(4):
                seats = [base[(index - rotation) % 4] for index in range(4)]
                schedule.append(
                    {
                        "game_id": f"g{len(schedule):05d}",
                        "seed": seed,
                        "matchup_id": arm,
                        "rotation": rotation,
                        "seats": seats,
                        "focal_seat": rotation,
                        "paired_group": None,
                        "paired_key": [arm, seed, rotation],
                    }
                )
    return schedule


def native_config(plan: dict, backends: dict, seeds: list[int]) -> dict:
    missing = {"v130", "B"} - set(backends)
    if missing:
        raise ValueError(f"W3e native config lacks backends: {sorted(missing)}")
    return {
        "config_version": 1,
        "engine": {
            "game_mode": plan["engine"]["game_mode"],
            "rule": plan["engine"]["rule"],
            "seeds": list(seeds),
        },
        "backends": {name: backends[name] for name in ("v130", "B")},
        "matchups": [],
        "evaluation": {
            "cluster": "seed_with_all_rotations",
            "confidence_level": 0.95,
            "primary_paired_group": "W3e",
        },
        "artifacts": {
            "max_bytes": plan.get("artifacts", {}).get("max_bytes", 2_000_000_000),
            "snapshot_stride": plan.get("artifacts", {}).get("snapshot_stride", 100),
            "full_snapshots_first_run": plan.get("artifacts", {}).get(
                "full_snapshots_first_run", False
            ),
        },
    }


def registration(
    plan: dict,
    plan_path: str | Path,
    service_config_paths: dict,
    schedule: list[dict],
    *,
    stage: str,
    seeds: list[int],
) -> dict:
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "stage": stage,
        "created_at": _utc_now(),
        "plan_path": str(Path(plan_path)),
        "plan_sha256": sha256_file(plan_path),
        "service_config_sha256": {
            name: sha256_file(path) for name, path in service_config_paths.items()
        },
        "arms": plan["arms"],
        "seeds": list(seeds),
        "seed_ranges": {
            "smoke": plan["engine"].get("smoke_seed_range"),
            "stage1": plan["engine"].get("stage1_seed_range"),
            "stage2": plan["engine"].get("stage2_seed_range"),
        },
        "schedule_games": len(schedule),
        "schedule_sha256": sha256_json(schedule),
        "capture": plan["capture"],
        "analysis": plan["analysis"],
        "notes": list(plan.get("notes", [])),
    }


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def capture_filter(spec: dict, record: dict) -> bool:
    """Keep full state only for focal-seat executed riichi declarations."""
    return (
        record.get("player_id") == spec.get("focal_seat")
        and record.get("task") == "turn"
        and (record.get("selected_action") or {}).get("kind") == RIICHI_KIND
    )


# ---------------------------------------------------------------------------
# shadow classification
# ---------------------------------------------------------------------------


def collect_events(
    batch_dir: str | Path,
    plan: dict,
    v130,
    v140,
) -> tuple[list[dict], dict]:
    batch = Path(batch_dir)
    summary = read_json(batch / "summary.json")
    games = summary.get("games", [])
    limits = {
        task: min(
            v130.metadata[task]["candidate_limit"],
            v140.metadata[task]["candidate_limit"],
        )
        for task in ("turn", "response")
    }
    meta = {
        "games": len(games),
        "captured_records": 0,
        "captured_with_state": 0,
        "excluded_common_support": 0,
        "missing_state": 0,
        "missing_candidates": 0,
        "shadow_errors": 0,
        "common_support_limit": limits,
    }
    events = []
    for game in games:
        focal = game.get("focal_seat")
        if focal is None:
            continue
        path = batch / "decisions" / f"{game['game_id']}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(f"missing decision spool: {path}")
        for record in read_jsonl_gz(path):
            if not capture_filter(game, record):
                continue
            meta["captured_records"] += 1
            base = {
                "game_id": game["game_id"],
                "seed": game.get("seed"),
                "arm": game.get("matchup_id"),
                "rotation": game.get("rotation"),
                "focal": focal,
                "decision_id": record.get("decision_id"),
                "hand_id": record.get("hand_id"),
                "task": "turn",
            }
            state = record.get("state")
            if not state:
                meta["missing_state"] += 1
                events.append({**base, "error": "missing_state"})
                continue
            meta["captured_with_state"] += 1
            candidates = record.get("candidates") or []
            if not candidates:
                meta["missing_candidates"] += 1
                events.append({**base, "error": "missing_candidates"})
                continue
            if len(candidates) > limits["turn"]:
                meta["excluded_common_support"] += 1
                continue
            source = record.get("selected_action") or {}
            try:
                v130_result = v130.predict(
                    "turn", normalize_state(state, "full_state_v2"), candidates
                )
                v140_result = v140.predict(
                    "turn", normalize_state(state, "full_state_v3"), candidates
                )
                v130_action = v130_result.get("selected_action")
                v140_action = v140_result.get("selected_action")
                if not isinstance(v130_action, dict) or not isinstance(v140_action, dict):
                    raise ValueError("service returned no selected_action")
            except BaseException as exc:
                meta["shadow_errors"] += 1
                events.append({**base, "error": f"{type(exc).__name__}: {exc}"})
                continue
            other = v130_action if base["arm"] == "B" else v140_action
            features = state_features(state)
            source_tile = source.get("tile")
            other_tile = other.get("tile")
            same_tile = (
                source_tile == other_tile
                if source_tile is not None and other_tile is not None
                else None
            )
            source_eval = (
                evaluate_post_discard(state, source_tile, riichi=True)
                if source_tile is not None
                else None
            )
            other_eval = (
                evaluate_post_discard(state, other_tile, riichi=True)
                if other.get("kind") == RIICHI_KIND and other_tile is not None
                else None
            )
            events.append(
                {
                    **base,
                    "source_selected": source,
                    "source_class": action_class("turn", source),
                    "v130_selected": v130_action,
                    "v140_selected": v140_action,
                    "v130_class": action_class("turn", v130_action),
                    "v140_class": action_class("turn", v140_action),
                    "other_selected": other,
                    "other_class": action_class("turn", other),
                    "both_riichi": (
                        source.get("kind") == RIICHI_KIND
                        and other.get("kind") == RIICHI_KIND
                    ),
                    "same_tile": same_tile,
                    "candidates": len(candidates),
                    "turn_number": features.get("turn_number"),
                    "opp_riichi_count": features.get("opp_riichi_count"),
                    "own_meld_count": features.get("own_meld_count"),
                    "dora_count": features.get("dora_count"),
                    "late": _is_late(features.get("turn_number")),
                    "opp_riichi_late": bool(features.get("opp_riichi_count"))
                    and _is_late(features.get("turn_number")),
                    "source_eval": source_eval,
                    "other_eval": other_eval,
                    "error": None,
                }
            )
    return events, meta


def _is_late(turn_number) -> bool:
    return isinstance(turn_number, (int, float)) and turn_number > 12


# ---------------------------------------------------------------------------
# count-only phase (no outcomes)
# ---------------------------------------------------------------------------


def _dedupe_hands(rows: list[dict]) -> list[dict]:
    seen = {}
    for row in sorted(
        rows,
        key=lambda item: (
            item.get("game_id") or "",
            item.get("hand_id") if isinstance(item.get("hand_id"), int) else 0,
            item.get("decision_id") if isinstance(item.get("decision_id"), int) else 0,
        ),
    ):
        key = (
            row.get("game_id"),
            row.get("hand_id")
            if isinstance(row.get("hand_id"), int)
            else f"decision:{row.get('decision_id')}",
        )
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    valid = [row for row in rows if not row.get("error")]
    source_riichi = [row for row in valid if row.get("source_class") == RIICHI_KIND]
    groups = {
        "B_extra": [
            row
            for row in source_riichi
            if row.get("arm") == "B" and row.get("other_class") == DISCARD_KIND
        ],
        "A_mirror": [
            row
            for row in source_riichi
            if row.get("arm") == "A" and row.get("other_class") == DISCARD_KIND
        ],
        "B_both_riichi": [
            row
            for row in source_riichi
            if row.get("arm") == "B" and row.get("both_riichi")
        ],
        "A_both_riichi": [
            row
            for row in source_riichi
            if row.get("arm") == "A" and row.get("both_riichi")
        ],
        "B_all_riichi": [row for row in source_riichi if row.get("arm") == "B"],
        "A_all_riichi": [row for row in source_riichi if row.get("arm") == "A"],
    }
    for prefix, base in (("B_extra", groups["B_extra"]), ("A_mirror", groups["A_mirror"])):
        groups[f"{prefix}_opp_riichi"] = [
            row for row in base if (row.get("opp_riichi_count") or 0) > 0
        ]
        groups[f"{prefix}_late"] = [row for row in base if row.get("late")]
        groups[f"{prefix}_opp_riichi_late"] = [
            row for row in base if row.get("opp_riichi_late")
        ]
        groups[f"{prefix}_not_opp_riichi_late"] = [
            row for row in base if not row.get("opp_riichi_late")
        ]
    return {name: _dedupe_hands(value) for name, value in groups.items()}


def count_events(
    events: list[dict],
    meta: dict,
    *,
    stage: str,
    plan: dict,
    summary: dict,
    audit_games: int,
    plan_sha256: str,
) -> dict:
    groups = group_rows(events)
    errors = [row for row in events if row.get("error")]
    by_arm = {}
    for arm in ARM_IDS:
        arm_rows = _dedupe_hands([row for row in events if row.get("arm") == arm])
        by_arm[arm] = {
            "captured_hands": len(arm_rows),
            "riichi_hands": len(
                [row for row in arm_rows if row.get("source_class") == RIICHI_KIND]
            ),
            "both_riichi": len([row for row in arm_rows if row.get("both_riichi")]),
        }
    b_extra = groups["B_extra"]
    a_mirror = groups["A_mirror"]
    extension_rule = plan["analysis"]["extension"]
    extend = stage == "stage1" and len(b_extra) < extension_rule["extend_if_below"]
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "stage": stage,
        "created_at": _utc_now(),
        "plan_sha256": plan_sha256,
        "seeds": list(plan[f"{stage}_seeds"]) if f"{stage}_seeds" in plan else [],
        "games": summary.get("completed_games"),
        "expected_games": summary.get("expected_games"),
        "fallbacks": sum(int(game.get("fallbacks", 0)) for game in summary.get("games", [])),
        "audit_games": audit_games,
        "captured_states": meta.get("captured_records", len(events)),
        "classified_states": len(events),
        "captured_with_state": meta.get("captured_with_state", 0),
        "excluded_common_support": meta.get("excluded_common_support", 0),
        "missing_state": meta.get("missing_state", 0),
        "missing_candidates": meta.get("missing_candidates", 0),
        "shadow_errors": meta.get("shadow_errors", 0),
        "error_rows": len(errors),
        "common_support_limit": meta.get("common_support_limit"),
        "by_arm": by_arm,
        "divergence": {
            "b_extra_unique_hands": len(b_extra),
            "a_mirror_unique_hands": len(a_mirror),
            "b_extra_opp_riichi_late_unique_hands": len(
                groups["B_extra_opp_riichi_late"]
            ),
            "a_mirror_opp_riichi_late_unique_hands": len(
                groups["A_mirror_opp_riichi_late"]
            ),
            "b_extra_share_of_b_riichi": (
                len(b_extra) / len(groups["B_all_riichi"])
                if groups["B_all_riichi"]
                else None
            ),
            "a_mirror_share_of_a_riichi": (
                len(a_mirror) / len(groups["A_all_riichi"])
                if groups["A_all_riichi"]
                else None
            ),
            "both_riichi": {
                "B": len(groups["B_both_riichi"]),
                "A": len(groups["A_both_riichi"]),
            },
        },
        "extension_rule": {
            "stage": "stage1" if stage == "stage1" else None,
            "count_field": extension_rule["count_field"],
            "extend_if_below": extension_rule["extend_if_below"],
            "current_value": len(b_extra) if stage == "stage1" else None,
            "extension_required": extend,
            "decision_basis": extension_rule["decision_basis"],
        },
        "outcomes_inspected": False,
        "errors": [
            {
                "game_id": row.get("game_id"),
                "decision_id": row.get("decision_id"),
                "error": row.get("error"),
            }
            for row in errors[:50]
        ],
    }


# ---------------------------------------------------------------------------
# outcomes and analysis
# ---------------------------------------------------------------------------


def attach_outcomes(events: list[dict], batch_dir: str | Path) -> list[dict]:
    batch = Path(batch_dir)
    cache = {}
    output = []
    for row in events:
        if row.get("error"):
            output.append({**row, "hand": None, "outcome_missing": True})
            continue
        game_id = row["game_id"]
        if game_id not in cache:
            path = batch / "games" / f"{game_id}.mjai.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"missing game log: {path}")
            cache[game_id] = hand_metrics(list(read_jsonl(path)), row["focal"])
        hands = cache[game_id]
        hand_id = row.get("hand_id")
        if not isinstance(hand_id, int) or not 1 <= hand_id <= len(hands):
            output.append({**row, "hand": None, "outcome_missing": True})
            continue
        output.append(
            {**row, "hand": hands[hand_id - 1], "outcome_missing": False}
        )
    return output


def wilson_interval(k: int, n: int, z: float = Z95) -> dict:
    if n <= 0:
        return {"low": None, "high": None}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return {"low": max(0.0, center - half), "high": min(1.0, center + half)}


def newcombe_difference(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> dict:
    if n1 <= 0 or n2 <= 0:
        return {"difference": None, "ci95": [None, None]}
    p1, p2 = k1 / n1, k2 / n2
    i1, i2 = wilson_interval(k1, n1, z), wilson_interval(k2, n2, z)
    lower = (p1 - p2) - math.sqrt(
        (p1 - i1["low"]) ** 2 + (i2["high"] - p2) ** 2
    )
    upper = (p1 - p2) + math.sqrt(
        (i1["high"] - p1) ** 2 + (p2 - i2["low"]) ** 2
    )
    return {"difference": p1 - p2, "ci95": [lower, upper]}


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float | None:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]]."""
    n = a + b + c + d
    row1 = a + b
    col1 = a + c
    if n == 0:
        return None
    denominator = math.comb(n, col1)
    if denominator == 0:
        return None

    def probability(x: int) -> float:
        return (
            math.comb(row1, x) * math.comb(n - row1, col1 - x) / denominator
        )

    observed = probability(a)
    total = 0.0
    low = max(0, col1 - (n - row1))
    high = min(row1, col1)
    for x in range(low, high + 1):
        p = probability(x)
        if p <= observed + 1e-12:
            total += p
    return min(1.0, total)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _seed_outcomes(rows: list[dict], field: str) -> dict:
    by_seed = defaultdict(list)
    for row in _dedupe_hands(rows):
        hand = row.get("hand")
        if hand is None:
            continue
        value = hand.get(field)
        if isinstance(value, bool):
            by_seed[row.get("seed")].append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            by_seed[row.get("seed")].append(float(value))
    return by_seed


def _cluster_bootstrap(by_seed1: dict, by_seed2: dict, *, resamples: int, seed: int):
    seeds = sorted(
        {key for key in by_seed1 if by_seed1.get(key)}
        | {key for key in by_seed2 if by_seed2.get(key)}
    )
    if not seeds:
        return {"mean": None, "ci95": [None, None], "valid_resamples": 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        draw = [rng.choice(seeds) for _ in seeds]
        left = [value for key in draw for value in by_seed1.get(key, [])]
        right = [value for key in draw for value in by_seed2.get(key, [])]
        if not left or not right:
            continue
        diffs.append(sum(left) / len(left) - sum(right) / len(right))
    if not diffs:
        return {"mean": None, "ci95": [None, None], "valid_resamples": 0}
    return {
        "mean": sum(diffs) / len(diffs),
        "ci95": [_percentile(diffs, 0.025), _percentile(diffs, 0.975)],
        "valid_resamples": len(diffs),
    }


def _cluster_bootstrap_mean(by_seed: dict, *, resamples: int, seed: int):
    seeds = sorted(key for key, values in by_seed.items() if values)
    if not seeds:
        return {"mean": None, "ci95": [None, None], "valid_resamples": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [rng.choice(seeds) for _ in seeds]
        values = [value for key in draw for value in by_seed.get(key, [])]
        if not values:
            continue
        means.append(sum(values) / len(values))
    if not means:
        return {"mean": None, "ci95": [None, None], "valid_resamples": 0}
    return {
        "mean": sum(means) / len(means),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "valid_resamples": len(means),
    }


def _outcome_stats(rows: list[dict], plan: dict) -> dict:
    rows = _dedupe_hands(rows)
    valid = [row for row in rows if row.get("hand") is not None]
    n = len(valid)
    wins = sum(1 for row in valid if row["hand"].get("win"))
    deals = sum(1 for row in valid if row["hand"].get("deal"))
    income = sum(row["hand"].get("income", 0) for row in valid)
    loss = sum(row["hand"].get("loss", 0) for row in valid)
    win_ci = wilson_interval(wins, n)
    deal_ci = wilson_interval(deals, n)
    income_by_seed = _seed_outcomes(rows, "income")
    loss_by_seed = _seed_outcomes(rows, "loss")
    return {
        "n": n,
        "unique_hands": len(rows),
        "missing_outcomes": len(rows) - n,
        "seeds": len({row.get("seed") for row in valid}),
        "wins": wins,
        "deals": deals,
        "win_rate": wins / n if n else None,
        "deal_rate": deals / n if n else None,
        "win_wilson_ci95": [win_ci["low"], win_ci["high"]],
        "deal_wilson_ci95": [deal_ci["low"], deal_ci["high"]],
        "income_per_hand": income / n if n else None,
        "loss_per_hand": loss / n if n else None,
        "mean_win_income": income / wins if wins else None,
        "mean_deal_loss": loss / deals if deals else None,
        "income_per_hand_bootstrap": _cluster_bootstrap_mean(
            income_by_seed,
            resamples=plan["analysis"]["bootstrap_resamples"],
            seed=plan["analysis"]["bootstrap_seed"],
        ),
        "loss_per_hand_bootstrap": _cluster_bootstrap_mean(
            loss_by_seed,
            resamples=plan["analysis"]["bootstrap_resamples"],
            seed=plan["analysis"]["bootstrap_seed"],
        ),
        "hand_riichi_rate": (
            sum(1 for row in valid if row["hand"].get("riichi")) / n if n else None
        ),
        "hand_open_rate": (
            sum(1 for row in valid if row["hand"].get("open")) / n if n else None
        ),
    }


def compare_binary(rows1: list[dict], rows2: list[dict], field: str, plan: dict) -> dict:
    left = _dedupe_hands([row for row in rows1 if row.get("hand") is not None])
    right = _dedupe_hands([row for row in rows2 if row.get("hand") is not None])
    k1 = sum(1 for row in left if row["hand"].get(field))
    k2 = sum(1 for row in right if row["hand"].get(field))
    n1, n2 = len(left), len(right)
    diff = newcombe_difference(k1, n1, k2, n2)
    bootstrap = _cluster_bootstrap(
        _seed_outcomes(left, field),
        _seed_outcomes(right, field),
        resamples=plan["analysis"]["bootstrap_resamples"],
        seed=plan["analysis"]["bootstrap_seed"],
    )
    return {
        "field": field,
        "left": {
            "n": n1,
            "count": k1,
            "rate": k1 / n1 if n1 else None,
            "wilson_ci95": list(wilson_interval(k1, n1).values()),
        },
        "right": {
            "n": n2,
            "count": k2,
            "rate": k2 / n2 if n2 else None,
            "wilson_ci95": list(wilson_interval(k2, n2).values()),
        },
        "difference": diff["difference"],
        "newcombe_ci95": diff["ci95"],
        "fisher_p": fisher_exact_two_sided(k1, n1 - k1, k2, n2 - k2),
        "seed_cluster_bootstrap": bootstrap,
    }


def compare_continuous(rows1: list[dict], rows2: list[dict], field: str, plan: dict) -> dict:
    left = _dedupe_hands([row for row in rows1 if row.get("hand") is not None])
    right = _dedupe_hands([row for row in rows2 if row.get("hand") is not None])
    values1 = [float(row["hand"].get(field, 0)) for row in left]
    values2 = [float(row["hand"].get(field, 0)) for row in right]
    return {
        "field": field,
        "left_mean": sum(values1) / len(values1) if values1 else None,
        "right_mean": sum(values2) / len(values2) if values2 else None,
        "difference": (
            (sum(values1) / len(values1) - sum(values2) / len(values2))
            if values1 and values2
            else None
        ),
        "seed_cluster_bootstrap": _cluster_bootstrap(
            _seed_outcomes(left, field),
            _seed_outcomes(right, field),
            resamples=plan["analysis"]["bootstrap_resamples"],
            seed=plan["analysis"]["bootstrap_seed"],
        ),
    }


def share_difference(rows1: list[dict], rows2: list[dict], predicate, plan: dict) -> dict:
    left = _dedupe_hands(rows1)
    right = _dedupe_hands(rows2)
    k1 = sum(1 for row in left if predicate(row))
    k2 = sum(1 for row in right if predicate(row))
    n1, n2 = len(left), len(right)
    diff = newcombe_difference(k1, n1, k2, n2)
    return {
        "left": {
            "n": n1,
            "count": k1,
            "rate": k1 / n1 if n1 else None,
            "wilson_ci95": list(wilson_interval(k1, n1).values()),
        },
        "right": {
            "n": n2,
            "count": k2,
            "rate": k2 / n2 if n2 else None,
            "wilson_ci95": list(wilson_interval(k2, n2).values()),
        },
        "difference": diff["difference"],
        "newcombe_ci95": diff["ci95"],
        "fisher_p": fisher_exact_two_sided(k1, n1 - k1, k2, n2 - k2),
    }


def _same_tile_stats(rows: list[dict]) -> dict:
    both = [row for row in _dedupe_hands(rows) if row.get("both_riichi")]
    same = [row for row in both if row.get("same_tile") is True]
    interval = wilson_interval(len(same), len(both))
    return {
        "both_riichi": len(both),
        "same_tile": len(same),
        "same_tile_rate": len(same) / len(both) if both else None,
        "same_tile_wilson_ci95": [interval["low"], interval["high"]],
    }


def primary_decision(comparison: dict, plan: dict) -> tuple[str, str]:
    minimum = plan["analysis"]["minimum_unique_hands_per_group"]
    n1 = comparison["left"]["n"]
    n2 = comparison["right"]["n"]
    if n1 < minimum or n2 < minimum:
        return (
            "underpowered",
            f"unique hands {n1} (B_extra) / {n2} (A_mirror) below {minimum}",
        )
    lower, upper = comparison["newcombe_ci95"]
    difference = comparison["difference"]
    if lower is None or upper is None or difference is None:
        return "inconclusive", "difference interval is undefined"
    if lower > 0 and difference >= plan["analysis"]["replication_min_risk_difference"]:
        return (
            "replicated",
            "Newcombe lower bound > 0 and point difference >= +0.10",
        )
    if upper <= plan["analysis"]["rejection_max_upper_bound"]:
        return (
            "rejected",
            "Newcombe upper bound <= +0.10; a >=10pp increase is ruled out",
        )
    return "inconclusive", "interval does not meet either pre-registered boundary"


def analyze_events(events: list[dict], plan: dict, *, scope: str, stages: list[str]) -> dict:
    groups = group_rows(events)
    group_stats = {
        name: _outcome_stats(rows, plan) for name, rows in groups.items()
    }
    b_extra = groups["B_extra"]
    a_mirror = groups["A_mirror"]
    comparison = compare_binary(b_extra, a_mirror, "deal", plan)
    win_comparison = compare_binary(b_extra, a_mirror, "win", plan)
    income_comparison = compare_continuous(b_extra, a_mirror, "income", plan)
    loss_comparison = compare_continuous(b_extra, a_mirror, "loss", plan)
    within_arm = compare_binary(b_extra, groups["B_both_riichi"], "deal", plan)
    late_share = share_difference(
        b_extra, a_mirror, lambda row: bool(row.get("opp_riichi_late")), plan
    )
    decision, explanation = primary_decision(comparison, plan)
    errors = [row for row in events if row.get("error")]
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "scope": scope,
        "stages": list(stages),
        "created_at": _utc_now(),
        "plan_sha256": plan.get("_plan_sha256"),
        "events": len(events),
        "errors": len(errors),
        "missing_outcomes": sum(1 for row in events if row.get("outcome_missing")),
        "groups": group_stats,
        "primary": {
            "name": plan["analysis"]["primary_endpoint"],
            "hypothesis": "B_extra deal rate > A_mirror deal rate",
            "groups": {
                "b_extra": group_stats["B_extra"],
                "a_mirror": group_stats["A_mirror"],
            },
            "comparison": comparison,
            "decision_rule": {
                "minimum_unique_hands_per_group": plan["analysis"][
                    "minimum_unique_hands_per_group"
                ],
                "replication_min_risk_difference": plan["analysis"][
                    "replication_min_risk_difference"
                ],
                "rejection_max_upper_bound": plan["analysis"][
                    "rejection_max_upper_bound"
                ],
            },
            "decision": decision,
            "decision_explanation": explanation,
        },
        "secondary": {
            "win": win_comparison,
            "income": income_comparison,
            "loss": loss_comparison,
            "b_extra_vs_b_both_riichi_deal": within_arm,
            "risk_concentration": {
                "opp_riichi_late_share": late_share,
                "b_extra_opp_riichi_late_deal": _outcome_stats(
                    groups["B_extra_opp_riichi_late"], plan
                ),
                "a_mirror_opp_riichi_late_deal": _outcome_stats(
                    groups["A_mirror_opp_riichi_late"], plan
                ),
                "b_extra_not_opp_riichi_late_deal": _outcome_stats(
                    groups["B_extra_not_opp_riichi_late"], plan
                ),
                "a_mirror_not_opp_riichi_late_deal": _outcome_stats(
                    groups["A_mirror_not_opp_riichi_late"], plan
                ),
            },
            "shape": {
                "b_extra": _shape_stats(b_extra, "source_eval"),
                "a_mirror": _shape_stats(a_mirror, "source_eval"),
                "b_extra_other": _shape_stats(b_extra, "other_eval"),
                "a_mirror_other": _shape_stats(a_mirror, "other_eval"),
                "b_extra_source_vs_other": _paired_stats(
                    b_extra, "source_eval", "other_eval"
                ),
                "a_mirror_source_vs_other": _paired_stats(
                    a_mirror, "source_eval", "other_eval"
                ),
                "b_both_riichi": _shape_stats(groups["B_both_riichi"], "source_eval"),
                "a_both_riichi": _shape_stats(groups["A_both_riichi"], "source_eval"),
            },
            "both_riichi_discard_agreement": {
                "B": _same_tile_stats(groups["B_all_riichi"]),
                "A": _same_tile_stats(groups["A_all_riichi"]),
            },
            "divergence_counts": {
                "b_extra": len(b_extra),
                "a_mirror": len(a_mirror),
                "b_all_riichi": len(groups["B_all_riichi"]),
                "a_all_riichi": len(groups["A_all_riichi"]),
                "b_extra_share_of_b_riichi": (
                    len(b_extra) / len(groups["B_all_riichi"])
                    if groups["B_all_riichi"]
                    else None
                ),
                "a_mirror_share_of_a_riichi": (
                    len(a_mirror) / len(groups["A_all_riichi"])
                    if groups["A_all_riichi"]
                    else None
                ),
            },
        },
        "limitations": [
            "B_extra and A_mirror are different states in different games; this is not a paired counterfactual and not a causal effect of Turn.",
            "The divergence groups can differ in turn, opponent-riichi, and hand-shape mix; the risk-concentration tables describe but do not remove that confounding.",
            "Hand outcomes are final hand results, not the result of the specific discard at the divergence state.",
            "No retraining, no calibration change, and no chi/pon/riichi penalty tuning was performed.",
            "Fixed v1.3.0 opponents, local CPU, 4p-red-half/default_tenhou; not online ranked strength.",
            "Response is fixed to v1.3.0; W3e says nothing about Response.",
            "max_han uses actual dora plus riichi only; ura dora, riichi sticks, and situational EV are not modelled.",
            "Secondary endpoints are exploratory and unadjusted.",
            "W3e alone does not change the W2 insufficient verdict or model release status.",
        ],
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _pct(value):
    return f"{value:.2%}" if isinstance(value, (int, float)) else "n/a"


def _pp(value):
    return f"{value * 100:+.2f}pp" if isinstance(value, (int, float)) else "n/a"


def _num(value, digits=2):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def _ci(pair):
    if not pair or pair[0] is None or pair[1] is None:
        return "n/a"
    return f"[{pair[0]:+.4f}, {pair[1]:+.4f}]"


def _ci_pct(pair):
    if not pair or pair[0] is None or pair[1] is None:
        return "n/a"
    return f"[{pair[0]:+.2%}, {pair[1]:+.2%}]"


def render_report(result: dict) -> str:
    primary = result["primary"]
    b = result["groups"]["B_extra"]
    a = result["groups"]["A_mirror"]
    comparison = primary["comparison"]
    lines = [
        "# W3e Turn-only riichi divergence replication",
        "",
        f"- scope: `{result['scope']}`; stages: `{', '.join(result['stages'])}`",
        f"- plan SHA-256: `{result.get('plan_sha256')}`",
        f"- captured states: `{result['events']}`; errors: `{result['errors']}`; missing outcomes: `{result['missing_outcomes']}`",
        "",
        "This is a descriptive replication of the W3d hypothesis. `B_extra` and `A_mirror` are different states in different games, not paired counterfactuals.",
        "",
        "## Primary: B_extra vs A_mirror deal rate",
        "",
        "| Group | Unique hands | Wins | Deals | Win rate | Deal rate | 95% Wilson (deal) | Income/hand | Loss/hand |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
        f"| B_extra | {b['n']} | {b['wins']} | {b['deals']} | {_pct(b['win_rate'])} | {_pct(b['deal_rate'])} | "
        f"{_ci_pct(b['deal_wilson_ci95'])} | {_num(b['income_per_hand'], 0)} | {_num(b['loss_per_hand'], 0)} |",
        f"| A_mirror | {a['n']} | {a['wins']} | {a['deals']} | {_pct(a['win_rate'])} | {_pct(a['deal_rate'])} | "
        f"{_ci_pct(a['deal_wilson_ci95'])} | {_num(a['income_per_hand'], 0)} | {_num(a['loss_per_hand'], 0)} |",
        "",
        f"- risk difference: `{_pp(comparison['difference'])}`; Newcombe 95% CI `{_ci_pct(comparison['newcombe_ci95'])}`",
        f"- Fisher exact two-sided p: `{_num(comparison['fisher_p'], 4)}`",
        f"- seed-cluster bootstrap difference: `{_pp(comparison['seed_cluster_bootstrap'].get('mean'))}` "
        f"[{_ci_pct(comparison['seed_cluster_bootstrap'].get('ci95'))}], "
        f"valid resamples `{comparison['seed_cluster_bootstrap'].get('valid_resamples')}`",
        f"- decision: **{primary['decision']}** ({primary['decision_explanation']})",
        "",
        "## Context groups",
        "",
        "| Group | Unique hands | Win rate | Deal rate | Income/hand | Loss/hand |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "B_extra",
        "A_mirror",
        "B_both_riichi",
        "A_both_riichi",
        "B_all_riichi",
        "A_all_riichi",
    ):
        item = result["groups"][name]
        lines.append(
            f"| {name} | {item['n']} | {_pct(item['win_rate'])} | {_pct(item['deal_rate'])} | "
            f"{_num(item['income_per_hand'], 0)} | {_num(item['loss_per_hand'], 0)} |"
        )
    lines += [
        "",
        "## Secondary outcomes",
        "",
        "| Metric | B_extra vs A_mirror | 95% Newcombe | Fisher p |",
        "|---|---:|---|---:|",
    ]
    for key, label in (
        ("win", "win rate"),
        ("b_extra_vs_b_both_riichi_deal", "B_extra vs B_both_riichi deal rate"),
    ):
        item = (
            result["secondary"][key]
            if key in result["secondary"]
            else result["secondary"]["win"]
        )
        lines.append(
            f"| {label} | {_pp(item['difference'])} | {_ci_pct(item['newcombe_ci95'])} | "
            f"{_num(item['fisher_p'], 4)} |"
        )
    for key, label in (("income", "income/hand"), ("loss", "loss/hand")):
        item = result["secondary"][key]
        lines.append(
            f"| {label} | {_num(item['difference'], 0)} | {_ci_pct(item['seed_cluster_bootstrap'].get('ci95'))} | n/a |"
        )
    lines += [
        "",
        "## Opponent-riichi + late subset",
        "",
        "| Group | Unique hands | Deal rate | Deal 95% Wilson |",
        "|---|---:|---:|---|",
    ]
    risk = result["secondary"]["risk_concentration"]
    for key, name in (
        ("b_extra_opp_riichi_late_deal", "B_extra opp_riichi_late"),
        ("a_mirror_opp_riichi_late_deal", "A_mirror opp_riichi_late"),
        ("b_extra_not_opp_riichi_late_deal", "B_extra not opp_riichi_late"),
        ("a_mirror_not_opp_riichi_late_deal", "A_mirror not opp_riichi_late"),
    ):
        item = risk[key]
        lines.append(
            f"| {name} | {item['n']} | {_pct(item['deal_rate'])} | {_ci_pct(item['deal_wilson_ci95'])} |"
        )
    share = risk["opp_riichi_late_share"]
    lines += [
        "",
        f"- `opp_riichi_late` share: B_extra `{_pct(share['left']['rate'])}` "
        f"({share['left']['count']}/{share['left']['n']}), A_mirror `{_pct(share['right']['rate'])}` "
        f"({share['right']['count']}/{share['right']['n']}); difference `{_pp(share['difference'])}` "
        f"`{_ci_pct(share['newcombe_ci95'])}`, Fisher p `{_num(share['fisher_p'], 4)}`",
        "",
        "## Hand shape of executed riichi discards",
        "",
        "`remaining` = W3d visibility rule; `max_han` = actual dora + riichi, no ura.",
        "",
        "| Group | Shaped | Tenpai | Wait tiles | Remaining waits | Max han |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, name in (
        ("b_extra", "B_extra"),
        ("a_mirror", "A_mirror"),
        ("b_both_riichi", "B_both_riichi"),
        ("a_both_riichi", "A_both_riichi"),
    ):
        shape = result["secondary"]["shape"][key]
        lines.append(
            f"| {name} | {shape['shaped']} | {_pct(shape['tenpai_rate'])} | "
            f"{_num(shape['mean_wait_count'])} | {_num(shape['mean_remaining'])} | "
            f"{_num(shape['mean_max_han'])} |"
        )
    lines += [
        "",
        "### Shadow riichi shape and paired agreement",
        "",
        "| Group | Paired n | Δremaining | source better/equal/worse | Δhan | source better/equal/worse |",
        "|---|---:|---:|---|---:|---|",
    ]
    for key, name in (
        ("b_extra_source_vs_other", "B_extra"),
        ("a_mirror_source_vs_other", "A_mirror"),
    ):
        paired = result["secondary"]["shape"][key]
        rem = paired["remaining_better_equal_worse"]
        han = paired["han_better_equal_worse"]
        lines.append(
            f"| {name} | {paired['n']} | {_num(paired['mean_delta_remaining'], 3)} | "
            f"{rem[0]}/{rem[1]}/{rem[2]} | {_num(paired['mean_delta_han'], 3)} | "
            f"{han[0]}/{han[1]}/{han[2]} |"
        )
    lines += [
        "",
        "## Both-riichi discard agreement",
        "",
        "| Arm | Both riichi | Same tile | Same-tile rate | 95% Wilson |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in ARM_IDS:
        item = result["secondary"]["both_riichi_discard_agreement"][arm]
        lines.append(
            f"| {arm} | {item['both_riichi']} | {item['same_tile']} | "
            f"{_pct(item['same_tile_rate'])} | {_ci_pct(item['same_tile_wilson_ci95'])} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# artifact helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_events(stage_dir: str | Path) -> list[dict]:
    path = Path(stage_dir) / "events.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing events.jsonl: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
