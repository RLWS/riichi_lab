"""W5: serving-time veto of late extra Riichi under opponent threat.

Design (pre-registered in
``docs/w5-riichi-veto-preregistration-2026-11-18.md``):

* Arm C = v1.4.0 Turn + v1.3.0 Response, no veto.
* Arm T = v1.4.0 Turn + v1.3.0 veto + v1.3.0 Response.
* The veto replaces a v1.4.0 ``riichi_discard`` with the v1.3.0 ``discard`` on
  the same state only when an opponent has declared Riichi and
  ``turn_number > 12``.
* Three fixed v1.3.0 opponents, four seat rotations per seed and arm.
* Primary endpoint: paired hand-level deal rate of the first-veto hand.
* Extension from Stage 1 to Stage 2 is count-only; outcomes are never read
  before the extension decision.

The module keeps deterministic pieces (plan/schedule/counts) separate from the
outcome analysis, mirroring W3e.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random

import yaml

from rilab.w2 import service_configs, sha256_file, sha256_json
from rilab.w3 import hand_metrics, read_json, read_jsonl, read_jsonl_gz
from rilab.w3e import wilson_interval

PURPOSE = "w5_riichi_veto"
ARM_IDS = ("C", "T")
EXPECTED_ARMS = {
    "C": {"turn": "v140", "response": "v130", "veto": None},
    "T": {"turn": "v140", "response": "v130", "veto": "v130"},
}
STAGE_SEED_COUNTS = {"smoke": 2, "stage1": 64, "stage2": 64}
FORBIDDEN_SEED_RANGES = (
    (40001, 40008),
    (41001, 41128),
    (42001, 42128),
    (43001, 43128),
)
EXPECTED_SERVICES = {
    "v130": {
        "adapter_profile": "full_state_v2",
        "commit": "1e5092f344fb69bcdf4697f683f2ecf15b9865d8",
        "turn_sha256": "30f2811cd86beebad57d4da1428272b25b598fb64a6231b5a1760e11ec08bf90",
        "response_sha256": "e1d8c58a2aa61cef23852f15fe5bb9c65014bc3916c52756fb8c4cd1cd956c51",
    },
    "v140": {
        "adapter_profile": "full_state_v3",
        "commit": "57e8fb336ed4aa7b19afd8ca1ef4ee4c41a96ed2",
        "turn_sha256": "eeaa87b9c122fd49dd917758f299430764d305acc69d46f8a84c333448646cfe",
        "response_sha256": "41403f23015ee6cc364f48975bbffd6124122a717a760eccc127d01efa18e3df",
    },
}
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
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("W5 plan must be a mapping")
    engine = plan.get("engine")
    if not isinstance(engine, dict):
        raise ValueError("W5 plan requires engine")
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
            raise ValueError(f"W5 plan requires {range_key} or {list_key}")
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
        raise ValueError("W5 plan_version must be 1")
    if plan.get("purpose") != PURPOSE:
        raise ValueError("W5 purpose must be w5_riichi_veto")
    engine = plan.get("engine")
    if not isinstance(engine, dict):
        raise ValueError("W5 plan requires engine")
    if engine.get("game_mode") != "4p-red-half" or engine.get("rule") != "default_tenhou":
        raise ValueError("W5 only supports 4p-red-half/default_tenhou")

    services = service_configs(plan)
    for name, expected in EXPECTED_SERVICES.items():
        service = services[name]
        if service["adapter_profile"] != expected["adapter_profile"]:
            raise ValueError(f"W5 service profile differs: {name}")
        if service["revision"].get("commit") != expected["commit"]:
            raise ValueError(f"W5 service commit differs: {name}")
        for task in ("turn", "response"):
            if service["model_identity"].get(f"{task}_sha256") != expected[
                f"{task}_sha256"
            ]:
                raise ValueError(f"W5 service weight hash differs: {name}/{task}")

    arms = plan.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_IDS):
        raise ValueError("W5 arms must be exactly C/T")
    for arm, expected in EXPECTED_ARMS.items():
        if arms.get(arm) != expected:
            raise ValueError(f"W5 arm definition differs: {arm}")

    veto = plan.get("veto")
    if not isinstance(veto, dict):
        raise ValueError("W5 plan requires veto")
    if veto.get("arm") != "T" or veto.get("scope") != "treatment_arm_only":
        raise ValueError("W5 veto must be treatment-arm only")
    condition = veto.get("condition")
    if not isinstance(condition, dict):
        raise ValueError("W5 veto requires condition")
    if condition.get("opp_riichi_count_min") != 1:
        raise ValueError("W5 veto opp_riichi_count_min must be 1")
    if condition.get("turn_number_min_exclusive") != 12:
        raise ValueError("W5 veto turn_number_min_exclusive must be 12")
    trigger = veto.get("trigger")
    if not isinstance(trigger, dict):
        raise ValueError("W5 veto requires trigger")
    if trigger.get("new_turn_kind") != RIICHI_KIND:
        raise ValueError("W5 veto new_turn_kind must be riichi_discard")
    if trigger.get("old_turn_kind") != DISCARD_KIND:
        raise ValueError("W5 veto old_turn_kind must be discard")
    if veto.get("replacement") != "old_turn_discard":
        raise ValueError("W5 veto replacement must be old_turn_discard")
    if veto.get("old_state_projection") != "full_state_v2":
        raise ValueError("W5 veto old_state_projection must be full_state_v2")

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
        raise ValueError("W5 plan requires capture")
    if capture.get("tasks") != ["turn"]:
        raise ValueError("W5 capture.tasks must be [turn]")
    if capture.get("players") != "focal":
        raise ValueError("W5 capture.players must be focal")
    if capture.get("store_state_when") != "veto_applied":
        raise ValueError("W5 capture.store_state_when must be veto_applied")

    analysis = plan.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("W5 plan requires analysis")
    if analysis.get("primary_endpoint") != "first_veto_pair_deal_rate":
        raise ValueError("W5 primary_endpoint is fixed")
    if analysis.get("primary_unit") != "first_veto_pair":
        raise ValueError("W5 primary_unit is fixed")
    if analysis.get("direction") != "treatment_lower_deal_rate":
        raise ValueError("W5 direction is fixed")
    if list(analysis.get("pairing", [])) != ["seed", "rotation", "hand_id"]:
        raise ValueError("W5 pairing must be seed/rotation/hand_id")
    if analysis.get("confidence_level") != 0.95:
        raise ValueError("W5 confidence_level must be 0.95")
    if analysis.get("bootstrap_resamples") != 10000:
        raise ValueError("W5 bootstrap_resamples must be 10000")
    if analysis.get("bootstrap_seed") != 20261118:
        raise ValueError("W5 bootstrap_seed must be 20261118")
    if analysis.get("minimum_pairs") != 24:
        raise ValueError("W5 minimum_pairs must be 24")
    if analysis.get("supported_min_risk_difference") != 0.10:
        raise ValueError("W5 supported_min_risk_difference must be 0.10")
    if analysis.get("supported_ci_upper_bound") != 0.0:
        raise ValueError("W5 supported_ci_upper_bound must be 0.0")
    extension = analysis.get("extension")
    if not isinstance(extension, dict):
        raise ValueError("W5 plan requires analysis.extension")
    if extension.get("count_field") != "first_veto_pairs":
        raise ValueError("W5 extension count_field must be first_veto_pairs")
    if extension.get("extend_if_below") != 24:
        raise ValueError("W5 extension.extend_if_below must be 24")
    if extension.get("decision_basis") != "count_only_no_outcomes":
        raise ValueError("W5 extension decision must use counts only")
    if analysis.get("forbidden_seed_ranges") != [
        list(pair) for pair in FORBIDDEN_SEED_RANGES
    ]:
        raise ValueError("W5 forbidden_seed_ranges differ from the pre-registration")


def build_schedule(plan: dict, seeds: list[int]) -> list[dict]:
    schedule = []
    for seed in seeds:
        for arm in ARM_IDS:
            base = [arm, "v130", "v130", "v130"]
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
    missing = {"v130", "C", "T"} - set(backends)
    if missing:
        raise ValueError(f"W5 native config lacks backends: {sorted(missing)}")
    return {
        "config_version": 1,
        "engine": {
            "game_mode": plan["engine"]["game_mode"],
            "rule": plan["engine"]["rule"],
            "seeds": list(seeds),
        },
        "backends": {name: backends[name] for name in ("v130", "C", "T")},
        "matchups": [],
        "evaluation": {
            "cluster": "seed_with_all_rotations",
            "confidence_level": 0.95,
            "primary_paired_group": "W5",
        },
        "artifacts": {
            "max_bytes": plan.get("artifacts", {}).get("max_bytes", 2_000_000_000),
            "snapshot_stride": plan.get("artifacts", {}).get("snapshot_stride", 100),
            "full_snapshots_first_run": plan.get("artifacts", {}).get(
                "full_snapshots_first_run", False
            ),
        },
    }


def _backend_base(service: dict, api_base: str) -> dict:
    return {
        "adapter_profile": service["adapter_profile"],
        "api_base": api_base.rstrip("/"),
        "timeout_seconds": service.get("timeout_seconds", 30.0),
        "response": service["response"],
        "revision": service["revision"],
        "model_identity": service["model_identity"],
    }


def build_backends(plan: dict, api_bases: dict) -> dict:
    if set(api_bases) != {"v130", "v140"}:
        raise ValueError("W5 api_bases must cover v130/v140")
    services = service_configs(plan)
    v130 = _backend_base(services["v130"], api_bases["v130"])
    v140 = _backend_base(services["v140"], api_bases["v140"])
    condition = plan["veto"]["condition"]
    return {
        "v130": v130,
        "C": {
            "backend_type": "task_routed",
            "turn_backend": dict(v140),
            "response_backend": dict(v130),
        },
        "T": {
            "backend_type": "veto_turn",
            "turn_backend": dict(v140),
            "veto_backend": dict(v130),
            "response_backend": dict(v130),
            "veto_policy": {
                "opp_riichi_count_min": condition["opp_riichi_count_min"],
                "turn_number_min_exclusive": condition[
                    "turn_number_min_exclusive"
                ],
            },
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
        "veto": plan["veto"],
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
    """Keep full state only for focal-seat decisions where the veto applied."""
    prediction = record.get("prediction") or {}
    veto = prediction.get("veto") or {}
    return (
        record.get("player_id") == spec.get("focal_seat")
        and record.get("task") == "turn"
        and veto.get("applied") is True
    )


# ---------------------------------------------------------------------------
# count-only phase (no outcomes)
# ---------------------------------------------------------------------------


def _game_index(summary: dict) -> dict:
    index = {}
    for game in summary.get("games", []):
        index[(game.get("seed"), game.get("rotation"), game.get("matchup_id"))] = game
    return index


def _first_veto(records: list[dict]):
    vetoes = [
        record
        for record in records
        if (record.get("prediction") or {}).get("veto", {}).get("applied") is True
    ]
    if not vetoes:
        return None, []
    first = min(vetoes, key=lambda record: record.get("decision_id", -1))
    return first, vetoes


def collect_first_vetoes(batch_dir: str | Path) -> tuple[list[dict], dict]:
    """Return one ID-only row per treatment game with an applied veto."""
    batch = Path(batch_dir)
    summary = read_json(batch / "summary.json")
    games = summary.get("games", [])
    rows = []
    meta = {
        "games": len(games),
        "treatment_games": sum(
            1 for game in games if game.get("matchup_id") == "T"
        ),
        "veto_events": 0,
        "games_with_veto": 0,
        "first_vetoes": 0,
        "anomalies": [],
    }
    for game in games:
        if game.get("matchup_id") != "T":
            continue
        path = batch / "decisions" / f"{game['game_id']}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(f"missing decision spool: {path}")
        first, vetoes = _first_veto(list(read_jsonl_gz(path)))
        if first is None:
            continue
        meta["veto_events"] += len(vetoes)
        meta["games_with_veto"] += 1
        focal = game.get("focal_seat")
        if first.get("player_id") != focal or first.get("task") != "turn":
            meta["anomalies"].append(
                {
                    "game_id": game["game_id"],
                    "decision_id": first.get("decision_id"),
                    "player_id": first.get("player_id"),
                    "task": first.get("task"),
                    "focal_seat": focal,
                }
            )
            continue
        rows.append(
            {
                "stage": None,
                "game_id": game["game_id"],
                "seed": game.get("seed"),
                "rotation": game.get("rotation"),
                "focal": focal,
                "decision_id": first.get("decision_id"),
                "hand_id": first.get("hand_id"),
                "veto_count": len(vetoes),
                "veto_decision_ids": [
                    record.get("decision_id") for record in vetoes
                ],
            }
        )
    meta["first_vetoes"] = len(rows)
    return rows, meta


def split_first_veto_pairs(
    rows: list[dict], summary: dict
) -> tuple[list[dict], list[dict]]:
    """A first-veto pair needs the matching control game in the same batch."""
    index = _game_index(summary)
    pairs = []
    unpaired = []
    for row in rows:
        control = index.get((row["seed"], row["rotation"], "C"))
        if control is None:
            unpaired.append({**row, "reason": "control_game_missing"})
            continue
        pairs.append({**row, "control_game_id": control["game_id"]})
    return pairs, unpaired


def count_document(
    rows: list[dict],
    meta: dict,
    *,
    stage: str,
    plan: dict,
    summary: dict,
    audit_games: int,
    plan_sha256: str,
) -> dict:
    pairs, unpaired = split_first_veto_pairs(rows, summary)
    extension = plan["analysis"]["extension"]
    extend = stage == "stage1" and len(pairs) < extension["extend_if_below"]
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "stage": stage,
        "created_at": _utc_now(),
        "plan_sha256": plan_sha256,
        "seeds": list(plan[f"{stage}_seeds"]) if f"{stage}_seeds" in plan else [],
        "games": summary.get("completed_games"),
        "expected_games": summary.get("expected_games"),
        "fallbacks": sum(
            int(game.get("fallbacks", 0)) for game in summary.get("games", [])
        ),
        "audit_games": audit_games,
        "treatment_games": meta.get("treatment_games"),
        "veto_events": meta.get("veto_events"),
        "games_with_veto": meta.get("games_with_veto"),
        "later_veto_events": (meta.get("veto_events") or 0)
        - (meta.get("first_vetoes") or 0),
        "first_vetoes": meta.get("first_vetoes"),
        "first_veto_pairs": len(pairs),
        "unpaired_first_vetoes": len(unpaired),
        "unpaired": unpaired,
        "anomalies": meta.get("anomalies", []),
        "pair_ids": pairs,
        "extension_rule": {
            "stage": "stage1" if stage == "stage1" else None,
            "count_field": extension["count_field"],
            "extend_if_below": extension["extend_if_below"],
            "current_value": len(pairs) if stage == "stage1" else None,
            "extension_required": extend,
            "decision_basis": extension["decision_basis"],
        },
        "outcomes_inspected": False,
    }


# ---------------------------------------------------------------------------
# outcome pairing and analysis
# ---------------------------------------------------------------------------


def _fingerprint(record: dict) -> str:
    payload = {
        "player_id": record.get("player_id"),
        "task": record.get("task"),
        "hand_id": record.get("hand_id"),
        "selected_action": record.get("selected_action"),
        "engine_action": record.get("engine_action"),
        "new_events": record.get("new_events"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def verify_prefix(
    treatment_records: list[dict],
    control_records: list[dict],
    stop_decision_id: int,
) -> tuple[bool, str | None]:
    """Prove the two games are identical up to (but excluding) the veto action."""
    treatment = {record.get("decision_id"): record for record in treatment_records}
    control = {record.get("decision_id"): record for record in control_records}
    for decision_id in range(stop_decision_id):
        left = treatment.get(decision_id)
        right = control.get(decision_id)
        if left is None or right is None:
            return False, f"missing_decision_{decision_id}"
        if _fingerprint(left) != _fingerprint(right):
            return False, f"decision_{decision_id}_differs"
    left_stop = treatment.get(stop_decision_id)
    right_stop = control.get(stop_decision_id)
    if left_stop is None or right_stop is None:
        return False, "missing_veto_decision"
    if (
        left_stop.get("player_id") != right_stop.get("player_id")
        or left_stop.get("task") != right_stop.get("task")
        or left_stop.get("hand_id") != right_stop.get("hand_id")
    ):
        return False, "veto_decision_context_differs"
    return True, None


def load_stage_pairs(batch_dir: str | Path) -> tuple[list[dict], list[dict], dict]:
    """Load paired first-veto hands and their full-hand outcomes."""
    batch = Path(batch_dir)
    summary = read_json(batch / "summary.json")
    rows, meta = collect_first_vetoes(batch)
    index = _game_index(summary)
    pairs = []
    unpaired = []
    for row in rows:
        control = index.get((row["seed"], row["rotation"], "C"))
        if control is None:
            unpaired.append({**row, "reason": "control_game_missing"})
            continue
        treatment_path = batch / "decisions" / f"{row['game_id']}.jsonl.gz"
        control_path = batch / "decisions" / f"{control['game_id']}.jsonl.gz"
        if not control_path.is_file():
            unpaired.append({**row, "reason": "control_decision_log_missing"})
            continue
        treatment_records = list(read_jsonl_gz(treatment_path))
        control_records = list(read_jsonl_gz(control_path))
        ok, reason = verify_prefix(
            treatment_records, control_records, int(row["decision_id"])
        )
        if not ok:
            unpaired.append({**row, "reason": f"prior_divergence:{reason}"})
            continue
        treatment_events = list(
            read_jsonl(batch / "games" / f"{row['game_id']}.mjai.jsonl")
        )
        control_events = list(
            read_jsonl(batch / "games" / f"{control['game_id']}.mjai.jsonl")
        )
        treatment_hands = hand_metrics(treatment_events, row["focal"])
        control_hands = hand_metrics(control_events, control["focal_seat"])
        hand_id = row.get("hand_id")
        if not isinstance(hand_id, int) or not 1 <= hand_id <= len(treatment_hands):
            unpaired.append({**row, "reason": "treatment_hand_missing"})
            continue
        if not isinstance(hand_id, int) or not 1 <= hand_id <= len(control_hands):
            unpaired.append({**row, "reason": "control_hand_missing"})
            continue
        pairs.append(
            {
                "stage": None,
                "game_id": row["game_id"],
                "control_game_id": control["game_id"],
                "seed": row["seed"],
                "rotation": row["rotation"],
                "focal": row["focal"],
                "decision_id": row["decision_id"],
                "hand_id": hand_id,
                "veto_count": row["veto_count"],
                "veto_decision_ids": row["veto_decision_ids"],
                "treatment": treatment_hands[hand_id - 1],
                "control": control_hands[hand_id - 1],
            }
        )
    return pairs, unpaired, meta


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap(
    rows: list[dict], value_fn, *, resamples: int, seed: int
) -> dict:
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row.get("seed")].append(float(value_fn(row)))
    seeds = sorted(key for key, values in by_seed.items() if values)
    if not seeds:
        return {
            "mean": None,
            "ci95": [None, None],
            "valid_resamples": 0,
            "seed_clusters": 0,
        }
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [rng.choice(seeds) for _ in seeds]
        values = [value for key in draw for value in by_seed.get(key, [])]
        if not values:
            continue
        means.append(sum(values) / len(values))
    if not means:
        return {
            "mean": None,
            "ci95": [None, None],
            "valid_resamples": 0,
            "seed_clusters": len(seeds),
        }
    return {
        "mean": sum(means) / len(means),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "valid_resamples": len(means),
        "seed_clusters": len(seeds),
    }


def mcnemar_exact(b: int, c: int) -> float | None:
    """Exact two-sided McNemar p-value for discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_binary(pairs: list[dict], field: str, plan: dict) -> dict:
    valid = [
        pair
        for pair in pairs
        if pair["treatment"].get(field) is not None
        and pair["control"].get(field) is not None
    ]
    n = len(valid)
    treatment = sum(1 for pair in valid if pair["treatment"].get(field))
    control = sum(1 for pair in valid if pair["control"].get(field))
    b = sum(
        1
        for pair in valid
        if pair["treatment"].get(field) and not pair["control"].get(field)
    )
    c = sum(
        1
        for pair in valid
        if pair["control"].get(field) and not pair["treatment"].get(field)
    )
    both = sum(
        1
        for pair in valid
        if pair["control"].get(field) and pair["treatment"].get(field)
    )
    neither = n - b - c - both
    difference = (treatment - control) / n if n else None
    bootstrap = _cluster_bootstrap(
        valid,
        lambda pair: (
            (1.0 if pair["treatment"].get(field) else 0.0)
            - (1.0 if pair["control"].get(field) else 0.0)
        ),
        resamples=plan["analysis"]["bootstrap_resamples"],
        seed=plan["analysis"]["bootstrap_seed"],
    )
    return {
        "field": field,
        "n_pairs": n,
        "unique_hands": n,
        "treatment": {
            "count": treatment,
            "rate": treatment / n if n else None,
            "wilson_ci95": [
                wilson_interval(treatment, n)["low"],
                wilson_interval(treatment, n)["high"],
            ],
        },
        "control": {
            "count": control,
            "rate": control / n if n else None,
            "wilson_ci95": [
                wilson_interval(control, n)["low"],
                wilson_interval(control, n)["high"],
            ],
        },
        "difference": difference,
        "discordant": {
            "treatment_only": b,
            "control_only": c,
            "both": both,
            "neither": neither,
        },
        "mcnemar_exact_p": mcnemar_exact(b, c),
        "seed_cluster_bootstrap": bootstrap,
    }


def paired_continuous(pairs: list[dict], field: str, plan: dict) -> dict:
    valid = [
        pair
        for pair in pairs
        if pair["treatment"].get(field) is not None
        and pair["control"].get(field) is not None
    ]
    treatment_values = [float(pair["treatment"].get(field, 0)) for pair in valid]
    control_values = [float(pair["control"].get(field, 0)) for pair in valid]
    n = len(valid)
    return {
        "field": field,
        "n_pairs": n,
        "treatment_mean": sum(treatment_values) / n if n else None,
        "control_mean": sum(control_values) / n if n else None,
        "difference": (
            (sum(treatment_values) - sum(control_values)) / n if n else None
        ),
        "seed_cluster_bootstrap": _cluster_bootstrap(
            valid,
            lambda pair: float(pair["treatment"].get(field, 0))
            - float(pair["control"].get(field, 0)),
            resamples=plan["analysis"]["bootstrap_resamples"],
            seed=plan["analysis"]["bootstrap_seed"],
        ),
    }


def _hand_outcome_stats(pairs: list[dict], side: str) -> dict:
    hands = [pair[side] for pair in pairs]
    n = len(hands)
    wins = sum(1 for hand in hands if hand.get("win"))
    deals = sum(1 for hand in hands if hand.get("deal"))
    income = sum(hand.get("income", 0) for hand in hands)
    loss = sum(hand.get("loss", 0) for hand in hands)
    return {
        "n": n,
        "wins": wins,
        "deals": deals,
        "win_rate": wins / n if n else None,
        "deal_rate": deals / n if n else None,
        "income": income,
        "loss": loss,
        "income_per_hand": income / n if n else None,
        "loss_per_hand": loss / n if n else None,
        "mean_win_income": income / wins if wins else None,
        "mean_deal_loss": loss / deals if deals else None,
    }


def _arm_game_stats(games: list[dict], arm: str) -> dict:
    rows = [game for game in games if game.get("matchup_id") == arm]
    if not rows:
        return {
            "games": 0,
            "mean_rank": None,
            "fourth_rate": None,
            "mean_score": None,
        }
    ranks = [game["ranks"][game["focal_seat"]] for game in rows]
    scores = [game["scores"][game["focal_seat"]] for game in rows]
    return {
        "games": len(rows),
        "mean_rank": sum(ranks) / len(ranks),
        "fourth_rate": sum(1 for rank in ranks if rank == 4) / len(ranks),
        "mean_score": sum(scores) / len(scores),
    }


def game_level(games: list[dict], pair_keys: set, plan: dict) -> dict:
    index = {}
    for game in games:
        index[(game.get("seed"), game.get("rotation"), game.get("matchup_id"))] = game
    paired = []
    for (seed, rotation, arm) in list(index):
        if arm != "T":
            continue
        control = index.get((seed, rotation, "C"))
        treatment = index.get((seed, rotation, "T"))
        if control is None or treatment is None:
            continue
        paired.append(
            {
                "seed": seed,
                "rotation": rotation,
                "rank_difference": (
                    treatment["ranks"][treatment["focal_seat"]]
                    - control["ranks"][control["focal_seat"]]
                ),
                "score_difference": (
                    treatment["scores"][treatment["focal_seat"]]
                    - control["scores"][control["focal_seat"]]
                ),
                "treatment_fourth": 1.0
                if treatment["ranks"][treatment["focal_seat"]] == 4
                else 0.0,
                "control_fourth": 1.0
                if control["ranks"][control["focal_seat"]] == 4
                else 0.0,
            }
        )
    affected_games = [
        game
        for game in games
        if (game.get("seed"), game.get("rotation")) in pair_keys
    ]
    config = plan["analysis"]
    return {
        "all_games": {
            "C": _arm_game_stats(games, "C"),
            "T": _arm_game_stats(games, "T"),
        },
        "paired": {
            "clusters": len({row["seed"] for row in paired}),
            "pairs": len(paired),
            "rank_difference": _cluster_bootstrap(
                paired,
                lambda row: row["rank_difference"],
                resamples=config["bootstrap_resamples"],
                seed=config["bootstrap_seed"],
            ),
            "score_difference": _cluster_bootstrap(
                paired,
                lambda row: row["score_difference"],
                resamples=config["bootstrap_resamples"],
                seed=config["bootstrap_seed"],
            ),
            "fourth_rate_difference": (
                sum(row["treatment_fourth"] for row in paired)
                - sum(row["control_fourth"] for row in paired)
            )
            / len(paired)
            if paired
            else None,
        },
        "affected_games": {
            "games": len({(game.get("seed"), game.get("rotation")) for game in affected_games}),
            "C": _arm_game_stats(affected_games, "C"),
            "T": _arm_game_stats(affected_games, "T"),
        },
    }


def primary_decision(primary: dict, plan: dict) -> tuple[str, str]:
    n = primary.get("n_pairs", 0)
    minimum = plan["analysis"]["minimum_pairs"]
    if n < minimum:
        return (
            "underpowered",
            f"first-veto pairs {n} below {minimum} after the allowed extension",
        )
    difference = primary.get("difference")
    upper = primary["seed_cluster_bootstrap"]["ci95"][1]
    if difference is None or upper is None:
        return "underpowered", "paired risk difference is undefined"
    threshold = -plan["analysis"]["supported_min_risk_difference"]
    if difference <= threshold and upper < plan["analysis"]["supported_ci_upper_bound"]:
        return (
            "supported",
            "treatment deal rate lower by >=10pp and paired CI upper bound < 0",
        )
    return "not_supported", "estimate does not meet the support rule"


def analyze_stages(stage_data: list[dict], plan: dict, *, scope: str) -> dict:
    pairs = []
    games = []
    unpaired = []
    veto_meta = {
        "veto_events": 0,
        "games_with_veto": 0,
        "first_vetoes": 0,
        "later_veto_events": 0,
    }
    stages = []
    for item in stage_data:
        stage_pairs = [{**pair, "stage": item["stage"]} for pair in item["pairs"]]
        pairs.extend(stage_pairs)
        games.extend(item["summary"].get("games", []))
        unpaired.extend(
            [{**row, "stage": item["stage"]} for row in item["unpaired"]]
        )
        meta = item["meta"]
        veto_meta["veto_events"] += meta.get("veto_events") or 0
        veto_meta["games_with_veto"] += meta.get("games_with_veto") or 0
        veto_meta["first_vetoes"] += meta.get("first_vetoes") or 0
        veto_meta["later_veto_events"] += (
            (meta.get("veto_events") or 0) - (meta.get("first_vetoes") or 0)
        )
        stages.append(item["stage"])
    primary = paired_binary(pairs, "deal", plan)
    win = paired_binary(pairs, "win", plan)
    income = paired_continuous(pairs, "income", plan)
    loss = paired_continuous(pairs, "loss", plan)
    decision, explanation = primary_decision(primary, plan)
    pair_keys = {(pair["seed"], pair["rotation"]) for pair in pairs}
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "scope": scope,
        "stages": stages,
        "created_at": _utc_now(),
        "first_veto_pairs": len(pairs),
        "first_veto_pair_ids": [
            {
                "stage": pair["stage"],
                "game_id": pair["game_id"],
                "control_game_id": pair["control_game_id"],
                "seed": pair["seed"],
                "rotation": pair["rotation"],
                "decision_id": pair["decision_id"],
                "hand_id": pair["hand_id"],
                "veto_count": pair["veto_count"],
            }
            for pair in pairs
        ],
        "unpaired": unpaired,
        "veto_events": {
            **veto_meta,
            "affected_games": len(pair_keys),
        },
        "primary": {
            "name": "first_veto_pair_deal_rate",
            **primary,
            "treatment_hands": _hand_outcome_stats(pairs, "treatment"),
            "control_hands": _hand_outcome_stats(pairs, "control"),
            "decision": decision,
            "explanation": explanation,
            "rule": {
                "underpowered": "n_pairs < 24 after the allowed extension",
                "supported": "RD <= -0.10 and paired 95% CI upper bound < 0",
                "else": "not_supported",
            },
        },
        "secondary": {
            "win": win,
            "income": income,
            "loss": loss,
            "treatment_hands": _hand_outcome_stats(pairs, "treatment"),
            "control_hands": _hand_outcome_stats(pairs, "control"),
            "game_level": game_level(games, pair_keys, plan),
            "exploratory": True,
            "unadjusted": True,
        },
        "limitations": [
            "Fixed v1.3.0 opponents; local game outcomes, not online ranked strength.",
            "The veto changes the game trajectory; only the first-veto hand is paired.",
            "Secondary endpoints are exploratory and unadjusted.",
            "Unpaired first-vetoes are reported, never silently dropped.",
        ],
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _pct(value):
    return "n/a" if value is None else f"{value:.2%}"


def _pp(value):
    return "n/a" if value is None else f"{value:+.2%}"


def _num(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(pair, transform=lambda value: f"{value:+.3f}"):
    if pair is None or pair[0] is None or pair[1] is None:
        return "n/a"
    return f"[{transform(pair[0])}, {transform(pair[1])}]"


def render_report(result: dict) -> str:
    primary = result["primary"]
    secondary = result["secondary"]
    lines = [
        "# W5 Riichi veto report",
        "",
        f"- scope: `{result['scope']}`",
        f"- stages: `{', '.join(result['stages'])}`",
        f"- first-veto pairs: `{result['first_veto_pairs']}`",
        f"- decision: **{primary['decision']}**",
        f"- explanation: {primary['explanation']}",
        "",
        "## Primary: first-veto hand deal rate",
        "",
        "| Group | n | Wins | Deals | Win rate | Deal rate | Income/hand | Loss/hand |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in (
        ("Control (C)", primary["control_hands"]),
        ("Treatment (T)", primary["treatment_hands"]),
    ):
        lines.append(
            f"| {name} | {stats['n']} | {stats['wins']} | {stats['deals']} | "
            f"{_pct(stats['win_rate'])} | {_pct(stats['deal_rate'])} | "
            f"{_num(stats['income_per_hand'], 0)} | "
            f"{_num(stats['loss_per_hand'], 0)} |"
        )
    lines += [
        "",
        f"- paired risk difference (T - C): `{_pp(primary['difference'])}`",
        f"- seed-cluster bootstrap 95% CI: "
        f"`{_ci(primary['seed_cluster_bootstrap']['ci95'], _pp)}` "
        f"({primary['seed_cluster_bootstrap']['valid_resamples']} resamples, "
        f"{primary['seed_cluster_bootstrap']['seed_clusters']} seed clusters)",
        f"- exact McNemar p: `{_num(primary['mcnemar_exact_p'], 4)}`",
        f"- discordant: treatment-only deal {primary['discordant']['treatment_only']}, "
        f"control-only deal {primary['discordant']['control_only']}, "
        f"both {primary['discordant']['both']}, neither {primary['discordant']['neither']}",
        "",
        "## Secondary (exploratory, unadjusted)",
        "",
        "| Endpoint | Treatment | Control | Difference | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---|",
        f"| Win rate | {_pct(secondary['win']['treatment']['rate'])} | "
        f"{_pct(secondary['win']['control']['rate'])} | "
        f"{_pp(secondary['win']['difference'])} | "
        f"{_ci(secondary['win']['seed_cluster_bootstrap']['ci95'], _pp)} |",
        f"| Income/hand | {_num(secondary['income']['treatment_mean'], 0)} | "
        f"{_num(secondary['income']['control_mean'], 0)} | "
        f"{_num(secondary['income']['difference'], 0)} | "
        f"{_ci(secondary['income']['seed_cluster_bootstrap']['ci95'])} |",
        f"| Loss/hand | {_num(secondary['loss']['treatment_mean'], 0)} | "
        f"{_num(secondary['loss']['control_mean'], 0)} | "
        f"{_num(secondary['loss']['difference'], 0)} | "
        f"{_ci(secondary['loss']['seed_cluster_bootstrap']['ci95'])} |",
        "",
        "### Game level and dilution check",
        "",
        "| Arm | Games | Mean rank | 4th rate | Mean score |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARM_IDS:
        stats = secondary["game_level"]["all_games"][arm]
        lines.append(
            f"| {arm} | {stats['games']} | {_num(stats['mean_rank'], 4)} | "
            f"{_pct(stats['fourth_rate'])} | {_num(stats['mean_score'], 1)} |"
        )
    paired = secondary["game_level"]["paired"]
    lines += [
        "",
        f"- paired game-level rank difference (T - C): "
        f"`{_num(paired['rank_difference']['mean'], 4)}` "
        f"{_ci(paired['rank_difference']['ci95'])}",
        f"- paired game-level score difference (T - C): "
        f"`{_num(paired['score_difference']['mean'], 1)}` "
        f"{_ci(paired['score_difference']['ci95'])}",
        f"- 4th-rate difference (T - C): `{_pp(paired['fourth_rate_difference'])}`",
        f"- affected games: `{secondary['game_level']['affected_games']['games']}` "
        f"(dilution check; most games are identical)",
        "",
        "## Veto events",
        "",
        f"- veto events: `{result['veto_events']['veto_events']}`",
        f"- games with at least one veto: `{result['veto_events']['games_with_veto']}`",
        f"- first-veto pairs: `{result['first_veto_pairs']}`",
        f"- later veto events (descriptive): "
        f"`{result['veto_events']['later_veto_events']}`",
        f"- unpaired first-vetoes: `{len(result['unpaired'])}`",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in result["limitations"]]
    lines.append("")
    return "\n".join(lines)
