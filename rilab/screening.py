"""v1.4.0 Step 1 local parameter screening.

This module owns the handoff-spec validation, native arena/protocol
generation, per-decision calibration audit, and the fixed-budget statistics.
It deliberately does not launch services or games; ``scripts/run_v140_screening.py``
does that and calls these pure functions.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

import yaml

from rilab.artifacts import audit_batch, encode, write_json
from rilab.config import (
    PROFILES,
    calibration_contract,
    effective_calibration,
    load_arena_config,
    validate_expected_calibration,
    valid_action_biases,
)
from rilab.evaluation import percentile, seat_counters
from rilab.scheduling import expand_schedule

BASELINE_ID = "baseline_v1_3_0"
BASELINE_COMMIT = "1e5092f344fb69bcdf4697f683f2ecf15b9865d8"
CANDIDATE_COMMIT = "0710aa0f31d1d3c8d96ddd7cc31bf832cd68ada6"
WEIGHT_PATHS = {
    "turn": "runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt",
    "response": "runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_response.pt",
}
WEIGHT_SHA256 = {
    "turn": "30f2811cd86beebad57d4da1428272b25b598fb64a6231b5a1760e11ec08bf90",
    "response": "e1d8c58a2aa61cef23852f15fe5bb9c65014bc3916c52756fb8c4cd1cd956c51",
}
CANDIDATES = [
    {
        "id": "A_more_calls",
        "service_config": "service_A_more_calls.yaml",
        "action_biases": {"ron": 0.0, "kan": 0.4, "pon": 1.1, "chi": 1.3},
        "no_yaku_call_penalty": 0.3,
    },
    {
        "id": "B_balanced",
        "service_config": "service_B_balanced.yaml",
        "action_biases": {"ron": 0.0, "kan": 0.4, "pon": 1.4, "chi": 1.6},
        "no_yaku_call_penalty": 0.3,
    },
    {
        "id": "C_fewer_calls",
        "service_config": "service_C_fewer_calls.yaml",
        "action_biases": {"ron": 0.0, "kan": 0.4, "pon": 1.7, "chi": 1.9},
        "no_yaku_call_penalty": 0.3,
    },
]
BASELINE_SERVICE_CONFIG = "service_baseline_v1_3_0.yaml"
BASELINE_PASS_BIAS = 1.5
BASELINE_PENALTY = 0.0
ENGINEERING_SEEDS = [20001, 20002]
SCREENING_SEEDS = list(range(21001, 21129))
RESERVED_SEEDS = list(range(31001, 31257))
KNOWN_EXCLUDED = list(range(1, 9)) + list(range(10001, 10129))
PLAN_MATCHUPS = [
    {
        "id": "candidate_vs_baseline",
        "seats": ["CANDIDATE", BASELINE_ID, BASELINE_ID, BASELINE_ID],
        "paired_group": "upgrade",
    },
    {
        "id": "baseline_control",
        "seats": [BASELINE_ID, BASELINE_ID, BASELINE_ID, BASELINE_ID],
        "paired_group": "upgrade",
    },
]
MATCHUPS = [
    {**spec, "focal_seat": 0, "rotation": "cyclic4"}
    for spec in PLAN_MATCHUPS
]

METRICS = {
    "mean_rank": ("rank_sum", "games", "lower_is_better"),
    "mean_final_points": ("point_sum", "games", "higher_is_better"),
    "first_rate": ("firsts", "games", "higher_is_better"),
    "top2_rate": ("top2", "games", "higher_is_better"),
    "last_place_rate": ("last_places", "games", "lower_is_better"),
    "bust_rate": ("busts", "games", "lower_is_better"),
    "game_bust_rate": ("game_busts", "games", "lower_is_better"),
    "win_rate": ("wins", "hands", "higher_is_better"),
    "deal_in_rate": ("deal_ins", "hands", "lower_is_better"),
    "riichi_accepted_rate": ("riichi_accepted", "hands", "descriptive"),
    "riichi_declaration_rate": ("riichi_declared", "hands", "descriptive"),
    "call_rate": ("called_hands", "hands", "descriptive"),
    "avg_win_income": ("win_income_sum", "wins", "higher_is_better"),
    "avg_deal_in_loss": ("deal_in_loss_sum", "deal_ins", "lower_is_better"),
    "deal_in_loss_per_hand": ("deal_in_loss_sum", "hands", "lower_is_better"),
    "response_switch_rate": ("response_switches", "response_decisions", "descriptive"),
    "response_multi_switch_rate": (
        "response_multi_switches",
        "response_multi_candidate",
        "descriptive",
    ),
    "forced_pass_rate": ("response_forced_pass", "response_decisions", "descriptive"),
}

COUNT_FIELDS = (
    "games",
    "rank_sum",
    "last_places",
    "firsts",
    "top2",
    "point_sum",
    "busts",
    "game_busts",
    "hands",
    "wins",
    "deal_ins",
    "riichi_accepted",
    "riichi_declared",
    "called_hands",
    "win_income_sum",
    "deal_in_loss_sum",
    "response_decisions",
    "response_forced_pass",
    "response_multi_candidate",
    "response_switches",
    "response_multi_switches",
    "response_opportunities_ron",
    "response_opportunities_chankan",
    "response_opportunities_daiminkan",
    "response_opportunities_pon",
    "response_opportunities_chi",
    "response_opportunities_pass",
    "response_selected_ron",
    "response_selected_chankan",
    "response_selected_daiminkan",
    "response_selected_pon",
    "response_selected_chi",
    "response_selected_pass",
    "response_raw_ron",
    "response_raw_chankan",
    "response_raw_daiminkan",
    "response_raw_pon",
    "response_raw_chi",
    "response_raw_pass",
)

RESPONSE_KINDS = ("ron", "chankan", "daiminkan", "pon", "chi", "pass")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_handoff(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("handoff plan must be a mapping")
    return data


def _same_list(value, expected, name):
    if value != expected:
        raise ValueError(f"handoff {name} does not match the frozen specification")


def validate_handoff(plan: dict, *, handoff_dir: str | Path | None = None) -> None:
    """Reject any change to the frozen source, parameters, seeds, or budget."""
    if plan.get("handoff_schema_version") != 1:
        raise ValueError("unsupported handoff schema")
    if plan.get("kind") != "local_screening_handoff_spec_not_a_rilab_cli_config":
        raise ValueError("handoff kind is not the local screening spec")
    if plan.get("status") != "planned_not_executed":
        raise ValueError("handoff status must remain planned_not_executed")
    if plan.get("experiment_id") != "v1_4_0_step1_three_profiles":
        raise ValueError("unexpected experiment id")
    if plan.get("training_repository") != "train_mj" or plan.get(
        "execution_repository"
    ) != "riichi_lab":
        raise ValueError("unexpected repository identity")

    runtime = plan.get("runtime", {})
    if runtime != {
        "device": "cpu",
        "precision": "float32",
        "candidate_execution": "sequential_A_B_C",
        "runs_per_group": 1,
        "timeout_seconds_per_group": 86400,
        "no_training": True,
        "no_online_ranked_games": True,
        "auto_run_confirmation": False,
    }:
        raise ValueError("runtime settings were changed")

    source = plan.get("source", {})
    if source.get("baseline", {}).get("ref") != "v1.3.0" or source.get(
        "baseline", {}
    ).get("commit") != BASELINE_COMMIT:
        raise ValueError("baseline source identity changed")
    if source.get("baseline", {}).get("calibration_contract") != "legacy_pass_bias":
        raise ValueError("baseline calibration contract changed")
    candidate_source = source.get("candidate", {})
    if (
        candidate_source.get("ref") != CANDIDATE_COMMIT
        or candidate_source.get("commit") != CANDIDATE_COMMIT
        or candidate_source.get("branch_hint") != "codex/v1.4.0"
        or candidate_source.get("release_tag_exists") is not False
        or candidate_source.get("calibration_contract") != "action_biases_v1"
    ):
        raise ValueError("candidate source identity changed")
    for item in (source["baseline"], candidate_source):
        if item.get("feature_schema") != "full_state_v2":
            raise ValueError("feature schema changed")

    weights = plan.get("weights", {})
    if weights.get("same_files_for_all_profiles") is not True:
        raise ValueError("all profiles must share the same weight files")
    if "rehash actual files" not in str(weights.get("hash_provenance", "")):
        raise ValueError("weight hash provenance was changed")
    for task in ("turn", "response"):
        item = weights.get(task, {})
        if item.get("path_relative_to_train_mj") != WEIGHT_PATHS[task]:
            raise ValueError(f"{task} weight path changed")
        if item.get("expected_file_sha256") != WEIGHT_SHA256[task]:
            raise ValueError(f"{task} weight hash changed")

    baseline = plan.get("baseline", {})
    if (
        baseline.get("id") != BASELINE_ID
        or baseline.get("service_config") != BASELINE_SERVICE_CONFIG
        or baseline.get("pass_bias") != BASELINE_PASS_BIAS
        or baseline.get("no_yaku_call_penalty") != BASELINE_PENALTY
    ):
        raise ValueError("baseline parameters changed")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(CANDIDATES):
        raise ValueError("candidate list changed")
    for actual, expected in zip(candidates, CANDIDATES, strict=True):
        if actual != expected:
            raise ValueError(f"candidate parameters changed: {expected['id']}")

    seeds = plan.get("seeds", {})
    _same_list(seeds.get("engineering"), ENGINEERING_SEEDS, "engineering seeds")
    _same_list(seeds.get("screening"), SCREENING_SEEDS, "screening seeds")
    _same_list(
        seeds.get("confirmation_reserved_do_not_run"),
        RESERVED_SEEDS,
        "reserved confirmation seeds",
    )
    _same_list(
        seeds.get("known_previously_used_excluded"),
        KNOWN_EXCLUDED,
        "known excluded seeds",
    )
    if seeds.get("must_check_local_run_history_for_other_overlap") is not True:
        raise ValueError("local seed-history check was removed")
    if (
        seeds.get("on_overlap")
        != "stop_before_games_and_report_do_not_silently_replace"
    ):
        raise ValueError("seed overlap policy changed")
    if set(ENGINEERING_SEEDS) & set(SCREENING_SEEDS):
        raise ValueError("engineering and screening seeds overlap")
    if set(SCREENING_SEEDS) & set(RESERVED_SEEDS):
        raise ValueError("screening seeds include reserved confirmation seeds")
    if (set(ENGINEERING_SEEDS) | set(SCREENING_SEEDS)) & set(KNOWN_EXCLUDED):
        raise ValueError("new seeds include previously used seeds")

    design = plan.get("design", {})
    if (
        design.get("game_mode") != "4p-red-half"
        or design.get("rule") != "default_tenhou"
        or design.get("cluster") != "seed_with_all_rotations"
        or design.get("focal_seat_before_rotation") != 0
        or design.get("rotation") != "cyclic4"
        or design.get("no_two_vs_two") is not True
        or design.get("engineering_half_games_per_candidate") != 16
        or design.get("engineering_total_half_games") != 48
        or design.get("screening_seed_clusters") != 128
        or design.get("screening_half_games_per_candidate") != 1024
        or design.get("screening_focal_candidate_half_games_per_candidate") != 512
        or design.get("screening_focal_control_half_games_per_candidate") != 512
        or design.get("screening_total_physical_half_games") != 3072
        or design.get("controls_repeated_per_candidate_are_not_independent_evidence")
        is not True
    ):
        raise ValueError("screening design or budget changed")
    _same_list(
        design.get("matchups_per_candidate"),
        PLAN_MATCHUPS,
        "matchup specification",
    )

    analysis = plan.get("analysis", {})
    if analysis != {
        "primary": "mean_rank_difference_candidate_minus_control",
        "negative_is_better": True,
        "tie_rule": "inherit_engine_official_rank_rule_and_record",
        "bootstrap_iterations": 10000,
        "bootstrap_rng_seed": 20261109,
        "confidence_level": 0.95,
        "resampling": "paired_seed_clusters_all_four_rotations_together",
        "cross_candidate_resampling": "same_resampled_seed_indices_across_all_candidates",
        "intervals_are_exploratory_not_confirmatory": True,
        "candidate_eligible_if": {
            "mean_rank_difference_strictly_less_than": 0.0,
            "last_place_rate_difference_at_most": 0.02,
            "engineering_and_artifact_checks_pass": True,
        },
        "eligible_sort_keys": [
            "mean_rank_difference_ascending",
            "last_place_rate_difference_ascending",
            "mean_final_points_descending",
            "candidate_id_ascending",
        ],
        "if_no_eligible_candidate": "recommend_none_do_not_change_parameters_or_extend_budget",
        "deal_in_and_call_rates": "report_with_denominators_and_intervals_not_optimization_targets",
    }:
        raise ValueError("analysis rules changed")
    if plan.get("stopping") != {
        "rule": "fixed_budget_no_optional_stopping",
        "on_runtime_failure": "stop_launching_new_games_preserve_artifacts_mark_incomplete",
        "resume": "only_if_exact_schedule_identity_hash_and_no_duplicates_are_verified_otherwise_require_review",
        "no_selective_deletion_or_seed_replacement": True,
    }:
        raise ValueError("stopping rules changed")
    _same_list(
        plan.get("review_required_before"),
        [
            "confirmation_games",
            "online_ranked_validation",
            "additional_parameter_search",
            "budget_extension",
        ],
        "review gates",
    )

    if handoff_dir is not None:
        root = Path(handoff_dir)
        for name in [BASELINE_SERVICE_CONFIG] + [
            c["service_config"] for c in CANDIDATES
        ]:
            if not (root / name).is_file():
                raise ValueError(f"missing handoff service config: {name}")


def load_service_configs(plan: dict, handoff_dir: str | Path) -> dict:
    """Load and freeze the four external service YAMLs before any game."""
    root = Path(handoff_dir)
    result = {"baseline": None, "candidates": {}}
    specs = [(BASELINE_ID, BASELINE_SERVICE_CONFIG, None)] + [
        (c["id"], c["service_config"], c) for c in CANDIDATES
    ]
    for backend_id, filename, candidate in specs:
        path = root / filename
        raw = path.read_bytes()
        values = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"service config is not a mapping: {filename}")
        if values.get("device") != "cpu":
            raise ValueError(f"service config must use CPU: {filename}")
        models = values.get("models", {})
        for task in ("turn", "response"):
            if models.get(task, {}).get("checkpoint") != WEIGHT_PATHS[task]:
                raise ValueError(f"{filename}: {task} checkpoint differs from plan")
        response = models.get("response", {})
        if candidate is None:
            if response.get("pass_bias", {}).get("default") != BASELINE_PASS_BIAS:
                raise ValueError(f"{filename}: baseline pass_bias must be 1.5")
            if response.get("no_yaku_call_penalty") != BASELINE_PENALTY:
                raise ValueError(f"{filename}: baseline penalty must be 0.0")
            if "action_biases" in response:
                raise ValueError(f"{filename}: baseline must not use action_biases")
        else:
            if response.get("action_biases") != candidate["action_biases"]:
                raise ValueError(f"{filename}: action_biases differ from plan")
            if response.get("no_yaku_call_penalty") != candidate[
                "no_yaku_call_penalty"
            ]:
                raise ValueError(f"{filename}: penalty differs from plan")
            if "pass_bias" in response:
                raise ValueError(f"{filename}: candidate must not use pass_bias")
        record = {
            "path": str(path),
            "filename": filename,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "values": values,
        }
        if candidate is None:
            result["baseline"] = record
        else:
            result["candidates"][candidate["id"]] = record
    return result


def _backend_config(
    plan: dict, candidate_id: str, *, api_base: str
) -> dict:
    candidate = next(c for c in CANDIDATES if c["id"] == candidate_id)
    return {
        "baseline": {
            "mode": "http",
            "api_base": api_base,
            "adapter_profile": "full_state_v2",
            "calibration_contract": "legacy_pass_bias",
            "revision": {
                "tag": "v1.3.0",
                "commit": BASELINE_COMMIT,
                "branch_hint": "codex/full-state-features",
            },
            "timeout_seconds": 30.0,
            "response": {
                "pass_bias": BASELINE_PASS_BIAS,
                "no_yaku_call_penalty": BASELINE_PENALTY,
            },
            "model_identity": {
                "turn_sha256": WEIGHT_SHA256["turn"],
                "response_sha256": WEIGHT_SHA256["response"],
            },
        },
        "candidate": {
            "mode": "http",
            "api_base": api_base,
            "adapter_profile": "full_state_v2_action_biases",
            "calibration_contract": "action_biases_v1",
            "revision": {
                "commit": CANDIDATE_COMMIT,
                "branch_hint": "codex/v1.4.0",
            },
            "timeout_seconds": 30.0,
            "response": {
                "action_biases": deepcopy(candidate["action_biases"]),
                "no_yaku_call_penalty": candidate["no_yaku_call_penalty"],
            },
            "model_identity": {
                "turn_sha256": WEIGHT_SHA256["turn"],
                "response_sha256": WEIGHT_SHA256["response"],
            },
        },
    }


def build_native_config(
    plan: dict,
    *,
    stage: str,
    candidate_id: str,
    baseline_api_base: str = "http://127.0.0.1:0",
    candidate_api_base: str = "http://127.0.0.1:1",
    handoff_sha256: str | None = None,
) -> dict:
    if stage not in ("smoke", "screen"):
        raise ValueError("stage must be smoke or screen")
    candidate = next((c for c in CANDIDATES if c["id"] == candidate_id), None)
    if candidate is None:
        raise ValueError(f"unknown candidate: {candidate_id}")
    seeds = ENGINEERING_SEEDS if stage == "smoke" else SCREENING_SEEDS
    backends = _backend_config(
        plan, candidate_id, api_base=baseline_api_base
    )
    baseline = backends["baseline"]
    candidate_backend = backends["candidate"]
    candidate_backend["api_base"] = candidate_api_base
    matchups = []
    for spec in MATCHUPS:
        seats = [
            candidate_id if seat == "CANDIDATE" else seat for seat in spec["seats"]
        ]
        matchups.append({**deepcopy(spec), "seats": seats})
    config = {
        "config_version": 1,
        "engine": {
            "game_mode": "4p-red-half",
            "rule": "default_tenhou",
            "seeds": list(seeds),
        },
        "backends": {BASELINE_ID: baseline, candidate_id: candidate_backend},
        "matchups": matchups,
        "artifacts": {
            "output_dir": "outputs/screening",
            "full_snapshots_first_run": True,
            "snapshot_stride": 100,
            "max_bytes": 2_000_000_000,
        },
        "evaluation": {
            "primary_paired_group": "upgrade",
            "cluster": "seed_with_all_rotations",
            "confidence_level": 0.95,
        },
        "screening": {
            "experiment_id": plan["experiment_id"],
            "stage": stage,
            "candidate_id": candidate_id,
            "handoff_plan_sha256": handoff_sha256,
        },
    }
    return load_arena_config_dict(config)


def load_arena_config_dict(config: dict) -> dict:
    """Validate an in-memory native config through the normal loader."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        path = Path(handle.name)
    try:
        load_arena_config(path)
    finally:
        path.unlink(missing_ok=True)
    return config


def build_native_protocol(
    plan: dict, *, stage: str, candidate_id: str
) -> dict:
    if stage not in ("smoke", "screen"):
        raise ValueError("stage must be smoke or screen")
    candidate = next((c for c in CANDIDATES if c["id"] == candidate_id), None)
    if candidate is None:
        raise ValueError(f"unknown candidate: {candidate_id}")
    seeds = ENGINEERING_SEEDS if stage == "smoke" else SCREENING_SEEDS
    clusters = len(seeds)
    budget = clusters * 2 * 4
    return {
        "status": "planned",
        "purpose": "local_parameter_screening",
        "experiment_id": plan["experiment_id"],
        "stage": stage,
        "comparison": {
            "candidate": candidate_id,
            "baseline": BASELINE_ID,
            "estimand": "complete_deployed_policies_including_calibration",
        },
        "design": {
            "primary": "candidate_vs_baseline_and_baseline_control_cyclic4",
            "supplementary": "none_no_two_vs_two",
            "cluster": "seed_with_all_rotations",
            "excluded_engineering_or_calibration_seeds": sorted(
                set(KNOWN_EXCLUDED) | set(ENGINEERING_SEEDS) | set(RESERVED_SEEDS)
            ),
            "final_seeds": list(seeds),
            "target_seed_clusters": clusters,
            "half_game_budget": budget,
            "no_two_vs_two": True,
            "controls_repeated_per_candidate_are_not_independent_evidence": True,
        },
        "decision": {
            "primary_metric": "mean_rank_difference_candidate_minus_control",
            "mean_rank_difference_strictly_less_than": 0.0,
            "last_place_rate_difference_at_most": 0.02,
            "eligible_sort_keys": [
                "mean_rank_difference_ascending",
                "last_place_rate_difference_ascending",
                "mean_final_points_descending",
                "candidate_id_ascending",
            ],
            "if_no_eligible_candidate": "recommend_none_do_not_change_parameters_or_extend_budget",
            "bootstrap_iterations": 10000,
            "bootstrap_rng_seed": 20261109,
            "confidence_level": 0.95,
            "intervals_are_exploratory_not_confirmatory": True,
        },
        "stopping": {
            "rule": "fixed_budget_no_optional_stopping",
            "on_runtime_failure": "stop_launching_new_games_preserve_artifacts_mark_incomplete",
            "resume": "not_implemented_no_resume",
            "no_selective_deletion_or_seed_replacement": True,
        },
        "expected_calibration": {
            BASELINE_ID: {
                "pass_bias": BASELINE_PASS_BIAS,
                "no_yaku_call_penalty": BASELINE_PENALTY,
            },
            candidate_id: {
                "calibration_contract": "action_biases_v1",
                "action_biases": deepcopy(candidate["action_biases"]),
                "no_yaku_call_penalty": candidate["no_yaku_call_penalty"],
            },
        },
        "confirmation_reserved_do_not_run": list(RESERVED_SEEDS),
        "review_required_before": list(plan["review_required_before"]),
    }


def validate_screening_plan(plan: dict, config: dict) -> list[dict]:
    if plan.get("purpose") != "local_parameter_screening":
        raise ValueError("not a local parameter screening protocol")
    design = plan.get("design", {})
    comparison = plan.get("comparison", {})
    if (
        comparison.get("baseline") != BASELINE_ID
        or comparison.get("candidate") not in {c["id"] for c in CANDIDATES}
        or comparison.get("estimand")
        != "complete_deployed_policies_including_calibration"
    ):
        raise ValueError("unsupported screening comparison")
    if (
        design.get("primary")
        != "candidate_vs_baseline_and_baseline_control_cyclic4"
        or design.get("supplementary") != "none_no_two_vs_two"
        or design.get("cluster") != "seed_with_all_rotations"
        or design.get("no_two_vs_two") is not True
        or design.get("final_seeds") != config["engine"]["seeds"]
        or design.get("target_seed_clusters") != len(config["engine"]["seeds"])
        or design.get("half_game_budget")
        != len(expand_schedule(config))
    ):
        raise ValueError("screening design/budget differs from native config")
    decision = plan.get("decision", {})
    if (
        decision.get("primary_metric")
        != "mean_rank_difference_candidate_minus_control"
        or decision.get("mean_rank_difference_strictly_less_than") != 0.0
        or decision.get("last_place_rate_difference_at_most") != 0.02
        or decision.get("bootstrap_iterations") != 10000
        or decision.get("bootstrap_rng_seed") != 20261109
        or decision.get("confidence_level") != 0.95
        or decision.get("intervals_are_exploratory_not_confirmatory") is not True
    ):
        raise ValueError("screening decision rules changed")
    expected = plan.get("expected_calibration")
    if not isinstance(expected, dict) or set(expected) != set(config["backends"]):
        raise ValueError("screening calibration does not cover backends")
    for name, backend in config["backends"].items():
        validate_expected_calibration(backend, expected[name])
    group = config["evaluation"]["primary_paired_group"]
    members = [m for m in config["matchups"] if m.get("paired_group") == group]
    if len(members) != 2:
        raise ValueError("screening requires exactly two paired matchups")
    candidate_id = comparison["candidate"]
    controls = [m for m in members if len(set(m["seats"])) == 1]
    candidates = [
        m
        for m in members
        if m["seats"][m["focal_seat"]] == candidate_id
        and all(
            seat == candidate_id if index == m["focal_seat"] else seat == BASELINE_ID
            for index, seat in enumerate(m["seats"])
        )
    ]
    if len(controls) != 1 or len(candidates) != 1:
        raise ValueError("screening matchups do not match the frozen design")
    if controls[0]["seats"] != [BASELINE_ID] * 4:
        raise ValueError("screening control must be four baselines")
    schedule = expand_schedule(config)
    if len(schedule) != design["half_game_budget"]:
        raise ValueError("expanded schedule budget differs from protocol")
    return schedule


def empty_counts() -> dict:
    return {field: 0 for field in COUNT_FIELDS}


def add_counts(target: dict, source: dict) -> None:
    for field in COUNT_FIELDS:
        target[field] = target.get(field, 0) + source.get(field, 0)


def _response_counts(records) -> dict:
    counts = empty_counts()
    for record in records:
        if record.get("task") != "response":
            continue
        candidates = record["candidates"]
        prediction = record.get("prediction")
        if prediction is None:
            raise ValueError("response decision has no prediction")
        selected = record["selected_action"]
        raw = prediction["raw_selected_action"]
        kinds = [candidate.get("kind") for candidate in candidates]
        selected_kind = selected.get("kind")
        raw_kind = raw.get("kind")
        counts["response_decisions"] += 1
        if kinds == ["pass"]:
            counts["response_forced_pass"] += 1
        if len(candidates) > 1:
            counts["response_multi_candidate"] += 1
        for kind in RESPONSE_KINDS:
            if kind in kinds:
                counts[f"response_opportunities_{kind}"] += 1
            if selected_kind == kind:
                counts[f"response_selected_{kind}"] += 1
            if raw_kind == kind:
                counts[f"response_raw_{kind}"] += 1
        if selected != raw:
            counts["response_switches"] += 1
            if len(candidates) > 1:
                counts["response_multi_switches"] += 1
    return counts


def _game_counts(events, focal_seat, ranks, scores, decision_records) -> dict:
    counters = seat_counters(events)[focal_seat]
    counts = empty_counts()
    counts.update(counters)
    counts["games"] = 1
    counts["rank_sum"] = ranks[focal_seat]
    counts["firsts"] = int(ranks[focal_seat] == 1)
    counts["top2"] = int(ranks[focal_seat] <= 2)
    counts["last_places"] = int(ranks[focal_seat] == 4)
    counts["point_sum"] = scores[focal_seat]
    counts["busts"] = int(scores[focal_seat] < 0)
    counts["game_busts"] = int(any(score < 0 for score in scores))
    for event in events:
        if event["type"] != "hora":
            continue
        actor = event["actor"]
        target = event.get("target")
        deltas = event["deltas"]
        if actor == focal_seat:
            counts["win_income_sum"] += max(0, deltas[actor])
        tsumo = event.get("tsumo", target == actor)
        if not tsumo and target == focal_seat:
            counts["deal_in_loss_sum"] += max(0, -deltas[target])
    add_counts(
        counts,
        _response_counts(
            record
            for record in decision_records
            if record.get("player_id") == focal_seat
        ),
    )
    return counts


def _decision_records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _expected_contract_views(config: dict, models: dict) -> dict:
    result = {}
    for name, backend in config["backends"].items():
        response = models[name]["response"]
        result[name] = {
            "contract": calibration_contract(backend),
            "calibration": effective_calibration(backend, response),
        }
    return result


def verify_group_decisions(run: str | Path, config: dict, models: dict) -> dict:
    """Audit every recorded request/result against the pinned calibration."""
    run = Path(run)
    expected = _expected_contract_views(config, models)
    result = {
        "valid": True,
        "decisions": 0,
        "turn_decisions": 0,
        "response_decisions": 0,
        "forced_riichi_discards": 0,
        "calibration_checked": 0,
        "calibration_mismatches": 0,
        "selected_action_mismatches": 0,
        "raw_action_mismatches": 0,
        "model_metadata_mismatches": 0,
        "task_mismatches": 0,
        "fallbacks": 0,
        "errors": [],
    }
    for game_id in sorted(p.stem for p in (run / "results").glob("*.json")):
        for record in _decision_records(run / "decisions" / f"{game_id}.jsonl.gz"):
            result["decisions"] += 1
            if record.get("fallback") is not False:
                result["fallbacks"] += 1
                result["errors"].append(f"{game_id}: fallback")
            backend_id = record.get("backend_id")
            if backend_id not in config["backends"]:
                result["errors"].append(f"{game_id}: unknown backend")
                result["valid"] = False
                continue
            task = record.get("task")
            prediction = record.get("prediction")
            if task == "turn":
                result["turn_decisions"] += 1
            elif task == "response":
                result["response_decisions"] += 1
            else:
                result["task_mismatches"] += 1
                result["errors"].append(f"{game_id}: unknown task")
                result["valid"] = False
                continue
            if prediction is None:
                if task == "turn" and record.get("forced_riichi_discard"):
                    result["forced_riichi_discards"] += 1
                    continue
                result["errors"].append(f"{game_id}: missing prediction")
                result["valid"] = False
                continue
            if prediction.get("task") != task:
                result["task_mismatches"] += 1
                result["valid"] = False
            if prediction.get("model") != models[backend_id][task]:
                result["model_metadata_mismatches"] += 1
                result["valid"] = False
            if prediction.get("selected_action") != record.get("selected_action"):
                result["selected_action_mismatches"] += 1
                result["valid"] = False
            if task == "response":
                view = expected[backend_id]
                calibration = view["calibration"]
                result["calibration_checked"] += 1
                if view["contract"] == "legacy_pass_bias":
                    if (
                        prediction.get("pass_bias") != calibration["pass_bias"]
                        or "action_biases" in prediction
                    ):
                        result["calibration_mismatches"] += 1
                        result["valid"] = False
                else:
                    if (
                        prediction.get("action_biases")
                        != calibration["action_biases"]
                        or "pass_bias" in prediction
                    ):
                        result["calibration_mismatches"] += 1
                        result["valid"] = False
                if prediction.get("no_yaku_call_penalty") != calibration[
                    "no_yaku_call_penalty"
                ]:
                    result["calibration_mismatches"] += 1
                    result["valid"] = False
                if prediction.get("raw_selected_action") not in record[
                    "candidates"
                ]:
                    result["raw_action_mismatches"] += 1
                    result["valid"] = False
    if result["fallbacks"]:
        result["valid"] = False
    return result


def _metric_value(counts: dict, metric: str):
    numerator, denominator, _ = METRICS[metric]
    den = counts.get(denominator, 0)
    if den <= 0:
        return None
    return counts.get(numerator, 0) / den


def point_metrics(counts: dict) -> dict:
    result = {}
    for metric, (numerator, denominator, direction) in METRICS.items():
        den = counts.get(denominator, 0)
        num = counts.get(numerator, 0)
        result[metric] = {
            "numerator": num,
            "denominator": den,
            "value": num / den if den > 0 else None,
            "direction": direction,
        }
    return result


def _ci(values):
    if not values:
        return None
    ordered = sorted(values)
    return [percentile(ordered, 0.025), percentile(ordered, 0.975)]


def bootstrap_screening(
    clusters_by_candidate: dict,
    *,
    iterations: int = 10000,
    rng_seed: int = 20261109,
) -> dict:
    """Paired seed-cluster bootstrap with identical indices for all candidates."""
    if type(iterations) is not int or iterations < 100:
        raise ValueError("invalid bootstrap iterations")
    ids = list(clusters_by_candidate)
    if not ids:
        raise ValueError("no candidate clusters")
    n = len(clusters_by_candidate[ids[0]])
    if n < 2 or any(len(clusters_by_candidate[cid]) != n for cid in ids):
        raise ValueError("candidate clusters must share the same seed count")
    rng = random.Random(rng_seed)
    draws = {
        cid: {
            "candidate": {metric: [] for metric in METRICS},
            "control": {metric: [] for metric in METRICS},
            "difference": {metric: [] for metric in METRICS},
            "difference_draws": {metric: [] for metric in METRICS},
        }
        for cid in ids
    }
    for draw_index in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        for cid in ids:
            candidate = empty_counts()
            control = empty_counts()
            for index in indices:
                add_counts(candidate, clusters_by_candidate[cid][index]["candidate"])
                add_counts(control, clusters_by_candidate[cid][index]["control"])
            for metric in METRICS:
                a = _metric_value(candidate, metric)
                b = _metric_value(control, metric)
                if a is not None:
                    draws[cid]["candidate"][metric].append(a)
                if b is not None:
                    draws[cid]["control"][metric].append(b)
                if a is not None and b is not None:
                    value = a - b
                    draws[cid]["difference"][metric].append(value)
                    draws[cid]["difference_draws"][metric].append(
                        (draw_index, value)
                    )
    result = {"iterations": iterations, "rng_seed": rng_seed, "clusters": n}
    for cid in ids:
        candidate = empty_counts()
        control = empty_counts()
        for cluster in clusters_by_candidate[cid]:
            add_counts(candidate, cluster["candidate"])
            add_counts(control, cluster["control"])
        a = point_metrics(candidate)
        b = point_metrics(control)
        differences = {}
        for metric in METRICS:
            av = a[metric]["value"]
            bv = b[metric]["value"]
            differences[metric] = {
                "numerator_candidate": a[metric]["numerator"],
                "denominator_candidate": a[metric]["denominator"],
                "numerator_control": b[metric]["numerator"],
                "denominator_control": b[metric]["denominator"],
                "candidate": av,
                "control": bv,
                "value": av - bv if av is not None and bv is not None else None,
                "ci95": _ci(draws[cid]["difference"][metric]),
                "draws_used": len(draws[cid]["difference"][metric]),
                "direction": METRICS[metric][2],
            }
        result[cid] = {
            "candidate": a,
            "control": b,
            "differences": differences,
            "candidate_ci95": {
                metric: _ci(draws[cid]["candidate"][metric]) for metric in METRICS
            },
            "control_ci95": {
                metric: _ci(draws[cid]["control"][metric]) for metric in METRICS
            },
        }
    # Cross-candidate contrasts reuse the same draw index for all candidates.
    cross = {}
    for left in ids:
        for right in ids:
            if left == right:
                continue
            key = f"{left}_minus_{right}"
            cross[key] = {}
            for metric in METRICS:
                right_by_draw = dict(
                    draws[right]["difference_draws"][metric]
                )
                diffs = [
                    a - right_by_draw[index]
                    for index, a in draws[left]["difference_draws"][metric]
                    if index in right_by_draw
                ]
                left_value = result[left]["differences"][metric]["value"]
                right_value = result[right]["differences"][metric]["value"]
                point = (
                    left_value - right_value
                    if left_value is not None and right_value is not None
                    else None
                )
                cross[key][metric] = {
                    "value": point,
                    "ci95": _ci(diffs),
                    "draws_used": len(diffs),
                    "scope": "correlated exploratory contrast of candidate-minus-control",
                }
    result["cross_candidate"] = cross
    return result


def analyze_group(
    run: str | Path,
    config: dict,
    models: dict,
    *,
    candidate_id: str,
    stage: str,
    iterations: int = 10000,
    rng_seed: int = 20261109,
) -> dict:
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if manifest["schedule"] != expand_schedule(config):
        raise ValueError("batch schedule differs from native config")
    if (
        summary.get("valid") is not True
        or summary.get("fallbacks") != 0
        or summary.get("completed_games") != len(manifest["schedule"])
    ):
        raise ValueError("incomplete or invalid batch summary")
    if manifest["models"] != models:
        raise ValueError("batch manifest models differ from live metadata")
    games = audit_batch(run)
    group = config["evaluation"]["primary_paired_group"]
    members = [m for m in config["matchups"] if m.get("paired_group") == group]
    control_matchup = next(m for m in members if len(set(m["seats"])) == 1)
    candidate_matchup = next(m for m in members if m is not control_matchup)
    by_seed = defaultdict(lambda: {"candidate": empty_counts(), "control": empty_counts()})
    per_game = []
    focused = {"candidate": empty_counts(), "control": empty_counts()}
    response_breakdown = {"candidate": empty_counts(), "control": empty_counts()}
    for game in games:
        if game["matchup_id"] == candidate_matchup["id"]:
            label = "candidate"
        elif game["matchup_id"] == control_matchup["id"]:
            label = "control"
        else:
            raise ValueError("unexpected matchup in screening batch")
        focal = game["focal_seat"]
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
        records = list(
            _decision_records(run / "decisions" / f"{game['game_id']}.jsonl.gz")
        )
        focal_records = [
            record for record in records if record.get("player_id") == focal
        ]
        counts = _game_counts(
            events, focal, game["ranks"], game["scores"], focal_records
        )
        add_counts(by_seed[game["seed"]][label], counts)
        add_counts(focused[label], counts)
        add_counts(response_breakdown[label], _response_counts(focal_records))
        per_game.append(
            {
                "candidate_id": candidate_id,
                "stage": stage,
                "seed": game["seed"],
                "rotation": game["rotation"],
                "label": label,
                "matchup_id": game["matchup_id"],
                "game_id": game["game_id"],
                "focal_seat": focal,
                "rank": game["ranks"][focal],
                "final_points": game["scores"][focal],
                "game_bust": int(any(score < 0 for score in game["scores"])),
                **{
                    key: counts[key]
                    for key in (
                        "hands",
                        "wins",
                        "deal_ins",
                        "riichi_accepted",
                        "riichi_declared",
                        "called_hands",
                        "win_income_sum",
                        "deal_in_loss_sum",
                        "busts",
                    )
                },
            }
        )
    seeds = config["engine"]["seeds"]
    clusters = []
    for seed in seeds:
        counts = by_seed[seed]
        if counts["candidate"]["games"] != 4 or counts["control"]["games"] != 4:
            raise ValueError(f"seed {seed} lacks both four-rotation arms")
        clusters.append(
            {
                "seed": seed,
                "candidate": counts["candidate"],
                "control": counts["control"],
            }
        )
    bootstrap = bootstrap_screening(
        {candidate_id: clusters}, iterations=iterations, rng_seed=rng_seed
    )
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "run": str(run.resolve()),
        "seed_clusters": len(clusters),
        "completed_half_games": len(games),
        "candidate_focal_games": sum(
            count["candidate"]["games"] for count in by_seed.values()
        ),
        "control_focal_games": sum(
            count["control"]["games"] for count in by_seed.values()
        ),
        "focused": {
            label: point_metrics(focused[label]) for label in ("candidate", "control")
        },
        "differences": bootstrap[candidate_id]["differences"],
        "candidate_ci95": bootstrap[candidate_id]["candidate_ci95"],
        "control_ci95": bootstrap[candidate_id]["control_ci95"],
        "bootstrap": {
            "iterations": iterations,
            "rng_seed": rng_seed,
            "clusters": len(clusters),
        },
        "response": {
            label: {
                "decisions": response_breakdown[label]["response_decisions"],
                "forced_pass": response_breakdown[label]["response_forced_pass"],
                "multi_candidate": response_breakdown[label][
                    "response_multi_candidate"
                ],
                "switches": response_breakdown[label]["response_switches"],
                "multi_switches": response_breakdown[label][
                    "response_multi_switches"
                ],
                "opportunities": {
                    kind: response_breakdown[label][
                        f"response_opportunities_{kind}"
                    ]
                    for kind in RESPONSE_KINDS
                },
                "selected": {
                    kind: response_breakdown[label][f"response_selected_{kind}"]
                    for kind in RESPONSE_KINDS
                },
                "raw_selected": {
                    kind: response_breakdown[label][f"response_raw_{kind}"]
                    for kind in RESPONSE_KINDS
                },
                "switch_rate": (
                    response_breakdown[label]["response_switches"]
                    / response_breakdown[label]["response_decisions"]
                    if response_breakdown[label]["response_decisions"]
                    else None
                ),
                "multi_candidate_switch_rate": (
                    response_breakdown[label]["response_multi_switches"]
                    / response_breakdown[label]["response_multi_candidate"]
                    if response_breakdown[label]["response_multi_candidate"]
                    else None
                ),
            }
            for label in ("candidate", "control")
        },
        "per_game": per_game,
        "clusters": clusters,
        "models": models,
        "manifest": manifest,
        "summary": summary,
    }


def recommendation(
    groups: dict, *, engineering_ok: bool, artifact_ok: bool
) -> dict:
    eligible = []
    for candidate_id, group in groups.items():
        mean_rank = group["differences"]["mean_rank"]["value"]
        last_place = group["differences"]["last_place_rate"]["value"]
        if mean_rank is None or last_place is None:
            continue
        if (
            engineering_ok
            and artifact_ok
            and mean_rank < 0
            and last_place <= 0.02
        ):
            final_points = group["focused"]["candidate"]["mean_final_points"][
                "value"
            ]
            eligible.append(
                {
                    "candidate_id": candidate_id,
                    "mean_rank_difference": mean_rank,
                    "last_place_rate_difference": last_place,
                    "mean_final_points_candidate": final_points,
                }
            )
    eligible.sort(
        key=lambda row: (
            row["mean_rank_difference"],
            row["last_place_rate_difference"],
            -row["mean_final_points_candidate"],
            row["candidate_id"],
        )
    )
    selected = eligible[0]["candidate_id"] if eligible else None
    return {
        "engineering_and_artifact_checks_pass": bool(engineering_ok and artifact_ok),
        "rule": {
            "mean_rank_difference_strictly_less_than": 0.0,
            "last_place_rate_difference_at_most": 0.02,
            "sort_keys": [
                "mean_rank_difference_ascending",
                "last_place_rate_difference_ascending",
                "mean_final_points_descending",
                "candidate_id_ascending",
            ],
            "if_no_eligible_candidate": "recommend_none_do_not_change_parameters_or_extend_budget",
        },
        "eligible": eligible,
        "selected": selected,
        "status": "pending_independent_confirmation"
        if selected
        else "no_recommendation",
        "note": "Point-estimate screening rule only; intervals are exploratory and do not authorize online ranked deployment.",
    }


def write_metrics_csv(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    rows = []
    for candidate_id, group in metrics["groups"].items():
        for metric, item in group["differences"].items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "scope": "candidate_vs_control",
                    "metric": metric,
                    "candidate_value": item["candidate"],
                    "control_value": item["control"],
                    "difference": item["value"],
                    "ci95_low": item["ci95"][0] if item["ci95"] else None,
                    "ci95_high": item["ci95"][1] if item["ci95"] else None,
                    "numerator_candidate": item["numerator_candidate"],
                    "denominator_candidate": item["denominator_candidate"],
                    "numerator_control": item["numerator_control"],
                    "denominator_control": item["denominator_control"],
                }
            )
    for pair, values in metrics["cross_candidate"].items():
        for metric, item in values.items():
            rows.append(
                {
                    "candidate_id": pair,
                    "scope": "cross_candidate_correlated",
                    "metric": metric,
                    "candidate_value": None,
                    "control_value": None,
                    "difference": item["value"],
                    "ci95_low": item["ci95"][0] if item["ci95"] else None,
                    "ci95_high": item["ci95"][1] if item["ci95"] else None,
                    "numerator_candidate": None,
                    "denominator_candidate": None,
                    "numerator_control": None,
                    "denominator_control": None,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def write_per_seed_csv(path: str | Path, groups: dict) -> None:
    rows = []
    for candidate_id, group in groups.items():
        rows.extend(group["per_game"])
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def render_report(metrics: dict) -> str:
    lines = [
        "# v1.4.0 Step 1 本地参数筛选报告",
        "",
        "本报告是固定预算的探索性参数筛选。结论不是线上发布许可；独立确认种子未运行。",
        "",
        "## 推荐规则结果",
        "",
    ]
    rec = metrics["recommendation"]
    lines.append(f"- 状态：`{rec['status']}`")
    lines.append(f"- 待独立确认候选：`{rec['selected']}`")
    lines.append(
        "- 规则：平均顺位差（候选−控制）严格 < 0，且末位率差 ≤ +0.02；"
        "按平均顺位差升序、末位率差升序、平均终局点数降序、候选 ID 升序排序。"
    )
    if not rec["eligible"]:
        lines.append("- 本轮没有合格候选，按预登记规则不推荐任何组。")
    lines.append("")
    lines.append("## 主指标")
    lines.append("")
    lines.append(
        "| 候选 | 候选平均顺位 | 控制平均顺位 | 顺位差 | 95% 区间 | 末位率差 | 95% 区间 | 候选平均终局点数 |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---|---:|")
    for candidate_id, group in metrics["groups"].items():
        rank = group["differences"]["mean_rank"]
        last = group["differences"]["last_place_rate"]
        lines.append(
            f"| {candidate_id} | {_fmt(rank['candidate'])} | {_fmt(rank['control'])} | "
            f"{_fmt(rank['value'])} | {_fmt_ci(rank['ci95'])} | "
            f"{_fmt_pct(last['value'])} | {_fmt_ci_pct(last['ci95'])} | "
            f"{_fmt(group['focused']['candidate']['mean_final_points']['value'])} |"
        )
    lines.extend(
        [
            "",
            "## 副作用与过程指标",
            "",
            "| 候选 | 和率差 | 放铳率差 | 立直率差 | 副露率差 | 平均和牌收入差 | 平均放铳失点差 | 每手放铳损失差 | 切换率差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id, group in metrics["groups"].items():
        d = group["differences"]
        lines.append(
            f"| {candidate_id} | {_fmt_pct(d['win_rate']['value'])} | "
            f"{_fmt_pct(d['deal_in_rate']['value'])} | "
            f"{_fmt_pct(d['riichi_accepted_rate']['value'])} | "
            f"{_fmt_pct(d['call_rate']['value'])} | "
            f"{_fmt(d['avg_win_income']['value'])} | "
            f"{_fmt(d['avg_deal_in_loss']['value'])} | "
            f"{_fmt(d['deal_in_loss_per_hand']['value'])} | "
            f"{_fmt_pct(d['response_switch_rate']['value'])} |"
        )
    lines.extend(
        [
            "",
            "## 各候选的 response 统计（焦点席）",
            "",
            "| 候选 | 组 | 决策数 | forced-pass | 多候选 | 切换 | 切换率 | 多候选切换率 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id, group in metrics["groups"].items():
        for label in ("candidate", "control"):
            row = group["response"][label]
            lines.append(
                f"| {candidate_id} | {label} | {row['decisions']} | "
                f"{row['forced_pass']} | {row['multi_candidate']} | {row['switches']} | "
                f"{_fmt_pct(row['switch_rate'])} | {_fmt_pct(row['multi_candidate_switch_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## 交付与解释",
            "",
            "- 三组各自与同一 v1.3.0 基线比较；控制局在三组重复运行，未当作三倍独立证据。",
            "- 跨候选区间使用同一组 seed 抽样索引，保留共同种子/共同控制的相关性；仍为探索性。",
            "- 所有差值均为候选减控制；平均顺位差为负数表示候选更好。",
            "- 本场棒、供托和多家荣和直接用引擎事件的 `deltas` 计入和牌收入/放铳损失。",
            "- 副露率只计至少一次吃/碰/大明杠的手数；暗杠不计副露。",
            "- DEF-001 未在本轮修复；离线回放只证明记录一致性，不重新验证网络 logits。",
            "",
            "完整分母、区间、逐种子/轮换记录见 `metrics.json`、`metrics.csv`、`per_seed.csv`。",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value):
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def _fmt_pct(value):
    if value is None:
        return "unavailable"
    return f"{100 * value:.4f}%"


def _fmt_ci(ci):
    if not ci:
        return "unavailable"
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _fmt_ci_pct(ci):
    if not ci:
        return "unavailable"
    return f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]"
