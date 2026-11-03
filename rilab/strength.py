"""Validate a fixed strength plan and seal it before launching observations."""

from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import yaml

from rilab.artifacts import encode, write_json
from rilab.config import load_arena_config
from rilab.scheduling import expand_schedule


def validate_plan(plan, config):
    schedule = expand_schedule(config)
    seeds = config["engine"]["seeds"]
    if (
        plan.get("status") != "planned"
        or plan.get("purpose") != "formal_strength_experiment"
    ):
        raise ValueError("requires a planned strength protocol")
    design = plan["design"]
    if (
        len(seeds) < 2
        or design["cluster"] != "seed_with_all_rotations"
        or design["primary"] != "candidate_vs_baseline_and_baseline_control_cyclic4"
        or design["supplementary"] != "two_vs_two_unique6_separate_report"
        or plan["comparison"]["estimand"]
        != "complete_deployed_policies_including_calibration"
        or plan["decision"]["primary_metric"]
        != "mean_rank_difference_candidate_minus_control"
        or type(plan["decision"]["bootstrap_rng_seed"]) is not int
    ):
        raise ValueError("unsupported strength experiment design")
    if (
        design["final_seeds"] != seeds
        or design["target_seed_clusters"] != len(seeds)
        or design["half_game_budget"] != len(schedule)
        or set(seeds) & set(design["excluded_engineering_or_calibration_seeds"])
        or not set(range(1, 9)).issubset(
            design["excluded_engineering_or_calibration_seeds"]
        )
    ):
        raise ValueError(
            "strength seeds/budget do not match plan or overlap engineering seeds"
        )
    decision = plan["decision"]
    gain, guard = (
        decision["minimum_meaningful_rank_gain"],
        decision["last_place_rate_guardrail"],
    )
    if (
        type(gain) not in (int, float)
        or not math.isfinite(gain)
        or gain <= 0
        or type(guard) not in (int, float)
        or not math.isfinite(guard)
        or not 0 <= guard <= 1
        or decision["confidence_level"] != 0.95
        or type(decision["bootstrap_iterations"]) is not int
        or decision["bootstrap_iterations"] < 10000
    ):
        raise ValueError("invalid strength decision thresholds")
    if plan["stopping"] != {
        "rule": "fixed_complete_seed_budget_no_optional_stopping",
        "on_runtime_failure": "invalidate_whole_run_and_stop_new_games",
        "no_peeking_to_change_budget_or_models": True,
    }:
        raise ValueError("invalid stopping rules")
    comparison = plan["comparison"]
    group = config["evaluation"]["primary_paired_group"]
    members = [m for m in config["matchups"] if m.get("paired_group") == group]
    baseline, candidate = comparison["baseline"], comparison["candidate"]
    if baseline == candidate or not any(m["seats"] == [baseline] * 4 for m in members):
        raise ValueError("missing registered baseline control")
    if not any(
        m["seats"][m["focal_seat"]] == candidate
        and all(
            s == m["focal_seat"] or name == baseline
            for s, name in enumerate(m["seats"])
        )
        for m in members
    ):
        raise ValueError("missing registered candidate comparison")
    return schedule


def register_protocol(plan_path, config_path, runtime_config, metadata, output_dir):
    """Called by the launcher after deployment validation, before batch execution."""
    planned_config = load_arena_config(config_path)
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    validate_plan(plan, planned_config)
    validate_plan(plan, runtime_config)
    normalized = deepcopy(runtime_config)
    if set(normalized["backends"]) != set(planned_config["backends"]):
        raise ValueError("runtime backend registry differs from plan")
    for name, backend in normalized["backends"].items():
        backend["api_base"] = planned_config["backends"][name]["api_base"]
        backend.pop("deployment", None)
    if normalized != planned_config:
        raise ValueError("runtime configuration differs from plan")
    for name, backend in runtime_config["backends"].items():
        for field in ("adapter_profile", "revision", "model_identity", "response"):
            if backend[field] != planned_config["backends"][name][field]:
                raise ValueError(
                    "runtime model identity differs from planned configuration"
                )
        deployment = backend.get("deployment", {})
        if (
            deployment.get("source_dirty") is not False
            or deployment.get("verified_weight_sha256") != backend["model_identity"]
        ):
            raise ValueError("missing clean deployment and verified weight hashes")
        response = metadata[name]["response"]
        actual_bias = backend["response"]["pass_bias"]
        if actual_bias is None:
            actual_bias = response["default_pass_bias"]
        actual = {
            "pass_bias": actual_bias,
            "no_yaku_call_penalty": response["no_yaku_call_penalty"],
        }
        if actual != plan["expected_calibration"][name]:
            raise ValueError(f"effective service calibration differs from plan: {name}")
    out = Path(output_dir)
    target = out / "protocol.registered.json"
    if target.exists() or (out / "batch").exists() or (out / "replay").exists():
        raise ValueError(
            "cannot register a protocol after a batch exists or overwrite registration"
        )
    record = {
        "status": "registered_before_launch",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "planned_config": planned_config,
        "model_metadata": metadata,
        "planned_config_file_sha256": hashlib.sha256(
            Path(config_path).read_bytes()
        ).hexdigest(),
        "plan_file_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        "runtime_config_sha256_before_registration_ref": hashlib.sha256(
            encode(runtime_config).encode()
        ).hexdigest(),
        "deployment": {
            name: b["deployment"] for name, b in runtime_config["backends"].items()
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).parents[1] / "scripts/run_a1_integration.py",
            )
        },
    }
    write_json(target, record)
    runtime_config["evaluation"]["strength_registration"] = {
        "path": str(target.resolve()),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    return record
