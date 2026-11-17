"""W2: fixed-budget Turn x Response 2x2 local component experiment.

The four arms are:
  A: old Turn + old Response (all-v1.3.0 control)
  B: new Turn + old Response
  C: old Turn + new Response
  D: new Turn + new Response (complete v1.4.0)

D-A is the only formal comparison.  B-A, C-A, D-B, D-C and the derived
component/interaction contrasts are exploratory; they are reported with
unadjusted intervals and never used as a confirmed component verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
from datetime import datetime, timezone

import yaml

from rilab.config import PROFILES, valid_action_biases

ROOT = Path(__file__).resolve().parents[1]
ARM_IDS = ("A", "B", "C", "D")
SERVICE_IDS = ("v130", "v140")
PRIMARY = "D-A"
COMPONENTS = ("B-A", "C-A", "D-B", "D-C")
CONTRASTS = (PRIMARY, *COMPONENTS)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_plan(path: str | Path) -> dict:
    plan_path = Path(path)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("W2 plan must be a mapping")
    engine = plan.get("engine", {})
    if isinstance(engine, dict):
        if "smoke_seeds" not in plan and "smoke_seeds" in engine:
            plan["smoke_seeds"] = engine["smoke_seeds"]
        if "formal_seed_range" not in plan and "formal_seed_range" in engine:
            plan["formal_seed_range"] = engine["formal_seed_range"]
    if "formal_seeds" not in plan and "formal_seed_range" in plan:
        start, end = plan["formal_seed_range"]
        plan["formal_seeds"] = list(range(int(start), int(end) + 1))
    return plan


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _service(plan: dict, name: str) -> dict:
    services = plan.get("services")
    if not isinstance(services, dict) or name not in services:
        raise ValueError(f"missing W2 service: {name}")
    service = services[name]
    if not isinstance(service, dict):
        raise ValueError(f"invalid W2 service: {name}")
    for key in (
        "source_root",
        "service_config",
        "weights_root",
        "adapter_profile",
        "revision",
        "model_identity",
        "response",
    ):
        if key not in service:
            raise ValueError(f"missing W2 service field: {name}/{key}")
    if service["adapter_profile"] not in PROFILES:
        raise ValueError(f"unknown W2 adapter profile: {name}")
    if service["adapter_profile"] not in ("full_state_v2", "full_state_v3"):
        raise ValueError(f"W2 service must use full_state_v2/v3: {name}")
    revision = service["revision"]
    if not isinstance(revision, dict) or not HEX40.fullmatch(
        str(revision.get("commit", ""))
    ):
        raise ValueError(f"invalid W2 revision: {name}")
    identity = service["model_identity"]
    if not isinstance(identity, dict):
        raise ValueError(f"invalid W2 model_identity: {name}")
    for task in ("turn", "response"):
        if not HEX64.fullmatch(str(identity.get(f"{task}_sha256", ""))):
            raise ValueError(f"invalid W2 weight hash: {name}/{task}")
    response = service["response"]
    if not isinstance(response, dict):
        raise ValueError(f"invalid W2 response calibration: {name}")
    if service["adapter_profile"] == "full_state_v2":
        if set(response) - {"pass_bias", "no_yaku_call_penalty"}:
            raise ValueError(f"invalid legacy W2 response fields: {name}")
        bias = response.get("pass_bias")
        if bias is not None and (
            type(bias) not in (int, float) or not math.isfinite(bias) or bias < 0
        ):
            raise ValueError(f"invalid W2 pass_bias: {name}")
    else:
        if set(response) - {"action_biases", "no_yaku_call_penalty"}:
            raise ValueError(f"invalid v3 W2 response fields: {name}")
        if not valid_action_biases(response.get("action_biases")):
            raise ValueError(f"invalid W2 action_biases: {name}")
    penalty = response.get("no_yaku_call_penalty")
    if penalty is not None and (
        type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty < 0
    ):
        raise ValueError(f"invalid W2 penalty: {name}")
    if not _resolve(service["source_root"]).is_dir():
        raise ValueError(f"W2 source_root does not exist: {name}")
    if not _resolve(service["service_config"]).is_file():
        raise ValueError(f"W2 service_config does not exist: {name}")
    if not _resolve(service["weights_root"]).is_dir():
        raise ValueError(f"W2 weights_root does not exist: {name}")
    return service


def validate_plan(plan: dict) -> None:
    if plan.get("plan_version") != 1:
        raise ValueError("W2 plan_version must be 1")
    engine = plan.get("engine")
    if not isinstance(engine, dict):
        raise ValueError("W2 plan requires engine")
    if engine.get("game_mode") != "4p-red-half" or engine.get("rule") != "default_tenhou":
        raise ValueError("W2 only supports 4p-red-half/default_tenhou")
    for name in SERVICE_IDS:
        _service(plan, name)
    arms = plan.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_IDS):
        raise ValueError("W2 arms must be exactly A/B/C/D")
    expected = {
        "A": {"turn": "v130", "response": "v130"},
        "B": {"turn": "v140", "response": "v130"},
        "C": {"turn": "v130", "response": "v140"},
        "D": {"turn": "v140", "response": "v140"},
    }
    for arm, value in expected.items():
        if arms[arm] != value:
            raise ValueError(f"W2 arm definition differs: {arm}")
    smoke = plan.get("smoke_seeds")
    formal = plan.get("formal_seeds")
    for name, seeds, count in (
        ("smoke_seeds", smoke, 2),
        ("formal_seeds", formal, 128),
    ):
        if (
            not isinstance(seeds, list)
            or len(seeds) != count
            or len(set(seeds)) != count
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        ):
            raise ValueError(f"{name} must contain {count} unique uint32 seeds")
    if set(smoke) & set(formal):
        raise ValueError("smoke_seeds and formal_seeds must be disjoint")
    if set(formal) & set(range(40001, 40009)):
        raise ValueError("formal_seeds must not reuse A3 engineering seeds")
    evaluation = plan.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("W2 plan requires evaluation")
    if evaluation.get("primary") != PRIMARY:
        raise ValueError("W2 primary must be D-A")
    if list(evaluation.get("components", [])) != list(COMPONENTS):
        raise ValueError("W2 components must be B-A/C-A/D-B/D-C")
    if evaluation.get("bootstrap_resamples") != 10000:
        raise ValueError("W2 bootstrap_resamples must be 10000")
    if evaluation.get("bootstrap_seed") != 20261117:
        raise ValueError("W2 bootstrap_seed must be 20261117")
    if evaluation.get("confidence_level") != 0.95:
        raise ValueError("W2 confidence_level must be 0.95")


def service_configs(plan: dict) -> dict:
    return {name: _service(plan, name) for name in SERVICE_IDS}


def _backend_base(service: dict, api_base: str) -> dict:
    return {
        "adapter_profile": service["adapter_profile"],
        "api_base": api_base,
        "timeout_seconds": service.get("timeout_seconds", 30.0),
        "response": service["response"],
        "revision": service["revision"],
        "model_identity": service["model_identity"],
    }


def build_backends(plan: dict, api_bases: dict) -> dict:
    if set(api_bases) != set(SERVICE_IDS):
        raise ValueError("W2 api_bases must cover v130/v140")
    services = service_configs(plan)
    v130 = _backend_base(services["v130"], api_bases["v130"])
    v140 = _backend_base(services["v140"], api_bases["v140"])
    return {
        "v130": v130,
        "B": {
            "backend_type": "task_routed",
            "turn_backend": dict(v140),
            "response_backend": dict(v130),
        },
        "C": {
            "backend_type": "task_routed",
            "turn_backend": dict(v130),
            "response_backend": dict(v140),
        },
        "D": v140,
    }


def build_schedule(plan: dict, seeds: list[int]) -> list[dict]:
    schedule = []
    for seed in seeds:
        for arm in ARM_IDS:
            base = (
                ["v130", "v130", "v130", "v130"]
                if arm == "A"
                else [arm, "v130", "v130", "v130"]
            )
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


def native_config(plan: dict, backends: dict) -> dict:
    return {
        "config_version": 1,
        "engine": {
            "game_mode": plan["engine"]["game_mode"],
            "rule": plan["engine"]["rule"],
            "seeds": list(plan["formal_seeds"]),
        },
        "backends": backends,
        "matchups": [],
        "evaluation": {
            "cluster": "seed_with_all_rotations",
            "confidence_level": 0.95,
            "primary_paired_group": "W2",
        },
        "artifacts": {
            "max_bytes": plan.get("artifacts", {}).get("max_bytes", 2_000_000_000),
            "snapshot_stride": plan.get("artifacts", {}).get("snapshot_stride", 100),
            "full_snapshots_first_run": True,
        },
    }


def registration(
    plan: dict,
    plan_path: str | Path,
    service_config_paths: dict,
    schedule: list[dict],
    *,
    stage: str,
) -> dict:
    service_hashes = {
        name: sha256_file(_resolve(path))
        for name, path in service_config_paths.items()
    }
    return {
        "schema_version": 1,
        "purpose": "w2_turn_response_2x2",
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(Path(plan_path)),
        "plan_sha256": sha256_file(plan_path),
        "service_config_sha256": service_hashes,
        "arms": plan["arms"],
        "smoke_seeds": list(plan["smoke_seeds"]),
        "formal_seeds": list(plan["formal_seeds"]),
        "schedule_games": len(schedule),
        "schedule_sha256": sha256_json(schedule),
        "evaluation": plan["evaluation"],
        "analysis": {
            "primary": PRIMARY,
            "components": list(COMPONENTS),
            "components_are_exploratory": True,
            "unadjusted_intervals": True,
        },
        "notes": [
            "A is the all-v1.3.0 control; B/C/D are component arms.",
            "D-A is the only formal comparison.",
            "B-A/C-A/D-B/D-C and interaction are exploratory.",
            "Smoke seeds are excluded from formal statistics.",
        ],
    }


def write_json(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_smoke(path: str | Path, plan: dict) -> dict:
    verification = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = len(plan["smoke_seeds"]) * len(ARM_IDS) * 4
    if verification.get("valid") is not True:
        raise ValueError("W2 smoke verification is not valid")
    if verification.get("completed_games") != expected:
        raise ValueError("W2 smoke completed_games mismatch")
    if verification.get("expected_games") != expected:
        raise ValueError("W2 smoke expected_games mismatch")
    if verification.get("fallbacks") not in (0, None):
        raise ValueError("W2 smoke has fallbacks")
    if verification.get("smoke_seeds") != list(plan["smoke_seeds"]):
        raise ValueError("W2 smoke seeds mismatch")
    return verification


def _seed_games(games: list[dict]) -> dict:
    grouped = {}
    for game in games:
        arm = game["matchup_id"]
        seed = game["seed"]
        focal = game["focal_seat"]
        if arm not in ARM_IDS or focal is None:
            raise ValueError("W2 game is missing arm/focal_seat")
        ranks = game.get("ranks")
        scores = game.get("scores")
        if not isinstance(ranks, list) or len(ranks) != 4:
            raise ValueError("W2 game is missing ranks")
        if not isinstance(scores, list) or len(scores) != 4:
            raise ValueError("W2 game is missing scores")
        grouped.setdefault(seed, {}).setdefault(arm, []).append(
            {
                "rank": ranks[focal],
                "points": scores[focal],
                "fourth": 1.0 if ranks[focal] == 4 else 0.0,
            }
        )
    return grouped


def _arm_seed_metrics(grouped: dict) -> dict:
    metrics = {}
    for seed, arms in grouped.items():
        metrics[seed] = {}
        for arm in ARM_IDS:
            rows = arms.get(arm, [])
            if len(rows) != 4:
                raise ValueError(f"W2 seed {seed}/{arm} does not have 4 rotations")
            metrics[seed][arm] = {
                "mean_rank": sum(row["rank"] for row in rows) / 4.0,
                "fourth_rate": sum(row["fourth"] for row in rows) / 4.0,
                "mean_points": sum(row["points"] for row in rows) / 4.0,
            }
    return metrics


def _seed_contrasts(metrics: dict) -> dict:
    values = {name: [] for name in CONTRASTS}
    for seed in sorted(metrics):
        arms = metrics[seed]
        a = arms["A"]
        b = arms["B"]
        c = arms["C"]
        d = arms["D"]
        values["B-A"].append(b["mean_rank"] - a["mean_rank"])
        values["C-A"].append(c["mean_rank"] - a["mean_rank"])
        values["D-A"].append(d["mean_rank"] - a["mean_rank"])
        values["D-B"].append(d["mean_rank"] - b["mean_rank"])
        values["D-C"].append(d["mean_rank"] - c["mean_rank"])
    return values


def _seed_last_contrasts(metrics: dict) -> list[float]:
    return [
        metrics[seed]["D"]["fourth_rate"] - metrics[seed]["A"]["fourth_rate"]
        for seed in sorted(metrics)
    ]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty bootstrap sample")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(
    contrasts: dict,
    last_contrasts: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict:
    count = len(contrasts["D-A"])
    rng = random.Random(seed)
    samples = {name: [] for name in CONTRASTS}
    last_samples = []
    for _ in range(resamples):
        draw = [rng.randrange(count) for _ in range(count)]
        for name in CONTRASTS:
            values = contrasts[name]
            samples[name].append(sum(values[i] for i in draw) / len(draw))
        last_samples.append(sum(last_contrasts[i] for i in draw) / len(draw))
    return {
        name: {
            "mean": sum(values) / len(values),
            "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
        }
        for name, values in samples.items()
    } | {
        "D-A:fourth_rate": {
            "mean": sum(last_samples) / len(last_samples),
            "ci95": [_percentile(last_samples, 0.025), _percentile(last_samples, 0.975)],
        }
    }


def _bootstrap_series(values: list[float], *, resamples: int, seed: int) -> dict:
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        draw = [rng.choice(values) for _ in values]
        samples.append(sum(draw) / len(draw))
    return {
        "mean": sum(samples) / len(samples),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
    }


def analyze_batch(batch_dir: str | Path, plan: dict) -> dict:
    batch = Path(batch_dir)
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    games = summary.get("games", [])
    expected = len(plan["formal_seeds"]) * len(ARM_IDS) * 4
    if len(games) != expected:
        raise ValueError(f"W2 analysis expected {expected} games, got {len(games)}")
    grouped = _seed_games(games)
    metrics = _arm_seed_metrics(grouped)
    contrasts = _seed_contrasts(metrics)
    last_contrasts = _seed_last_contrasts(metrics)
    bootstrap = _bootstrap(
        contrasts,
        last_contrasts,
        resamples=plan["evaluation"]["bootstrap_resamples"],
        seed=plan["evaluation"]["bootstrap_seed"],
    )
    rank = bootstrap[PRIMARY]
    last = bootstrap["D-A:fourth_rate"]
    if rank["ci95"][0] >= 0.05:
        decision = "regression"
    elif rank["ci95"][1] <= -0.05 and last["ci95"][1] <= 0.02:
        decision = "improvement"
    else:
        decision = "insufficient"
    arm_summary = {}
    for arm in ARM_IDS:
        arm_summary[arm] = {
            "mean_rank": sum(metrics[seed][arm]["mean_rank"] for seed in metrics) / len(metrics),
            "fourth_rate": sum(metrics[seed][arm]["fourth_rate"] for seed in metrics) / len(metrics),
            "mean_points": sum(metrics[seed][arm]["mean_points"] for seed in metrics) / len(metrics),
            "seed_clusters": len(metrics),
        }
    turn = [
        ((metrics[seed]["B"]["mean_rank"] - metrics[seed]["A"]["mean_rank"])
         + (metrics[seed]["D"]["mean_rank"] - metrics[seed]["C"]["mean_rank"])) / 2.0
        for seed in sorted(metrics)
    ]
    response = [
        ((metrics[seed]["C"]["mean_rank"] - metrics[seed]["A"]["mean_rank"])
         + (metrics[seed]["D"]["mean_rank"] - metrics[seed]["B"]["mean_rank"])) / 2.0
        for seed in sorted(metrics)
    ]
    interaction = [
        (metrics[seed]["D"]["mean_rank"] - metrics[seed]["C"]["mean_rank"])
        - (metrics[seed]["B"]["mean_rank"] - metrics[seed]["A"]["mean_rank"])
        for seed in sorted(metrics)
    ]
    return {
        "schema_version": 1,
        "purpose": "w2_turn_response_2x2",
        "stage": summary.get("stage"),
        "expected_games": expected,
        "completed_games": len(games),
        "seed_clusters": len(metrics),
        "arms": arm_summary,
        "primary": {
            "name": PRIMARY,
            "rank_difference": rank,
            "fourth_rate_difference": last,
            "decision": decision,
            "rule": {
                "regression": "rank CI lower >= +0.05",
                "improvement": "rank CI upper <= -0.05 and fourth-rate CI upper <= +0.02",
                "else": "insufficient",
            },
        },
        "components": {
            name: {
                **bootstrap[name],
                "exploratory": True,
                "unadjusted": True,
            }
            for name in COMPONENTS
        },
        "derived": {
            "turn_effect": {
                **_bootstrap_series(
                    turn,
                    resamples=plan["evaluation"]["bootstrap_resamples"],
                    seed=plan["evaluation"]["bootstrap_seed"],
                ),
                "exploratory": True,
                "unadjusted": True,
            },
            "response_effect": {
                **_bootstrap_series(
                    response,
                    resamples=plan["evaluation"]["bootstrap_resamples"],
                    seed=plan["evaluation"]["bootstrap_seed"],
                ),
                "exploratory": True,
                "unadjusted": True,
            },
            "interaction": {
                **_bootstrap_series(
                    interaction,
                    resamples=plan["evaluation"]["bootstrap_resamples"],
                    seed=plan["evaluation"]["bootstrap_seed"],
                ),
                "exploratory": True,
                "unadjusted": True,
            },
        },
        "per_seed": {
            str(seed): metrics[seed]
            for seed in sorted(metrics)
        },
        "limitations": [
            "Fixed v1.3.0 opponents; local four-player performance, not real ranked strength.",
            "D-A is the only formal comparison.",
            "Component and interaction contrasts are exploratory with unadjusted intervals.",
            "No fallback/illegal game is part of the formal sample.",
        ],
    }


def render_report(result: dict) -> str:
    lines = [
        "# W2 Turn x Response 2x2 report",
        "",
        f"- stage: `{result['stage']}`",
        f"- games: `{result['completed_games']}/{result['expected_games']}`",
        f"- seed clusters: `{result['seed_clusters']}`",
        "",
        "## Arms",
        "",
        "| Arm | Mean rank | 4th rate | Mean points |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARM_IDS:
        row = result["arms"][arm]
        lines.append(
            f"| {arm} | {row['mean_rank']:.4f} | {row['fourth_rate']:.4%} | "
            f"{row['mean_points']:.1f} |"
        )
    primary = result["primary"]
    rank = primary["rank_difference"]
    last = primary["fourth_rate_difference"]
    lines += [
        "",
        "## Primary: D-A",
        "",
        f"- rank difference: `{rank['mean']:+.4f}` "
        f"[{rank['ci95'][0]:+.4f}, {rank['ci95'][1]:+.4f}]",
        f"- 4th-rate difference: `{last['mean']:+.4%}` "
        f"[{last['ci95'][0]:+.4%}, {last['ci95'][1]:+.4%}]",
        f"- decision: **{primary['decision']}**",
        "",
        "## Exploratory components",
        "",
        "| Contrast | Mean rank diff | 95% unadjusted CI |",
        "|---|---:|---:|",
    ]
    for name in COMPONENTS:
        item = result["components"][name]
        lines.append(
            f"| {name} | {item['mean']:+.4f} | "
            f"[{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}] |"
        )
    lines += [
        "",
        "## Derived effects (exploratory)",
        "",
        "| Effect | Mean rank diff | 95% unadjusted CI |",
        "|---|---:|---:|",
    ]
    for name, item in result["derived"].items():
        lines.append(
            f"| {name} | {item['mean']:+.4f} | "
            f"[{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}] |"
        )
    lines += [
        "",
        "Component and interaction contrasts are exploratory and unadjusted; "
        "they are not confirmed component regressions.",
        "",
        "## Limitations",
        "",
        "- Fixed v1.3.0 opponents; local performance, not real ranked strength.",
        "- D-A is the only formal comparison.",
        "- No fallback/illegal game is part of the formal sample.",
        "",
    ]
    return "\n".join(lines)
