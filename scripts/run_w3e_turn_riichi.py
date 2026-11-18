"""Run the pre-registered W3e Turn-only riichi-divergence replication.

Phases are deliberately separate so the extension decision cannot be driven by
outcome rates:

  run      -> play the fixed schedule and write an audited batch
  count    -> same-state shadow replay; writes events.jsonl + counts.json (no outcomes)
  analyze  -> join captured states to hand outcomes; writes report.json/report.md

Examples:
  D:\\Anaconda3\\python.exe scripts\\run_w3e_turn_riichi.py --stage stage1 --phase run --dry-run
  D:\\Anaconda3\\python.exe scripts\\run_w3e_turn_riichi.py --stage stage1 --phase run
  D:\\Anaconda3\\python.exe scripts\\run_w3e_turn_riichi.py --stage stage1 --phase count
  D:\\Anaconda3\\python.exe scripts\\run_w3e_turn_riichi.py --stage stage1 --phase analyze
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.artifacts import audit_batch
from rilab.backends.http_backend import HttpPredictBackend
from rilab.batch import run_batch
from rilab.w2 import build_backends, sha256_file, write_json
from rilab.w3 import read_json
from rilab.w3e import (
    ARM_IDS,
    PURPOSE,
    analyze_events,
    attach_outcomes,
    build_schedule,
    capture_filter,
    collect_events,
    count_events,
    group_rows,
    load_events,
    load_plan,
    native_config,
    registration,
    render_report,
    validate_plan,
    write_jsonl,
)
from scripts.run_v140_validation import start_service

DEFAULT_PLAN = ROOT / "configs/w3e/w3e_turn_riichi_plan.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/w3e_turn_20261118"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _backend_config(service: dict, api_base: str) -> dict:
    return {
        "adapter_profile": service["adapter_profile"],
        "api_base": api_base.rstrip("/"),
        "timeout_seconds": service.get("timeout_seconds", 30.0),
        "response": service["response"],
    }


def _service_records(services: list) -> dict:
    return {
        service.service_id: {
            "api_base": service.api_base,
            "pid": service.process.pid,
            "log_path": str(service.log_path),
            "resolved_config": str(service.resolved_path),
            "resolved_config_sha256": (
                sha256_file(service.resolved_path)
                if service.resolved_path.is_file()
                else None
            ),
        }
        for service in services
    }


def _write_verification(
    stage_dir: Path,
    *,
    plan: dict,
    plan_path: Path,
    stage: str,
    seeds: list[int],
    summary: dict,
    audit_games: list[dict],
    registration_sha: str,
    services: dict,
) -> dict:
    fallbacks = sum(int(game.get("fallbacks", 0)) for game in summary["games"])
    expected = summary.get("expected_games")
    valid = (
        summary.get("valid") is True
        and summary.get("completed_games") == expected
        and fallbacks == 0
        and len(audit_games) == expected
        and all(game.get("valid") is True for game in audit_games)
    )
    manifest = read_json(stage_dir / "manifest.json")
    verification = {
        "schema_version": 1,
        "valid": valid,
        "stage": stage,
        "purpose": PURPOSE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "expected_games": expected,
        "completed_games": summary.get("completed_games"),
        "fallbacks": fallbacks,
        "audit_games": len(audit_games),
        "seeds": list(seeds),
        "protocol_registration_sha256": registration_sha,
        "replay_digest": summary.get("replay_digest"),
        "rilab_commit": manifest.get("rilab_commit"),
        "rilab_dirty": manifest.get("rilab_dirty"),
        "capture": plan["capture"],
        "models": manifest.get("models"),
        "services": services,
    }
    write_json(stage_dir / "verification.json", verification)
    return verification


def verify_stage(stage_root: Path, plan: dict, stage: str) -> dict:
    if stage not in ("smoke", "stage1", "stage2"):
        raise ValueError(f"stage has no batch verification: {stage}")
    path = stage_root / "batch" / "verification.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing stage verification: {path}")
    verification = read_json(path)
    expected = len(plan[f"{stage}_seeds"]) * len(ARM_IDS) * 4
    if verification.get("valid") is not True:
        raise ValueError(f"W3e stage {stage} verification is not valid")
    if verification.get("purpose") != PURPOSE:
        raise ValueError(f"W3e stage {stage} purpose mismatch")
    if verification.get("completed_games") != expected:
        raise ValueError(f"W3e stage {stage} completed_games mismatch")
    if verification.get("expected_games") != expected:
        raise ValueError(f"W3e stage {stage} expected_games mismatch")
    if verification.get("fallbacks") != 0:
        raise ValueError(f"W3e stage {stage} has fallbacks")
    if verification.get("audit_games") != expected:
        raise ValueError(f"W3e stage {stage} audit_games mismatch")
    if verification.get("seeds") != list(plan[f"{stage}_seeds"]):
        raise ValueError(f"W3e stage {stage} seeds mismatch")
    return verification


def run_stage(plan, plan_path: Path, stage: str, output_root: Path, args) -> dict:
    if stage == "pooled":
        raise ValueError("run phase does not support pooled")
    seeds = list(plan[f"{stage}_seeds"])
    schedule = build_schedule(plan, seeds)
    stage_root = output_root / stage
    batch_dir = stage_root / "batch"
    if batch_dir.exists():
        raise FileExistsError(f"stage batch directory already exists: {batch_dir}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "stage": stage,
                    "purpose": PURPOSE,
                    "seed_clusters": len(seeds),
                    "seeds": seeds,
                    "arms": list(ARM_IDS),
                    "rotations": 4,
                    "games": len(schedule),
                    "expected_games": len(schedule),
                    "capture": plan["capture"],
                    "extension_threshold": plan["analysis"]["extension"][
                        "extend_if_below"
                    ],
                    "output_root": str(output_root),
                    "plan": str(plan_path),
                    "service_configs": {
                        name: str(_resolve(service["service_config"]))
                        for name, service in plan["services"].items()
                    },
                },
                indent=2,
            )
        )
        return {}

    stage_root.mkdir(parents=True, exist_ok=True)
    services = []
    try:
        if args.v130_api_base and args.v140_api_base:
            api_bases = {
                "v130": args.v130_api_base.rstrip("/"),
                "v140": args.v140_api_base.rstrip("/"),
            }
        else:
            services = [
                start_service("v130", plan["services"]["v130"], stage_root),
                start_service("v140", plan["services"]["v140"], stage_root),
            ]
            api_bases = {service.service_id: service.api_base for service in services}
        backends = build_backends(plan, api_bases)
        config = native_config(plan, backends, seeds)
        write_json(stage_root / "native_config.json", config)
        reg = registration(
            plan,
            plan_path,
            {
                name: _resolve(service["service_config"])
                for name, service in plan["services"].items()
            },
            schedule,
            stage=stage,
            seeds=seeds,
        )
        reg_path = stage_root / "protocol.registered.json"
        write_json(reg_path, reg)
        manifest_extra = {
            "protocol_registration": {
                "path": str(reg_path),
                "sha256": sha256_file(reg_path),
            },
            "w3e": {
                "stage": stage,
                "capture": plan["capture"],
                "preregistration": "docs/w3e-turn-riichi-preregistration-2026-11-18.md",
            },
        }
        summary = run_batch(
            config,
            batch_dir,
            manifest_extra=manifest_extra,
            schedule=schedule,
            stage=f"W3e-{stage}",
            purpose=PURPOSE,
            capture_filter=capture_filter,
        )
        audit_games = audit_batch(batch_dir)
        external_services = (
            {
                name: {"api_base": base, "external": True}
                for name, base in api_bases.items()
            }
            if not services
            else {}
        )
        service_records = _service_records(services) | external_services
        verification = _write_verification(
            batch_dir,
            plan=plan,
            plan_path=plan_path,
            stage=stage,
            seeds=seeds,
            summary=summary,
            audit_games=audit_games,
            registration_sha=sha256_file(reg_path),
            services=service_records,
        )
        if not verification["valid"]:
            raise RuntimeError(f"W3e stage {stage} verification is not valid")
        print(
            json.dumps(
                {
                    "valid": True,
                    "stage": stage,
                    "games": summary.get("completed_games"),
                    "expected_games": summary.get("expected_games"),
                    "fallbacks": verification["fallbacks"],
                    "verification": str(batch_dir / "verification.json"),
                },
                indent=2,
            )
        )
        return verification
    finally:
        for service in services:
            service.stop()


def _open_prediction_backends(plan, args, stage_root: Path):
    services = []
    if args.v130_api_base and args.v140_api_base:
        api_bases = {
            "v130": args.v130_api_base.rstrip("/"),
            "v140": args.v140_api_base.rstrip("/"),
        }
    else:
        services = [
            start_service("v130", plan["services"]["v130"], stage_root),
            start_service("v140", plan["services"]["v140"], stage_root),
        ]
        api_bases = {service.service_id: service.api_base for service in services}
    v130 = HttpPredictBackend(
        "v130", _backend_config(plan["services"]["v130"], api_bases["v130"])
    )
    v140 = HttpPredictBackend(
        "v140", _backend_config(plan["services"]["v140"], api_bases["v140"])
    )
    return v130, v140, services


def count_stage(plan, plan_path: Path, stage: str, output_root: Path, args) -> dict:
    if stage == "pooled":
        raise ValueError("count phase does not support pooled")
    stage_root = output_root / stage
    verification = verify_stage(stage_root, plan, stage)
    batch_dir = stage_root / "batch"
    v130 = v140 = None
    services = []
    try:
        v130, v140, services = _open_prediction_backends(plan, args, stage_root)
        events, meta = collect_events(batch_dir, plan, v130, v140)
        v130.check_unchanged()
        v140.check_unchanged()
        write_jsonl(stage_root / "events.jsonl", events)
        summary = read_json(batch_dir / "summary.json")
        audit_games = audit_batch(batch_dir)
        counts = count_events(
            events,
            meta,
            stage=stage,
            plan=plan,
            summary=summary,
            audit_games=len(audit_games),
            plan_sha256=sha256_file(plan_path),
        )
        counts["verification_replay_digest"] = verification.get("replay_digest")
        write_json(stage_root / "counts.json", counts)
        if counts["shadow_errors"] or counts["missing_state"] or counts["missing_candidates"]:
            raise RuntimeError(
                "W3e shadow classification has errors: "
                f"shadow_errors={counts['shadow_errors']} "
                f"missing_state={counts['missing_state']} "
                f"missing_candidates={counts['missing_candidates']}"
            )
        print(
            json.dumps(
                {
                    "valid": True,
                    "stage": stage,
                    "captured_states": counts["captured_states"],
                    "captured_with_state": counts["captured_with_state"],
                    "excluded_common_support": counts["excluded_common_support"],
                    "shadow_errors": counts["shadow_errors"],
                    "error_rows": counts["error_rows"],
                    "divergence": counts["divergence"],
                    "extension_rule": counts["extension_rule"],
                    "counts": str(stage_root / "counts.json"),
                },
                indent=2,
            )
        )
        return counts
    finally:
        if v130 is not None:
            v130.close()
        if v140 is not None:
            v140.close()
        for service in services:
            service.stop()


def analyze_stage(plan, plan_path: Path, stage: str, output_root: Path) -> dict:
    if stage == "pooled":
        stage_names = ["stage1", "stage2"]
        for name in stage_names:
            counts_path = output_root / name / "counts.json"
            if not counts_path.is_file():
                raise FileNotFoundError(
                    f"pooled analysis requires completed stage: {counts_path}"
                )
    else:
        stage_names = [stage]
        counts_path = output_root / stage / "counts.json"
        if not counts_path.is_file():
            raise FileNotFoundError(
                f"analyze requires count phase first: {counts_path}"
            )
    events = []
    integrity = {}
    for name in stage_names:
        stage_root = output_root / name
        verification = verify_stage(stage_root, plan, name)
        counts = read_json(stage_root / "counts.json")
        if counts.get("shadow_errors") or counts.get("error_rows"):
            raise RuntimeError(
                f"stage {name} has shadow errors; refusing outcome analysis"
            )
        stage_events = load_events(stage_root)
        stage_events = attach_outcomes(stage_events, stage_root / "batch")
        events.extend(stage_events)
        integrity[name] = {
            "verification_valid": verification.get("valid"),
            "games": verification.get("completed_games"),
            "fallbacks": verification.get("fallbacks"),
            "captured_states": counts.get("captured_states"),
            "shadow_errors": counts.get("shadow_errors"),
            "excluded_common_support": counts.get("excluded_common_support"),
            "extension_rule": counts.get("extension_rule"),
            "replay_digest": verification.get("replay_digest"),
        }
    plan = {**plan, "_plan_sha256": sha256_file(plan_path)}
    result = analyze_events(events, plan, scope=stage, stages=stage_names)
    result["integrity"] = integrity
    result["plan_path"] = str(plan_path)
    stage_root = output_root / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    write_json(stage_root / "report.json", result)
    (stage_root / "report.md").write_text(render_report(result), encoding="utf-8")
    labels = {}
    for name, rows in group_rows(events).items():
        for row in rows:
            labels.setdefault(id(row), []).append(name)
    write_jsonl(
        stage_root / "hands.jsonl",
        [{**row, "groups": sorted(labels.get(id(row), []))} for row in events],
    )
    print(
        json.dumps(
            {
                "valid": True,
                "stage": stage,
                "scope": stage,
                "stages": stage_names,
                "events": result["events"],
                "captured_groups": {
                    "B_extra": result["groups"]["B_extra"]["n"],
                    "A_mirror": result["groups"]["A_mirror"]["n"],
                },
                "primary": {
                    "decision": result["primary"]["decision"],
                    "b_extra_deal_rate": result["primary"]["groups"]["b_extra"][
                        "deal_rate"
                    ],
                    "a_mirror_deal_rate": result["primary"]["groups"]["a_mirror"][
                        "deal_rate"
                    ],
                    "difference": result["primary"]["comparison"]["difference"],
                    "newcombe_ci95": result["primary"]["comparison"]["newcombe_ci95"],
                    "fisher_p": result["primary"]["comparison"]["fisher_p"],
                },
                "report": str(stage_root / "report.json"),
                "report_md": str(stage_root / "report.md"),
            },
            indent=2,
        )
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--stage", required=True, choices=["smoke", "stage1", "stage2", "pooled"]
    )
    parser.add_argument(
        "--phase", required=True, choices=["run", "count", "analyze", "all"]
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--v130-api-base")
    parser.add_argument("--v140-api-base")
    args = parser.parse_args(argv)

    if bool(args.v130_api_base) != bool(args.v140_api_base):
        parser.error("provide both --v130-api-base and --v140-api-base, or neither")
    if args.phase == "all" and args.stage != "smoke":
        parser.error(
            "--phase all is allowed only for smoke; formal stages must run "
            "run -> count -> extension decision -> analyze"
        )

    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    validate_plan(plan)
    output_root = args.output_root.resolve()
    if args.phase == "run":
        run_stage(plan, plan_path, args.stage, output_root, args)
        return 0
    if args.phase == "count":
        count_stage(plan, plan_path, args.stage, output_root, args)
        return 0
    if args.phase == "analyze":
        analyze_stage(plan, plan_path, args.stage, output_root)
        return 0
    # smoke only
    run_stage(plan, plan_path, args.stage, output_root, args)
    count_stage(plan, plan_path, args.stage, output_root, args)
    analyze_stage(plan, plan_path, args.stage, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
