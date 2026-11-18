"""Run the pre-registered W5 serving-time Riichi veto experiment.

Phases are deliberately separate so the extension decision cannot be driven by
outcomes:

  run      -> play the fixed schedule and write an audited batch
  count    -> count first-veto pairs from decision artifacts (no outcomes)
  analyze  -> pair first-veto hands with control hands; write report.json/report.md

Examples:
  D:\\Anaconda3\\python.exe scripts\\run_w5_riichi_veto.py --stage stage1 --phase run --dry-run
  D:\\Anaconda3\\python.exe scripts\\run_w5_riichi_veto.py --stage smoke --phase all
  D:\\Anaconda3\\python.exe scripts\\run_w5_riichi_veto.py --stage stage1 --phase run
  D:\\Anaconda3\\python.exe scripts\\run_w5_riichi_veto.py --stage stage1 --phase count
  D:\\Anaconda3\\python.exe scripts\\run_w5_riichi_veto.py --stage stage1 --phase analyze
  D:\\Anaconda3\\python.exe scripts\\run_w5_riichi_veto.py --stage pooled --phase analyze
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
from rilab.batch import run_batch
from rilab.w2 import sha256_file, write_json
from rilab.w3 import read_json
from rilab.w5 import (
    ARM_IDS,
    PURPOSE,
    analyze_stages,
    build_backends,
    build_schedule,
    capture_filter,
    collect_first_vetoes,
    count_document,
    load_plan,
    load_stage_pairs,
    native_config,
    registration,
    render_report,
    validate_plan,
)
from scripts.run_v140_validation import start_service

DEFAULT_PLAN = ROOT / "configs/w5/w5_riichi_veto_plan.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/w5_riichi_veto_20261118"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


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
        "arms": plan["arms"],
        "veto": plan["veto"],
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
        raise ValueError(f"W5 stage {stage} verification is not valid")
    if verification.get("purpose") != PURPOSE:
        raise ValueError(f"W5 stage {stage} purpose mismatch")
    if verification.get("completed_games") != expected:
        raise ValueError(f"W5 stage {stage} completed_games mismatch")
    if verification.get("expected_games") != expected:
        raise ValueError(f"W5 stage {stage} expected_games mismatch")
    if verification.get("fallbacks") != 0:
        raise ValueError(f"W5 stage {stage} has fallbacks")
    if verification.get("audit_games") != expected:
        raise ValueError(f"W5 stage {stage} audit_games mismatch")
    if verification.get("seeds") != list(plan[f"{stage}_seeds"]):
        raise ValueError(f"W5 stage {stage} seeds mismatch")
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
                    "veto": plan["veto"],
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
            "w5": {
                "stage": stage,
                "arms": plan["arms"],
                "veto": plan["veto"],
                "capture": plan["capture"],
                "preregistration": "docs/w5-riichi-veto-preregistration-2026-11-18.md",
            },
        }
        summary = run_batch(
            config,
            batch_dir,
            manifest_extra=manifest_extra,
            schedule=schedule,
            stage=f"W5-{stage}",
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
            raise RuntimeError(f"W5 stage {stage} verification is not valid")
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


def count_stage(plan, plan_path: Path, stage: str, output_root: Path) -> dict:
    if stage == "pooled":
        raise ValueError("count phase does not support pooled")
    stage_root = output_root / stage
    verification = verify_stage(stage_root, plan, stage)
    batch_dir = stage_root / "batch"
    summary = read_json(batch_dir / "summary.json")
    audit_games = audit_batch(batch_dir)
    rows, meta = collect_first_vetoes(batch_dir)
    counts = count_document(
        rows,
        meta,
        stage=stage,
        plan=plan,
        summary=summary,
        audit_games=len(audit_games),
        plan_sha256=sha256_file(plan_path),
    )
    counts["verification_replay_digest"] = verification.get("replay_digest")
    write_json(stage_root / "counts.json", counts)
    if counts["anomalies"] or counts["unpaired"]:
        raise RuntimeError(
            "W5 first-veto count has anomalies/unpaired rows: "
            f"anomalies={len(counts['anomalies'])} "
            f"unpaired={len(counts['unpaired'])}"
        )
    print(
        json.dumps(
            {
                "valid": True,
                "stage": stage,
                "treatment_games": counts["treatment_games"],
                "veto_events": counts["veto_events"],
                "games_with_veto": counts["games_with_veto"],
                "later_veto_events": counts["later_veto_events"],
                "first_veto_pairs": counts["first_veto_pairs"],
                "unpaired_first_vetoes": counts["unpaired_first_vetoes"],
                "extension_rule": counts["extension_rule"],
                "counts": str(stage_root / "counts.json"),
            },
            indent=2,
        )
    )
    return counts


def _stage_data(name: str, output_root: Path, plan: dict) -> dict:
    stage_root = output_root / name
    verification = verify_stage(stage_root, plan, name)
    counts_path = stage_root / "counts.json"
    if not counts_path.is_file():
        raise FileNotFoundError(
            f"analyze requires count phase first: {counts_path}"
        )
    counts = read_json(counts_path)
    if counts.get("outcomes_inspected") is not False:
        raise RuntimeError(f"stage {name} count phase was not outcome-blind")
    if counts.get("anomalies"):
        raise RuntimeError(f"stage {name} has first-veto anomalies")
    batch_dir = stage_root / "batch"
    pairs, unpaired, meta = load_stage_pairs(batch_dir)
    if len(pairs) + len(unpaired) != counts["first_vetoes"]:
        raise RuntimeError(
            f"stage {name} first-veto count changed between count and analyze"
        )
    summary = read_json(batch_dir / "summary.json")
    return {
        "stage": name,
        "verification": verification,
        "counts": counts,
        "summary": summary,
        "meta": meta,
        "pairs": pairs,
        "unpaired": unpaired,
    }


def analyze_stage(plan, plan_path: Path, stage: str, output_root: Path) -> dict:
    if stage == "stage2":
        raise ValueError("Stage 2 outcomes are analyzed pooled only")
    if stage == "stage1":
        counts = read_json(output_root / "stage1" / "counts.json")
        if counts["extension_rule"]["extension_required"]:
            raise RuntimeError(
                "Stage 1 count requires Stage 2; run stage2, then analyze pooled"
            )
        stage_names = ["stage1"]
    elif stage == "pooled":
        stage1_counts = read_json(output_root / "stage1" / "counts.json")
        if not stage1_counts["extension_rule"]["extension_required"]:
            raise RuntimeError(
                "pooled analysis requires a Stage 1 count that triggered extension"
            )
        stage2_counts = output_root / "stage2" / "counts.json"
        if not stage2_counts.is_file():
            raise FileNotFoundError(
                f"pooled analysis requires completed stage: {stage2_counts}"
            )
        stage_names = ["stage1", "stage2"]
    else:
        stage_names = [stage]

    stage_data = [_stage_data(name, output_root, plan) for name in stage_names]
    plan = {**plan, "_plan_sha256": sha256_file(plan_path)}
    result = analyze_stages(stage_data, plan, scope=stage)
    result["plan_path"] = str(plan_path)
    result["plan_sha256"] = sha256_file(plan_path)
    result["integrity"] = {
        item["stage"]: {
            "verification_valid": item["verification"].get("valid"),
            "games": item["verification"].get("completed_games"),
            "fallbacks": item["verification"].get("fallbacks"),
            "veto_events": item["meta"].get("veto_events"),
            "games_with_veto": item["meta"].get("games_with_veto"),
            "first_veto_pairs": item["counts"].get("first_veto_pairs"),
            "count_outcomes_inspected": item["counts"].get("outcomes_inspected"),
            "extension_rule": item["counts"].get("extension_rule"),
            "replay_digest": item["verification"].get("replay_digest"),
        }
        for item in stage_data
    }
    stage_root = output_root / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    write_json(stage_root / "report.json", result)
    (stage_root / "report.md").write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "valid": True,
                "stage": stage,
                "scope": stage,
                "stages": stage_names,
                "first_veto_pairs": result["first_veto_pairs"],
                "primary": {
                    "decision": result["primary"]["decision"],
                    "treatment_deal_rate": result["primary"]["treatment"]["rate"],
                    "control_deal_rate": result["primary"]["control"]["rate"],
                    "difference": result["primary"]["difference"],
                    "bootstrap_ci95": result["primary"]["seed_cluster_bootstrap"][
                        "ci95"
                    ],
                    "mcnemar_exact_p": result["primary"]["mcnemar_exact_p"],
                },
                "unpaired": len(result["unpaired"]),
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
    if args.stage == "pooled" and args.phase != "analyze":
        parser.error("pooled supports only the analyze phase")
    if args.stage == "stage2" and args.phase == "analyze":
        parser.error("Stage 2 outcomes are analyzed as pooled, not alone")
    if args.dry_run and args.phase != "run":
        parser.error("--dry-run is supported only for the run phase")

    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    validate_plan(plan)
    output_root = args.output_root.resolve()
    if args.phase == "run":
        run_stage(plan, plan_path, args.stage, output_root, args)
        return 0
    if args.phase == "count":
        count_stage(plan, plan_path, args.stage, output_root)
        return 0
    if args.phase == "analyze":
        analyze_stage(plan, plan_path, args.stage, output_root)
        return 0
    # smoke only
    run_stage(plan, plan_path, args.stage, output_root, args)
    count_stage(plan, plan_path, args.stage, output_root)
    analyze_stage(plan, plan_path, args.stage, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
