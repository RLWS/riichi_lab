"""Read-only audit for a completed v1.4.0 Step 1 screening output directory."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.artifacts import audit_batch  # noqa: E402
from rilab.screening import (  # noqa: E402
    BASELINE_ID,
    CANDIDATE_IDS,
    recommendation,
    verify_group_decisions,
)


def hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="run the offline decision replay (no model calls, no new games)",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    checks = []
    errors = []

    def add(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    checksums_path = out / "artifact_checksums.json"
    add("artifact_checksums_exists", checksums_path.is_file())
    if checksums_path.is_file():
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        missing = []
        changed = []
        for relative, item in checksums["files"].items():
            path = out / relative
            if not path.is_file():
                missing.append(relative)
            elif hash_file(path) != item["sha256"]:
                changed.append(relative)
        add("artifact_checksums_missing", not missing, missing[:20])
        add("artifact_checksums_changed", not changed, changed[:20])

    protocol_path = out / "protocol.registered.json"
    add("protocol_exists", protocol_path.is_file())
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.is_file()
        else {}
    )
    add(
        "protocol_status",
        protocol.get("status") == "registered_before_launch",
        protocol.get("status"),
    )
    plan_path = out / "screening_plan.original.yaml"
    add("handoff_copy_exists", plan_path.is_file())
    if plan_path.is_file():
        add(
            "handoff_hash",
            protocol.get("handoff_plan_sha256") == hash_file(plan_path),
            protocol.get("handoff_plan_sha256"),
        )

    smoke_path = out / "smoke/summary.json"
    smoke = (
        json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke_path.is_file()
        else {}
    )
    add("smoke_valid", smoke.get("valid") is True, smoke.get("total_half_games"))
    add(
        "smoke_half_games",
        smoke.get("total_half_games") == 48,
        smoke.get("total_half_games"),
    )

    aggregate_path = out / "aggregate/metrics.json"
    metrics = (
        json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate_path.is_file()
        else {}
    )
    add("aggregate_exists", aggregate_path.is_file())
    screen_total = 0
    replay_results = {}
    for candidate_id in CANDIDATE_IDS:
        group_dir = out / "screen" / candidate_id
        batch = group_dir / "batch"
        verification_path = group_dir / "verification.json"
        group_ok = True
        detail = {}
        if not batch.is_dir() or not verification_path.is_file():
            add(f"{candidate_id}_group_exists", False)
            continue
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
        games = audit_batch(batch)
        screen_total += summary.get("completed_games", 0)
        config = manifest["config"]
        models = manifest["models"]
        decision_audit = verify_group_decisions(batch, config, models)
        detail = {
            "completed_games": summary.get("completed_games"),
            "expected_games": summary.get("expected_games"),
            "seed_clusters": len(config["engine"]["seeds"]),
            "verification_valid": verification.get("valid"),
            "decision_audit_valid": decision_audit["valid"],
            "decision_calibration_mismatches": decision_audit[
                "calibration_mismatches"
            ],
            "protocol_reference": manifest.get("protocol_registration"),
        }
        group_ok = (
            verification.get("valid") is True
            and summary.get("valid") is True
            and summary.get("completed_games") == 1024
            and len(games) == 1024
            and len(config["engine"]["seeds"]) == 128
            and decision_audit["valid"]
        )
        add(f"{candidate_id}_group_valid", group_ok, detail)
        if args.replay:
            from rilab.formal import replay_recorded_games

            replay = replay_recorded_games(batch, manifest, games, workers=args.workers)
            replay_results[candidate_id] = replay
            add(
                f"{candidate_id}_offline_replay",
                replay.get("valid") is True
                and replay.get("games") == 1024
                and replay.get("model_calls") == 0,
                {key: replay.get(key) for key in ("games", "decisions", "model_calls")},
            )
    add("screen_total_half_games", screen_total == 3072, screen_total)
    if metrics:
        groups = metrics["groups"]
        recomputed = recommendation(
            groups,
            engineering_ok=smoke.get("valid") is True,
            artifact_ok=all(
                group.get("verification", {}).get("valid") is True
                for group in groups.values()
            ),
        )
        add(
            "recommendation_matches",
            recomputed["selected"] == metrics["recommendation"]["selected"],
            {
                "recorded": metrics["recommendation"]["selected"],
                "recomputed": recomputed["selected"],
            },
        )
        add(
            "cross_candidate_rule",
            set(metrics.get("cross_candidate", {}))
            == {
                f"{left}_minus_{right}"
                for left in CANDIDATE_IDS
                for right in CANDIDATE_IDS
                if left != right
            },
        )
    payload = {
        "valid": not errors,
        "output_dir": str(out),
        "checks": checks,
        "errors": errors,
        "offline_replay": replay_results,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
