"""Independently audit both real HTTP batches and export compact A3 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rilab.artifacts import audit_batch, write_json  # noqa: E402


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.run_dir
    root = Path(__file__).resolve().parents[1]
    manifests = [load(out / label / "manifest.json") for label in ("batch", "replay")]
    summaries = [load(out / label / "summary.json") for label in ("batch", "replay")]
    results = [audit_batch(out / label) for label in ("batch", "replay")]
    verification = load(out / "verification.json")
    sources = manifests[0]["runtime_source_sha256"]
    unchanged = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == value
        for name, value in sources.items()
    )
    checks = {
        "both_runs_valid": all(s["valid"] for s in summaries),
        "at_least_100_half_games": len(results[0]) >= 100,
        "identical_schedule": manifests[0]["schedule"] == manifests[1]["schedule"],
        "identical_results_and_all_decision_event_digests": results[0] == results[1],
        "persisted_summaries_match": all(
            s["games"] == r for s, r in zip(summaries, results)
        ),
        "identical_runtime_sources": sources == manifests[1]["runtime_source_sha256"],
        "runtime_sources_match_workspace": unchanged,
        "deployment_verified": verification["valid"] is True,
        "zero_fallbacks": all(g["fallbacks"] == 0 for r in results for g in r),
    }
    events = Counter()
    for g in results[0]:
        for line in (
            (out / "batch/games" / f"{g['game_id']}.mjai.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            events[json.loads(line)["type"]] += 1
    requests = Counter()
    for game in results[0]:
        requests.update(game["backend_requests"])
    report = {
        "stage": "A3",
        "valid": all(checks.values()),
        "purpose": "engineering_acceptance_only; no strength conclusion",
        "checks": checks,
        "seeds": manifests[0]["config"]["engine"]["seeds"],
        "games_per_run": len(results[0]),
        "total_games_including_replay": sum(map(len, results)),
        "decisions_per_run": [sum(g["decisions"] for g in r) for r in results],
        "backend_requests_first_run_excluding_warmup": dict(requests),
        "events_first_run": dict(events),
        "elapsed_seconds_per_run": [s["elapsed_seconds"] for s in summaries],
        "replay_digests": [s["replay_digest"] for s in summaries],
        "production_environment": {
            k: manifests[0][k] for k in ("python", "platform", "dependencies", "rule")
        },
        "serving_environment": load(out / "serving_environment.json"),
        "backends": manifests[0]["config"]["backends"],
        "model_metadata": manifests[0]["models"],
        "known_exceptions": manifests[0]["known_exceptions"],
        "tested_source": {
            "base_commit": manifests[0]["rilab_commit"],
            "dirty": manifests[0]["rilab_dirty"],
            "runtime_source_sha256": sources,
        },
        "local_artifacts": str(out.resolve()),
        "first_run_results": results[0],
    }
    write_json(args.output, report)
    print(json.dumps({"valid": report["valid"], "checks": checks}, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
