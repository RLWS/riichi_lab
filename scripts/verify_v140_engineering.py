"""Audit both 112-game engineering batches and replay every recorded decision."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.artifacts import audit_batch, encode, write_json  # noqa: E402
from rilab.backends.http_backend import HttpPredictBackend  # noqa: E402
from rilab.config import (  # noqa: E402
    PROFILES,
    calibration_contract,
    effective_calibration,
    load_arena_config,
)
from rilab.formal import replay_recorded_games  # noqa: E402
from rilab.scheduling import expand_schedule  # noqa: E402


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run_dir, config_path, *, workers=4):
    run_dir = Path(run_dir)
    planned = load_arena_config(config_path)
    schedule = expand_schedule(planned)
    if planned["engine"]["seeds"] != list(range(40001, 40009)) or len(schedule) != 112:
        raise ValueError("requires the fixed eight-seed, 112-game engineering plan")
    launch = load(run_dir / "verification.json")
    if not (
        launch["valid"]
        and launch["replay_identical"]
        and launch["runs"] == 2
        and launch["games_per_run"] == 112
    ):
        raise ValueError("launcher did not verify two complete identical runs")
    reports, all_games, manifests = {}, [], []
    for label in ("batch", "replay"):
        run = run_dir / label
        manifest, summary = load(run / "manifest.json"), load(run / "summary.json")
        manifests.append(manifest)
        normalized = deepcopy(manifest["config"])
        if set(normalized["backends"]) != set(planned["backends"]):
            raise ValueError("backend registry differs from plan")
        for name, backend in normalized["backends"].items():
            backend.pop("deployment", None)
            backend["api_base"] = planned["backends"][name]["api_base"]
        if normalized != planned or manifest["schedule"] != schedule:
            raise ValueError("run configuration/schedule differs from plan")
        if any(
            importlib.metadata.version(n) != v
            for n, v in manifest["dependencies"].items()
        ):
            raise ValueError("replay dependency versions differ from run")
        if any(
            sha(ROOT / n) != v for n, v in manifest["runtime_source_sha256"].items()
        ):
            raise ValueError("current replay sources differ from run")
        games = audit_batch(run)
        all_games.append(games)
        digest = hashlib.sha256(encode(games).encode()).hexdigest()
        if not (
            summary["valid"]
            and summary["expected_games"] == 112
            and summary["completed_games"] == 112
            and summary["fallbacks"] == 0
            and summary["games"] == games
            and summary["replay_digest"] == digest
            and digest == launch["digests"][len(all_games) - 1]
        ):
            raise ValueError("summary/digest differs from audited results")
        validators = {}
        for name, backend in manifest["config"]["backends"].items():
            deployment = backend["deployment"]
            if not (
                deployment["source_dirty"] is False
                and deployment["source_commit"] == backend["revision"]["commit"]
                and deployment["verified_weight_sha256"] == backend["model_identity"]
                and sha(run_dir / f"{name}.zip") == deployment["archive_sha256"]
            ):
                raise ValueError(f"deployment identity mismatch: {name}")
            service = (
                ROOT / backend["service_config"]
                if backend.get("service_config")
                else run_dir / name / "configs/api_local.yaml"
            )
            if sha(service) != deployment["service_config_sha256"]:
                raise ValueError(f"service configuration changed: {name}")
            # Reuse production validation without opening an HTTP connection.
            validator = HttpPredictBackend.__new__(HttpPredictBackend)
            validator.backend_id, validator.config = name, backend
            validator.expected = PROFILES[backend["adapter_profile"]]
            validator.calibration_contract = calibration_contract(backend)
            validator.metadata = validator._validate_models(
                {"models": manifest["models"][name]}
            )
            validator.calibration = validator._expected_calibration()
            validator.requests = 0
            expected = deepcopy(backend["response"])
            if validator.calibration_contract == "action_biases_v1":
                expected["calibration_contract"] = "action_biases_v1"
            if (
                effective_calibration(backend, validator.metadata["response"])
                != expected
            ):
                raise ValueError(f"effective calibration differs from plan: {name}")
            validators[name] = validator
        events, predictions = Counter(), Counter()
        for game in games:
            for line in (
                (run / "games" / f"{game['game_id']}.mjai.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ):
                events[json.loads(line)["type"]] += 1
            with gzip.open(
                run / "decisions" / f"{game['game_id']}.jsonl.gz",
                "rt",
                encoding="utf-8",
            ) as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("prediction") is not None:
                        name = record["backend_id"]
                        validators[name]._validate_prediction(
                            record["task"], record["candidates"], record["prediction"]
                        )
                        predictions[name] += 1
            if sum(game["backend_requests"].values()) <= 0:
                raise ValueError("missing model requests")
        expected_requests = Counter()
        for game in games:
            expected_requests.update(game["backend_requests"])
        if predictions != expected_requests:
            raise ValueError("prediction records differ from request counters")
        reports[label] = {
            "valid": True,
            "games": len(games),
            "decisions": sum(g["decisions"] for g in games),
            "fallbacks": 0,
            "predictions_audited": dict(predictions),
            "events": dict(events),
            "elapsed_seconds": summary["elapsed_seconds"],
            "offline_replay": replay_recorded_games(
                run, manifest, games, workers=workers
            ),
        }
        print(f"{label}: audited and replayed {len(games)} games", flush=True)
    if (
        all_games[0] != all_games[1]
        or manifests[0]["runtime_source_sha256"]
        != manifests[1]["runtime_source_sha256"]
        or manifests[0]["models"] != manifests[1]["models"]
        or manifests[0]["dependencies"] != manifests[1]["dependencies"]
    ):
        raise ValueError("runs differ in results, code, models or dependencies")
    return {
        "valid": True,
        "purpose": "engineering_acceptance_only; no strength conclusion",
        "seeds": planned["engine"]["seeds"],
        "config_sha256": sha(Path(config_path)),
        "games_per_run": 112,
        "total_games": 224,
        "replay_identical": True,
        "runs": reports,
        "rilab_commit": manifests[0]["rilab_commit"],
        "rilab_dirty": manifests[0]["rilab_dirty"],
        "runtime_source_sha256": manifests[0]["runtime_source_sha256"],
        "dependencies": manifests[0]["dependencies"],
        "known_exceptions": manifests[0]["known_exceptions"],
        "boundary": "Old v1.3.0 retains DEF-001; v3 directed parity passed separately. Recorded replay does not recompute neural logits.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/arena_v140_engineering.yaml"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = verify(args.run_dir, args.config, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"valid": report["valid"], "total_games": report["total_games"]}))


if __name__ == "__main__":
    main()
