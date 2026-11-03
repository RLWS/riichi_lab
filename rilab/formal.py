"""Eligibility and deterministic offline replay for preregistered strength runs."""

from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from riichienv import GameRule, RiichiEnv

from rilab.adapters import create_adapter
from rilab.agent import MjaiBotAgent
from rilab.artifacts import encode
from rilab.strength import validate_plan

ROOT = Path(__file__).resolve().parents[1]


def committed_bytes(commit, name):
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed_hash_matches(commit, name, expected):
    raw = committed_bytes(commit, name)
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {
        hashlib.sha256(b).hexdigest() for b in (raw, lf, lf.replace(b"\n", b"\r\n"))
    }


def validate_completed_registration(run, manifest):
    """Use the sealed artifact, never today's mutable protocol/config defaults."""
    run = Path(run)
    config = deepcopy(manifest["config"])
    ref = config["evaluation"].pop("strength_registration", None)
    registration_path = run.parent / "protocol.registered.json"
    if not ref or sha(registration_path) != ref["sha256"]:
        raise ValueError("missing or changed preregistration artifact")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if (
        registration["status"] != "registered_before_launch"
        or not datetime.fromisoformat(registration["registered_at"]).tzinfo
    ):
        raise ValueError("invalid preregistration status or timestamp")
    if (
        hashlib.sha256(encode(config).encode()).hexdigest()
        != registration["runtime_config_sha256_before_registration_ref"]
    ):
        raise ValueError("runtime configuration differs from sealed preregistration")
    plan, planned = registration["plan"], registration["planned_config"]
    validate_plan(plan, config)
    validate_plan(plan, planned)
    normalized = deepcopy(config)
    for name, backend in normalized["backends"].items():
        backend["api_base"] = planned["backends"][name]["api_base"]
        backend.pop("deployment", None)
    if normalized != planned or manifest["models"] != registration["model_metadata"]:
        raise ValueError("planned identities or sealed model metadata changed")
    if manifest.get("rilab_dirty") is not False:
        raise ValueError("formal run requires a clean pinned code commit")
    commit = manifest["rilab_commit"]
    import yaml

    for name, expected_hash, expected_value in (
        (
            "configs/arena_strength.yaml",
            registration["planned_config_file_sha256"],
            planned,
        ),
        ("configs/strength_protocol.yaml", registration["plan_file_sha256"], plan),
    ):
        if (
            not committed_hash_matches(commit, name, expected_hash)
            or yaml.safe_load(committed_bytes(commit, name)) != expected_value
        ):
            raise ValueError("sealed plan/config does not match clean run commit")
    for name, expected in manifest["runtime_source_sha256"].items():
        if not committed_hash_matches(commit, name, expected):
            raise ValueError(f"run source does not match pinned commit: {name}")
        # Analysis modules may evolve; decision/replay modules must match the run.
        if (
            name not in ("rilab/evaluation.py", "rilab/cli.py")
            and sha(ROOT / name) != expected
        ):
            raise ValueError(f"current replay implementation differs from run: {name}")
    for name, expected in registration["source_sha256"].items():
        relative = (
            "rilab/strength.py"
            if name == "strength.py"
            else "scripts/run_a1_integration.py"
            if name == "run_a1_integration.py"
            else None
        )
        if relative is None or not committed_hash_matches(commit, relative, expected):
            raise ValueError("registration implementation not anchored to run commit")
    for name, backend in config["backends"].items():
        deployment = backend["deployment"]
        if (
            deployment != registration["deployment"][name]
            or deployment.get("source_dirty") is not False
            or deployment["verified_weight_sha256"] != backend["model_identity"]
            or sha(run.parent / f"{name}.zip") != deployment["archive_sha256"]
        ):
            raise ValueError("deployment/archive/weight identity mismatch")
        response = manifest["models"][name]["response"]
        bias = backend["response"]["pass_bias"]
        calibration = {
            "pass_bias": response["default_pass_bias"] if bias is None else bias,
            "no_yaku_call_penalty": response["no_yaku_call_penalty"],
        }
        if calibration != plan["expected_calibration"][name]:
            raise ValueError("effective calibration differs from preregistration")
    verification = json.loads(
        (run.parent / "verification.json").read_text(encoding="utf-8")
    )
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if (
        verification.get("valid") is not True
        or verification.get("runs") != 1
        or verification.get("games_per_run") != plan["design"]["half_game_budget"]
        or verification.get("digests") != [summary["replay_digest"]]
    ):
        raise ValueError("launcher completion verification mismatch")
    for name, backend in config["backends"].items():
        v = verification["deployment"][name]
        if (
            v["revision"] != backend["revision"]
            or v["verified_weight_sha256"] != backend["model_identity"]
            or v["source_dirty"] is not False
        ):
            raise ValueError("launcher deployment verification mismatch")
    for name, version in manifest["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("offline replay dependency version differs from run")
    return {
        "eligible": True,
        "registration": registration,
        "registration_sha256": sha(registration_path),
        "checks": {
            key: True
            for key in (
                "registration_hash_anchored_in_manifest",
                "runtime_config_sealed",
                "independent_fixed_seed_budget",
                "clean_run_sources_match_commit",
                "registration_code_precedes_launch_in_pinned_source",
                "replay_sources_unchanged",
                "deployment_archives_and_weight_identities",
                "effective_calibration",
                "single_complete_run_verified",
                "replay_dependency_versions",
            )
        },
    }


class RecordedBackend:
    """Replay recorded selections only; contains no HTTP client or model."""

    def __init__(self, name, config, metadata):
        self.backend_id = name
        self.profile = config["adapter_profile"]
        self.metadata = metadata
        self.record = None
        self.requests = 0

    def predict(self, task, state, candidates):
        record = self.record
        result = record["prediction"]
        if (
            record["task"] != task
            or record["candidates"] != candidates
            or result is None
            or result["task"] != task
            or result["model"] != self.metadata[task]
            or result["selected_action"] not in candidates
        ):
            raise ValueError("recorded model request/result differs from replay")
        self.requests += 1
        return result


_REPLAY_CONTEXT = None


def _set_replay_context(run, manifest):
    global _REPLAY_CONTEXT
    _REPLAY_CONTEXT = (Path(run), manifest)


def _replay_one(game):
    import gzip

    run, manifest = _REPLAY_CONTEXT
    config = manifest["config"]
    backends = {
        name: RecordedBackend(name, backend, manifest["models"][name])
        for name, backend in config["backends"].items()
    }
    agents = [
        MjaiBotAgent(
            p,
            backends[name],
            create_adapter(config["backends"][name]["adapter_profile"]),
        )
        for p, name in enumerate(game["seats"])
    ]
    env = RiichiEnv(
        game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=game["seed"]
    )
    observations = env.reset()
    digest = hashlib.sha256()
    step = count = 0
    cursors = [0] * 4
    with gzip.open(
        run / "decisions" / f"{game['game_id']}.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        while not env.done():
            actions = {}
            for p, obs in sorted(observations.items()):
                line = stream.readline()
                if not line:
                    raise ValueError(
                        "decision stream ended before engine terminal state"
                    )
                record = json.loads(line)
                if (
                    record["player_id"] != p
                    or record["step"] != step
                    or record["decision_id"] != count
                    or record["fallback"] is not False
                    or record["backend_id"] != game["seats"][p]
                    or record["event_range"]
                    != [cursors[p], cursors[p] + len(record["new_events"])]
                ):
                    raise ValueError("recorded decision sequencing differs from replay")
                cursors[p] = record["event_range"][1]
                backends[game["seats"][p]].record = record
                actions[p] = agents[p].act(obs)
                actual = agents[p].last_decision
                for key, value in actual.items():
                    if key in record and record[key] != value:
                        raise ValueError(
                            f"{game['game_id']} decision {count}: replay differs in {key}"
                        )
                stable = {
                    k: v
                    for k, v in record.items()
                    if k not in ("prediction", "elapsed_ms")
                }
                stable["state"] = actual["state"]
                digest.update((encode(stable) + "\n").encode())
                count += 1
            observations = env.step(actions)
            for p in actions:
                agents[p].confirm_action(accepted=True)
            step += 1
            if step > 10000:
                raise ValueError("offline replay step limit exceeded")
        if stream.readline():
            raise ValueError("extra decision after engine terminal state")
    event_digest = hashlib.sha256()
    with (run / "games" / f"{game['game_id']}.mjai.jsonl").open("rb") as file:
        first = file.readline()
    newline = b"\r\n" if first.endswith(b"\r\n") else b"\n"
    for event in env.mjai_log:
        event_digest.update(encode(event).encode("utf-8") + newline)
    if (
        digest.hexdigest() != game["decision_digest"]
        or event_digest.hexdigest() != game["event_digest"]
        or count != game["decisions"]
        or step != game["steps"]
        or list(env.scores()) != game["scores"]
        or list(env.ranks()) != game["ranks"]
        or {name: b.requests for name, b in backends.items()}
        != game["backend_requests"]
    ):
        raise ValueError(f"offline replay output/digest mismatch: {game['game_id']}")
    return {
        "game_id": game["game_id"],
        "decisions": count,
        "decision_digest": digest.hexdigest(),
        "event_digest": event_digest.hexdigest(),
    }


def replay_recorded_games(run, manifest, games, *, workers=None):
    """Reconstruct all inputs in isolated CPU workers, with no model inference."""
    from concurrent.futures import ProcessPoolExecutor
    import os

    if workers is None:
        workers = min(4, os.cpu_count() or 1) if len(games) > 8 else 1
    if type(workers) is not int or workers <= 0:
        raise ValueError("invalid replay worker count")
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_set_replay_context,
            initargs=(str(run), manifest),
        )
        rows = pool.map(_replay_one, games, chunksize=1)
    else:
        _set_replay_context(run, manifest)
        rows = map(_replay_one, games)
    total = 0
    per_game = []
    try:
        for index, row in enumerate(rows):
            total += row["decisions"]
            per_game.append(row)
            if (index + 1) % 100 == 0 or index + 1 == len(games):
                print(
                    encode(
                        {
                            "offline_replay_games": index + 1,
                            "total": len(games),
                            "decisions": total,
                        }
                    ),
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return {
        "valid": True,
        "games": len(games),
        "decisions": total,
        "model_calls": 0,
        "workers": workers,
        "all_full_projected_inputs_and_action_bindings_verified": True,
        "event_and_terminal_scores_ranks_verified": True,
        "per_game_digest_sha256": hashlib.sha256(encode(per_game).encode()).hexdigest(),
    }
