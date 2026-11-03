"""A3 sequential HTTP batch runner with bounded synchronous artifacts."""

from copy import deepcopy
import gzip
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import time

from riichienv import GameRule, RiichiEnv

from rilab.adapters import create_adapter
from rilab.agent import MjaiBotAgent
from rilab.artifacts import audit_batch, encode, write_json
from rilab.backends.http_backend import BackendRegistry, HttpPredictBackend
from rilab.decision import AdapterError
from rilab.scheduling import expand_schedule
from rilab.selfplay import warmup_backends


def run_batch(config, output_dir, *, backend_factory=HttpPredictBackend):
    schedule = expand_schedule(config)
    if (
        config["engine"]["game_mode"] != "4p-red-half"
        or config["engine"]["rule"] != "default_tenhou"
    ):
        raise ValueError("unsupported engine rules")
    policy = config.get("artifacts", {})
    limit = policy.get("max_bytes", 2_000_000_000)
    stride = policy.get("snapshot_stride", 100)
    if type(limit) is not int or limit <= 0 or type(stride) is not int or stride <= 0:
        raise ValueError("invalid artifact limits")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("games", "decisions", "results"):
        (out / folder).mkdir()
    summary = {
        "stage": "A3",
        "valid": False,
        "expected_games": len(schedule),
        "completed_games": 0,
        "games": [],
        "fallbacks": 0,
        "purpose": "engineering_acceptance_only",
    }
    registry = None
    started = time.monotonic()
    try:
        rule = GameRule.default_tenhou()
        root = Path(__file__).resolve().parents[1]
        manifest = {
            "stage": "A3",
            "config": deepcopy(config),
            "schedule": schedule,
            "rilab_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "rilab_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root
                ).strip()
            ),
            "runtime_source_sha256": {
                path.relative_to(root).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted((root / "rilab").rglob("*.py"))
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("riichienv", "httpx", "PyYAML", "websockets")
            },
            "rule": {
                key: getattr(rule, key)
                for key in dir(rule)
                if not key.startswith("_") and isinstance(getattr(rule, key), bool)
            },
            "wall_identity": None,
            "pairing_boundary": "same initial seed; calls/kan may change subsequent draws; no common future wall guarantee",
            "known_exceptions": [
                "DEF-001: frozen training samplers use incorrect red chankan target; no A3 waiver of other failures"
            ],
            "logging": {
                "queue_capacity": 0,
                "mode": "synchronous_backpressure",
                "max_bytes": limit,
                "snapshot_stride": stride,
                "error_game": "full compressed decision spool",
            },
            "weight_verification": "expected hashes only; HTTP metadata cannot prove remote file hashes; see deployment evidence",
        }
        write_json(out / "manifest.json", manifest)
        registry = BackendRegistry(config["backends"], factory=backend_factory)
        warmup_backends(registry)
        manifest["models"] = {n: b.metadata for n, b in registry.backends.items()}
        write_json(out / "manifest.json", manifest)
        used_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        for spec in schedule:
            result = _run_game(
                config,
                registry,
                spec,
                out,
                stride,
                full=policy.get("full_snapshots_first_run", True),
                budget=limit - used_bytes,
            )
            summary["games"].append(result)
            summary["completed_games"] += 1
            used_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            if used_bytes > limit:
                raise AdapterError("artifact capacity exceeded")
            write_json(out / "summary.json", summary)
            print(
                encode(
                    {
                        "completed": summary["completed_games"],
                        "total": len(schedule),
                        "game_id": spec["game_id"],
                    }
                ),
                flush=True,
            )
        for backend in registry.backends.values():
            backend.check_unchanged()
        if audit_batch(out) != summary["games"]:
            raise AdapterError("persisted results differ from completed schedule")
        summary["valid"] = len(summary["games"]) == len(schedule)
        summary["replay_digest"] = hashlib.sha256(
            encode(summary["games"]).encode()
        ).hexdigest()
        return summary
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        try:
            write_json(out / "summary.json", summary)
        finally:
            if registry is not None:
                registry.close()


def _run_game(config, registry, spec, out, stride, *, full, budget):
    game_id = spec["game_id"]
    spool = out / "decisions" / f"{game_id}.full.jsonl.gz"
    sampled = out / "decisions" / f"{game_id}.jsonl.gz"
    result = {**spec, "valid": False, "steps": 0, "decisions": 0, "fallbacks": 0}
    env = RiichiEnv(
        game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=spec["seed"]
    )
    agents = [
        MjaiBotAgent(
            p,
            registry[name],
            create_adapter(config["backends"][name]["adapter_profile"]),
        )
        for p, name in enumerate(spec["seats"])
    ]
    digest = hashlib.sha256()
    written = 0
    seat_events = [0] * 4
    request_start = {
        name: backend.requests for name, backend in registry.backends.items()
    }
    try:
        observations = env.reset()
        with (
            gzip.open(spool, "wt", encoding="utf-8") as complete,
            gzip.open(sampled, "wt", encoding="utf-8") as compact,
        ):
            while not env.done():
                actions = {}
                for p, obs in sorted(observations.items()):
                    tick = time.monotonic()
                    try:
                        actions[p] = agents[p].act(obs)
                    except BaseException as exc:
                        complete.write(
                            encode(
                                {
                                    "decision_id": result["decisions"],
                                    "player_id": p,
                                    "error": str(exc),
                                    "observation": obs.serialize_to_base64(),
                                }
                            )
                            + "\n"
                        )
                        raise
                    record = {
                        "decision_id": result["decisions"],
                        "step": result["steps"],
                        **agents[p].last_decision,
                        "fallback": False,
                    }
                    record["event_range"] = [
                        seat_events[p],
                        seat_events[p] + len(record["new_events"]),
                    ]
                    seat_events[p] = record["event_range"][1]
                    # Stable digest includes every projected input, candidate, selection and physical action.
                    stable = {k: v for k, v in record.items() if k != "prediction"}
                    digest.update((encode(stable) + "\n").encode())
                    record["elapsed_ms"] = 1000 * (time.monotonic() - tick)
                    text = encode(record) + "\n"
                    complete.write(text)
                    if full or result["decisions"] % stride == 0:
                        compact.write(text)
                    else:
                        compact.write(
                            encode({k: v for k, v in record.items() if k != "state"})
                            + "\n"
                        )
                    written += len(text.encode())
                    # Conservative uncompressed bound also caps a single in-flight record/spool.
                    if 2 * written > budget:
                        raise AdapterError("artifact capacity exceeded during game")
                    result["decisions"] += 1
                observations = env.step(actions)
                for p in actions:
                    agents[p].confirm_action(accepted=True)
                result["steps"] += 1
                if result["steps"] > 10000:
                    raise AdapterError("game exceeded step limit")
        for backend in registry.backends.values():
            backend.check_unchanged()
        result.update(
            valid=True,
            scores=list(env.scores()),
            ranks=list(env.ranks()),
            decision_digest=digest.hexdigest(),
            backend_requests={
                name: backend.requests - request_start[name]
                for name, backend in registry.backends.items()
            },
        )
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (out / "games" / f"{game_id}.mjai.jsonl").open(
            "w", encoding="utf-8"
        ) as log:
            for event in env.mjai_log:
                log.write(encode(event) + "\n")
        result["event_digest"] = hashlib.sha256(
            (out / "games" / f"{game_id}.mjai.jsonl").read_bytes()
        ).hexdigest()
        write_json(out / "results" / f"{game_id}.json", result)
    if result["valid"]:
        spool.unlink()
    return result
