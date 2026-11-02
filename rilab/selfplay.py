"""A1 manual-seat single-game runner. Automatic scheduling belongs to A3."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

from riichienv import GameRule, RiichiEnv, ActionType

from rilab.adapters import create_adapter
from rilab.agent import MjaiBotAgent
from rilab.backends.http_backend import BackendRegistry, HttpPredictBackend
from rilab.decision import AdapterError
from rilab.mapping.action_mapping import bind_candidates
from rilab.tiles import from_physical
from rilab.tracker.state_tracker import StateTracker


def warmup_backends(registry) -> None:
    """Warm both heads from real, isolated initial-window observations."""
    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=1)
    turn = env.reset()[0]
    discard = next(
        a
        for a in turn.legal_actions()
        if a.action_type == ActionType.Discard and from_physical(a.tile) == "3m"
    )
    response = env.step({0: discard})[1]
    for backend in registry.backends.values():
        adapter = create_adapter(backend.profile)
        for obs in (turn, response):
            tracker = StateTracker(obs.player_id)
            tracker.consume(obs.new_events())
            tracker.check_observation(obs)
            context = tracker.context(obs.legal_actions())
            canonical = tracker.snapshot(context)
            bindings = bind_candidates(obs, context, canonical)
            backend.predict(
                context.task,
                adapter.project(canonical),
                [b.candidate for b in bindings],
            )


def run_single_game(
    config: dict,
    *,
    seats: list[str],
    seed: int,
    output_dir: str | Path,
    backend_factory=HttpPredictBackend,
) -> dict:
    if len(seats) != 4 or any(name not in config["backends"] for name in seats):
        raise AdapterError("single game requires four registered backend IDs")
    if (
        config["engine"]["game_mode"] != "4p-red-half"
        or config["engine"]["rule"] != "default_tenhou"
    ):
        raise AdapterError("unsupported engine rules")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    summary = {"valid": False, "seed": seed, "seats": seats, "steps": 0, "decisions": 0}
    registry = None
    env = None
    start = time.monotonic()
    try:
        configs = {
            name: deepcopy(config["backends"][name]) for name in dict.fromkeys(seats)
        }
        registry = BackendRegistry(configs, factory=backend_factory)
        warmup_backends(registry)
        warmup_requests = {
            name: backend.requests for name, backend in registry.backends.items()
        }
        metadata = {
            name: backend.metadata for name, backend in registry.backends.items()
        }
        manifest = {
            "stage": "A1",
            "seed": seed,
            "seats": seats,
            "backends": configs,
            "models": metadata,
            "engine": config["engine"],
            "rilab_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "rilab_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"]).strip()
            ),
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        agents = [
            MjaiBotAgent(
                p, registry[name], create_adapter(configs[name]["adapter_profile"])
            )
            for p, name in enumerate(seats)
        ]
        env = RiichiEnv(
            game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=seed
        )
        observations = env.reset()
        with (out / "decisions.jsonl").open("w", encoding="utf-8") as log:
            while not env.done():
                actions = {}
                for player_id, obs in sorted(observations.items()):
                    tick = time.monotonic()
                    actions[player_id] = agents[player_id].act(obs)
                    record = {
                        "step": summary["steps"],
                        **agents[player_id].last_decision,
                        "elapsed_ms": 1000 * (time.monotonic() - tick),
                    }
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                    summary["decisions"] += 1
                observations = env.step(actions)
                for player_id in actions:
                    agents[player_id].confirm_action(accepted=True)
                summary["steps"] += 1
                if summary["steps"] > 10000:
                    raise AdapterError("single game exceeded step limit")
        for backend in registry.backends.values():
            backend.check_unchanged()
        summary.update(
            valid=True,
            scores=list(env.scores()),
            ranks=list(env.ranks()),
            backend_requests={
                name: b.requests - warmup_requests[name]
                for name, b in registry.backends.items()
            },
            warmup_requests=warmup_requests,
        )
        return summary
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        try:
            if env is not None:
                with (out / "game.mjai.jsonl").open("w", encoding="utf-8") as f:
                    for event in env.mjai_log:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")
            (out / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        finally:
            if registry:
                registry.close()
