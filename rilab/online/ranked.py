"""Ranked batches with optional time limits checked between complete sessions."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from importlib.metadata import version
import platform
import re
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

import websockets
import yaml

from rilab.backends.http_backend import AsyncHttpPredictBackend
from rilab.config import PROFILES
from rilab.online.client import (
    connect_session,
    credentials,
    make_session,
    persist_session,
)
from rilab.online.session import SessionState
from rilab.online.validation import review_session


def positive(value, name, *, zero=False, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero)
    ):
        raise ValueError(f"invalid ranked {name}")


def load_ranked_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("ranked config must be an object")
    online, backend, batch = config["online"], config["backend"], config["ranked"]
    for key, scheme in (("endpoint_ranked", ("ws", "wss")),):
        url = urlsplit(online[key])
        if (
            url.scheme not in scheme
            or not url.netloc
            or url.path != "/ws/ranked"
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("invalid ranked endpoint")
        if url.scheme == "ws" and url.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("ranked credentials require TLS except on loopback")
    api = urlsplit(backend["api_base"])
    if (
        api.scheme not in ("http", "https")
        or not api.netloc
        or api.username
        or api.password
        or api.query
        or api.fragment
    ):
        raise ValueError("invalid ranked backend URL")
    if backend["adapter_profile"] not in PROFILES:
        raise ValueError("unknown adapter profile")
    profile = PROFILES[backend["adapter_profile"]]
    if any(backend["revision"].get(k) != profile[k] for k in ("tag", "commit")):
        raise ValueError("ranked profile/revision mismatch")
    for task in ("turn", "response"):
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(backend["model_identity"].get(task + "_sha256", ""))
        ):
            raise ValueError("ranked deployment requires declared weight SHA256")
    if online["match_rounds"] != 8:
        raise ValueError("ranked requires East-South match_rounds=8")
    if online.get("token_env", "RIICHI_DEV_BOT_TOKEN") == backend.get("api_key_env"):
        raise ValueError("platform and HTTP credentials must be independent")
    for key in ("send_reserve_seconds", "inbound_estimate_seconds"):
        positive(online[key], key, zero=True)
    positive(online["finish_timeout_seconds"], "finish_timeout_seconds")
    positive(backend["timeout_seconds"], "backend timeout")
    positive(batch["max_games"], "max_games", integer=True)
    batch.setdefault("max_total_seconds", 0)
    positive(batch["max_total_seconds"], "max_total_seconds", zero=True)
    for key in (
        "session_timeout_seconds",
        "matchmaking_timeout_seconds",
    ):
        positive(batch[key], key)
    limits = batch["stop"]
    expected = {
        "max_fallbacks",
        "max_defaulted",
        "max_server_elapsed_ms",
        "min_bank_ms",
        "min_deadline_margin_ms",
    }
    if set(limits) != expected:
        raise ValueError("ranked stop thresholds must be explicit and complete")
    for key in expected:
        positive(
            limits[key],
            key,
            zero=key != "max_server_elapsed_ms",
            integer=key in {"max_fallbacks", "max_defaulted"},
        )
    return deepcopy(config)


def stop_reasons(report, records, limits, totals):
    reasons = []
    if not report["protocol_passed"]:
        reasons.append("session_or_protocol_failure")
    if any(
        r["kind"] == "inference_failure" or r.get("reason") == "prediction_error"
        for r in records
    ):
        reasons.append("inference_failure")
    acks = [
        r["message"]
        for r in records
        if r["kind"] == "received" and r["message"]["type"] == "action_ack"
    ]
    if any(
        any(
            type(a.get(key)) not in (int, float)
            or not math.isfinite(a[key])
            or a[key] < 0
            for key in ("elapsed_ms", "bank_ms", "bank_consumed_ms")
        )
        for a in acks
    ):
        reasons.append("missing_or_invalid_ack_timing")
    if totals["fallbacks"] > limits["max_fallbacks"]:
        reasons.append("fallback_limit")
    if totals["defaulted"] > limits["max_defaulted"]:
        reasons.append("defaulted_limit")
    elapsed = report["server_elapsed_ms"]
    if elapsed and elapsed["max"] > limits["max_server_elapsed_ms"]:
        reasons.append("server_latency_limit")
    for field, key, reason in (
        ("minimum_ack_bank_ms", "min_bank_ms", "bank_limit"),
        (
            "minimum_deadline_margin_ms",
            "min_deadline_margin_ms",
            "deadline_margin_limit",
        ),
    ):
        if report[field] is None:
            reasons.append("missing_" + field)
        elif report[field] < limits[key]:
            reasons.append(reason)
    return reasons


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


async def run_ranked(
    config_path,
    output_dir,
    *,
    backend_factory=AsyncHttpPredictBackend,
    connector=websockets.connect,
):
    config = load_ranked_config(config_path)
    token, secrets = credentials(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    backend = None
    session = None
    games = []
    totals = {"fallbacks": 0, "defaulted": 0}
    batch = config["ranked"]
    started = time.perf_counter()
    deadline = (
        started + batch["max_total_seconds"] if batch["max_total_seconds"] else None
    )
    summary = {
        "status": "running",
        "completed": False,
        "attempted": 0,
        "games_started": 0,
        "games_completed": 0,
        "stop_reasons": [],
        "games": games,
        **totals,
    }

    def batch_expired():
        return deadline is not None and time.perf_counter() >= deadline

    def checkpoint():
        summary.update(totals, elapsed_seconds=time.perf_counter() - started)
        write_json(output / "summary.json", summary)

    try:
        backend = backend_factory("ranked", config["backend"])
        sanitizer = make_session(config, backend, validation=False, secrets=secrets)
        root = Path(__file__).resolve().parents[2]
        manifest = sanitizer.redact(
            {
                "mode": "ranked",
                "python": platform.python_version(),
                "dependencies": {
                    name: version(name)
                    for name in ("riichienv", "httpx", "websockets", "PyYAML")
                },
                "model_identity_source": "operator deployment declaration; /v1/models does not verify weight SHA256",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "config_sha256": hashlib.sha256(
                    Path(config_path).read_bytes()
                ).hexdigest(),
                "rilab_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "rilab_dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=root, text=True
                    )
                ),
                "source_sha256": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((root / "rilab").rglob("*.py"))
                },
            }
        )
        write_json(
            output / "manifest.json", manifest
        )  # Seal limits before any connection.
        checkpoint()
        await backend.warmup()
        manifest["models"] = sanitizer.redact(backend.metadata)
        write_json(output / "manifest.json", manifest)
        for index in range(batch["max_games"]):
            if batch_expired():
                summary["stop_reasons"] = ["batch_time_limit"]
                break
            await backend.check_unchanged()
            if batch_expired():
                summary["stop_reasons"] = ["batch_time_limit"]
                break
            game_dir = output / f"game_{index + 1:04d}"
            game_dir.mkdir()
            session = make_session(
                config,
                backend,
                validation=False,
                secrets=secrets,
                start_timeout_seconds=batch["matchmaking_timeout_seconds"],
            )
            summary["attempted"] += 1
            checkpoint()  # If interrupted, never silently reattempt this slot.
            try:
                # The batch limit only gates new connections. An admitted session
                # keeps its full independent timeout, including matchmaking.
                async with asyncio.timeout(batch["session_timeout_seconds"]):
                    await connect_session(
                        config["online"]["endpoint_ranked"],
                        token,
                        session,
                        connector=connector,
                    )
            except TimeoutError:
                session.state = SessionState.Disconnected
                session.fault("session_time_limit")
            except asyncio.CancelledError:
                session.state = SessionState.Disconnected
                session.fault("batch_cancelled")
                raise
            finally:
                summary["games_started"] += int(
                    any(
                        r["kind"] == "received" and r["message"]["type"] == "start_game"
                        for r in session.records
                    )
                )
                summary["games_completed"] += int(session.completed)
                await asyncio.to_thread(persist_session, game_dir, session)
            report = await asyncio.to_thread(
                review_session, game_dir, require_validation=False
            )
            totals["fallbacks"] += len(report["fallback_request_ids"])
            totals["defaulted"] += len(report["defaulted_request_ids"])
            reasons = stop_reasons(report, session.records, batch["stop"], totals)
            write_json(game_dir / "review.json", report)
            games.append(
                {
                    "game": game_dir.name,
                    "protocol_passed": report["protocol_passed"],
                    "stop_reasons": reasons,
                    "trace_sha256": report["trace_sha256"],
                    "summary_sha256": report["summary_sha256"],
                }
            )
            checkpoint()
            await backend.check_unchanged()
            if reasons:
                summary["stop_reasons"] = reasons
                break
        else:
            summary["status"] = "completed_budget"
            summary["completed"] = True
        if summary["status"] == "running":
            summary["status"] = "stopped"
    except asyncio.CancelledError:
        summary.update(
            status="cancelled", completed=False, stop_reasons=["batch_cancelled"]
        )
        raise
    except Exception as exc:
        summary.update(
            status="failed", completed=False, stop_reasons=[type(exc).__name__]
        )
    finally:
        if backend is not None:
            try:
                await backend.close()
            except Exception as exc:
                summary.update(
                    status="failed",
                    completed=False,
                    stop_reasons=["backend_close_" + type(exc).__name__],
                )
        checkpoint()  # Persistence failure propagates; no subsequent connection is possible.
    return summary
