"""One WebSocket connection per game; deliberately no reconnect or ranked loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import websockets
import yaml

from rilab.adapters import create_adapter
from rilab.backends.http_backend import AsyncHttpPredictBackend
from rilab.online.policy import OnlinePolicy
from rilab.online.session import OnlineSession, SessionState


async def connect_session(endpoint, token, session, *, connector=websockets.connect):
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in ("ws", "wss")
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid WebSocket endpoint (credentials must be separate)")
    if not token:
        raise ValueError("platform token is empty")
    if token not in session.secrets:
        session.secrets += (token,)
    try:
        async with connector(
            endpoint,
            additional_headers={"Authorization": f"Bearer {token}"},
            max_queue=16,
            open_timeout=10,
            close_timeout=2,
        ) as ws:
            return await session.run(ws)
    except Exception as exc:
        session.state = SessionState.Disconnected
        session.fault("connection_" + type(exc).__name__)
        return session.summary()


def credentials(config):
    online, backend_config = config["online"], config["backend"]
    token_env = online.get("token_env", "RIICHI_DEV_BOT_TOKEN")
    if token_env == backend_config.get("api_key_env"):
        raise ValueError(
            "platform token and HTTP key must use independent environment variables"
        )
    token = os.environ[token_env]
    secrets = [token]
    if backend_config.get("api_key_env"):
        secrets.append(os.environ[backend_config["api_key_env"]])
    return token, secrets


def make_session(config, backend, *, validation=True, secrets=(), **limits):
    online = config["online"]
    return OnlineSession(
        OnlinePolicy(
            create_adapter(backend.profile), match_rounds=online.get("match_rounds", 4)
        ),
        backend,
        validation=validation,
        send_reserve_seconds=online["send_reserve_seconds"],
        inbound_estimate_seconds=online["inbound_estimate_seconds"],
        finish_timeout_seconds=online["finish_timeout_seconds"],
        secrets=secrets,
        **limits,
    )


def persist_session(output, session):
    (output / "session.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in session.records),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(session.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def run_validation(config_path, output_dir):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    online, backend_config = config["online"], config["backend"]
    token, secrets = credentials(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    backend = AsyncHttpPredictBackend("online", backend_config)
    session = make_session(config, backend, secrets=secrets)
    try:
        await backend.warmup()
        summary = await connect_session(online["endpoint_validate"], token, session)
    except Exception as exc:
        session.state = SessionState.Disconnected
        session.fault("startup_" + type(exc).__name__)
        summary = session.summary()
    finally:
        await backend.close()
        # B0 keeps a bounded in-memory trace and flushes after connection closure:
        # disk latency never holds up an action. Persistence errors fail the command.
        persist_session(output, session)
    return summary
