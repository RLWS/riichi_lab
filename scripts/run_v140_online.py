"""Activate a pending v3 bot, then run the configured bounded ranked batch."""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.backends.http_backend import HttpPredictBackend  # noqa: E402
from rilab.online.client import run_validation  # noqa: E402
from rilab.online.ranked import load_ranked_config, run_ranked  # noqa: E402
from rilab.online.validation import review_session  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def loopback_config(config):
    """The dedicated local server disables HTTP auth; preserve the deployment config."""
    result = deepcopy(config)
    result["backend"].pop("api_key_env", None)
    return result


def validation_config(config):
    result = deepcopy(config)
    result["online"]["endpoint_validate"] = "wss://game.riichi.dev/ws/validate"
    result["online"]["match_rounds"] = 4
    result.pop("ranked", None)
    return result


def validation_passed(review):
    return bool(
        review["protocol_passed"]
        and not review["fallback_request_ids"]
        and not review["defaulted_request_ids"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/arena_ranked_v140.yaml"
    )
    parser.add_argument("--train-mj-root", type=Path, default=ROOT.parent / "train_mj")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--token-stdin",
        action="store_true",
        help="read token from a private pipe, not command arguments",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="only for a bot already confirmed active",
    )
    args = parser.parse_args()
    config = loopback_config(load_ranked_config(args.config))
    backend = config["backend"]
    if backend["adapter_profile"] != "full_state_v3":
        raise ValueError("this launcher requires full_state_v3")
    token_env = config["online"]["token_env"]
    if args.token_stdin:
        os.environ[token_env] = sys.stdin.readline().strip()
    if not os.environ.get(token_env):
        raise ValueError(f"{token_env} is required")
    train, out = args.train_mj_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commit = backend["revision"]["commit"]
    resolved = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=train, text=True
    ).strip()
    if resolved != commit:
        raise ValueError("training commit identity mismatch")
    archive = out / "source.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", "--output", str(archive), commit],
        cwd=train,
        check=True,
    )
    source = out / "source"
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(source)
    service_path = ROOT / backend["service_config"]
    service = yaml.safe_load(service_path.read_text(encoding="utf-8"))
    for task in ("turn", "response"):
        path = train / service["models"][task]["checkpoint"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != backend["model_identity"][task + "_sha256"]
        ):
            raise ValueError(f"weight hash mismatch: {task}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    backend["api_base"] = f"http://127.0.0.1:{port}"
    config_path = out / "ranked.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    validate_path = out / "validation.yaml"
    validate_path.write_text(
        yaml.safe_dump(validation_config(config), sort_keys=False), encoding="utf-8"
    )
    write_json(
        out / "deployment.json",
        {
            "revision": backend["revision"],
            "weight_sha256": backend["model_identity"],
            "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "service_config_sha256": hashlib.sha256(
                service_path.read_bytes()
            ).hexdigest(),
            "rilab_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "rilab_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            ),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        },
    )
    server_env = dict(os.environ)
    server_env.pop(token_env, None)
    state = {
        "status": "starting",
        "completed": False,
        "activation_confirmed": False,
        "ranked_budget": config["ranked"]["max_games"],
    }
    write_json(out / "summary.json", state)
    with (out / "backend.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts/serve_integration_backend.py"),
                "--source-root",
                str(source),
                "--weights-root",
                str(train),
                "--port",
                str(port),
                "--service-config",
                str(service_path),
                "--resolved-config-out",
                str(out / "resolved-service.yaml"),
            ],
            env=server_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("model service exited; see backend.log")
                try:
                    response = httpx.get(backend["api_base"] + "/v1/models", timeout=1)
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("model service startup timed out")
                    time.sleep(0.1)
            connection = HttpPredictBackend("v1_4_0", backend)
            try:
                write_json(out / "models.before.json", connection.metadata)
            finally:
                connection.close()
            if not args.skip_validation:
                state["status"] = "validating"
                write_json(out / "summary.json", state)
                print("Verified frozen model ready; validating bot.", flush=True)
                asyncio.run(
                    asyncio.wait_for(
                        run_validation(validate_path, out / "validation"), timeout=900
                    )
                )
                review = review_session(out / "validation")
                write_json(out / "validation_review.json", review)
                if not validation_passed(review):
                    raise RuntimeError(
                        "validation did not pass cleanly; ranked not started"
                    )
            state.update(activation_confirmed=not args.skip_validation, status="ranked")
            write_json(out / "summary.json", state)
            print(
                f"Starting ranked budget: {config['ranked']['max_games']} games.",
                flush=True,
            )
            ranked = asyncio.run(run_ranked(config_path, out / "ranked"))
            after = (
                httpx.get(backend["api_base"] + "/v1/models", timeout=10)
                .raise_for_status()
                .json()["models"]
            )
            write_json(out / "models.after.json", after)
            if after != json.loads(
                (out / "models.before.json").read_text(encoding="utf-8")
            ):
                raise RuntimeError("model metadata changed during online run")
            state.update(
                status=ranked["status"], completed=ranked["completed"], ranked=ranked
            )
            print(
                json.dumps(
                    {
                        "status": ranked["status"],
                        "games_completed": ranked["games_completed"],
                        "stop_reasons": ranked["stop_reasons"],
                    }
                ),
                flush=True,
            )
            return 0 if ranked["completed"] else 1
        except Exception as exc:
            state.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            write_json(out / "summary.json", state)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            os.environ.pop(token_env, None)


if __name__ == "__main__":
    raise SystemExit(main())
