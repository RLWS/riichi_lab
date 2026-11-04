"""Run one real B1 validation with a verified, isolated v1.3.0 HTTP deployment."""

from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-mj-root", type=Path, default=ROOT.parent / "train_mj")
    args = parser.parse_args()
    if not os.environ.get("RIICHI_DEV_BOT_TOKEN"):
        raise ValueError("RIICHI_DEV_BOT_TOKEN is required")
    out, train = args.output_dir.resolve(), args.train_mj_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    backend = yaml.safe_load(
        (ROOT / "configs/arena_strength.yaml").read_text(encoding="utf-8")
    )["backends"]["v1_3_0"]
    tag = backend["revision"]["tag"]
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{tag}^{{commit}}"], cwd=train, text=True
    ).strip()
    if commit != backend["revision"]["commit"]:
        raise ValueError("training tag moved")
    archive = out / "source.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", "--output", str(archive), tag],
        cwd=train,
        check=True,
    )
    source = out / "source"
    with zipfile.ZipFile(archive) as z:
        z.extractall(source)
    service_config = yaml.safe_load(
        (source / "configs/api_local.yaml").read_text(encoding="utf-8")
    )
    for task in ("turn", "response"):
        path = train / service_config["models"][task]["checkpoint"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != backend["model_identity"][f"{task}_sha256"]
        ):
            raise ValueError(f"weight hash mismatch: {task}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = yaml.safe_load(
        (ROOT / "configs/arena_online.yaml").read_text(encoding="utf-8")
    )
    config["backend"]["api_base"] = f"http://127.0.0.1:{port}"
    config["backend"].pop(
        "api_key_env", None
    )  # Dedicated loopback-only service uses no HTTP auth.
    config_path = out / "online.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    identity = {
        "revision": backend["revision"],
        "weight_sha256": backend["model_identity"],
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "rilab_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "rilab_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        ),
        "endpoint": config["online"]["endpoint_validate"],
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                *sorted((ROOT / "rilab").rglob("*.py")),
            ]
        },
    }
    (out / "deployment.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    # Never pass the platform credential to the model server.
    server_env = dict(os.environ)
    server_env.pop("RIICHI_DEV_BOT_TOKEN", None)
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
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=server_env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("model service exited; see backend.log")
                try:
                    models = (
                        httpx.get(
                            config["backend"]["api_base"] + "/v1/models", timeout=1
                        )
                        .raise_for_status()
                        .json()
                    )
                    break
                except httpx.HTTPError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("model service startup timed out")
                    time.sleep(0.1)
            (out / "models.before.json").write_text(
                json.dumps(models, indent=2), encoding="utf-8"
            )
            print("Verified model ready; starting one validation session.", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rilab.cli",
                    "arena-online",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(out / "session"),
                ],
                cwd=ROOT,
                timeout=900,
            )
            after = (
                httpx.get(config["backend"]["api_base"] + "/v1/models", timeout=10)
                .raise_for_status()
                .json()
            )
            (out / "models.after.json").write_text(
                json.dumps(after, indent=2), encoding="utf-8"
            )
            if after != models:
                raise RuntimeError("model metadata changed")
            review = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rilab.online.validation",
                    "--session-dir",
                    str(out / "session"),
                    "--output",
                    str(out / "review.json"),
                ],
                cwd=ROOT,
                check=False,
            )
            return result.returncode or review.returncode
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
