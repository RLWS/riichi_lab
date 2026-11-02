"""Run a real-weight dual-HTTP A1 game without modifying train_mj checkout.

Use the training development Python (torch/FastAPI/uvicorn/httpx installed).
python scripts/run_a1_integration.py --train-mj-root <repo> --output-dir <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile

import httpx
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mj-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    train = args.train_mj_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(
        (repo / "configs/arena_selfplay.yaml").read_text(encoding="utf-8")
    )
    processes = []
    logs = []
    used_ports = set()
    try:
        for name, backend in config["backends"].items():
            tag = backend["revision"]["tag"]
            commit = subprocess.check_output(
                ["git", "rev-parse", f"{tag}^{{commit}}"], cwd=train, text=True
            ).strip()
            if commit != backend["revision"]["commit"]:
                raise RuntimeError(f"tag moved: {tag}")
            archive = out / f"{name}.zip"
            subprocess.run(
                ["git", "archive", "--format=zip", "--output", str(archive), tag],
                cwd=train,
                check=True,
            )
            source = out / name
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
                    raise RuntimeError(f"weight hash mismatch: {name}/{task}")
            while True:
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    port = s.getsockname()[1]
                if port not in used_ports:
                    break
            used_ports.add(port)
            backend["api_base"] = f"http://127.0.0.1:{port}"
            log = (out / f"{name}.log").open("w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(repo / "scripts/serve_integration_backend.py"),
                    "--source-root",
                    str(source),
                    "--weights-root",
                    str(train),
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            processes.append(process)
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited: see {name}.log")
                try:
                    response = httpx.get(backend["api_base"] + "/v1/models", timeout=1)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name} startup timed out")
                time.sleep(0.1)
        config_path = out / "arena.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        command = [
            str(repo / ".venv/Scripts/python.exe"),
            "-m",
            "rilab.cli",
            "arena-selfplay",
            "--config",
            str(config_path),
            "--seed",
            str(args.seed),
            "--seats",
            "v1_3_0,v1_2_0,v1_2_0,v1_2_0",
            "--output-dir",
            str(out / "game"),
        ]
        subprocess.run(command, cwd=repo, check=True, timeout=300)
        evidence = {
            "seed": args.seed,
            "versions": {name: b["revision"] for name, b in config["backends"].items()},
            "summary": json.loads(
                (out / "game/summary.json").read_text(encoding="utf-8")
            ),
        }
        (out / "verification.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
    finally:
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
