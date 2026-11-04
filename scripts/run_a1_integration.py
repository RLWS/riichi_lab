"""Run a real-weight dual-HTTP A1 game without modifying train_mj checkout.

Use a venv with rilab and CPU inference dependencies installed.
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mj-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/arena_selfplay.yaml",
    )
    parser.add_argument("--batch-runs", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--batch-timeout", type=int, default=7200, help="seconds per batch"
    )
    parser.add_argument(
        "--strength-protocol",
        type=Path,
        help="seal planned strength protocol before first game",
    )
    parser.add_argument(
        "--batch", action="store_true", help="A3: full configured schedule, run twice"
    )
    args = parser.parse_args(argv)
    if args.batch_timeout <= 0:
        parser.error("--batch-timeout must be positive")
    if args.strength_protocol and (not args.batch or args.batch_runs != 1):
        parser.error("strength experiment requires --batch --batch-runs 1")
    repo = Path(__file__).resolve().parents[1]
    train = args.train_mj_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.strength_protocol:
        sys.path.insert(0, str(repo))
        from rilab.strength import validate_plan

        validate_plan(
            yaml.safe_load(args.strength_protocol.read_text(encoding="utf-8")), config
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
            backend["deployment"] = {
                "source": "git archive of verified immutable tag",
                "source_dirty": False,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "verified_weight_sha256": backend["model_identity"].copy(),
            }
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
        if args.strength_protocol:
            from rilab.strength import register_protocol

            metadata = {
                name: httpx.get(b["api_base"] + "/v1/models", timeout=10)
                .raise_for_status()
                .json()["models"]
                for name, b in config["backends"].items()
            }
            register_protocol(
                args.strength_protocol, args.config, config, metadata, out
            )
        config_path = out / "arena.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        command = [
            sys.executable,
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
        if args.batch:
            command = command[: command.index("--seed")]
            summaries = []
            for label in ("batch", "replay") if args.batch_runs == 2 else ("batch",):
                subprocess.run(
                    command + ["--output-dir", str(out / label)],
                    cwd=repo,
                    check=True,
                    timeout=args.batch_timeout,
                )
                summaries.append(
                    json.loads(
                        (out / label / "summary.json").read_text(encoding="utf-8")
                    )
                )
            same = (
                summaries[0]["games"] == summaries[1]["games"]
                if len(summaries) == 2
                else None
            )
            evidence = {
                "valid": all(s["valid"] for s in summaries) and same is not False,
                "runs": len(summaries),
                "replay_identical": same,
                "games_per_run": len(summaries[0]["games"]),
                "digests": [s["replay_digest"] for s in summaries],
                "deployment": {
                    name: {
                        "revision": b["revision"],
                        "verified_weight_sha256": b["model_identity"],
                        "source": "git archive of verified immutable tag",
                        "source_dirty": False,
                    }
                    for name, b in config["backends"].items()
                },
            }
            (out / "verification.json").write_text(
                json.dumps(evidence, indent=2), encoding="utf-8"
            )
            if not evidence["valid"]:
                raise RuntimeError("replay mismatch or invalid batch")
            return
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
