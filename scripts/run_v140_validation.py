"""Run the W2 Turn x Response 2x2 local component experiment.

Examples:
  python scripts/run_v140_validation.py --plan configs/v1_4_0_validation/w2_plan.yaml --stage 2x2 --smoke --output-dir outputs/w2_v140_smoke --dry-run
  python scripts/run_v140_validation.py --plan configs/v1_4_0_validation/w2_plan.yaml --stage 2x2 --smoke --output-dir outputs/w2_v140_smoke
  python scripts/run_v140_validation.py --plan configs/v1_4_0_validation/w2_plan.yaml --stage 2x2 --formal --output-dir outputs/w2_v140_20261118

The runner starts the two native services unless --v130-api-base and
--v140-api-base are supplied.  It pre-registers the four arms and schedule
before the first game, then runs the fixed schedule, audits the artifacts and
(for the formal stage) writes report.json/report.md.  It never changes the
model, calibration, seeds, arms or thresholds after seeing results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.artifacts import audit_batch  # noqa: E402
from rilab.backends import create_backend  # noqa: E402
from rilab.batch import run_batch  # noqa: E402
from rilab.w2 import (  # noqa: E402
    ARM_IDS,
    analyze_batch,
    build_backends,
    build_schedule,
    load_plan,
    native_config,
    registration,
    render_report,
    sha256_file,
    validate_plan,
    verify_smoke,
    write_json,
)

SERVE = ROOT / "scripts/serve_integration_backend.py"


class RunningService:
    def __init__(self, service_id, api_base, process, handle, log_path, resolved_path):
        self.service_id = service_id
        self.api_base = api_base
        self.process = process
        self.handle = handle
        self.log_path = log_path
        self.resolved_path = resolved_path

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.handle.close()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def start_service(service_id: str, service: dict, output_dir: Path) -> RunningService:
    source = _resolve(service["source_root"])
    weights = _resolve(service["weights_root"])
    config = _resolve(service["service_config"])
    port = free_port()
    services_dir = output_dir / "services"
    services_dir.mkdir(parents=True, exist_ok=True)
    resolved = services_dir / f"{service_id}.resolved.yaml"
    log = services_dir / f"{service_id}.log"
    cmd = [
        sys.executable,
        str(SERVE),
        "--source-root",
        str(source),
        "--weights-root",
        str(weights),
        "--port",
        str(port),
        "--service-config",
        str(config),
        "--resolved-config-out",
        str(resolved),
    ]
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    api_base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{service_id} service exited during startup:\n"
                    + log.read_text(encoding="utf-8", errors="replace")[-4000:]
                )
            try:
                response = httpx.get(api_base + "/v1/models", timeout=2.0)
                if response.status_code == 200:
                    return RunningService(
                        service_id, api_base, process, handle, log, resolved
                    )
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise TimeoutError(
            f"{service_id} service startup timed out:\n"
            + log.read_text(encoding="utf-8", errors="replace")[-4000:]
        )
    except BaseException:
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        finally:
            handle.close()
        raise


def _service_records(services: list[RunningService]) -> dict:
    return {
        service.service_id: {
            "api_base": service.api_base,
            "pid": service.process.pid,
            "log_path": str(service.log_path),
            "resolved_config": str(service.resolved_path),
            "resolved_config_sha256": sha256_file(service.resolved_path)
            if service.resolved_path.is_file()
            else None,
        }
        for service in services
    }


def _write_verification(
    stage_dir: Path,
    *,
    plan: dict,
    stage: str,
    purpose: str,
    summary: dict,
    audit_games: list[dict],
    registration_sha: str,
    services: dict,
) -> dict:
    fallbacks = sum(int(game.get("fallbacks", 0)) for game in summary["games"])
    valid = (
        summary.get("valid") is True
        and summary.get("completed_games") == summary.get("expected_games")
        and fallbacks == 0
        and len(audit_games) == summary.get("expected_games")
    )
    manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    verification = {
        "schema_version": 1,
        "valid": valid,
        "stage": stage,
        "purpose": purpose,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_games": summary.get("expected_games"),
        "completed_games": summary.get("completed_games"),
        "fallbacks": fallbacks,
        "audit_games": len(audit_games),
        "smoke_seeds": list(plan["smoke_seeds"]),
        "formal_seeds": list(plan["formal_seeds"]),
        "protocol_registration_sha256": registration_sha,
        "replay_digest": summary.get("replay_digest"),
        "models": manifest.get("models"),
        "services": services,
    }
    write_json(stage_dir / "verification.json", verification)
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--stage", default="2x2", choices=["2x2"])
    stage_flag = parser.add_mutually_exclusive_group(required=True)
    stage_flag.add_argument("--smoke", action="store_true")
    stage_flag.add_argument("--formal", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--v130-api-base")
    parser.add_argument("--v140-api-base")
    args = parser.parse_args(argv)

    plan = load_plan(args.plan)
    validate_plan(plan)
    seeds = plan["smoke_seeds"] if args.smoke else plan["formal_seeds"]
    schedule = build_schedule(plan, seeds)
    stage = "W2-smoke" if args.smoke else "W2"
    purpose = "w2_turn_response_2x2_smoke" if args.smoke else "w2_turn_response_2x2"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "stage": stage,
                    "purpose": purpose,
                    "seed_clusters": len(seeds),
                    "arms": list(ARM_IDS),
                    "games": len(schedule),
                    "smoke_seeds": plan["smoke_seeds"],
                    "formal_seeds_count": len(plan["formal_seeds"]),
                    "service_configs": {
                        name: str(_resolve(service["service_config"]))
                        for name, service in plan["services"].items()
                    },
                },
                indent=2,
            )
        )
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir / ("smoke" if args.smoke else "batch")
    if stage_dir.exists():
        parser.error(f"stage directory already exists: {stage_dir}")
    if args.formal:
        smoke_root = (
            args.smoke_dir.resolve() if args.smoke_dir else output_dir / "smoke"
        )
        smoke_verification = smoke_root / "verification.json"
        if not smoke_verification.is_file():
            parser.error("formal stage requires a valid smoke verification.json")
        verify_smoke(smoke_verification, plan)

    if bool(args.v130_api_base) != bool(args.v140_api_base):
        parser.error("provide both --v130-api-base and --v140-api-base, or neither")
    services = []
    try:
        if args.v130_api_base and args.v140_api_base:
            api_bases = {
                "v130": args.v130_api_base.rstrip("/"),
                "v140": args.v140_api_base.rstrip("/"),
            }
        else:
            services = []
            services.append(start_service("v130", plan["services"]["v130"], output_dir))
            services.append(start_service("v140", plan["services"]["v140"], output_dir))
            api_bases = {service.service_id: service.api_base for service in services}
        backends = build_backends(plan, api_bases)
        config = native_config(plan, backends)
        write_json(output_dir / "native_config.json", config)
        reg = registration(
            plan,
            args.plan,
            {
                name: service["service_config"]
                for name, service in plan["services"].items()
            },
            schedule,
            stage=stage,
        )
        reg_path = output_dir / "protocol.registered.json"
        write_json(reg_path, reg)
        manifest_extra = {
            "protocol_registration": {
                "path": str(reg_path),
                "sha256": sha256_file(reg_path),
            }
        }
        summary = run_batch(
            config,
            stage_dir,
            backend_factory=create_backend,
            manifest_extra=manifest_extra,
            schedule=schedule,
            stage=stage,
            purpose=purpose,
        )
        audit_games = audit_batch(stage_dir)
        external_services = (
            {
                name: {"api_base": base, "external": True}
                for name, base in api_bases.items()
            }
            if not services
            else {}
        )
        service_records = _service_records(services) | external_services
        verification = _write_verification(
            stage_dir,
            plan=plan,
            stage=stage,
            purpose=purpose,
            summary=summary,
            audit_games=audit_games,
            registration_sha=sha256_file(reg_path),
            services=service_records,
        )
        if not verification["valid"]:
            raise RuntimeError("W2 run verification is not valid")
        if args.smoke:
            print(json.dumps(verification, indent=2))
            return 0
        result = analyze_batch(stage_dir, plan)
        write_json(output_dir / "report.json", result)
        (output_dir / "report.md").write_text(
            render_report(result), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "stage": stage,
                    "games": result["completed_games"],
                    "seed_clusters": result["seed_clusters"],
                    "primary": result["primary"],
                    "components": result["components"],
                    "derived": result["derived"],
                    "report": str(output_dir / "report.json"),
                },
                indent=2,
            )
        )
        return 0
    finally:
        for service in services:
            service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
