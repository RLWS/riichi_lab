"""One-click v1.4.0 Step 1 local screening entry point.

Examples (exact interfaces from the handoff prompt):
  python scripts/run_v140_screening.py --handoff-plan <screening_plan.yaml> --stage smoke --output-dir <new-dir> --dry-run
  python scripts/run_v140_screening.py --handoff-plan <screening_plan.yaml> --stage smoke --output-dir <new-dir>
  python scripts/run_v140_screening.py --handoff-plan <screening_plan.yaml> --stage screen --output-dir <new-dir> --dry-run
  python scripts/run_v140_screening.py --handoff-plan <screening_plan.yaml> --stage screen --output-dir <new-dir>

The screen stage expects the same output directory to contain a valid smoke
stage and the sealed protocol.registration artifact.  There is deliberately no
resume mode and no overwrite mode.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.artifacts import audit_batch, encode, write_json  # noqa: E402
from rilab.backends.http_backend import HttpPredictBackend  # noqa: E402
from rilab.config import effective_calibration, load_arena_config  # noqa: E402
from rilab.screening import (  # noqa: E402
    BASELINE_ID,
    CANDIDATES,
    ENGINEERING_SEEDS,
    RESERVED_SEEDS,
    SCREENING_SEEDS,
    analyze_group,
    bootstrap_screening,
    build_native_config,
    build_native_protocol,
    empty_counts,
    load_handoff,
    load_service_configs,
    point_metrics,
    recommendation,
    render_report,
    sha256_file,
    validate_handoff,
    validate_screening_plan,
    verify_group_decisions,
    write_metrics_csv,
    write_per_seed_csv,
)

SERVE = ROOT / "scripts/serve_integration_backend.py"
BATCH = ROOT / "scripts/run_screening_batch.py"
REPLAY = ROOT / "scripts/replay_screening_group.py"
VERIFY_SERVICES = ROOT / "scripts/verify_v140_services.py"
VERIFY_SELECTOR = ROOT / "scripts/verify_v140_selector.py"
CHECK_LEGACY = ROOT / "scripts/check_http_contract.py"
CHECK_CONFIG_CONTRACT = ROOT / "scripts/check_service_config_contract.py"
TRAIN_TESTS = [
    "tests/test_response_calibration.py",
    "tests/test_api_routes.py",
    "tests/test_serving_predictor.py",
    "tests/test_serving_validation.py",
]
CANDIDATE_IDS = [c["id"] for c in CANDIDATES]
CANDIDATE_SOURCE_ID = "candidate_v1_4_0"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CommandLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.sequence = 0

    def _write(self, value: dict) -> None:
        self.sequence += 1
        value = {"sequence": self.sequence, **value}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    def record_inproc(self, name, cwd, started, ended, exit_code, note=None):
        self._write(
            {
                "event": "in_process",
                "name": name,
                "cwd": str(cwd),
                "started_at": started,
                "ended_at": ended,
                "duration_seconds": _seconds(started, ended),
                "exit_code": exit_code,
                "note": note,
            }
        )

    def run(self, cmd, cwd, *, log_path=None, timeout=None, note=None):
        started = utcnow()
        self._write(
            {
                "event": "command_start",
                "cmd": [str(item) for item in cmd],
                "cwd": str(cwd),
                "started_at": started,
                "note": note,
            }
        )
        try:
            if log_path:
                log_path = Path(log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as handle:
                    process = subprocess.run(
                        [str(item) for item in cmd],
                        cwd=str(cwd),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                        text=True,
                    )
            else:
                process = subprocess.run(
                    [str(item) for item in cmd],
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
            self._write(
                {
                    "event": "command_end",
                    "cmd": [str(item) for item in cmd],
                    "cwd": str(cwd),
                    "started_at": started,
                    "ended_at": utcnow(),
                    "duration_seconds": _seconds(started, utcnow()),
                    "exit_code": code,
                    "note": note,
                }
            )
            raise
        self._write(
            {
                "event": "command_end",
                "cmd": [str(item) for item in cmd],
                "cwd": str(cwd),
                "started_at": started,
                "ended_at": utcnow(),
                "duration_seconds": _seconds(started, utcnow()),
                "exit_code": code,
                "note": note,
            }
        )
        return process

    def start_service(self, cmd, cwd, *, log_path, note=None):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        started = utcnow()
        process = subprocess.Popen(
            [str(item) for item in cmd],
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._write(
            {
                "event": "service_start",
                "cmd": [str(item) for item in cmd],
                "cwd": str(cwd),
                "pid": process.pid,
                "started_at": started,
                "log_path": str(log_path),
                "note": note,
            }
        )
        return process, handle, started

    def stop_service(self, process, handle, started, *, note=None):
        code = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        code = process.returncode
        handle.close()
        self._write(
            {
                "event": "service_stop",
                "pid": process.pid,
                "ended_at": utcnow(),
                "duration_seconds": _seconds(started, utcnow()),
                "exit_code": code,
                "note": note,
            }
        )
        return code


def _seconds(start, end):
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        p
        for p in Path(root).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and path_suffix(p) != ".pyc"
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hash_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def path_suffix(path: Path) -> str:
    return path.suffix.lower()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def git_text(cmdlog, cwd, args, note=None):
    result = cmdlog.run(["git", "-C", str(cwd), *args], cwd=cwd, note=note)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_info(cmdlog, root: Path) -> dict:
    root = Path(root)
    head = git_text(cmdlog, root, ["rev-parse", "HEAD"], "repository identity")
    branch = git_text(cmdlog, root, ["branch", "--show-current"], "repository branch")
    status = git_text(
        cmdlog, root, ["status", "--porcelain"], "repository dirty status"
    )
    return {
        "root": str(root.resolve()),
        "commit": head,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def verify_source_refs(cmdlog, train: Path) -> None:
    baseline = git_text(
        cmdlog, train, ["rev-parse", "v1.3.0^{commit}"], "baseline tag"
    )
    expected = "1e5092f344fb69bcdf4697f683f2ecf15b9865d8"
    if baseline != expected:
        raise RuntimeError(f"v1.3.0 moved: {baseline}")
    candidate = "0710aa0f31d1d3c8d96ddd7cc31bf832cd68ada6"
    resolved = git_text(
        cmdlog, train, ["rev-parse", f"{candidate}^{{commit}}"], "candidate commit"
    )
    if resolved != candidate:
        raise RuntimeError(f"candidate commit does not resolve: {resolved}")
    tag_check = subprocess.run(
        ["git", "-C", str(train), "rev-parse", "-q", "--verify", "refs/tags/v1.4.0"],
        capture_output=True,
        text=True,
    )
    if tag_check.returncode == 0:
        raise RuntimeError("v1.4.0 tag exists; refusing to borrow a release tag")


def compute_weights(cmdlog, train: Path, plan: dict) -> dict:
    result = {}
    for task in ("turn", "response"):
        path = train / plan["weights"][task]["path_relative_to_train_mj"]
        if not path.is_file():
            raise RuntimeError(f"missing weight: {path}")
        started = utcnow()
        actual = hash_file(path)
        ended = utcnow()
        cmdlog.record_inproc(
            f"sha256 {task}",
            train,
            started,
            ended,
            0,
            note=str(path),
        )
        expected = plan["weights"][task]["expected_file_sha256"]
        if actual != expected:
            raise RuntimeError(
                f"{task} weight hash mismatch: actual={actual} expected={expected}"
            )
        result[task] = {
            "path": str(path.resolve()),
            "path_relative_to_train_mj": plan["weights"][task][
                "path_relative_to_train_mj"
            ],
            "size_bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
            "matches": True,
        }
    return result


def load_archive_manifest(out: Path) -> dict:
    path = out / "source_archives/SHA256SUMS.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def export_source(cmdlog, train: Path, out: Path, backend_id: str, commit: str,
                  *, tag: str | None, reuse: bool) -> dict:
    archives = out / "source_archives"
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / f"{backend_id}.zip"
    source = archives / backend_id
    existing = load_archive_manifest(out).get(backend_id)
    if reuse:
        if not existing or not archive.is_file() or not source.is_dir():
            raise RuntimeError(f"cannot reuse missing source export: {backend_id}")
        if hash_file(archive) != existing["archive_sha256"]:
            raise RuntimeError(f"source archive changed: {backend_id}")
        if tree_hash(source) != existing["source_tree_sha256"]:
            raise RuntimeError(f"source tree changed: {backend_id}")
        return existing
    if archive.exists() or source.exists():
        raise RuntimeError(f"source export already exists: {backend_id}")
    if tag:
        resolved = git_text(
            cmdlog, train, ["rev-parse", f"{tag}^{{commit}}"], f"{backend_id} tag"
        )
        if resolved != commit:
            raise RuntimeError(f"tag moved: {tag}")
    cmdlog.run(
        [
            "git",
            "-C",
            str(train),
            "archive",
            "--format=zip",
            "--output",
            str(archive),
            commit,
        ],
        cwd=train,
        note=f"export {backend_id} from fixed commit",
    )
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(source)
    record = {
        "backend_id": backend_id,
        "ref": tag or commit,
        "tag": tag,
        "commit": commit,
        "archive": str(archive),
        "archive_sha256": hash_file(archive),
        "source_dir": str(source),
        "source_tree_sha256": tree_hash(source),
        "source_dirty": False,
        "source_label": "git archive of verified immutable tag"
        if tag
        else "git archive of verified fixed commit",
    }
    manifest = load_archive_manifest(out)
    manifest[backend_id] = record
    write_json(archives / "SHA256SUMS.json", manifest)
    return record


def seed_values(value):
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(seed_values(item))
        return result
    return []


SEED_KEYS = {
    "seed",
    "seeds",
    "final_seeds",
    "engineering",
    "screening",
    "confirmation_reserved_do_not_run",
    "known_previously_used_excluded",
    "excluded_engineering_or_calibration_seeds",
}


def collect_seed_hits(value, path, hits):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key).lower() in SEED_KEYS or str(key).lower().endswith("seeds"):
                for number in seed_values(item):
                    hits.append((child, number))
            collect_seed_hits(item, child, hits)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            collect_seed_hits(item, f"{path}[{index}]", hits)


def history_seed_file(path: Path) -> bool:
    """Only actual run manifests/results are history, not planned configs."""
    if path.name in {"manifest.json", "summary.json"}:
        return True
    return path.parent.name == "results" and path.suffix.lower() == ".json"


def scan_seed_history(
    out: Path,
    roots: list[Path],
    blocking_seeds: set[int],
    reserved_seeds: set[int],
    engineering_seeds: set[int] | None = None,
) -> dict:
    engineering_seeds = set(engineering_seeds or ())
    matches = []
    parse_errors = []
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        for directory, dirnames, filenames in os.walk(root):
            current = Path(directory)
            if out.resolve() in current.resolve().parents or current.resolve() == out.resolve():
                dirnames[:] = []
                continue
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {".git", ".venv", "__pycache__", "node_modules"}
            ]
            for filename in filenames:
                path = current / filename
                if path.suffix.lower() != ".json" or not history_seed_file(path):
                    continue
                try:
                    if path.stat().st_size > 20_000_000:
                        continue
                except OSError:
                    continue
                scanned += 1
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    parse_errors.append({"path": str(path), "error": str(exc)})
                    continue
                hits = []
                collect_seed_hits(value, "", hits)
                for key_path, number in hits:
                    if (
                        number in blocking_seeds
                        or number in reserved_seeds
                        or number in engineering_seeds
                    ):
                        matches.append(
                            {
                                "path": str(path),
                                "key": key_path,
                                "seed": number,
                                "blocking": number in blocking_seeds,
                                "reserved": number in reserved_seeds,
                                "engineering": number in engineering_seeds,
                            }
                        )
    blocking = [item for item in matches if item["blocking"]]
    reserved = [item for item in matches if item["reserved"]]
    engineering = [item for item in matches if item["engineering"]]
    return {
        "valid": not blocking,
        "scanned_files": scanned,
        "parse_errors": parse_errors[:50],
        "roots": [str(root) for root in roots],
        "blocking_seeds": sorted(blocking_seeds),
        "reserved_seeds": sorted(reserved_seeds),
        "engineering_seeds": sorted(engineering_seeds),
        "matches": matches[:500],
        "blocking_matches": blocking[:200],
        "reserved_matches": reserved[:200],
        "engineering_matches": engineering[:200],
        "overlap": bool(blocking),
        "policy": "stop_before_games_and_report_do_not_silently_replace",
        "scope": "actual batch/summary/results artifacts; planned configs are not history",
    }


def environment_snapshot(cmdlog, train: Path, out: Path, plan: dict,
                         service_configs: dict, sources: dict, weights: dict,
                         handoff_sha: str) -> dict:
    import torch
    from riichienv import GameRule

    rule = GameRule.default_tenhou()
    dependencies = {}
    for name in (
        "torch",
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "PyYAML",
        "numpy",
        "riichienv",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    disk = shutil.disk_usage(out)
    return {
        "generated_at": utcnow(),
        "experiment_id": plan["experiment_id"],
        "handoff_plan_sha256": handoff_sha,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "device": "cpu",
        "precision": "float32",
        "service_torch_num_threads": 1,
        "torch": {
            "version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_used": False,
        },
        "dependencies": dependencies,
        "riichi_lab": git_info(cmdlog, ROOT),
        "train_mj": git_info(cmdlog, train),
        "train_mj_v1_3_0_resolves": git_text(
            cmdlog, train, ["rev-parse", "v1.3.0^{commit}"], "baseline tag evidence"
        ),
        "engine": {
            "game_mode": plan["design"]["game_mode"],
            "rule": plan["design"]["rule"],
            "rule_flags": {
                key: getattr(rule, key)
                for key in dir(rule)
                if not key.startswith("_") and isinstance(getattr(rule, key), bool)
            },
        },
        "weights": weights,
        "sources": sources,
        "service_configs": {
            "baseline": {
                "path": service_configs["baseline"]["path"],
                "sha256": service_configs["baseline"]["sha256"],
            },
            "candidates": {
                name: {"path": item["path"], "sha256": item["sha256"]}
                for name, item in service_configs["candidates"].items()
            },
        },
        "confirmation_reserved_do_not_run": RESERVED_SEEDS,
    }


def write_planned_configs(out: Path, plan: dict, handoff_sha: str) -> dict:
    result = {}
    for stage in ("smoke", "screen"):
        result[stage] = {}
        for candidate_id in CANDIDATE_IDS:
            config = build_native_config(
                plan,
                stage=stage,
                candidate_id=candidate_id,
                handoff_sha256=handoff_sha,
            )
            protocol = build_native_protocol(
                plan, stage=stage, candidate_id=candidate_id
            )
            validate_screening_plan(protocol, config)
            directory = out / "resolved_configs" / stage / candidate_id
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / "arena.planned.yaml"
            protocol_path = directory / "protocol.planned.yaml"
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            protocol_path.write_text(
                yaml.safe_dump(protocol, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            result[stage][candidate_id] = {
                "config": config,
                "protocol": protocol,
                "config_path": str(config_path),
                "config_sha256": hash_file(config_path),
                "protocol_path": str(protocol_path),
                "protocol_sha256": hash_file(protocol_path),
            }
    return result


def write_handoff_copy(out: Path, handoff: Path) -> str:
    target = out / "screening_plan.original.yaml"
    target.write_bytes(handoff.read_bytes())
    digest = hash_file(target)
    (out / "screening_plan.original.yaml.sha256").write_text(
        f"{digest}  {target.name}\n", encoding="ascii"
    )
    return digest


def register_protocol(out: Path, plan: dict, handoff_sha: str,
                      service_configs: dict, sources: dict, weights: dict,
                      environment: dict, planned: dict,
                      contract_evidence_sha: str,
                      tests_summary: dict) -> str:
    target = out / "protocol.registered.json"
    if target.exists():
        raise RuntimeError("protocol.registered.json already exists; no resume/overwrite")
    record = {
        "schema_version": 1,
        "status": "registered_before_launch",
        "registered_at": utcnow(),
        "experiment_id": plan["experiment_id"],
        "handoff_plan": plan,
        "handoff_plan_sha256": handoff_sha,
        "handoff_plan_file": str((out / "screening_plan.original.yaml").resolve()),
        "service_configs": {
            "baseline": service_configs["baseline"],
            "candidates": service_configs["candidates"],
        },
        "source_identity": sources,
        "weights": weights,
        "environment": environment,
        "planned_configs": {
            stage: {
                candidate_id: {
                    "config": item["config"],
                    "config_sha256": item["config_sha256"],
                    "protocol": item["protocol"],
                    "protocol_sha256": item["protocol_sha256"],
                }
                for candidate_id, item in candidates.items()
            }
            for stage, candidates in planned.items()
        },
        "contract_evidence_sha256": contract_evidence_sha,
        "engineering_tests": tests_summary,
        "analysis_rules": plan["analysis"],
        "stopping": plan["stopping"],
        "seeds": {
            "engineering": ENGINEERING_SEEDS,
            "screening": SCREENING_SEEDS,
            "confirmation_reserved_do_not_run": RESERVED_SEEDS,
        },
        "review_required_before": plan["review_required_before"],
        "resume": "not_implemented_exact_schedule_identity_not_available",
        "no_online_ranked_games": True,
        "no_training": True,
    }
    write_json(target, record)
    return hash_file(target)


class RunningService:
    def __init__(self, backend_id, port, api_base, process, handle, started,
                 log_path, resolved_config_path, service_config_path):
        self.backend_id = backend_id
        self.port = port
        self.api_base = api_base
        self.process = process
        self.handle = handle
        self.started = started
        self.log_path = log_path
        self.resolved_config_path = resolved_config_path
        self.service_config_path = service_config_path

    def stop(self, cmdlog, note=None):
        return cmdlog.stop_service(
            self.process, self.handle, self.started, note=note or self.backend_id
        )


def tail(path: Path, lines=40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def start_service(cmdlog, train: Path, group_dir: Path, backend_id: str,
                  source_dir: Path, service_config_path: Path,
                  expected_backend_config: dict) -> RunningService:
    port = free_port()
    resolved = group_dir / "resolved_service_configs" / f"{backend_id}.yaml"
    log = group_dir / "services" / f"{backend_id}.stdout.log"
    cmd = [
        sys.executable,
        str(SERVE),
        "--source-root",
        str(source_dir),
        "--weights-root",
        str(train),
        "--port",
        str(port),
        "--service-config",
        str(service_config_path),
        "--resolved-config-out",
        str(resolved),
    ]
    process, handle, started = cmdlog.start_service(
        cmd, cwd=ROOT, log_path=log, note=f"{backend_id} service"
    )
    service = RunningService(
        backend_id,
        port,
        f"http://127.0.0.1:{port}",
        process,
        handle,
        started,
        log,
        resolved,
        service_config_path,
    )
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{backend_id} service exited during startup: {tail(log)}"
                )
            try:
                response = httpx.get(service.api_base + "/v1/models", timeout=2.0)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError(
                f"{backend_id} service startup timed out: {tail(log)}"
            )
        backend_config = deepcopy(expected_backend_config)
        backend_config["api_base"] = service.api_base
        backend = HttpPredictBackend(backend_id, backend_config)
        try:
            backend.metadata
        finally:
            backend.close()
        return service
    except Exception:
        try:
            cmdlog.stop_service(
                process, handle, started, note=f"{backend_id} startup cleanup"
            )
        except Exception:
            pass
        raise


def read_metadata(service: RunningService, expected_backend_config: dict) -> dict:
    backend_config = deepcopy(expected_backend_config)
    backend_config["api_base"] = service.api_base
    backend = HttpPredictBackend(service.backend_id, backend_config)
    try:
        return backend.metadata
    finally:
        backend.close()


def service_record(service: RunningService, config: dict) -> dict:
    return {
        "backend_id": service.backend_id,
        "port": service.port,
        "api_base": service.api_base,
        "pid": service.process.pid,
        "log_path": str(service.log_path),
        "external_service_config": str(service.service_config_path),
        "external_service_config_sha256": hash_file(service.service_config_path),
        "resolved_service_config": str(service.resolved_config_path),
        "resolved_service_config_sha256": hash_file(service.resolved_config_path)
        if service.resolved_config_path.is_file()
        else None,
        "calibration_contract": config["calibration_contract"],
    }


def deployment_record(source: dict, service: RunningService, config: dict) -> dict:
    return {
        "source": source["source_label"],
        "source_ref": source["ref"],
        "source_commit": source["commit"],
        "source_dirty": False,
        "archive_sha256": source["archive_sha256"],
        "source_tree_sha256": source["source_tree_sha256"],
        "service_config_sha256": hash_file(service.service_config_path),
        "resolved_service_config_sha256": hash_file(service.resolved_config_path),
        "verified_weight_sha256": config["model_identity"].copy(),
    }


def run_tests(cmdlog, train: Path, smoke_dir: Path) -> dict:
    tests_dir = smoke_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        {
            "name": "riichi_lab",
            "cmd": [sys.executable, "-m", "pytest", "-q"],
            "cwd": ROOT,
            "log": tests_dir / "riichi_lab_pytest.log",
        },
        {
            "name": "train_mj_serving",
            "cmd": [sys.executable, "-m", "pytest", *TRAIN_TESTS, "-q"],
            "cwd": train,
            "log": tests_dir / "train_mj_serving_pytest.log",
        },
    ]
    results = []
    for item in commands:
        process = cmdlog.run(
            item["cmd"],
            cwd=item["cwd"],
            log_path=item["log"],
            note=f"pytest {item['name']}",
        )
        results.append(
            {
                "name": item["name"],
                "exit_code": process.returncode,
                "log": str(item["log"]),
                "passed": process.returncode == 0,
            }
        )
    summary = {
        "valid": all(item["passed"] for item in results),
        "results": results,
    }
    write_json(tests_dir / "summary.json", summary)
    return summary


def run_contract(cmdlog, train: Path, out: Path, smoke_dir: Path,
                 sources: dict, service_configs: dict) -> dict:
    contract_dir = smoke_dir / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    services = {}
    running = []
    try:
        for name in ["baseline", *CANDIDATE_IDS]:
            backend_id = BASELINE_ID if name == "baseline" else name
            source = (
                sources[BASELINE_ID]
                if name == "baseline"
                else sources[CANDIDATE_SOURCE_ID]
            )
            service_config = (
                service_configs["baseline"]
                if name == "baseline"
                else service_configs["candidates"][name]
            )
            planned = build_native_config(
                {"experiment_id": "contract_only"},
                stage="smoke",
                candidate_id=name if name != "baseline" else CANDIDATE_IDS[0],
            )
            expected = planned["backends"][
                BASELINE_ID if name == "baseline" else name
            ]
            service = start_service(
                cmdlog,
                train,
                contract_dir / "services" / name,
                backend_id,
                Path(source["source_dir"]),
                Path(service_config["path"]),
                expected,
            )
            running.append(service)
            services[backend_id] = {
                "api_base": service.api_base,
                "calibration_contract": expected["calibration_contract"],
                "response": expected["response"],
            }
        services_json = contract_dir / "services.json"
        services_json.write_text(json.dumps(services, indent=2), encoding="utf-8")
        payload_source = ROOT / "outputs/strength_20261103_143025/batch"
        service_evidence = contract_dir / "service_contract.json"
        process = cmdlog.run(
            [
                sys.executable,
                str(VERIFY_SERVICES),
                "--services",
                str(services_json),
                "--output",
                str(service_evidence),
                *(
                    ["--payload-source", str(payload_source)]
                    if payload_source.is_dir()
                    else []
                ),
            ],
            cwd=ROOT,
            log_path=contract_dir / "service_contract.log",
            note="v1.4.0 HTTP service contract",
        )
        selector_evidence = contract_dir / "selector.json"
        selector_process = cmdlog.run(
            [
                sys.executable,
                str(VERIFY_SELECTOR),
                "--source-root",
                str(Path(sources[CANDIDATE_SOURCE_ID]["source_dir"])),
                "--output",
                str(selector_evidence),
            ],
            cwd=ROOT,
            log_path=contract_dir / "selector.log",
            note="v1.4.0 selector semantics",
        )
        legacy_evidence = contract_dir / "legacy_http_contract.json"
        legacy_process = cmdlog.run(
            [
                sys.executable,
                str(CHECK_LEGACY),
                "--source-root",
                str(Path(sources[BASELINE_ID]["source_dir"])),
                "--weights-root",
                str(train),
                "--profile",
                "full_state_v2",
                "--output",
                str(legacy_evidence),
            ],
            cwd=ROOT,
            log_path=contract_dir / "legacy_http_contract.log",
            note="legacy v1.3.0 HTTP contract",
        )
        rejection_evidence = contract_dir / "new_source_rejects_legacy_config.json"
        rejection_process = cmdlog.run(
            [
                sys.executable,
                str(CHECK_CONFIG_CONTRACT),
                "--source-root",
                str(Path(sources[CANDIDATE_SOURCE_ID]["source_dir"])),
                "--weights-root",
                str(train),
                "--service-config",
                str(Path(service_configs["baseline"]["path"])),
                "--output",
                str(rejection_evidence),
                "--expect-error",
                "pass_bias was removed",
            ],
            cwd=ROOT,
            log_path=contract_dir / "new_source_rejects_legacy_config.log",
            note="new source rejects legacy calibration config",
        )
        checks = {
            "service_contract": process.returncode == 0
            and _json_valid(service_evidence),
            "selector": selector_process.returncode == 0
            and _json_valid(selector_evidence),
            "legacy_http_contract": legacy_process.returncode == 0
            and _json_valid(legacy_evidence),
            "new_source_rejects_legacy_config": rejection_process.returncode == 0
            and _json_valid(rejection_evidence),
        }
        evidence = {
            "valid": all(checks.values()),
            "checks": checks,
            "services": {
                item.backend_id: {
                    "api_base": item.api_base,
                    "port": item.port,
                    "pid": item.process.pid,
                    "external_config": str(item.service_config_path),
                    "resolved_config": str(item.resolved_config_path),
                }
                for item in running
            },
            "evidence_files": {
                "service_contract": str(service_evidence),
                "selector": str(selector_evidence),
                "legacy_http_contract": str(legacy_evidence),
                "new_source_rejects_legacy_config": str(rejection_evidence),
            },
        }
        write_json(contract_dir / "evidence.json", evidence)
        write_json(
            contract_dir / "running_services.json",
            {
                item.backend_id: service_record(item, services[item.backend_id])
                for item in running
            },
        )
        return evidence
    finally:
        for service in running:
            try:
                service.stop(cmdlog, note="contract phase cleanup")
            except Exception:
                pass


def run_group(cmdlog, train: Path, out: Path, plan: dict, stage: str,
              candidate_id: str, sources: dict, service_configs: dict,
              weights: dict, protocol_sha: str, timeout: int,
              replay_workers: int) -> dict:
    group_dir = out / stage / candidate_id
    if group_dir.exists():
        raise RuntimeError(f"group directory already exists: {group_dir}")
    group_dir.mkdir(parents=True)
    (group_dir / "services").mkdir()
    (group_dir / "resolved_service_configs").mkdir()
    disk = shutil.disk_usage(group_dir)
    if disk.free < 5_000_000_000:
        raise RuntimeError(f"insufficient free disk: {disk.free} bytes")
    baseline_service = None
    candidate_service = None
    started = utcnow()
    try:
        handoff_sha = hash_file(out / "screening_plan.original.yaml")
        baseline_expected = build_native_config(
            plan,
            stage=stage,
            candidate_id=candidate_id,
            handoff_sha256=handoff_sha,
        )
        baseline_service = start_service(
            cmdlog,
            train,
            group_dir,
            BASELINE_ID,
            Path(sources[BASELINE_ID]["source_dir"]),
            Path(service_configs["baseline"]["path"]),
            baseline_expected["backends"][BASELINE_ID],
        )
        candidate_expected = build_native_config(
            plan,
            stage=stage,
            candidate_id=candidate_id,
            handoff_sha256=handoff_sha,
        )
        candidate_service = start_service(
            cmdlog,
            train,
            group_dir,
            candidate_id,
            Path(sources[CANDIDATE_SOURCE_ID]["source_dir"]),
            Path(service_configs["candidates"][candidate_id]["path"]),
            candidate_expected["backends"][candidate_id],
        )
        config = build_native_config(
            plan,
            stage=stage,
            candidate_id=candidate_id,
            baseline_api_base=baseline_service.api_base,
            candidate_api_base=candidate_service.api_base,
            handoff_sha256=handoff_sha,
        )
        config["backends"][BASELINE_ID]["deployment"] = deployment_record(
            sources[BASELINE_ID],
            baseline_service,
            config["backends"][BASELINE_ID],
        )
        config["backends"][candidate_id]["deployment"] = deployment_record(
            sources[CANDIDATE_SOURCE_ID], candidate_service, config["backends"][candidate_id]
        )
        config = _validate_config_dict(config)
        protocol = build_native_protocol(plan, stage=stage, candidate_id=candidate_id)
        validate_screening_plan(protocol, config)
        config_path = group_dir / "arena.yaml"
        protocol_path = group_dir / "protocol.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        protocol_path.write_text(
            yaml.safe_dump(protocol, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        models_before = {
            BASELINE_ID: read_metadata(
                baseline_service, config["backends"][BASELINE_ID]
            ),
            candidate_id: read_metadata(
                candidate_service, config["backends"][candidate_id]
            ),
        }
        write_json(group_dir / "models.before.json", models_before)
        group_record = {
            "experiment_id": plan["experiment_id"],
            "stage": stage,
            "candidate_id": candidate_id,
            "status": "launched",
            "started_at": started,
            "config": str(config_path),
            "config_sha256": hash_file(config_path),
            "protocol": str(protocol_path),
            "protocol_sha256": hash_file(protocol_path),
            "protocol_registration_sha256": protocol_sha,
            "seed_clusters": len(config["engine"]["seeds"]),
            "expected_half_games": len(_expand(config)),
            "models_before": models_before,
            "services": {
                BASELINE_ID: service_record(
                    baseline_service, config["backends"][BASELINE_ID]
                ),
                candidate_id: service_record(
                    candidate_service, config["backends"][candidate_id]
                ),
            },
            "source_identity": sources,
            "weights": weights,
        }
        write_json(group_dir / "group.json", group_record)
        extra = {
            "screening": {
                "experiment_id": plan["experiment_id"],
                "stage": stage,
                "candidate_id": candidate_id,
                "seed_clusters": len(config["engine"]["seeds"]),
                "expected_half_games": len(_expand(config)),
            },
            "source_identity": sources,
            "service_configs": {
                "baseline": service_configs["baseline"]["sha256"],
                "candidate": service_configs["candidates"][candidate_id]["sha256"],
            },
        }
        batch_cmd = [
            sys.executable,
            str(BATCH),
            "--config",
            str(config_path),
            "--output-dir",
            str(group_dir / "batch"),
            "--manifest-extra",
            json.dumps(extra, ensure_ascii=False),
            "--protocol-registration",
            str(out / "protocol.registered.json"),
        ]
        process = cmdlog.run(
            batch_cmd,
            cwd=ROOT,
            log_path=group_dir / "batch.log",
            timeout=timeout,
            note=f"{stage}/{candidate_id} batch",
        )
        if process.returncode != 0:
            raise RuntimeError(f"batch process failed with {process.returncode}")
        models_after = {
            BASELINE_ID: read_metadata(
                baseline_service, config["backends"][BASELINE_ID]
            ),
            candidate_id: read_metadata(
                candidate_service, config["backends"][candidate_id]
            ),
        }
        write_json(group_dir / "models.after.json", models_after)
        if models_after != models_before:
            raise RuntimeError("service metadata changed during batch")
        batch_manifest = json.loads(
            (group_dir / "batch/manifest.json").read_text(encoding="utf-8")
        )
        batch_summary = json.loads(
            (group_dir / "batch/summary.json").read_text(encoding="utf-8")
        )
        expected_games = len(_expand(config))
        if (
            batch_summary.get("valid") is not True
            or batch_summary.get("fallbacks") != 0
            or batch_summary.get("completed_games") != expected_games
            or batch_summary.get("expected_games") != expected_games
        ):
            raise RuntimeError("batch summary is incomplete or invalid")
        if batch_manifest.get("schedule") != _expand(config):
            raise RuntimeError("batch schedule differs from native config")
        if batch_manifest.get("models") != models_before:
            raise RuntimeError("batch models differ from pre-launch metadata")
        if batch_manifest.get("protocol_registration", {}).get("sha256") != protocol_sha:
            raise RuntimeError("batch manifest does not reference the sealed protocol")
        games = audit_batch(group_dir / "batch")
        if batch_summary.get("replay_digest") != hashlib.sha256(
            encode(games).encode()
        ).hexdigest():
            raise RuntimeError("batch replay digest mismatch")
        decision_audit = verify_group_decisions(
            group_dir / "batch", config, models_before
        )
        if not decision_audit["valid"]:
            raise RuntimeError("per-decision calibration audit failed")
        replay_process = cmdlog.run(
            [
                sys.executable,
                str(REPLAY),
                "--run-dir",
                str(group_dir / "batch"),
                "--output",
                str(group_dir / "offline_replay.json"),
                "--workers",
                str(replay_workers),
            ],
            cwd=ROOT,
            log_path=group_dir / "offline_replay.log",
            timeout=7200,
            note=f"{stage}/{candidate_id} offline replay",
        )
        if replay_process.returncode != 0:
            raise RuntimeError("offline replay failed")
        replay = json.loads(
            (group_dir / "offline_replay.json").read_text(encoding="utf-8")
        )
        if (
            replay.get("valid") is not True
            or replay.get("games") != expected_games
            or replay.get("model_calls") != 0
        ):
            raise RuntimeError("offline replay result is invalid")
        analysis = analyze_group(
            group_dir / "batch",
            config,
            models_before,
            candidate_id=candidate_id,
            stage=stage,
        )
        write_json(group_dir / "metrics.json", analysis)
        verification = {
            "valid": True,
            "stage": stage,
            "candidate_id": candidate_id,
            "runs": 1,
            "replay_identical": None,
            "games_per_run": expected_games,
            "completed_games": batch_summary["completed_games"],
            "expected_games": expected_games,
            "fallbacks": batch_summary["fallbacks"],
            "digests": [batch_summary["replay_digest"]],
            "deployment": {
                name: config["backends"][name]["deployment"]
                for name in config["backends"]
            },
            "models_before_sha256": hashlib.sha256(
                encode(models_before).encode()
            ).hexdigest(),
            "models_after_equal": True,
            "protocol_registration_sha256": protocol_sha,
            "artifact_audit": {"valid": True, "games": len(games)},
            "decision_audit": decision_audit,
            "offline_replay": replay,
            "source_identity": sources,
            "weights": weights,
        }
        write_json(group_dir / "verification.json", verification)
        group_record.update(
            status="completed",
            completed_at=utcnow(),
            completed_half_games=batch_summary["completed_games"],
            models_after=models_after,
            batch_manifest_sha256=hash_file(group_dir / "batch/manifest.json"),
            batch_summary_sha256=hash_file(group_dir / "batch/summary.json"),
            verification_sha256=hash_file(group_dir / "verification.json"),
        )
        write_json(group_dir / "group.json", group_record)
        return {
            "valid": True,
            "candidate_id": candidate_id,
            "stage": stage,
            "group_dir": str(group_dir),
            "verification": verification,
            "metrics": analysis,
            "summary": batch_summary,
        }
    except Exception as exc:
        if candidate_service is not None:
            try:
                candidate_service.stop(cmdlog, note="group failure cleanup")
            except Exception:
                pass
        if baseline_service is not None:
            try:
                baseline_service.stop(cmdlog, note="group failure cleanup")
            except Exception:
                pass
        failure = {
            "valid": False,
            "stage": stage,
            "candidate_id": candidate_id,
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at": utcnow(),
            "tail": tail(group_dir / "batch.log")
            if (group_dir / "batch.log").is_file()
            else None,
        }
        write_json(group_dir / "group_failure.json", failure)
        return {
            "valid": False,
            "candidate_id": candidate_id,
            "stage": stage,
            "group_dir": str(group_dir),
            "error": failure["error"],
        }
    finally:
        for service in (candidate_service, baseline_service):
            if service is not None and service.process.poll() is None:
                try:
                    service.stop(cmdlog, note=f"{stage}/{candidate_id} cleanup")
                except Exception:
                    pass


def _json_valid(path: Path) -> bool:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(value.get("valid") or value.get("passed"))


def _validate_config_dict(config: dict) -> dict:
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        path = Path(handle.name)
    try:
        load_arena_config(path)
    finally:
        path.unlink(missing_ok=True)
    return config


def _expand(config: dict):
    from rilab.scheduling import expand_schedule

    return expand_schedule(config)


def analyze_screen(out: Path, smoke_summary: dict) -> dict:
    group_metrics = {}
    for candidate_id in CANDIDATE_IDS:
        path = out / "screen" / candidate_id / "metrics.json"
        group_metrics[candidate_id] = json.loads(path.read_text(encoding="utf-8"))
    clusters = {
        candidate_id: group_metrics[candidate_id]["clusters"]
        for candidate_id in CANDIDATE_IDS
    }
    multi = bootstrap_screening(
        clusters, iterations=10000, rng_seed=20261109
    )
    groups = {}
    for candidate_id in CANDIDATE_IDS:
        detail = group_metrics[candidate_id]
        groups[candidate_id] = {
            "run": detail["run"],
            "seed_clusters": detail["seed_clusters"],
            "completed_half_games": detail["completed_half_games"],
            "candidate_focal_games": detail["candidate_focal_games"],
            "control_focal_games": detail["control_focal_games"],
            "focused": detail["focused"],
            "differences": multi[candidate_id]["differences"],
            "candidate_ci95": multi[candidate_id]["candidate_ci95"],
            "control_ci95": multi[candidate_id]["control_ci95"],
            "response": detail["response"],
            "per_game": detail["per_game"],
            "verification": json.loads(
                (out / "screen" / candidate_id / "verification.json").read_text(
                    encoding="utf-8"
                )
            ),
        }
    artifact_ok = all(
        group["verification"].get("valid") is True for group in groups.values()
    )
    engineering_ok = smoke_summary.get("valid") is True
    rec = recommendation(
        groups, engineering_ok=engineering_ok, artifact_ok=artifact_ok
    )
    metrics = {
        "schema_version": 1,
        "experiment_id": "v1_4_0_step1_three_profiles",
        "stage": "screen",
        "analysis_kind": "exploratory_parameter_screening",
        "tie_rule": (
            "authoritative RiichiEnv.ranks() order; the engine's official "
            "four-player tie-break is inherited and no re-ranking is applied"
        ),
        "bootstrap": {
            "method": "paired seed-cluster percentile; ratio-of-sums recomputed per draw",
            "iterations": 10000,
            "rng_seed": 20261109,
            "confidence_level": 0.95,
            "independent_units": "128 seeds; all four rotations in both arms resampled together",
            "cross_candidate": "same resampled seed indices across all candidates",
            "interval_scope": "exploratory; not a confirmatory or online-release test",
        },
        "groups": groups,
        "cross_candidate": multi["cross_candidate"],
        "recommendation": rec,
        "smoke_engineering_gate": smoke_summary,
        "artifact_verification": {
            "valid": artifact_ok and engineering_ok,
            "smoke_valid": engineering_ok,
            "all_group_verifications_valid": artifact_ok,
        },
        "notes": [
            "Control games repeat in each candidate group and are not independent evidence.",
            "Each candidate is compared with its own four-baseline control using paired seed clusters.",
            "Cross-candidate intervals reuse the same resampled seed indices and remain exploratory.",
            "DEF-001 remains a known training-side exception; this run does not claim it was fixed.",
            "Offline replay verifies recorded inputs/actions only and does not re-run model logits.",
            "No online ranked games or confirmation seeds were run.",
        ],
    }
    aggregate = out / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    write_json(aggregate / "metrics.json", metrics)
    write_metrics_csv(aggregate / "metrics.csv", metrics)
    write_per_seed_csv(aggregate / "per_seed.csv", groups)
    (aggregate / "report.md").write_text(render_report(metrics), encoding="utf-8")
    return metrics


def smoke_summary(
    out: Path, groups: dict, contract: dict, tests: dict, protocol_sha: str | None = None
) -> dict:
    total = sum(
        group["summary"]["completed_games"]
        for group in groups.values()
        if group.get("valid")
    )
    summary = {
        "valid": (
            all(group.get("valid") for group in groups.values())
            and contract.get("valid") is True
            and tests.get("valid") is True
            and total == 48
        ),
        "total_half_games": total,
        "planned_half_games": 48,
        "groups": {
            candidate_id: {
                "valid": group.get("valid"),
                "completed_half_games": group.get("summary", {}).get(
                    "completed_games", 0
                ),
                "verification": group.get("verification"),
                "error": group.get("error"),
            }
            for candidate_id, group in groups.items()
        },
        "contract": contract,
        "tests": tests,
        "protocol_registration_sha256": protocol_sha,
    }
    write_json(out / "smoke/summary.json", summary)
    return summary


def write_review(out: Path, plan: dict, status: str, *, environment=None,
                 sources=None, weights=None, smoke=None, metrics=None,
                 protocol_sha=None, blocked=None) -> dict:
    actual_smoke = None
    actual_screen = 0
    if smoke:
        actual_smoke = smoke.get("total_half_games")
    if metrics:
        actual_screen = sum(
            group["completed_half_games"] for group in metrics["groups"].values()
        )
    failures = {
        "blocked": blocked,
        "missing_smoke_or_screen_games": None,
        "fallbacks": 0,
        "illegal_actions": 0,
        "calibration_mismatches": 0,
        "service_errors": 0,
    }
    replay = {}
    unresolved = [
        "DEF-001 remains a known training-side exception; it was not fixed in this task.",
        "Intervals are exploratory and do not constitute confirmation or online-release evidence.",
        "The control arm repeats across candidates and is not three times independent evidence.",
        "No confirmation seeds (31001-31256) and no online ranked games were run.",
    ]
    if metrics:
        for candidate_id, group in metrics["groups"].items():
            verification = group["verification"]
            replay[candidate_id] = verification.get("offline_replay")
            failures["calibration_mismatches"] += verification.get(
                "decision_audit", {}
            ).get("calibration_mismatches", 0)
            failures["fallbacks"] += verification.get("fallbacks", 0)
        failures["missing_smoke_or_screen_games"] = (
            3072 - actual_screen if metrics else None
        )
    review = {
        "experiment_id": plan["experiment_id"],
        "status": status,
        "generated_at": utcnow(),
        "output_dir": str(out.resolve()),
        "stage_dirs": {
            "smoke": str((out / "smoke").resolve()),
            "screen": str((out / "screen").resolve()),
            "aggregate": str((out / "aggregate").resolve()),
        },
        "source_identity": sources,
        "train_mj": {
            "baseline": {"ref": "v1.3.0", "commit": "1e5092f344fb69bcdf4697f683f2ecf15b9865d8"},
            "candidate": {
                "ref": "0710aa0f31d1d3c8d96ddd7cc31bf832cd68ada6",
                "branch_hint": "codex/v1.4.0",
                "release_tag_exists": False,
            },
        },
        "weights": weights,
        "effective_parameters": {
            "baseline_v1_3_0": {
                "calibration_contract": "legacy_pass_bias",
                "pass_bias": 1.5,
                "no_yaku_call_penalty": 0.0,
            },
            **{
                item["id"]: {
                    "calibration_contract": "action_biases_v1",
                    "action_biases": item["action_biases"],
                    "no_yaku_call_penalty": item["no_yaku_call_penalty"],
                }
                for item in CANDIDATES
            },
        },
        "planned_games": {"smoke": 48, "screen": 3072, "total": 3120},
        "actual_games": {
            "smoke": actual_smoke,
            "screen": actual_screen,
            "total": (actual_smoke or 0) + (actual_screen or 0) if smoke or metrics else None,
        },
        "independent_seed_clusters": 128,
        "failures": failures,
        "protocol_registration_sha256": protocol_sha,
        "replay_verification": replay,
        "aggregate_paths": {
            "metrics_json": str((out / "aggregate/metrics.json").resolve()),
            "metrics_csv": str((out / "aggregate/metrics.csv").resolve()),
            "per_seed_csv": str((out / "aggregate/per_seed.csv").resolve()),
            "report_md": str((out / "aggregate/report.md").resolve()),
        },
        "recommendation": metrics["recommendation"] if metrics else None,
        "unresolved_issues": unresolved,
        "reproduction": {
            "dry_run_smoke": (
                f"python scripts/run_v140_screening.py --handoff-plan {out / 'screening_plan.original.yaml'} "
                f"--stage smoke --output-dir {out} --dry-run"
            ),
            "full_run": (
                f"python scripts/run_v140_screening.py --handoff-plan <handoff>/screening_plan.yaml "
                f"--stage smoke --output-dir {out} && "
                f"python scripts/run_v140_screening.py --handoff-plan <handoff>/screening_plan.yaml "
                f"--stage screen --output-dir {out}"
            ),
            "artifact_audit": (
                f"python scripts/verify_v140_artifacts.py --output-dir {out}"
            ),
        },
        "notes": [
            "No model training, finetuning, weight replacement, or online ranked games.",
            "Service code is frozen to fixed commits; handoff service YAMLs are external inputs.",
            "Screen groups ran sequentially A -> B -> C with isolated service instances per group.",
        ],
    }
    write_json(out / "handoff_review.json", review)
    (out / "handoff_review.md").write_text(
        render_handoff_review(review, metrics), encoding="utf-8"
    )
    return review


def render_handoff_review(review: dict, metrics: dict | None) -> str:
    lines = [
        "# v1.4.0 Step 1 本地筛选交接复核",
        "",
        f"- 状态：`{review['status']}`",
        f"- 输出目录：`{review['output_dir']}`",
        f"- 实验：`{review['experiment_id']}`",
        "",
        "## 先看这些路径",
        "",
        "| 内容 | 路径 |",
        "|---|---|",
        f"| 封存协议 | `{review['output_dir']}/protocol.registered.json` |",
        f"| 原始交接规范 | `{review['output_dir']}/screening_plan.original.yaml` |",
        f"| 环境与身份 | `{review['output_dir']}/environment.json` |",
        f"| 命令审计 | `{review['output_dir']}/commands.jsonl` |",
        f"| smoke 证据 | `{review['output_dir']}/smoke/` |",
        f"| 筛选原始运行 | `{review['output_dir']}/screen/` |",
        f"| 汇总指标 | `{review['output_dir']}/aggregate/metrics.json` |",
        f"| 汇总表 | `{review['output_dir']}/aggregate/metrics.csv` |",
        f"| 逐种子 | `{review['output_dir']}/aggregate/per_seed.csv` |",
        f"| 汇总报告 | `{review['output_dir']}/aggregate/report.md` |",
        f"| 产物哈希 | `{review['output_dir']}/artifact_checksums.json` |",
        "",
        "## 源码与权重身份",
        "",
        "| 项目 | 身份 |",
        "|---|---|",
        "| 基线 | train_mj `v1.3.0` / `1e5092f344fb69bcdf4697f683f2ecf15b9865d8` |",
        "| 候选 | train_mj `0710aa0f31d1d3c8d96ddd7cc31bf832cd68ada6`（无 v1.4.0 tag） |",
        "| turn 权重 | `30f2811cd86beebad57d4da1428272b25b598fb64a6231b5a1760e11ec08bf90` |",
        "| response 权重 | `e1d8c58a2aa61cef23852f15fe5bb9c65014bc3916c52756fb8c4cd1cd956c51` |",
        "",
        "## 实际局数",
        "",
        f"- smoke：{review['actual_games']['smoke']} / 48 半庄",
        f"- screen：{review['actual_games']['screen']} / 3072 半庄",
        f"- 独立种子簇：128（筛选阶段）",
        "",
    ]
    if metrics:
        rec = metrics["recommendation"]
        lines.extend(
            [
                "## 推荐结果",
                "",
                f"- 状态：`{rec['status']}`",
                f"- 待独立确认候选：`{rec['selected']}`",
                "",
                "| 候选 | 平均顺位差 | 95% 区间 | 末位率差 | 95% 区间 | 候选平均终局点数 |",
                "|---|---:|---|---:|---|---:|",
            ]
        )
        for candidate_id, group in metrics["groups"].items():
            rank = group["differences"]["mean_rank"]
            last = group["differences"]["last_place_rate"]
            lines.append(
                f"| {candidate_id} | {rank['value']:.6f} | "
                f"[{rank['ci95'][0]:.6f}, {rank['ci95'][1]:.6f}] | "
                f"{100 * last['value']:.4f}% | "
                f"[{100 * last['ci95'][0]:.4f}%, {100 * last['ci95'][1]:.4f}%] | "
                f"{group['focused']['candidate']['mean_final_points']['value']:.2f} |"
            )
        lines.extend(
            [
                "",
                "所有差值均为候选减控制；平均顺位差为负数表示候选更好。区间为探索性，不构成独立确认或线上发布许可。",
                "",
            ]
        )
    lines.extend(
        [
            "## 只读审计命令",
            "",
            "```powershell",
            f"python scripts/verify_v140_artifacts.py --output-dir \"{review['output_dir']}\"",
            f"python scripts/verify_v140_artifacts.py --output-dir \"{review['output_dir']}\" --replay",
            f"python -m pytest -q",
            "```",
            "",
            "`--replay` 会离线重放所有已记录决策，不调用模型、不新开对局；默认只核对哈希、日程、摘要和逐决策校准字段。",
            "",
            "## 未解决问题与解释边界",
            "",
        ]
    )
    for item in review["unresolved_issues"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "完整分子/分母、逐种子/轮换、response 机会与切换统计见 `aggregate/` 与各 `screen/*/metrics.json`。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_checksums(out: Path) -> dict:
    target = out / "artifact_checksums.json"
    files = {}
    total = 0
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path == target:
            continue
        relative = path.relative_to(out).as_posix()
        digest = hash_file(path)
        files[relative] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
        total += path.stat().st_size
    record = {
        "schema_version": 1,
        "generated_at": utcnow(),
        "root": str(out.resolve()),
        "file_count": len(files),
        "total_bytes": total,
        "excludes": ["artifact_checksums.json"],
        "files": files,
    }
    write_json(target, record)
    return record


def init_output(out: Path, handoff: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    for name in ("source_archives", "resolved_configs", "smoke", "screen", "aggregate"):
        (out / name).mkdir()
    write_handoff_copy(out, handoff)


def verify_sources(out: Path, sources: dict) -> None:
    for backend_id, record in sources.items():
        archive = Path(record["archive"])
        source = Path(record["source_dir"])
        if not archive.is_file() or not source.is_dir():
            raise RuntimeError(f"missing source export: {backend_id}")
        if hash_file(archive) != record["archive_sha256"]:
            raise RuntimeError(f"source archive hash changed: {backend_id}")
        if tree_hash(source) != record["source_tree_sha256"]:
            raise RuntimeError(f"source tree hash changed: {backend_id}")


def verify_screen_ready(out: Path, handoff_sha: str, train: Path,
                        cmdlog, service_configs: dict) -> dict:
    required = [
        out / "protocol.registered.json",
        out / "environment.json",
        out / "smoke/summary.json",
        out / "source_archives/SHA256SUMS.json",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"screen stage requires completed smoke artifact: {path}")
    registration = json.loads(
        (out / "protocol.registered.json").read_text(encoding="utf-8")
    )
    if registration.get("handoff_plan_sha256") != handoff_sha:
        raise RuntimeError("handoff plan changed after protocol registration")
    if registration.get("status") != "registered_before_launch":
        raise RuntimeError("invalid protocol registration status")
    smoke = json.loads((out / "smoke/summary.json").read_text(encoding="utf-8"))
    if smoke.get("valid") is not True or smoke.get("total_half_games") != 48:
        raise RuntimeError("smoke engineering gate is not valid")
    if smoke.get("protocol_registration_sha256") != hash_file(
        out / "protocol.registered.json"
    ):
        raise RuntimeError("sealed protocol changed after smoke")
    if registration.get("contract_evidence_sha256") != hash_file(
        out / "smoke/contract/evidence.json"
    ):
        raise RuntimeError("contract evidence changed after smoke")
    environment = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    current = git_info(cmdlog, ROOT)
    if (
        environment.get("riichi_lab", {}).get("commit") != current["commit"]
        or current["dirty"]
    ):
        raise RuntimeError("riichi_lab code changed after smoke registration")
    registered_train = environment.get("train_mj", {}).get("commit")
    current_train = git_info(cmdlog, train)["commit"]
    if registered_train != current_train:
        raise RuntimeError("train_mj worktree commit changed after registration")
    for name, item in service_configs["candidates"].items():
        registered = registration["service_configs"]["candidates"][name]["sha256"]
        if item["sha256"] != registered:
            raise RuntimeError(f"service config changed after registration: {name}")
    if service_configs["baseline"]["sha256"] != registration["service_configs"][
        "baseline"
    ]["sha256"]:
        raise RuntimeError("baseline service config changed after registration")
    return registration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-plan", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("smoke", "screen"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--train-mj-root", type=Path, default=ROOT.parent / "train_mj"
    )
    parser.add_argument(
        "--handoff-dir",
        type=Path,
        help="directory containing the four service YAMLs (default: plan parent)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replay-workers", type=int, default=4)
    args = parser.parse_args(argv)
    handoff = args.handoff_plan.resolve()
    handoff_dir = (args.handoff_dir or handoff.parent).resolve()
    train = args.train_mj_root.resolve()
    out = args.output_dir.resolve()
    plan = load_handoff(handoff)
    validate_handoff(plan, handoff_dir=handoff_dir)
    handoff_sha = sha256_file(handoff)
    service_configs = load_service_configs(plan, handoff_dir)
    if args.dry_run:
        temp_log = CommandLog(Path(os.devnull))
        verify_source_refs(temp_log, train)
        sources = {}
        for backend_id, commit, tag in (
            (BASELINE_ID, "1e5092f344fb69bcdf4697f683f2ecf15b9865d8", "v1.3.0"),
            (
                "candidate_v1_4_0",
                "0710aa0f31d1d3c8d96ddd7cc31bf832cd68ada6",
                None,
            ),
        ):
            sources[backend_id] = {
                "commit": commit,
                "tag": tag,
                "source_dir": "dry-run",
            }
        weights = {}
        for task in ("turn", "response"):
            path = train / plan["weights"][task]["path_relative_to_train_mj"]
            if not path.is_file():
                raise RuntimeError(f"missing weight: {path}")
            actual = hash_file(path)
            if actual != plan["weights"][task]["expected_file_sha256"]:
                raise RuntimeError(f"{task} weight hash mismatch")
            weights[task] = {"path": str(path), "sha256": actual, "matches": True}
        # Validate every native config for the requested stage without writing.
        for candidate_id in CANDIDATE_IDS:
            config = build_native_config(
                plan,
                stage=args.stage,
                candidate_id=candidate_id,
                handoff_sha256=handoff_sha,
            )
            protocol = build_native_protocol(
                plan, stage=args.stage, candidate_id=candidate_id
            )
            validate_screening_plan(protocol, config)
        print(
            json.dumps(
                {
                    "valid": True,
                    "experiment_id": plan["experiment_id"],
                    "stage": args.stage,
                    "handoff_plan_sha256": handoff_sha,
                    "seeds": (
                        ENGINEERING_SEEDS
                        if args.stage == "smoke"
                        else SCREENING_SEEDS
                    ),
                    "seed_clusters": 2 if args.stage == "smoke" else 128,
                    "planned_half_games": 48 if args.stage == "smoke" else 3072,
                    "candidates": CANDIDATE_IDS,
                    "weights": weights,
                    "confirmation_reserved_do_not_run": RESERVED_SEEDS,
                    "note": "dry-run validated plan, service configs, source refs and weight hashes; no services or games started",
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.stage == "smoke":
        if out.exists():
            raise RuntimeError("output directory already exists; refusing to overwrite")
        init_output(out, handoff)
        cmdlog = CommandLog(out / "commands.jsonl")
        cmdlog._write(
            {
                "event": "run_start",
                "experiment_id": plan["experiment_id"],
                "stage": "smoke",
                "handoff_plan": str(handoff),
                "handoff_plan_sha256": handoff_sha,
            }
        )
        verify_source_refs(cmdlog, train)
        seed_check = scan_seed_history(
            out,
            [ROOT / "outputs", train / "outputs"],
            set(SCREENING_SEEDS),
            set(RESERVED_SEEDS),
            engineering_seeds=set(ENGINEERING_SEEDS),
        )
        write_json(out / "seed_history_check.json", seed_check)
        if not seed_check["valid"]:
            write_review(
                out,
                plan,
                "blocked_seed_overlap",
                blocked=seed_check["blocking_matches"],
                protocol_sha=None,
            )
            write_checksums(out)
            return 1
        sources = {
            BASELINE_ID: export_source(
                cmdlog,
                train,
                out,
                BASELINE_ID,
                "1e5092f344fb69bcdf4697f683f2ecf15b9865d8",
                tag="v1.3.0",
                reuse=False,
            ),
            "candidate": export_source(
                cmdlog,
                train,
                out,
                "candidate_v1_4_0",
                "0710aa0f31d1d3c8d96ddd7cc31bf832cd68ada6",
                tag=None,
                reuse=False,
            ),
        }
        # The candidate backend ID is the profile selector in most helpers.
        sources["candidate_v1_4_0"] = sources.pop("candidate")
        weights = compute_weights(cmdlog, train, plan)
        copy_handoff_service_configs(out, service_configs)
        environment = environment_snapshot(
            cmdlog, train, out, plan, service_configs, sources, weights, handoff_sha
        )
        write_json(out / "environment.json", environment)
        planned = write_planned_configs(out, plan, handoff_sha)
        tests = run_tests(cmdlog, train, out / "smoke")
        if not tests["valid"]:
            write_review(
                out, plan, "tests_failed", environment=environment,
                sources=sources, weights=weights, protocol_sha=None,
            )
            write_checksums(out)
            return 1
        contract = run_contract(
            cmdlog, train, out, out / "smoke", sources, service_configs
        )
        if not contract["valid"]:
            write_review(
                out, plan, "contract_failed", environment=environment,
                sources=sources, weights=weights, protocol_sha=None,
            )
            write_checksums(out)
            return 1
        protocol_sha = register_protocol(
            out,
            plan,
            handoff_sha,
            service_configs,
            sources,
            weights,
            environment,
            planned,
            hash_file(out / "smoke/contract/evidence.json"),
            tests,
        )
        groups = {}
        for candidate_id in CANDIDATE_IDS:
            print(
                f"[smoke] starting {candidate_id}: 16 half-games", flush=True
            )
            groups[candidate_id] = run_group(
                cmdlog,
                train,
                out,
                plan,
                "smoke",
                candidate_id,
                sources,
                service_configs,
                weights,
                protocol_sha,
                plan["runtime"]["timeout_seconds_per_group"],
                args.replay_workers,
            )
            print(
                f"[smoke] {candidate_id}: valid={groups[candidate_id].get('valid')}",
                flush=True,
            )
            if not groups[candidate_id].get("valid"):
                break
        smoke = smoke_summary(out, groups, contract, tests, protocol_sha)
        write_review(
            out,
            plan,
            "smoke_completed" if smoke["valid"] else "smoke_failed",
            environment=environment,
            sources=sources,
            weights=weights,
            smoke=smoke,
            protocol_sha=protocol_sha,
        )
        write_checksums(out)
        return 0 if smoke["valid"] else 1

    # screen stage
    if not out.exists():
        raise RuntimeError("screen stage requires the smoke output directory")
    cmdlog = CommandLog(out / "commands.jsonl")
    cmdlog._write(
        {"event": "run_start", "experiment_id": plan["experiment_id"], "stage": "screen"}
    )
    registration = verify_screen_ready(
        out, handoff_sha, train, cmdlog, service_configs
    )
    protocol_sha = hash_file(out / "protocol.registered.json")
    sources = json.loads(
        (out / "source_archives/SHA256SUMS.json").read_text(encoding="utf-8")
    )
    verify_sources(out, sources)
    weights = json.loads(
        (out / "environment.json").read_text(encoding="utf-8")
    )["weights"]
    recomputed = compute_weights(cmdlog, train, plan)
    if recomputed != weights:
        raise RuntimeError("weight identity changed after smoke registration")
    for candidate_id in CANDIDATE_IDS:
        if (out / "screen" / candidate_id).exists():
            raise RuntimeError(f"screen group already exists: {candidate_id}")
    groups = {}
    for candidate_id in CANDIDATE_IDS:
        print(
            f"[screen] starting {candidate_id}: 1024 half-games", flush=True
        )
        groups[candidate_id] = run_group(
            cmdlog,
            train,
            out,
            plan,
            "screen",
            candidate_id,
            sources,
            service_configs,
            weights,
            protocol_sha,
            plan["runtime"]["timeout_seconds_per_group"],
            args.replay_workers,
        )
        print(
            f"[screen] {candidate_id}: valid={groups[candidate_id].get('valid')}",
            flush=True,
        )
        if not groups[candidate_id].get("valid"):
            break
    smoke = json.loads((out / "smoke/summary.json").read_text(encoding="utf-8"))
    if not all(group.get("valid") for group in groups.values()):
        write_review(
            out,
            plan,
            "incomplete",
            sources=sources,
            weights=weights,
            smoke=smoke,
            protocol_sha=protocol_sha,
        )
        write_checksums(out)
        return 1
    metrics = analyze_screen(out, smoke)
    write_review(
        out,
        plan,
        "completed",
        sources=sources,
        weights=weights,
        smoke=smoke,
        metrics=metrics,
        protocol_sha=protocol_sha,
    )
    cmdlog.record_inproc(
        "finalize_artifacts",
        out,
        utcnow(),
        utcnow(),
        0,
        note="handoff review and checksums complete",
    )
    write_checksums(out)
    return 0


def copy_handoff_service_configs(out: Path, service_configs: dict) -> None:
    target = out / "resolved_configs" / "handoff"
    target.mkdir(parents=True, exist_ok=True)
    records = {}
    for item in [service_configs["baseline"], *service_configs["candidates"].values()]:
        source = Path(item["path"])
        destination = target / source.name
        destination.write_bytes(source.read_bytes())
        digest = hash_file(destination)
        if digest != item["sha256"]:
            raise RuntimeError(f"copied service config hash mismatch: {source.name}")
        records[source.name] = {
            "sha256": digest,
            "source": str(source),
        }
    write_json(target / "SHA256SUMS.json", records)


if __name__ == "__main__":
    raise SystemExit(main())
