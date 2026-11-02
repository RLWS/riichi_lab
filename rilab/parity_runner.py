"""Run each immutable train_mj revision in a separate development process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import yaml

from rilab.config import PROFILES


def run_parity(config_path: str | Path, output_dir: str | Path) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["parity"]
    configured_root = config.get("train_mj_root") or os.environ.get("TRAIN_MJ_ROOT")
    if not configured_root:
        raise ValueError("parity requires train_mj_root or TRAIN_MJ_ROOT")
    root = Path(configured_root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    fixtures = root / "tests/fixtures/tenhou"
    if not fixtures.is_dir():
        raise ValueError("training XML fixtures are missing")
    reports = []
    for profile in config.get("profiles", PROFILES):
        expected = PROFILES[profile]
        commit = subprocess.check_output(
            ["git", "rev-parse", f"{expected['tag']}^{{commit}}"], cwd=root, text=True
        ).strip()
        if commit != expected["commit"]:
            raise ValueError(f"tag moved: {expected['tag']}")
        source = out / f"source_{profile}"
        archive = out / f"{profile}.zip"
        subprocess.run(
            [
                "git",
                "archive",
                "--format=zip",
                "--output",
                str(archive),
                expected["tag"],
            ],
            cwd=root,
            check=True,
        )
        with zipfile.ZipFile(archive) as z:
            z.extractall(source)
        report_dir = out / profile
        with (out / f"{profile}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [
                    config.get("python") or sys.executable,
                    str(repo / "scripts/check_xml_parity.py"),
                    "--source-root",
                    str(source),
                    "--fixtures-root",
                    str(fixtures),
                    "--profile",
                    profile,
                    "--output-dir",
                    str(report_dir),
                ],
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        summary_path = report_dir / "summary.json"
        if not summary_path.exists():
            reports.append(
                {
                    "profile": profile,
                    "valid": False,
                    "error": "worker failed; see log",
                    "exit_code": process.returncode,
                }
            )
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["revision"] = {"tag": expected["tag"], "commit": commit}
            summary["exit_code"] = process.returncode
            http_output = out / f"{profile}_http.json"
            with (out / f"{profile}.log").open("a", encoding="utf-8") as log:
                http_process = subprocess.run(
                    [
                        config.get("python") or sys.executable,
                        str(repo / "scripts/check_http_contract.py"),
                        "--source-root",
                        str(source),
                        "--weights-root",
                        str(root),
                        "--profile",
                        profile,
                        "--output",
                        str(http_output),
                    ],
                    cwd=repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                )
            summary["http_contract"] = (
                json.loads(http_output.read_text(encoding="utf-8"))
                if http_output.exists()
                else {"valid": False, "error": "HTTP contract worker failed"}
            )
            summary["valid"] = (
                summary["valid"]
                and http_process.returncode == 0
                and summary["http_contract"]["valid"]
            )
            reports.append(summary)
        print(f"{profile}: {'PASS' if reports[-1]['valid'] else 'FAIL'}", flush=True)
    report = {
        "valid": bool(reports) and all(r["valid"] for r in reports),
        "profiles": reports,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
