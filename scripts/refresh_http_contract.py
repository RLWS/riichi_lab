"""Refresh real-weight A0 responses through FastAPI TestClient, without a port.

Run in the train_mj development Python environment:
python scripts/refresh_http_contract.py --train-mj-root <checkout>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mj-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.train_mj_root.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    expected = subprocess.check_output(
        ["git", "rev-parse", "v1.3.0^{commit}"], cwd=root, text=True
    ).strip()
    if (
        commit != expected
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
    ):
        raise RuntimeError("requires a clean v1.3.0 checkout")
    sys.path.insert(0, str(root))
    from fastapi.testclient import TestClient
    from mjai.api.app import create_app
    from mjai.serving.predictor import build_predictor_from_config
    from mjai.utils.config import load_resolved_yaml

    config = load_resolved_yaml(root / "configs/api_local.yaml")
    config["device"] = "cpu"
    config["api"]["debug_log_requests"] = False
    config["api"]["debug_log_responses"] = False
    config["security"] = {"allow_without_api_key": True}
    predictor = build_predictor_from_config(config, base_dir=root)
    client = TestClient(create_app(config=config, predictor=predictor))
    out = Path(__file__).resolve().parents[1] / "tests/fixtures/http_contract"
    models = client.get("/v1/models")
    models.raise_for_status()
    health = client.get("/health")
    health.raise_for_status()
    report = {
        "api_base": "asgi://train_mj/v1.3.0",
        "transport": "fastapi_testclient",
        "health": health.json(),
        "models": models.json(),
    }
    hashes = {}
    for task in ("turn", "response"):
        path = root / config["models"][task]["checkpoint"]
        hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["models"]["models"][task]["checkpoint"] = path.relative_to(
            root
        ).as_posix()
        request = json.loads(
            (out / f"golden_{task}_request.json").read_text(encoding="utf-8")
        )
        response = client.post(f"/v1/predict/{task}", json=request)
        response.raise_for_status()
        result = response.json()
        assert result["selected_action"] in request["legal_actions"]
        result["model"]["checkpoint"] = path.relative_to(root).as_posix()
        (out / f"baseline_{task}_response.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    (out / "baseline_models.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    evidence = {
        "train_mj_tag": "v1.3.0",
        "train_mj_commit": commit,
        "device": "cpu",
        "transport": "fastapi_testclient",
        "checkpoint_sha256": hashes,
        "turn_status": 200,
        "response_status": 200,
    }
    (out / "verification.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
