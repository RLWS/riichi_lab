"""Real-checkpoint A2 API contract checks in an isolated source environment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.append(str(repo / ".venv/Lib/site-packages"))
    import torch
    from fastapi.testclient import TestClient
    from mjai.api.app import create_app
    from mjai.utils.config import load_resolved_yaml
    from rilab.adapters import create_adapter
    from rilab.config import PROFILES

    torch.set_num_threads(1)
    config = load_resolved_yaml(args.source_root / "configs/api_local.yaml")
    config["device"] = "cpu"
    config["api"].update(debug_log_requests=False, debug_log_responses=False)
    config["security"] = {"allow_without_api_key": True}
    for task in ("turn", "response"):
        config["models"][task]["checkpoint"] = str(
            args.weights_root / config["models"][task]["checkpoint"]
        )
    client = TestClient(create_app(config=config))
    models = client.get("/v1/models").json()["models"]
    expected = PROFILES[args.profile]
    for index, task in enumerate(("turn", "response")):
        assert models[task]["state_dim"] == expected["state_dims"][index]
        assert models[task]["action_dim"] == expected["action_dims"][index]
        assert models[task].get("feature_schema") == expected["feature_schema"]
    fixtures = repo / "tests/fixtures/http_contract"
    requests = {
        task: json.loads(
            (fixtures / f"golden_{task}_request.json").read_text(encoding="utf-8")
        )
        for task in ("turn", "response")
    }
    for request in requests.values():
        request["state"] = create_adapter(args.profile).project(request["state"])
    checks = []
    for task, request in requests.items():
        response = client.post(f"/v1/predict/{task}", json=request)
        assert response.status_code == 200, response.text
        assert response.json()["selected_action"] in request["legal_actions"]
        checks.append({"case": f"{task}_valid", "status": 200})
    for bias in (None, 0):
        request = deepcopy(requests["response"])
        request["pass_bias"] = bias
        response = client.post("/v1/predict/response", json=request)
        assert response.status_code == 200
        assert response.json()["pass_bias"] == (
            models["response"]["default_pass_bias"] if bias is None else 0
        )
        checks.append({"case": f"calibration_{bias}", "status": 200})
    cases = []
    request = deepcopy(requests["response"])
    del request["state"]["round_wind"]
    cases.append(("missing_state", "response", request, 422))
    request = deepcopy(requests["response"])
    request["legal_actions"] *= models["response"]["candidate_limit"] + 1
    cases.append(("candidate_limit", "response", request, 413))
    request = deepcopy(requests["turn"])
    request["legal_actions"] = []
    cases.append(("empty_candidates", "turn", request, 400))
    request = deepcopy(requests["turn"])
    request["top_k"] = 0
    cases.append(("invalid_top_k", "turn", request, 400))
    request = deepcopy(requests["turn"])
    request["legal_actions"] = [{"kind": "chi"}]
    cases.append(("wrong_task_action", "turn", request, 422))
    for name, task, request, status in cases:
        response = client.post(f"/v1/predict/{task}", json=request)
        assert response.status_code == status, (
            name,
            response.status_code,
            response.text,
        )
        assert "error" in response.json()
        checks.append({"case": name, "status": status})
    result = {
        "profile": args.profile,
        "valid": True,
        "checks": checks,
        "transport": "FastAPI TestClient",
        "device": "cpu",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
