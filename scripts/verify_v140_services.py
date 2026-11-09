"""HTTP contract and raw-consistency checks for the v1.4.0 screening services."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACTION_KEYS = {"ron", "kan", "pon", "chi"}
EXPECTED_DIMS = {
    "turn": (11969, 89),
    "response": (12012, 91),
}


def load_payloads(payload_source: Path | None, target: Path) -> dict:
    fixtures = ROOT / "tests/fixtures/http_contract"
    turn = json.loads((fixtures / "golden_turn_request.json").read_text(encoding="utf-8"))
    response = json.loads(
        (fixtures / "golden_response_request.json").read_text(encoding="utf-8")
    )
    response.pop("pass_bias", None)
    payloads = {
        "turn": [{"payload": turn, "source": "tests/fixtures/http_contract/golden_turn_request.json"}],
        "response": [
            {
                "payload": response,
                "source": "tests/fixtures/http_contract/golden_response_request.json",
            }
        ],
    }
    if payload_source and payload_source.is_dir():
        wanted = {"ron", "chankan", "daiminkan", "pon", "chi"}
        found = set()
        files = sorted((payload_source / "decisions").glob("*.jsonl.gz"))[:80]
        for path in files:
            try:
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        record = json.loads(line)
                        if record.get("task") != "response" or "state" not in record:
                            continue
                        if record["state"].get("feature_schema") != "full_state_v2":
                            continue
                        kinds = {c.get("kind") for c in record.get("candidates", [])}
                        if not (kinds & wanted):
                            continue
                        kind = sorted(kinds & wanted)[0]
                        if kind in found:
                            continue
                        found.add(kind)
                        payloads["response"].append(
                            {
                                "payload": {
                                    "state": record["state"],
                                    "legal_actions": record["candidates"],
                                    "top_k": min(5, len(record["candidates"])),
                                },
                                "source": f"{path.name}:decision_{record['decision_id']}",
                            }
                        )
                        if len(found) == len(wanted):
                            break
            except (OSError, json.JSONDecodeError):
                continue
            if len(found) == len(wanted):
                break
    target.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    return payloads


def request_for(payload: dict, service: dict) -> dict:
    request = deepcopy(payload)
    request.pop("pass_bias", None)
    request.pop("action_biases", None)
    if service["calibration_contract"] == "legacy_pass_bias":
        request["pass_bias"] = service["response"]["pass_bias"]
    else:
        request["action_biases"] = deepcopy(service["response"]["action_biases"])
    return request


def check_models(service: dict, client: httpx.Client) -> dict:
    response = client.get(service["api_base"].rstrip("/") + "/v1/models")
    response.raise_for_status()
    models = response.json()["models"]
    errors = []
    for task, (state_dim, action_dim) in EXPECTED_DIMS.items():
        info = models.get(task, {})
        if (
            info.get("task") != task
            or info.get("feature_schema") != "full_state_v2"
            or info.get("state_dim") != state_dim
            or info.get("action_dim") != action_dim
            or not isinstance(info.get("candidate_limit"), int)
            or info["candidate_limit"] <= 0
        ):
            errors.append(f"{task} model contract mismatch")
    response_info = models.get("response", {})
    if service["calibration_contract"] == "legacy_pass_bias":
        if "action_biases" in response_info:
            errors.append("legacy metadata exposed action_biases")
        if response_info.get("default_pass_bias") != service["response"]["pass_bias"]:
            errors.append("legacy default_pass_bias mismatch")
    else:
        if "default_pass_bias" in response_info:
            errors.append("action_biases metadata exposed default_pass_bias")
        if response_info.get("action_biases") != service["response"]["action_biases"]:
            errors.append("service default action_biases mismatch")
    if response_info.get("no_yaku_call_penalty") != service["response"][
        "no_yaku_call_penalty"
    ]:
        errors.append("service no_yaku_call_penalty mismatch")
    return {"passed": not errors, "errors": errors, "models": models}


def check_prediction_echo(service: dict, result: dict) -> list[str]:
    errors = []
    if result.get("task") != "response":
        errors.append("response task mismatch")
    if service["calibration_contract"] == "legacy_pass_bias":
        if result.get("pass_bias") != service["response"]["pass_bias"]:
            errors.append("response pass_bias mismatch")
        if "action_biases" in result:
            errors.append("legacy response exposed action_biases")
    else:
        if result.get("action_biases") != service["response"]["action_biases"]:
            errors.append("response action_biases mismatch")
        if "pass_bias" in result:
            errors.append("action_biases response exposed pass_bias")
    if result.get("no_yaku_call_penalty") != service["response"][
        "no_yaku_call_penalty"
    ]:
        errors.append("response no_yaku_call_penalty mismatch")
    return errors


def raw_view(result: dict) -> list[dict]:
    return [
        {
            "rank": item.get("rank"),
            "index": item.get("index"),
            "action": item.get("action"),
            "score": item.get("score"),
            "prob": item.get("prob"),
        }
        for item in result.get("predictions", [])
    ]


def compare_raw(views: dict) -> dict:
    names = list(views)
    max_score = 0.0
    max_prob = 0.0
    mismatches = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            a, b = views[left], views[right]
            if len(a) != len(b):
                mismatches.append(f"{left} vs {right}: prediction count differs")
                continue
            for item_a, item_b in zip(a, b, strict=True):
                if (
                    item_a["rank"] != item_b["rank"]
                    or item_a["index"] != item_b["index"]
                    or item_a["action"] != item_b["action"]
                ):
                    mismatches.append(f"{left} vs {right}: rank/index/action differs")
                if item_a["score"] is not None and item_b["score"] is not None:
                    max_score = max(max_score, abs(item_a["score"] - item_b["score"]))
                if item_a["prob"] is not None and item_b["prob"] is not None:
                    max_prob = max(max_prob, abs(item_a["prob"] - item_b["prob"]))
    return {
        "valid": not mismatches and max_score <= 1e-6 and max_prob <= 1e-7,
        "max_abs_score_diff": max_score,
        "max_abs_prob_diff": max_prob,
        "mismatches": mismatches,
    }


def send_raw(client: httpx.Client, path: str, payload: dict):
    return client.post(
        path,
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )


def rejection_cases(service: dict, payload: dict) -> list[dict]:
    base = request_for(payload, service)
    cases = []

    def add(name, mutate):
        request = deepcopy(base)
        mutate(request)
        cases.append((name, request, False))

    add("pass_bias_null", lambda r: r.__setitem__("pass_bias", None))
    add("pass_bias_scalar", lambda r: r.__setitem__("pass_bias", 1.5))
    add(
        "action_biases_missing_chi",
        lambda r: r["action_biases"].pop("chi"),
    )
    add(
        "action_biases_unknown_key",
        lambda r: r["action_biases"].__setitem__("foo", 1.0),
    )
    add(
        "action_biases_negative",
        lambda r: r["action_biases"].__setitem__("ron", -0.1),
    )
    add(
        "request_penalty_override",
        lambda r: r.__setitem__("no_yaku_call_penalty", 0.1),
    )
    add(
        "action_biases_nan",
        lambda r: r["action_biases"].__setitem__("ron", float("nan")),
    )
    add(
        "action_biases_infinity",
        lambda r: r["action_biases"].__setitem__("ron", float("inf")),
    )
    cases.append(("action_biases_null_uses_default", deepcopy(base) | {"action_biases": None}, True))
    omitted = deepcopy(base)
    omitted.pop("action_biases", None)
    cases.append(("action_biases_omitted_uses_default", omitted, True))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--services", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--payload-source",
        type=Path,
        help="optional batch directory used to recover real response states",
    )
    args = parser.parse_args(argv)
    services = json.loads(args.services.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads(args.payload_source, output.parent / "payloads.json")
    checks = []
    raw_consistency = {}
    requests_evidence = {}
    with httpx.Client(timeout=60.0) as client:
        models = {}
        for name, service in services.items():
            try:
                model_check = check_models(service, client)
            except Exception as exc:  # pragma: no cover - live service failure
                model_check = {"passed": False, "errors": [str(exc)], "models": None}
            models[name] = model_check
            checks.append({"name": "models", "service": name, **model_check})
        for task in ("turn", "response"):
            for index, item in enumerate(payloads[task]):
                views = {}
                for name, service in services.items():
                    request = (
                        deepcopy(item["payload"])
                        if task == "turn"
                        else request_for(item["payload"], service)
                    )
                    try:
                        response = client.post(
                            service["api_base"].rstrip("/")
                            + f"/v1/predict/{task}",
                            json=request,
                        )
                        response.raise_for_status()
                        result = response.json()
                    except Exception as exc:
                        checks.append(
                            {
                                "name": f"{task}_{index}_valid",
                                "service": name,
                                "passed": False,
                                "errors": [str(exc)],
                            }
                        )
                        continue
                    errors = []
                    if result.get("selected_action") not in request["legal_actions"]:
                        errors.append("selected_action not in legal_actions")
                    if task == "response":
                        errors.extend(check_prediction_echo(service, result))
                    views[name] = raw_view(result)
                    if index == 0:
                        requests_evidence[f"{name}_{task}"] = {
                            "task": task,
                            "request": request,
                            "response": result,
                        }
                    checks.append(
                        {
                            "name": f"{task}_{index}_valid",
                            "service": name,
                            "passed": not errors,
                            "errors": errors,
                        }
                    )
                if len(views) == len(services):
                    comparison = compare_raw(views)
                    raw_consistency[f"{task}_{index}"] = {
                        "source": item["source"],
                        **comparison,
                    }
                    checks.append(
                        {
                            "name": f"{task}_{index}_raw_consistency",
                            "service": "all",
                            "passed": comparison["valid"],
                            "errors": comparison["mismatches"],
                        }
                    )
        for name, service in services.items():
            if service["calibration_contract"] != "action_biases_v1":
                continue
            for case_name, request, should_succeed in rejection_cases(
                service, payloads["response"][0]["payload"]
            ):
                try:
                    response = send_raw(
                        client,
                        service["api_base"].rstrip("/") + "/v1/predict/response",
                        request,
                    )
                    result = response.json()
                except Exception as exc:
                    checks.append(
                        {
                            "name": case_name,
                            "service": name,
                            "passed": False,
                            "errors": [str(exc)],
                        }
                    )
                    continue
                if should_succeed:
                    errors = []
                    if response.status_code != 200:
                        errors.append(f"HTTP {response.status_code}")
                    else:
                        errors.extend(check_prediction_echo(service, result))
                    checks.append(
                        {
                            "name": case_name,
                            "service": name,
                            "passed": not errors,
                            "errors": errors,
                            "status": response.status_code,
                        }
                    )
                else:
                    passed = response.status_code >= 400 and "error" in result
                    checks.append(
                        {
                            "name": case_name,
                            "service": name,
                            "passed": passed,
                            "errors": []
                            if passed
                            else [f"expected rejection, got HTTP {response.status_code}"],
                            "status": response.status_code,
                            "error": result.get("error"),
                        }
                    )
        for name, service in services.items():
            if service["calibration_contract"] != "legacy_pass_bias":
                continue
            for label, request in (
                (
                    "legacy_explicit_pass_bias",
                    request_for(payloads["response"][0]["payload"], service),
                ),
                (
                    "legacy_omitted_pass_bias",
                    {
                        key: value
                        for key, value in request_for(
                            payloads["response"][0]["payload"], service
                        ).items()
                        if key != "pass_bias"
                    },
                ),
            ):
                response = client.post(
                    service["api_base"].rstrip("/") + "/v1/predict/response",
                    json=request,
                )
                result = response.json() if response.content else {}
                errors = check_prediction_echo(service, result) if response.status_code == 200 else [
                    f"HTTP {response.status_code}"
                ]
                checks.append(
                    {
                        "name": label,
                        "service": name,
                        "passed": not errors,
                        "errors": errors,
                        "status": response.status_code,
                    }
                )
    payload = {
        "valid": all(check["passed"] for check in checks),
        "services": {
            name: {
                "api_base": service["api_base"],
                "calibration_contract": service["calibration_contract"],
                "response": service["response"],
            }
            for name, service in services.items()
        },
        "models": models,
        "checks": checks,
        "raw_consistency": raw_consistency,
        "requests": requests_evidence,
        "payload_sources": {
            task: [item["source"] for item in items] for task, items in payloads.items()
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"valid": payload["valid"], "output": str(output)}))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
