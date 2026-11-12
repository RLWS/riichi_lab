"""A0 arena configuration contract; no inference or scheduling is run here."""

from __future__ import annotations

import math
from pathlib import Path
import re
from urllib.parse import urlparse

import yaml

PROFILES = {
    "full_state_v3": {
        "tag": None,
        "commit": "57e8fb336ed4aa7b19afd8ca1ef4ee4c41a96ed2",
        "feature_schema": "full_state_v3",
        "state_dims": (11969, 12012),
        "action_dims": (249, 251),
        "calibration_contract": "action_biases_v1",
        "source_kind": "commit",
    },
    "legacy_v1_2": {
        "tag": "v1.2.0",
        "commit": "758981ab9fb457c882ab9c577c9377976f97f42b",
        "feature_schema": None,
        "state_dims": (8173, 8216),
        "action_dims": (89, 91),
        "calibration_contract": "legacy_pass_bias",
        "source_kind": "tag",
    },
    "full_state_v2": {
        "tag": "v1.3.0",
        "commit": "1e5092f344fb69bcdf4697f683f2ecf15b9865d8",
        "feature_schema": "full_state_v2",
        "state_dims": (11969, 12012),
        "action_dims": (89, 91),
        "calibration_contract": "legacy_pass_bias",
        "source_kind": "tag",
    },
    # v1.4.0 keeps the full_state_v2 tensor layout but changes only the
    # response calibration contract.  It is deliberately a separate deployment
    # identity: the old full_state_v2 profile stays pinned to v1.3.0.
    "full_state_v2_action_biases": {
        "tag": None,
        "commit": "0710aa0f31d1d3c8d96ddd7cc31bf832cd68ada6",
        "feature_schema": "full_state_v2",
        "state_dims": (11969, 12012),
        "action_dims": (89, 91),
        "calibration_contract": "action_biases_v1",
        "source_kind": "commit",
    },
}

CALIBRATION_CONTRACTS = ("legacy_pass_bias", "action_biases_v1")
ACTION_BIAS_KEYS = frozenset({"ron", "kan", "pon", "chi"})


def calibration_contract(backend: dict) -> str:
    profile = PROFILES[backend["adapter_profile"]]
    return backend.get("calibration_contract", profile["calibration_contract"])


def effective_calibration(backend: dict, response_metadata: dict) -> dict:
    """Return the contract-shaped calibration actually used by a deployment."""
    if calibration_contract(backend) == "legacy_pass_bias":
        bias = backend.get("response", {}).get("pass_bias")
        if bias is None:
            bias = response_metadata.get("default_pass_bias")
        return {
            "pass_bias": bias,
            "no_yaku_call_penalty": response_metadata.get("no_yaku_call_penalty"),
        }
    return {
        "calibration_contract": "action_biases_v1",
        "action_biases": response_metadata.get("action_biases"),
        "no_yaku_call_penalty": response_metadata.get("no_yaku_call_penalty"),
    }


def validate_expected_calibration(backend: dict, expected: dict) -> None:
    """Reject a protocol whose frozen calibration cannot match the deployment."""
    name = backend.get("adapter_profile", "backend")
    contract = calibration_contract(backend)
    if not isinstance(expected, dict):
        raise ValueError(f"invalid expected calibration: {name}")
    if contract == "legacy_pass_bias":
        if set(expected) != {"pass_bias", "no_yaku_call_penalty"}:
            raise ValueError(f"invalid legacy expected calibration: {name}")
        bias = expected["pass_bias"]
        if (
            type(bias) not in (int, float)
            or not math.isfinite(bias)
            or bias < 0
        ):
            raise ValueError(f"invalid expected pass_bias: {name}")
        configured = backend.get("response", {}).get("pass_bias")
        if configured is not None and configured != bias:
            raise ValueError(f"configured pass_bias differs from protocol: {name}")
    elif contract == "action_biases_v1":
        if set(expected) != {
            "calibration_contract",
            "action_biases",
            "no_yaku_call_penalty",
        } or expected["calibration_contract"] != "action_biases_v1":
            raise ValueError(f"invalid action_biases expected calibration: {name}")
        if expected["action_biases"] != backend.get("response", {}).get(
            "action_biases"
        ):
            raise ValueError(f"action_biases differ from protocol: {name}")
        if expected["no_yaku_call_penalty"] != backend.get("response", {}).get(
            "no_yaku_call_penalty"
        ):
            raise ValueError(f"no_yaku_call_penalty differs from protocol: {name}")
        if not valid_action_biases(expected["action_biases"]):
            raise ValueError(f"invalid expected action_biases: {name}")
    else:  # pragma: no cover - guarded by load_arena_config
        raise ValueError(f"unknown calibration contract: {contract}")
    penalty = expected["no_yaku_call_penalty"]
    if (
        type(penalty) not in (int, float)
        or not math.isfinite(penalty)
        or penalty < 0
    ):
        raise ValueError(f"invalid expected no_yaku_call_penalty: {name}")


def valid_action_biases(value) -> bool:
    if not isinstance(value, dict) or set(value) != ACTION_BIAS_KEYS:
        return False
    for raw in value.values():
        if type(raw) not in (int, float) or not math.isfinite(raw) or raw < 0:
            return False
    return True


def load_arena_config(path: str | Path) -> dict:
    """Validate backend identities/references now; A3 will require runnable seeds."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or type(config.get("config_version")) is not int
        or config["config_version"] != 1
    ):
        raise ValueError("arena config_version must be 1")
    backends = config.get("backends")
    if "backend" in config or not isinstance(backends, dict) or not backends:
        raise ValueError(
            "use the backends registry; the singleton backend contract is unsupported"
        )
    urls = set()
    for name, backend in backends.items():
        if not isinstance(name, str) or not isinstance(backend, dict):
            raise ValueError("invalid backend entry")
        profile = PROFILES.get(backend.get("adapter_profile"))
        if profile is None:
            raise ValueError(f"unknown adapter_profile: {name}")
        revision = backend.get("revision", {})
        if not isinstance(revision, dict) or revision.get("commit") != profile["commit"]:
            raise ValueError(f"profile/revision mismatch: {name}")
        if profile["tag"] is not None:
            if revision.get("tag") != profile["tag"]:
                raise ValueError(f"profile/revision mismatch: {name}")
        elif revision.get("tag") is not None:
            raise ValueError(f"commit-pinned profile must not claim a tag: {name}")
        contract = backend.get(
            "calibration_contract", profile["calibration_contract"]
        )
        if (
            contract not in CALIBRATION_CONTRACTS
            or contract != profile["calibration_contract"]
        ):
            raise ValueError(f"profile/calibration contract mismatch: {name}")
        url = backend.get("api_base", "")
        if not isinstance(url, str):
            raise ValueError(f"invalid api_base: {name}")
        parsed = urlparse(url)
        if (
            backend.get("mode") != "http"
            or parsed.scheme not in ("http", "https")
            or not parsed.netloc
        ):
            raise ValueError(f"invalid HTTP backend: {name}")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("api_base cannot embed credentials, query or fragment")
        if url.rstrip("/") in urls:
            raise ValueError("backend instances must have distinct endpoints")
        urls.add(url.rstrip("/"))
        timeout = backend.get("timeout_seconds")
        if (
            type(timeout) not in (float, int)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(f"invalid timeout: {name}")
        response = backend.get("response", {})
        if not isinstance(response, dict):
            raise ValueError(f"invalid response config: {name}")
        if contract == "legacy_pass_bias":
            if "action_biases" in response:
                raise ValueError(
                    f"legacy calibration must not carry action_biases: {name}"
                )
            if set(response) - {"pass_bias", "no_yaku_call_penalty"}:
                raise ValueError(f"unknown legacy response field: {name}")
            bias = response.get("pass_bias")
            if bias is not None and (
                type(bias) not in (int, float)
                or not math.isfinite(bias)
                or bias < 0
            ):
                raise ValueError(f"invalid pass_bias: {name}")
        else:
            if "pass_bias" in response:
                raise ValueError(
                    f"action_biases calibration must not carry pass_bias: {name}"
                )
            if set(response) - {"action_biases", "no_yaku_call_penalty"}:
                raise ValueError(f"unknown action_biases response field: {name}")
            if not valid_action_biases(response.get("action_biases")):
                raise ValueError(f"invalid action_biases: {name}")
            if "no_yaku_call_penalty" not in response:
                raise ValueError(f"action_biases requires explicit penalty: {name}")
        if "no_yaku_call_penalty" in response:
            penalty = response["no_yaku_call_penalty"]
            if (
                type(penalty) not in (int, float)
                or not math.isfinite(penalty)
                or penalty < 0
            ):
                raise ValueError(f"invalid no_yaku_call_penalty: {name}")
        identity = backend.get("model_identity", {})
        if not isinstance(identity, dict):
            raise ValueError(f"invalid model_identity: {name}")
        for task in ("turn", "response"):
            if not re.fullmatch(
                r"[0-9a-f]{64}", str(identity.get(f"{task}_sha256", ""))
            ):
                raise ValueError(f"missing weight hash: {name}/{task}")
    matchups = config.get("matchups")
    if not isinstance(matchups, list) or not matchups:
        raise ValueError("matchups must be nonempty")
    ids = set()
    for matchup in matchups:
        if (
            not isinstance(matchup, dict)
            or not isinstance(matchup.get("id"), str)
            or matchup["id"] in ids
        ):
            raise ValueError("matchup IDs must be unique strings")
        ids.add(matchup["id"])
        seats = matchup.get("seats")
        if (
            not isinstance(seats, list)
            or len(seats) != 4
            or any(not isinstance(s, str) or s not in backends for s in seats)
        ):
            raise ValueError("every matchup needs four known backend references")
        focal = matchup.get("focal_seat")
        rotation = matchup.get("rotation")
        if rotation == "cyclic4":
            if type(focal) is not int or not 0 <= focal < 4:
                raise ValueError("cyclic4 requires a focal seat")
        elif rotation == "unique6":
            if (
                focal is not None
                or len(set(seats)) != 2
                or any(seats.count(s) != 2 for s in set(seats))
            ):
                raise ValueError("unique6 requires 2+2 seats and no focal seat")
        else:
            raise ValueError("unknown rotation")
    return config
