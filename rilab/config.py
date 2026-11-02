"""A0 arena configuration contract; no inference or scheduling is run here."""

from __future__ import annotations

import math
from pathlib import Path
import re
from urllib.parse import urlparse

import yaml

PROFILES = {
    "legacy_v1_2": {
        "tag": "v1.2.0",
        "commit": "758981ab9fb457c882ab9c577c9377976f97f42b",
        "feature_schema": None,
        "state_dims": (8173, 8216),
        "action_dims": (89, 91),
    },
    "full_state_v2": {
        "tag": "v1.3.0",
        "commit": "1e5092f344fb69bcdf4697f683f2ecf15b9865d8",
        "feature_schema": "full_state_v2",
        "state_dims": (11969, 12012),
        "action_dims": (89, 91),
    },
}


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
        if not isinstance(revision, dict) or any(
            revision.get(k) != profile[k] for k in ("tag", "commit")
        ):
            raise ValueError(f"profile/revision mismatch: {name}")
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
        bias = response.get("pass_bias")
        if bias is not None and (
            type(bias) not in (int, float) or not math.isfinite(bias) or bias < 0
        ):
            raise ValueError(f"invalid pass_bias: {name}")
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
