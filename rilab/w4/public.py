"""Frozen, read-only W4 public account history and table adapter."""

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

CONFIG = Path("configs/v1_4_0_validation/w4.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf8")
    temp.replace(path)


class PublicAPI:
    def __init__(self, root, interval):
        self.root = root
        self.interval = interval
        self.session = httpx.Client(follow_redirects=True)
        self.last_start = 0.0

    def get(self, endpoint, **params):
        url = str(
            httpx.URL(
                "https://api.riichi.dev/api/v1/" + endpoint, params=params or None
            )
        )
        path = self.root / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists():
            return read(path)["data"]
        for attempt in range(5):
            time.sleep(
                max(
                    0,
                    self.interval - (time.monotonic() - self.last_start),
                    2**attempt if attempt else 0,
                )
            )
            self.last_start = time.monotonic()
            try:
                response = self.session.get(url, timeout=35)
                response.raise_for_status()
                result = response.json()
                if not result["ok"]:
                    raise ValueError("unsuccessful API response")
                break
            except (ValueError, httpx.HTTPError):
                if attempt == 4:
                    raise
        save(
            path,
            dict(
                url=url,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                data=result["data"],
            ),
        )
        return result["data"]


def before(stamp, game_id):
    return f"{datetime.fromisoformat(stamp) + timedelta(seconds=1)}|{game_id}"


def collect(cfg):
    root = Path(cfg["output"])
    api = PublicAPI(root / "cache", cfg["request_interval_seconds"])
    games = []
    for bot in cfg["accounts"]:
        cursor = before(cfg["cutoff_utc"], "ffffffff-ffff-ffff-ffff-ffffffffffff")
        unique = {}
        while True:
            page = api.get(f"bots/{bot}/games", limit=50, cursor=cursor)
            rows = page["games"]
            for g in rows:
                if (
                    g["played_at"] <= cfg["cutoff_utc"]
                    and g["game_type"] == "ranked"
                    and g["player_count"] == 4
                ):
                    if g["game_id"] in unique and unique[g["game_id"]] != g:
                        raise ValueError("conflicting game history")
                    unique[g["game_id"]] = g
            if not rows or not page["has_more"]:
                break
            last = rows[-1]
            next_cursor = last["played_at"] + "|" + last["game_id"]
            if next_cursor >= cursor:
                raise ValueError("history cursor stalled")
            cursor = next_cursor
        games.extend(
            dict(g, bot=bot, recent=i < cfg["window"])
            for i, g in enumerate(
                sorted(
                    unique.values(),
                    key=lambda g: (g["played_at"], g["game_id"]),
                    reverse=True,
                )
            )
        )
    save(root / "history.json", games)
    # Process recent games first. A cursor page also supplies nearby games of repeat opponents.
    ratings = {}
    recent_count = sum(g["recent"] for g in games)
    ordered = sorted(games, key=lambda g: (g["recent"], g["played_at"]), reverse=True)
    for number, g in enumerate(ordered, start=1):
        detail = api.get("games/" + g["game_id"])
        players = sorted(detail["players"], key=lambda p: p["seat"])
        focal = players[g["seat"]]
        if (
            [p["seat"] for p in players] != list(range(4))
            or focal["bot_id"] != g["bot"]
            or any(focal[k] != g[k] for k in ("rank", "score", "rating_delta"))
            or detail["id"] != g["game_id"]
        ):
            raise ValueError(f"table mismatch {g['game_id']}")
        g["scores"] = [p["score"] for p in players]
        g["bot_name"] = focal["bot_name"]
        g["opponents"] = []
        for p in players:
            if p["seat"] == g["seat"]:
                continue
            opp = {
                k: p[k]
                for k in (
                    "bot_id",
                    "seat",
                    "rank",
                    "score",
                    "is_disconnected",
                    "is_penalized",
                )
            }
            if g["recent"]:
                bot = p["bot_id"]
                key = (bot, g["game_id"])
                if key not in ratings:
                    page = api.get(
                        f"bots/{bot}/games",
                        limit=50,
                        cursor=before(g["played_at"], g["game_id"]),
                    )
                    ratings.update(((bot, r["game_id"]), r) for r in page["games"])
                r = ratings.get(key, {})
                if r and any(r[k] != p[k] for k in ("rank", "score", "seat")):
                    raise ValueError("opponent join mismatch")
                opp.update({k: r.get(k) for k in ("rating_before", "mu_before")})
            g["opponents"].append(opp)
        if number == recent_count:
            save(root / "recent.json", [x for x in games if x["recent"]])
    save(root / "tables.json", games)


if __name__ == "__main__":
    collect(read(CONFIG))
