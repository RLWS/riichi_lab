"""Deterministic seat schedules, preserving focal identity in controls."""

from itertools import combinations


def expand_schedule(config: dict) -> list[dict]:
    seeds = config["engine"].get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be nonempty unique uint32 integers")
    protocol = config.get("evaluation", {})
    if (
        protocol.get("cluster") != "seed_with_all_rotations"
        or protocol.get("confidence_level") != 0.95
        or not protocol.get("primary_paired_group")
    ):
        raise ValueError("missing evaluation protocol")
    groups = {}
    for m in config["matchups"]:
        group = m.get("paired_group")
        if group:
            if m["rotation"] != "cyclic4":
                raise ValueError("paired groups require cyclic4")
            groups.setdefault(group, []).append(m)
    if protocol["primary_paired_group"] not in groups:
        raise ValueError("primary paired group missing")
    for members in groups.values():
        if len(members) != 2 or members[0]["focal_seat"] != members[1]["focal_seat"]:
            raise ValueError("paired group requires two aligned focal matchups")
        a, b = members
        focal = a["focal_seat"]
        if any(a["seats"][s] != b["seats"][s] for s in range(4) if s != focal):
            raise ValueError("paired opponents must match")
    schedule = []
    for seed in seeds:
        for m in config["matchups"]:
            for rotation in range(4 if m["rotation"] == "cyclic4" else 6):
                if m["rotation"] == "cyclic4":
                    seats = [m["seats"][(s - rotation) % 4] for s in range(4)]
                    focal = (m["focal_seat"] + rotation) % 4
                else:
                    first, second = dict.fromkeys(m["seats"])
                    selected = list(combinations(range(4), 2))[rotation]
                    seats = [first if s in selected else second for s in range(4)]
                    focal = None
                schedule.append(
                    {
                        "game_id": f"g{len(schedule):05d}",
                        "seed": seed,
                        "matchup_id": m["id"],
                        "rotation": rotation,
                        "seats": seats,
                        "focal_seat": focal,
                        "paired_group": m.get("paired_group"),
                        "paired_key": [m["paired_group"], seed, rotation]
                        if m.get("paired_group")
                        else None,
                    }
                )
    return schedule
