"""A3 artifact serialization and independent completeness audit."""

import hashlib
import gzip
import json
from pathlib import Path


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_batch(output_dir):
    """Reject missing, duplicated or altered game outputs before accepting a run."""
    out = Path(output_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    schedule = manifest["schedule"]
    expected = {s["game_id"] for s in schedule}
    actual = {p.stem for p in (out / "results").glob("*.json")}
    if len(expected) != len(schedule) or expected != actual:
        raise ValueError("missing, duplicate or unexpected game results")
    results = []
    for spec in schedule:
        game_id = spec["game_id"]
        result = json.loads(
            (out / "results" / f"{game_id}.json").read_text(encoding="utf-8")
        )
        if (
            any(result.get(k) != v for k, v in spec.items())
            or result.get("valid") is not True
            or result.get("fallbacks") != 0
            or result.get("decisions", 0) <= 0
        ):
            raise ValueError(f"invalid game result: {game_id}")
        events = out / "games" / f"{game_id}.mjai.jsonl"
        if hashlib.sha256(events.read_bytes()).hexdigest() != result["event_digest"]:
            raise ValueError(f"event digest mismatch: {game_id}")
        decisions = out / "decisions" / f"{game_id}.jsonl.gz"
        if not decisions.is_file():
            raise ValueError(f"missing decision log: {game_id}")
        digest = hashlib.sha256()
        all_states = True
        count = 0
        event_cursors = [0] * 4
        with gzip.open(decisions, "rt", encoding="utf-8") as log:
            for line in log:
                record = json.loads(line)
                p = record["player_id"]
                if (
                    record["decision_id"] != count
                    or record.get("fallback") is not False
                    or record["backend_id"] != spec["seats"][p]
                    or record["event_range"]
                    != [event_cursors[p], event_cursors[p] + len(record["new_events"])]
                ):
                    raise ValueError(f"invalid decision sequence: {game_id}")
                event_cursors[p] = record["event_range"][1]
                stable = {
                    k: v
                    for k, v in record.items()
                    if k not in ("prediction", "elapsed_ms")
                }
                digest.update((encode(stable) + "\n").encode())
                all_states = all_states and "state" in record
                count += 1
        full_required = manifest["config"]["artifacts"].get(
            "full_snapshots_first_run", True
        )
        if (
            count != result["decisions"]
            or (full_required and not all_states)
            or (all_states and digest.hexdigest() != result["decision_digest"])
        ):
            raise ValueError(f"decision digest/count mismatch: {game_id}")
        results.append(result)
    return results
