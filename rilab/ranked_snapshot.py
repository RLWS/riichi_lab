"""Read-only, standard-library snapshot extraction; never imports a model/client."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def extract_session(path):
    captured_stat = path.stat()
    size = captured_stat.st_size
    digest = hashlib.sha256()
    seat = None
    scores = ranks = match_id = None
    hands = wins = deal_ins = riichi = calls = 0
    flags = None
    invalid_lines = 0
    with path.open("rb") as stream:
        remaining = size
        while remaining > 0:
            line = stream.readline(remaining)
            if not line:
                break
            remaining -= len(line)
            digest.update(line)
            try:
                record = json.loads(line)
            except ValueError:
                invalid_lines += 1
                continue
            if record.get("kind") != "received":
                continue
            event = record.get("message", {})
            kind = event.get("type")
            if kind == "start_game":
                seat = event.get("id")
                match_id = event.get("game_id", event.get("match_id"))
            elif kind == "start_kyoku":
                flags = dict.fromkeys(("win", "deal_in", "riichi", "call"), False)
            elif flags is not None and kind == "hora":
                flags["win"] |= event.get("actor") == seat
                flags["deal_in"] |= (
                    event.get("target") == seat and event.get("actor") != seat
                )
            elif flags is not None and kind == "reach_accepted":
                flags["riichi"] |= event.get("actor") == seat
            elif flags is not None and kind in ("chi", "pon", "daiminkan"):
                flags["call"] |= event.get("actor") == seat
            elif kind == "end_kyoku" and flags is not None:
                hands += 1
                wins += flags["win"]
                deal_ins += flags["deal_in"]
                riichi += flags["riichi"]
                calls += flags["call"]
                flags = None
            elif kind == "end_game":
                scores, ranks = event.get("scores"), event.get("ranks")
    return {
        "seat": seat,
        "scores": scores,
        "ranks": ranks,
        "match_id": match_id,
        "hands": hands,
        "wins": wins,
        "deal_ins": deal_ins,
        "riichi": riichi,
        "calls": calls,
        "trace_sha256": digest.hexdigest(),
        "trace_bytes_read": size,
        "invalid_or_partial_lines": invalid_lines,
        "trace_mtime_utc": captured_stat.st_mtime,
        "changed_during_read": path.stat().st_size != size,
    }


def snapshot(root):
    root = Path(root).resolve()
    batches, errors = [], []
    weight_cache = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        manifest = read_json(directory / "manifest.json")
        if not manifest or manifest.get("mode") != "ranked":
            continue
        summary = read_json(directory / "summary.json") or {}
        backend = manifest["config"]["backend"]
        model_metadata = manifest.get("models", {})
        verification = {}
        for task in ("turn", "response"):
            raw_path = model_metadata.get(task, {}).get("checkpoint")
            path = Path(raw_path).resolve() if isinstance(raw_path, str) else None
            allowed = (
                path is not None
                and path.suffix in (".pt", ".pth")
                and any(
                    path.is_relative_to(root.parent.parent / name)
                    for name in ("train_mj", "train_mj_120")
                )
            )
            actual = None
            if allowed and path.is_file():
                if str(path) not in weight_cache:
                    h = hashlib.sha256()
                    with path.open("rb") as file:
                        for chunk in iter(lambda: file.read(1024 * 1024), b""):
                            h.update(chunk)
                    weight_cache[str(path)] = h.hexdigest()
                actual = weight_cache[str(path)]
            expected = backend.get("model_identity", {}).get(task + "_sha256")
            verification[task] = {
                "declared_sha256": expected,
                "current_file_sha256": actual,
                "current_file_matches": actual == expected if actual else None,
            }
        batch = {
            "batch_id": directory.name,
            "created_at": manifest.get("created_at"),
            "status": summary.get("status"),
            "started": summary.get("games_started"),
            "declared_completed": summary.get("games_completed"),
            "stop_reasons": summary.get("stop_reasons"),
            "backend": {
                k: backend.get(k)
                for k in ("adapter_profile", "revision", "model_identity", "response")
            },
            "models": model_metadata,
            "current_weight_checks": verification,
            "manifest_sha256": hashlib.sha256(
                (directory / "manifest.json").read_bytes()
            ).hexdigest(),
            "games": [],
        }
        for folder in sorted(directory.glob("game_*")):
            trace = folder / "session.jsonl"
            if not trace.is_file():
                errors.append(
                    {"path": str(folder.relative_to(root)), "reason": "missing_trace"}
                )
                continue
            review = read_json(folder / "review.json") or {}
            state = read_json(folder / "summary.json") or {}
            try:
                extracted = extract_session(trace)
            except (OSError, ValueError) as exc:
                errors.append(
                    {
                        "path": str(folder.relative_to(root)),
                        "reason": type(exc).__name__,
                    }
                )
                continue
            batch["games"].append(
                {
                    "source_id": str(folder.relative_to(root)),
                    **extracted,
                    "client_completed": state.get("completed"),
                    "protocol_passed": review.get("protocol_passed"),
                    "faults": len(state.get("faults", [])),
                    "fallbacks": len(review.get("fallback_request_ids", [])),
                    "defaulted": len(review.get("defaulted_request_ids", [])),
                }
            )
        batches.append(batch)
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "read_only": True,
        "batches": batches,
        "read_errors": errors,
        "provenance_note": "Current checkpoint hashes verify current files, not historical in-memory weights. No auth messages, tokens, hands or model inputs exported.",
    }
